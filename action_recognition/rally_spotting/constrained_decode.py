"""Constraint-aware Viterbi decoding for phase/contact label sequences.

The structure a hand-written state machine encodes by hand - which state may
follow which, how long a state may last, what may open or close a video - is
estimated here by counting human-labelled segments instead.

Method follows "Improving Temporal Action Segmentation via Constraint-Aware
Decoding" (arXiv 2605.10149): maximise

    sum_t log P[t, y_t] + w * sum_t log Conf(y_{t-1} -> y_t)

subject to hard constraints on valid start labels, valid end labels, observed
transitions and per-class segment duration. Two deliberate deviations:

* duration bounds are kept in **frames**, not as a fraction of video length.
  Our classes have physical durations (a contact is a few frames at 29.97 fps
  whatever the video length), so normalising by length would couple them to
  how long a match happens to be.
* the terminal minimum-duration check is available but off by default. The
  published recurrence checks only the maximum for the final segment, and for
  continuous match video that is the right choice: the last segment is cut off
  by where the file ends, so holding it to a full minimum duration can make an
  otherwise fine decode infeasible.

As in the paper, the per-cell duration is carried greedily rather than being
part of the Viterbi state, so the result is the optimum over paths reachable
under that bookkeeping, not provably the constrained optimum. Decoding is
O(T * C^2).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from action_recognition.rally_spotting.dense_contacts import (
    IGNORE_INDEX,
    LABEL_TYPES,
    NUM_LABELS,
    is_contact_label,
    label_to_contact,
)

SCHEMA = "constraint_aware_decode_v1"
NEG_INF = -1.0e30


class InfeasibleConstraints(RuntimeError):
    """No label sequence satisfies the estimated constraints."""


def segments_of(labels: Sequence[int]) -> list[tuple[int, int, int]]:
    """Contiguous ``(label, start, end_inclusive)`` runs, skipping masked frames.

    A masked run breaks the sequence: the labels on either side are not treated
    as adjacent, because what happened in between is unknown.
    """
    values = np.asarray(labels, dtype=np.int64)
    result: list[tuple[int, int, int]] = []
    start = None
    for index, value in enumerate(values.tolist()):
        if value == IGNORE_INDEX:
            if start is not None:
                result.append((int(values[start]), start, index - 1))
                start = None
            continue
        if start is None:
            start = index
        elif values[index] != values[start]:
            result.append((int(values[start]), start, index - 1))
            start = index
    if start is not None:
        result.append((int(values[start]), start, len(values) - 1))
    return result


def estimate_constraints(
    label_sequences: Iterable[Sequence[int]],
    num_labels: int = NUM_LABELS,
    duration_quantile: float = 0.0,
    duration_slack: float = 1.0,
    self_loop_labels: Iterable[int] = (),
) -> dict[str, Any]:
    """Count transitions, start/end labels and duration bounds.

    Call this on the **training fold only**. ``duration_quantile`` trims the
    observed extremes (0.0 keeps the plain minimum and maximum);
    ``duration_slack`` multiplies the upper bound and divides the lower one.
    ``self_loop_labels`` are labels whose upper duration bound is dropped,
    for states that may legitimately run longer than anything observed.
    """
    if not 0.0 <= duration_quantile < 0.5:
        raise ValueError("duration_quantile must be within [0, 0.5)")
    if duration_slack < 1.0:
        raise ValueError("duration_slack must be at least 1.0")
    transitions = Counter()
    sources = Counter()
    starts = Counter()
    ends = Counter()
    durations: dict[int, list[int]] = {}
    sequence_count = 0

    for labels in label_sequences:
        runs = segments_of(labels)
        if not runs:
            continue
        sequence_count += 1
        starts[runs[0][0]] += 1
        ends[runs[-1][0]] += 1
        for label, start, end in runs:
            durations.setdefault(label, []).append(end - start + 1)
        previous_end = None
        previous_label = None
        for label, start, end in runs:
            if previous_label is not None and start == previous_end + 1:
                transitions[(previous_label, label)] += 1
                sources[previous_label] += 1
            previous_label, previous_end = label, end

    if not sequence_count:
        raise ValueError("no usable label sequence")

    confidence = np.full((num_labels, num_labels), NEG_INF, dtype=np.float64)
    for (source, target), count in transitions.items():
        confidence[source, target] = float(np.log(count / sources[source]))

    unbounded = set(int(value) for value in self_loop_labels)
    minimum = np.ones(num_labels, dtype=np.int64)
    maximum = np.zeros(num_labels, dtype=np.int64)
    for label in range(num_labels):
        observed = durations.get(label)
        if not observed:
            maximum[label] = 0  # Never observed: forbidden below.
            continue
        values = np.asarray(sorted(observed), dtype=np.float64)
        low = float(np.quantile(values, duration_quantile))
        high = float(np.quantile(values, 1.0 - duration_quantile))
        minimum[label] = max(1, int(np.floor(low / duration_slack)))
        maximum[label] = (
            np.iinfo(np.int64).max if label in unbounded
            else max(int(minimum[label]), int(np.ceil(high * duration_slack)))
        )

    return {
        "schema": SCHEMA,
        "num_labels": int(num_labels),
        "log_transition": confidence,
        "transition_counts": {
            f"{LABEL_TYPES[s]}->{LABEL_TYPES[t]}": int(c)
            for (s, t), c in sorted(transitions.items())
        },
        "valid_start": sorted(starts),
        "valid_end": sorted(ends),
        "duration_min": minimum,
        "duration_max": maximum,
        "observed_labels": sorted(durations),
        "sequences": sequence_count,
        "duration_quantile": float(duration_quantile),
        "duration_slack": float(duration_slack),
        "unbounded_labels": sorted(unbounded),
    }


def unseen_transitions(
    constraints: Mapping[str, Any], label_sequences: Iterable[Sequence[int]],
) -> dict[str, int]:
    """Transitions present in held-out labels but forbidden by the constraints.

    With a few hundred labelled contacts a legal-but-unobserved transition is
    forbidden outright, so this has to be measured, not assumed away.
    """
    allowed = np.asarray(constraints["log_transition"]) > NEG_INF
    missing = Counter()
    for labels in label_sequences:
        runs = segments_of(labels)
        for (source, _, end), (target, start, _) in zip(runs, runs[1:]):
            if start == end + 1 and not allowed[source, target]:
                missing[f"{LABEL_TYPES[source]}->{LABEL_TYPES[target]}"] += 1
    return dict(missing)


def constrained_viterbi(
    log_probabilities: np.ndarray,
    constraints: Mapping[str, Any],
    transition_weight: float = 1.0,
    enforce_terminal_minimum: bool = False,
) -> np.ndarray:
    """Best constrained label sequence for one video.

    ``log_probabilities`` is ``(frames, labels)``. With
    ``enforce_terminal_minimum`` the final segment must also meet its minimum
    duration; by default it need not, because the video end truncates it.
    Raises :class:`InfeasibleConstraints` when no path satisfies the
    constraints.
    """
    scores = np.asarray(log_probabilities, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("log_probabilities must be (frames, labels)")
    frames, labels = scores.shape
    if labels != int(constraints["num_labels"]):
        raise ValueError("label count does not match the constraints")
    if frames == 0:
        raise ValueError("no frames to decode")

    log_transition = np.asarray(constraints["log_transition"], dtype=np.float64)
    duration_min = np.asarray(constraints["duration_min"], dtype=np.int64)
    duration_max = np.asarray(constraints["duration_max"], dtype=np.int64)
    valid_start = np.asarray(constraints["valid_start"], dtype=np.int64)
    valid_end = np.asarray(constraints["valid_end"], dtype=np.int64)
    weighted = log_transition * float(transition_weight)

    value = np.full((frames, labels), NEG_INF, dtype=np.float64)
    duration = np.zeros((frames, labels), dtype=np.int64)
    back = np.full((frames, labels), -1, dtype=np.int64)

    value[0, valid_start] = scores[0, valid_start]
    duration[0, valid_start] = 1

    for frame in range(1, frames):
        previous = value[frame - 1]
        previous_duration = duration[frame - 1]
        # A class change is allowed only out of a segment that already met its
        # minimum duration, and only along an observed transition.
        eligible = (previous > NEG_INF) & (previous_duration >= duration_min)
        candidate = np.where(
            eligible[:, None], previous[:, None] + weighted, NEG_INF,
        )
        np.fill_diagonal(candidate, NEG_INF)
        source = np.argmax(candidate, axis=0)
        best_change = candidate[source, np.arange(labels)]

        # Staying in the same segment needs room under the maximum duration.
        can_stay = (previous > NEG_INF) & (previous_duration < duration_max)
        stay = np.where(can_stay, previous, NEG_INF)

        take_stay = stay >= best_change
        chosen = np.where(take_stay, stay, best_change)
        feasible = chosen > NEG_INF
        value[frame] = np.where(feasible, chosen + scores[frame], NEG_INF)
        duration[frame] = np.where(
            feasible, np.where(take_stay, previous_duration + 1, 1), 0,
        )
        back[frame] = np.where(
            feasible, np.where(take_stay, np.arange(labels), source), -1,
        )
        if not feasible.any():
            raise InfeasibleConstraints(
                f"no feasible label at frame {frame}"
            )

    final_duration = duration[frames - 1]
    terminal = np.full(labels, NEG_INF, dtype=np.float64)
    for label in valid_end.tolist():
        lower = duration_min[label] if enforce_terminal_minimum else 1
        if (
            value[frames - 1, label] > NEG_INF
            and lower <= final_duration[label] <= duration_max[label]
        ):
            terminal[label] = value[frames - 1, label]
    if not np.any(terminal > NEG_INF):
        raise InfeasibleConstraints(
            "no valid end label satisfies its duration bounds"
        )

    result = np.empty(frames, dtype=np.int64)
    result[frames - 1] = int(np.argmax(terminal))
    for frame in range(frames - 2, -1, -1):
        result[frame] = back[frame + 1, result[frame + 1]]
        if result[frame] < 0:
            raise InfeasibleConstraints(f"broken backpointer at frame {frame}")
    return result


def events_from_labels(
    labels: Sequence[int], probabilities: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """One event per decoded contact segment, timed at the segment centre.

    With ``probabilities`` given, the event frame is the highest-scoring frame
    of its own class inside the segment instead of the midpoint.
    """
    events: list[dict[str, Any]] = []
    for label, start, end in segments_of(labels):
        if not is_contact_label(label):
            continue
        if probabilities is None:
            frame = (start + end) // 2
            confidence = None
        else:
            window = np.asarray(probabilities)[start:end + 1, label]
            frame = start + int(np.argmax(window))
            confidence = float(window.max())
        events.append({
            "frame": int(frame),
            "type": label_to_contact(label),
            "start_frame": int(start),
            "end_frame": int(end),
            "confidence": confidence,
        })
    return events


def argmax_events(
    probabilities: np.ndarray, threshold: float = 0.5, radius: int = 5,
) -> list[dict[str, Any]]:
    """Unconstrained baseline: per-class peak picking over contact channels.

    This is what the dense model gives without any structural decoding, and is
    the control the constrained decoder has to beat.
    """
    from action_recognition.rally_spotting.metrics import nms_peaks

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != NUM_LABELS:
        raise ValueError("probabilities must be (frames, labels)")
    phase_total = values[:, :3].sum(axis=1)
    events: list[dict[str, Any]] = []
    for label in range(3, NUM_LABELS):
        channel = np.where(values[:, label] > phase_total, values[:, label], 0.0)
        for frame in nms_peaks(channel, threshold, radius):
            events.append({
                "frame": int(frame),
                "type": label_to_contact(label),
                "confidence": float(values[frame, label]),
            })
    return sorted(events, key=lambda event: event["frame"])
