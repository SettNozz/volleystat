"""Serve-anchored rallies built from the dense contact spotter's output.

A rally is a point that was actually played: a serve that someone on the
court then touched. Two things follow from that definition and decide
everything in this module:

* a rally opens on its serve. Contacts are grouped by time, and a group that
  does not begin with a serve is first given one (see below) or else left out
  of the rally list altogether.
* a serve that no contact follows is a service attempt, not a rally. The ball
  went out, into the net, or landed untouched; the spotter cannot tell those
  apart, so the attempt says ``serve_not_played`` and nothing more specific.

The spotter misses serves more often than any other contact, mostly the ones
struck from the far end line where the ball is a few pixels wide. Its decode
then enters ``serve_flight`` straight from ``between`` and emits the
reception, so the group starts with a reception that clearly had a serve in
front of it. The same model's per-frame ``serve`` probability still peaks
about a second before that reception. Measured on ``JIZ3exHe3hA_4k``, 8 of 9
serve-less groups carried a peak of 0.18-0.79 at 1.0-3.0 s before their first
contact, and the last, a lone reception, 0.04. That peak is taken as the
serve, marked ``inferred`` with its probability as confidence, and never
presented as a decoded contact.

Forcing the decoder itself to place a serve before every ``serve_flight`` was
measured and rejected (see ``require_contact_before_phase``); this works after
decoding and leaves the event list it was given untouched.

Nothing here attributes a side. The spotter emits ``team: None``, so the
per-team fields stay ``None`` rather than carrying a guess.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from action_recognition.rally_spotting import dense_contacts as dc

SCHEMA = "spotter_rallies_v1"
SPOTTER_SOURCE = "dense_contact_spotter"
INFERRED_SERVE_SOURCE = "spotter_serve_probability_peak"


@dataclass(frozen=True)
class RallyRules:
    """Timing rules, in seconds, so they hold at any frame rate."""

    # Consecutive contacts of one rally were at most 2.1 s apart on JIZ3, and
    # separate rallies at least 9.9 s. Anything in between splits.
    max_contact_gap_sec: float = 5.0
    # Serve to first contact was 0.7-1.4 s for decoded serves; a missed
    # reception can push the first decoded contact to ~3 s.
    serve_search_sec: float = 3.5
    # A peak this close to the first contact is that contact's own shoulder.
    min_serve_lead_sec: float = 0.3
    min_inferred_serve_probability: float = 0.15
    # The decoded phase track says when the ball went dead; this caps it.
    max_tail_sec: float = 3.0


def segment_rallies(
    events: Sequence[Mapping[str, Any]],
    fps: float,
    serve_probability: Sequence[float] | None = None,
    phase_labels: Sequence[int] | None = None,
    frame_numbers: Sequence[int] | None = None,
    rules: RallyRules = RallyRules(),
) -> dict[str, Any]:
    """Group spotter contacts into serve-anchored rallies.

    ``serve_probability``, ``phase_labels`` and ``frame_numbers`` are indexed
    the same way (the spotter's feature rows); ``frame_numbers`` maps a row to
    the video's own frame. Without probabilities no serve is inferred; without
    phase labels a rally ends on its last contact.

    Returns the rallies (playable only), every service attempt, the counters,
    the inferred serve events to add to the event list, and the contact groups
    that had no serve and so belong to no rally.
    """
    fps = float(fps) if fps else 30.0
    contacts = sorted(
        (dict(event) for event in events), key=lambda event: int(event["frame"]),
    )
    row_of: dict[int, int] = {}
    if frame_numbers is not None:
        row_of = {int(frame): row for row, frame in enumerate(frame_numbers)}
    probability = (
        np.asarray(serve_probability, dtype=np.float64)
        if serve_probability is not None else None
    )
    labels = np.asarray(phase_labels, dtype=np.int64) if phase_labels is not None else None

    groups = _group_contacts(contacts, round(rules.max_contact_gap_sec * fps))

    attempts: list[dict[str, Any]] = []
    rallies: list[dict[str, Any]] = []
    inferred: list[dict[str, Any]] = []
    unanchored: list[dict[str, Any]] = []
    previous_end = -1
    for group in groups:
        serve = group[0] if group[0]["type"] == "serve" else None
        if serve is None:
            serve = _infer_serve(
                first_frame=int(group[0]["frame"]),
                not_before=previous_end,
                probability=probability,
                row_of=row_of,
                frame_numbers=frame_numbers,
                fps=fps,
                rules=rules,
            )
            if serve is not None:
                inferred.append(serve)
        previous_end = int(group[-1]["frame"])
        if serve is None:
            unanchored.append({
                "start_frame": int(group[0]["frame"]),
                "end_frame": int(group[-1]["frame"]),
                "contacts": [
                    {"frame": int(event["frame"]), "type": event["type"]}
                    for event in group
                ],
                "reason": "no_serve_evidence",
            })
            continue

        played = [event for event in group if event is not serve]
        attempt_number = len(attempts) + 1
        end_frame = _rally_end(
            int(group[-1]["frame"]), labels, row_of, frame_numbers,
            round(rules.max_tail_sec * fps),
        )
        attempt = _attempt_record(
            attempt_number, serve, played, end_frame, fps,
        )
        attempts.append(attempt)
        if played:
            rally = dict(attempt)
            rally["rally_id"] = f"rally_{len(rallies) + 1:04d}"
            rallies.append(rally)
            attempt["rally_id"] = rally["rally_id"]

    decoded_serves = sum(1 for a in attempts if a["serve_source"] == SPOTTER_SOURCE)
    lifecycle = {
        "schema": SCHEMA,
        "source": SPOTTER_SOURCE,
        "service_attempts": len(attempts),
        "confirmed_serves": decoded_serves,
        "inferred_serves": len(attempts) - decoded_serves,
        "playable_rallies": len(rallies),
        "serves_not_played": len(attempts) - len(rallies),
        "contact_groups_without_serve": len(unanchored),
    }
    return {
        "rallies": rallies,
        "service_attempts": attempts,
        "rally_lifecycle": lifecycle,
        "inferred_serves": inferred,
        "unanchored_groups": unanchored,
        "rules": asdict(rules),
    }


def _group_contacts(
    contacts: list[dict[str, Any]], max_gap_frames: int,
) -> list[list[dict[str, Any]]]:
    """Split on long pauses, and before every serve: a serve starts a point."""
    groups: list[list[dict[str, Any]]] = []
    for event in contacts:
        starts_new = (
            not groups
            or event["type"] == "serve"
            or int(event["frame"]) - int(groups[-1][-1]["frame"]) > max_gap_frames
        )
        if starts_new:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def _infer_serve(
    first_frame: int,
    not_before: int,
    probability: np.ndarray | None,
    row_of: Mapping[int, int],
    frame_numbers: Sequence[int] | None,
    fps: float,
    rules: RallyRules,
) -> dict[str, Any] | None:
    """The spotter's own strongest ``serve`` frame shortly before a group."""
    if probability is None or frame_numbers is None or first_frame not in row_of:
        return None
    first_row = row_of[first_frame]
    lead = max(1, round(rules.min_serve_lead_sec * fps))
    low = max(0, first_row - round(rules.serve_search_sec * fps))
    high = first_row - lead
    # Never reach back into the previous group's contacts.
    while low < high and int(frame_numbers[low]) <= not_before:
        low += 1
    if high <= low:
        return None
    window = probability[low:high]
    peak = int(np.argmax(window))
    confidence = float(window[peak])
    if confidence < rules.min_inferred_serve_probability:
        return None
    frame = int(frame_numbers[low + peak])
    return {
        "frame": frame,
        "time": round(frame / fps, 2),
        "type": "serve",
        "core_type": dc.to_core("serve"),
        "confidence": confidence,
        "source": INFERRED_SERVE_SOURCE,
        "inferred": True,
        "team": None,
        "game_state": None,
        "rally_phase": None,
    }


def _rally_end(
    last_contact: int,
    labels: np.ndarray | None,
    row_of: Mapping[int, int],
    frame_numbers: Sequence[int] | None,
    max_tail_frames: int,
) -> int:
    """Last frame before the decoded track returns to ``between``."""
    if labels is None or frame_numbers is None or last_contact not in row_of:
        return last_contact
    between = dc.LABEL_TYPES.index("between")
    row = row_of[last_contact]
    limit = min(len(labels) - 1, row + max_tail_frames)
    while row < limit and int(labels[row + 1]) not in (between, dc.IGNORE_INDEX):
        row += 1
    return int(frame_numbers[row])


def _attempt_record(
    number: int,
    serve: Mapping[str, Any],
    played: Sequence[Mapping[str, Any]],
    end_frame: int,
    fps: float,
) -> dict[str, Any]:
    serve_frame = int(serve["frame"])
    decoded = serve.get("source") == SPOTTER_SOURCE
    serve_source = SPOTTER_SOURCE if decoded else INFERRED_SERVE_SOURCE
    first = int(played[0]["frame"]) if played else None
    second = int(played[1]["frame"]) if len(played) > 1 else None
    is_rally = bool(played)
    status = "playable_rally" if is_rally else "serve_not_played"
    end_frame = max(end_frame, serve_frame)
    duration = end_frame - serve_frame
    return {
        "service_attempt_id": f"service_attempt_{number:04d}",
        "start_frame": serve_frame,
        "end_frame": end_frame,
        "duration_frames": duration,
        "duration_sec": round(duration / fps, 2),
        "events": [dict(serve), *(dict(event) for event in played)],
        "contacts_after_serve": len(played),
        # No side attribution exists for spotter contacts.
        "contacts_team_a": None,
        "contacts_team_b": None,
        "serve_source": serve_source,
        "serve_confidence": serve.get("confidence"),
        "source": SPOTTER_SOURCE,
        "lifecycle": {
            "status": status,
            "is_playable": is_rally,
            "serve_confirmed": decoded,
            "start_frame": serve_frame,
            "serve_frame": serve_frame,
            "net_cross_frame": None,
            "reception_frame": first,
            "followup_frame": second,
            "serving_team": None,
            "receiving_team": None,
            "end_reason": "ball_dead" if is_rally else "no_contact_after_serve",
            # Out, net and an untouched ace all look the same here.
            "serve_outcome": "in_play" if is_rally else "unknown",
            "reception_quality": "unknown",
        },
        "serve_reception": {
            "serve": {"frame": serve_frame, "team": None, "confirmed": decoded},
            "reception": (
                {"frame": first, "team": None, "outcome": status}
                if first is not None else None
            ),
            "outcome": status,
        },
    }
