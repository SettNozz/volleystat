"""Production contact spotter: one checkpoint, one call, events out.

The experiment in ``scripts/evaluation/run_dense_contact_spotting.py`` trains a
model per held-out video and scores it. This is the other half: a checkpoint
that carries everything inference needs -- the trained members, the
normalisation fitted on their training data, and the counted decode
constraints -- so a new video can be processed without any human labels.

The constraints must travel with the checkpoint. They are counted from the
training videos, and at inference there is no training fold to count them from,
so a checkpoint without them cannot be decoded.

What this does **not** do:

* it does not tell warm-up from play. Measured on the one gold video with a
  real warm-up, the decoder labels those frames ``rally_live`` and emits
  contacts there; 35 of 36 such false positives sit inside a decoded in-play
  phase, so no phase gate here can remove them. Excluding non-game time is
  ``--match-config``'s job in the pipeline.
* it does not attribute a contact to a team or a player.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from action_recognition.rally_spotting import constrained_decode as cd
from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.model import RallySpottingTCN
from action_recognition.rally_spotting.prepare import (
    FEATURE_NAMES, build_continuous_features,
)

SCHEMA = "contact_spotter_v1"
DEFAULT_WINDOW = 240
DEFAULT_STRIDE = 120
# ``between`` and ``rally_live`` may legitimately run longer than anything
# observed, so their upper duration bound is dropped rather than guessed.
UNBOUNDED = (0, 2)


def window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def predict_dense(
    model: nn.Module, features: np.ndarray, window: int, stride: int,
    device: torch.device,
) -> np.ndarray:
    """Per-frame class probabilities, averaged over overlapping windows."""
    model.eval()
    length = features.shape[0]
    total = np.zeros((length, dc.NUM_LABELS), dtype=np.float64)
    counts = np.zeros((length, 1), dtype=np.float64)
    with torch.no_grad():
        for start in window_starts(length, window, stride):
            end = min(start + window, length)
            padded = np.zeros((window, features.shape[1]), dtype=np.float32)
            padded[:end - start] = features[start:end]
            logits = model(torch.from_numpy(padded[None]).to(device))
            probabilities = torch.softmax(logits, dim=-1)[0, :end - start]
            total[start:end] += probabilities.double().cpu().numpy()
            counts[start:end] += 1.0
    if (counts == 0).any():
        raise ValueError("uncovered frame in sliding-window inference")
    return total / counts


def save_checkpoint(
    path: str | Path,
    members: Sequence[Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
    mean: np.ndarray,
    deviation: np.ndarray,
    constraints: Mapping[str, Any],
    provenance: Mapping[str, Any],
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    transition_weight: float = 1.0,
) -> None:
    """Write a checkpoint that inference can use on its own."""
    if not members:
        raise ValueError("a checkpoint needs at least one trained member")
    torch.save(
        {
            "schema": SCHEMA,
            "members": [dict(state) for state in members],
            "config": dict(config),
            "mean": np.asarray(mean, dtype=np.float64),
            "std": np.asarray(deviation, dtype=np.float64),
            "constraints": dict(constraints),
            "feature_names": list(FEATURE_NAMES),
            "label_types": list(dc.LABEL_TYPES),
            "window": int(window),
            "stride": int(stride),
            "transition_weight": float(transition_weight),
            "provenance": dict(provenance),
        },
        str(path),
    )


def require_contact_before_phase(
    constraints: Mapping[str, Any], phase: str = "serve_flight",
    contact: str = "serve",
) -> dict[str, Any]:
    """Forbid entering ``phase`` from anything but ``contact``.

    The counted grammar permits ``between -> serve_flight`` because the human
    targets contain nine of them: where a serve's timing was uncertain the
    contact is masked, leaving a gap that reads as the phase starting on its
    own. The transition is an artefact of masking, not of volleyball -- a ball
    is not in serve flight until someone has served it.

    It matters because the decoder will take it. On footage where the serve
    contact is less confident the cheap path wins, and a video comes back with
    27 serve flights and 8 serves.

    Returns a copy; the checkpoint's own constraints are left alone.
    """
    tightened = dict(constraints)
    log_transition = np.array(constraints["log_transition"], dtype=np.float64)
    phase_index = dc.LABEL_TYPES.index(phase)
    contact_index = dc.LABEL_TYPES.index(contact)
    removed = []
    for source in range(log_transition.shape[0]):
        if source == contact_index or source == phase_index:
            continue
        if np.isfinite(log_transition[source, phase_index]):
            removed.append(f"{dc.LABEL_TYPES[source]}->{phase}")
            log_transition[source, phase_index] = -np.inf
    tightened["log_transition"] = log_transition
    tightened["tightened"] = {
        "rule": f"{phase} may only be entered from {contact}",
        "removed_transitions": removed,
    }
    return tightened


@dataclass
class SpottedEvents:
    """What one video yielded, with enough detail to audit it."""

    events: list[dict[str, Any]]
    phase_labels: np.ndarray
    frame_numbers: np.ndarray
    decode_status: str
    diagnostics: dict[str, Any]
    # Member-averaged per-row class probabilities, aligned with
    # ``phase_labels``. Rally segmentation reads the ``serve`` column to find
    # serves the decode left out.
    probabilities: np.ndarray | None = None


class ContactSpotter:
    """Loaded checkpoint, ready to run over a finished analysis."""

    def __init__(self, payload: Mapping[str, Any], device: torch.device | None = None):
        schema = payload.get("schema")
        if schema != SCHEMA:
            raise ValueError(
                f"checkpoint schema {schema!r} is not {SCHEMA!r}; an "
                "experiment checkpoint is not production-loadable"
            )
        saved_features = list(payload.get("feature_names") or [])
        if saved_features != list(FEATURE_NAMES):
            raise ValueError(
                "checkpoint was trained on different feature channels "
                f"({len(saved_features)} vs {len(FEATURE_NAMES)}); refusing to "
                "run rather than silently feeding it the wrong inputs"
            )
        if list(payload.get("label_types") or []) != list(dc.LABEL_TYPES):
            raise ValueError("checkpoint label space does not match this code")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = dict(payload["config"])
        self.mean = np.asarray(payload["mean"], dtype=np.float64)
        self.deviation = np.asarray(payload["std"], dtype=np.float64)
        self.constraints = dict(payload["constraints"])
        self.window = int(payload.get("window", DEFAULT_WINDOW))
        self.stride = int(payload.get("stride", DEFAULT_STRIDE))
        self.transition_weight = float(payload.get("transition_weight", 1.0))
        self.provenance = dict(payload.get("provenance") or {})
        self.members: list[nn.Module] = []
        for state in payload["members"]:
            model = RallySpottingTCN(**self.config).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.members.append(model)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> "ContactSpotter":
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(payload, device=device)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """Averaged member probabilities over already-built features."""
        normalised = ((features - self.mean) / self.deviation).astype(np.float32)
        stacked = [
            predict_dense(model, normalised, self.window, self.stride, self.device)
            for model in self.members
        ]
        # Each member's rows sum to one, so the mean is still a distribution.
        return np.mean(stacked, axis=0)

    def run(self, annotations: Mapping[str, Any]) -> SpottedEvents:
        """Spot contacts over one finished analysis.

        ``annotations`` is the whole annotation payload, not just its frames:
        the feature builder needs ``frame_width``, ``frame_height``, ``fps``
        and ``net_y`` to produce resolution-normalised channels.
        """
        frame_numbers, features = build_continuous_features(annotations)
        probabilities = self.probabilities(features)
        log_probabilities = np.log(np.clip(probabilities, 1e-12, None))
        decode_status = "ok"
        try:
            labels = cd.constrained_viterbi(
                log_probabilities, self.constraints,
                transition_weight=self.transition_weight,
            )
            decoded = cd.events_from_labels(labels, probabilities)
        except cd.InfeasibleConstraints as error:
            decode_status = f"infeasible: {error}"
            labels = np.full(features.shape[0], dc.IGNORE_INDEX, dtype=np.int64)
            decoded = []

        # Index space -> the video's own frame numbers. They coincide for a
        # full analysis starting at frame 0, and must not be assumed to.
        events: list[dict[str, Any]] = []
        inconsistent = 0
        for event in decoded:
            index = int(event["frame"])
            phase_before, phase_after = self._surrounding_phases(labels, index)
            # A contact the decoder itself surrounds with ``between`` on both
            # sides contradicts its own phase track. This catches very little
            # -- 1 of 36 warm-up false positives when measured -- and is not a
            # warm-up gate; it only drops self-inconsistent output.
            if phase_before == "between" and phase_after == "between":
                inconsistent += 1
                continue
            events.append({
                "frame": int(frame_numbers[index]),
                "type": event["type"],
                "core_type": dc.to_core(event["type"]),
                "confidence": event["confidence"],
                "start_frame": int(frame_numbers[int(event["start_frame"])]),
                "end_frame": int(frame_numbers[int(event["end_frame"])]),
                "phase_before": phase_before,
                "phase_after": phase_after,
                "source": "dense_contact_spotter",
            })

        counts: dict[str, int] = {}
        for event in events:
            counts[event["type"]] = counts.get(event["type"], 0) + 1
        return SpottedEvents(
            events=events,
            phase_labels=labels,
            frame_numbers=np.asarray(frame_numbers),
            probabilities=probabilities,
            decode_status=decode_status,
            diagnostics={
                "schema": SCHEMA,
                "frames": int(features.shape[0]),
                "members": len(self.members),
                "decode_status": decode_status,
                "events": len(events),
                "per_type": counts,
                "dropped_self_inconsistent": inconsistent,
                "warm_up_note": (
                    "contacts in non-game time are not removed here; the "
                    "model cannot tell warm-up from play. Use --match-config."
                ),
                "provenance": self.provenance,
            },
        )

    @staticmethod
    def _surrounding_phases(labels: np.ndarray, index: int) -> tuple[str, str]:
        """The phase labels either side of the contact segment at ``index``."""
        length = labels.shape[0]
        left = index
        while left > 0 and dc.is_contact_label(int(labels[left])):
            left -= 1
        right = index
        while right < length - 1 and dc.is_contact_label(int(labels[right])):
            right += 1
        def name(position: int) -> str:
            value = int(labels[position])
            if value < 0 or dc.is_contact_label(value):
                return "unknown"
            return dc.LABEL_TYPES[value]
        return name(left), name(right)


def phase_segments(
    labels: np.ndarray, frame_numbers: Sequence[int],
) -> list[dict[str, Any]]:
    """Decoded phase runs as frame ranges, contacts excluded.

    The same model that spots contacts also tracks the rally phase, so rally
    segmentation comes out of this pass for free.
    """
    segments = []
    for label, start, end in cd.segments_of(labels):
        if dc.is_contact_label(label) or label < 0:
            continue
        segments.append({
            "phase": dc.LABEL_TYPES[label],
            "start_frame": int(frame_numbers[start]),
            "end_frame": int(frame_numbers[end]),
            "frames": int(end - start + 1),
        })
    return segments
