"""Assemble one source video into features, dense targets and decode structure.

Both the leave-one-video-out experiment and the production final fit need the
same thing from a gold video: resolution-normalised per-frame features from a
finished analysis, single-label targets over phases and verified contacts, and
an unguarded copy of those targets for counting decode constraints.

The guarded and unguarded copies are both required and are not
interchangeable. The guard ring keeps a one-frame timing disagreement out of
the loss, but it also separates every contact from its phase, so constraints
counted on the guarded copy never observe ``rally_live -> attack`` and the
decoder is then forbidden to emit any contact at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import json

import numpy as np

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.prepare import (
    FEATURE_NAMES, build_continuous_features,
)

DENSE_TARGETS_SUBPATH = "derived_7d09206_v2/dense_targets.npz"
# The eight pose channels start here; blanking them is the matched control
# that attributes a gain to poses rather than to a changed trajectory.
POSE_OFFSET = FEATURE_NAMES.index("np_wrist_ball_dist")
POSE_CHANNELS = 8


def load_source(
    stem: str,
    annotations_dir: str | Path,
    gold_dir: str | Path,
    contacts: Sequence[Mapping[str, Any]],
    masked: Sequence[int],
    human_source: str | Path,
    contact_radius: int = 2,
    guard: int = 3,
    zero_pose_channels: bool = False,
) -> dict[str, Any]:
    """Features, targets, decode structure and provenance for one video."""
    annotation_path = Path(annotations_dir) / f"{stem}_annotations.json"
    dense_path = Path(gold_dir) / stem / DENSE_TARGETS_SUBPATH
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame_numbers, features = build_continuous_features(annotations)
    if int(frame_numbers[0]) != 0:
        raise ValueError(f"{stem}: analyses must start at frame 0")

    with np.load(dense_path) as arrays:
        phase_targets = arrays["phase_targets"]
        dense_frames = arrays["frame_numbers"]
    if int(dense_frames[0]) != 0 or dense_frames.shape[0] != phase_targets.shape[0]:
        raise ValueError(f"{stem}: exported phase channel is not frame-aligned")

    length = int(min(features.shape[0], phase_targets.shape[0]))
    if abs(features.shape[0] - phase_targets.shape[0]) > 2:
        raise ValueError(
            f"{stem}: analysis has {features.shape[0]} frames, human export "
            f"has {phase_targets.shape[0]}; more than container rounding"
        )

    covered = sorted({
        identity for contact in contacts for identity in contact["annotation_ids"]
    })
    uncovered = dc.uncovered_human_frames(human_source, stem, covered)
    in_range = [c for c in contacts if c["frame"] < length]
    if len(in_range) != len(contacts):
        raise ValueError(f"{stem}: verified contact past the decoded video")
    masked_frames = [f for f in list(masked) + uncovered if f < length]

    targets, audit = dc.build_targets(
        length, phase_targets, in_range, masked_frames=masked_frames,
        contact_radius=contact_radius, guard=guard,
    )
    structure, _ = dc.build_targets(
        length, phase_targets, in_range, masked_frames=masked_frames,
        contact_radius=contact_radius, guard=0,
    )

    pose_block = features[:length, POSE_OFFSET:POSE_OFFSET + POSE_CHANNELS]
    pose_frames = int(np.count_nonzero(np.any(pose_block != 0.0, axis=1)))
    if zero_pose_channels:
        features = features.copy()
        features[:, POSE_OFFSET:POSE_OFFSET + POSE_CHANNELS] = 0.0

    return {
        "stem": stem,
        "frames": length,
        "features": features[:length],
        "targets": targets,
        "structure": structure,
        "truth": dc.truth_events(in_range),
        "verified_contacts": len(in_range),
        "uncovered_human_frames": len(uncovered),
        "pose_feature_frames": pose_frames,
        "pose_channels_zeroed": bool(zero_pose_channels),
        "net_y": float(annotations.get("net_y", 0.0)),
        "target_audit": audit,
        "annotation_path": str(annotation_path.resolve()),
        "annotation_sha256": dc.file_sha256(annotation_path),
        "dense_targets_sha256": dc.file_sha256(dense_path),
    }


def training_windows(
    sources: Sequence[Mapping[str, Any]], window: int, stride: int,
    contact_repeat: int,
) -> list[tuple[int, int]]:
    """Window starts, repeating contact-bearing windows to fight 1% positives.

    A ``sparse`` source (a reviewed queue, see ``reviewed_contacts``) is masked
    everywhere but at its contacts, so a window without one carries no target
    at all and is skipped rather than spent on an empty loss.
    """
    from action_recognition.rally_spotting.contact_spotter import window_starts

    items: list[tuple[int, int]] = []
    for index, source in enumerate(sources):
        contact_frames = np.flatnonzero(source["targets"] >= dc.CONTACT_OFFSET)
        sparse = bool(source.get("sparse"))
        for start in window_starts(source["frames"], window, stride):
            has_contact = bool(
                contact_frames.size
                and ((contact_frames >= start)
                     & (contact_frames < start + window)).any()
            )
            if sparse and not has_contact:
                continue
            items.extend([(index, start)] * (contact_repeat if has_contact else 1))
    return items


def class_weights(
    targets: Sequence[np.ndarray], maximum: float, mode: str = "max",
) -> np.ndarray:
    """Inverse-frequency weights over observed labels, capped.

    A contact occupies about one percent of frames even after widening, so an
    unweighted cross entropy collapses onto the phase classes. ``mode="max"``
    scales against the largest class, which is what lifts the contact classes.
    """
    counts = np.zeros(dc.NUM_LABELS, dtype=np.float64)
    for values in targets:
        observed = np.asarray(values)
        for label in range(dc.NUM_LABELS):
            counts[label] += int((observed == label).sum())
    weights = np.ones(dc.NUM_LABELS, dtype=np.float32)
    present = counts > 0
    if not present.any():
        return weights
    reference = counts[present].max() if mode == "max" else np.median(counts[present])
    weights[present] = np.minimum(
        reference / counts[present], maximum,
    ).astype(np.float32)
    weights[~present] = 1.0
    return weights


def normalisation(sources: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation over the training sources only."""
    stacked = np.concatenate([source["features"] for source in sources], axis=0)
    mean = stacked.mean(axis=0)
    deviation = stacked.std(axis=0)
    deviation[deviation < 1e-6] = 1.0
    return mean, deviation
