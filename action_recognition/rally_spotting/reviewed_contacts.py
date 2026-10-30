"""Sparse training targets from a human-reviewed spotter queue.

A ``contact_review_v1`` file (``contact_review_annotator``) holds candidates the
spotter proposed on a video nobody had annotated, each verified by a human:
kept with a type, a side and possibly a corrected frame, or rejected. That is
far less than a full-video annotation, and the difference decides what may be
trained on:

* the reviewer saw only candidate frames. A frame the queue never offered may
  hold a contact the model missed, so it is masked, never background;
* there is no human phase track, so every phase frame is masked too;
* a rejection is not a negative. On the first batch 140 of 149 rejected
  candidates sat within ten frames of a kept contact: ``X`` there meant
  "duplicate of that touch", not "no touch here". Rejections are counted in
  the audit and otherwise ignored;
* two candidates the reviewer moved onto one frame are one touch. When their
  types disagree the frame is kept as a contact and its type is masked.

What remains is one short segment per kept contact at the frame the human
chose, with a guard ring masked around it, using the same rules as the gold
targets in ``dense_contacts``. Nothing here reads model output as a label: the
model's predicted type stays beside the human's in the review file and is
never copied into a target.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from action_recognition.rally_spotting import dense_contacts as dc

REVIEW_SCHEMA = "contact_review_v1"
SCHEMA = "reviewed_sparse_contact_targets_v1"
# A rejected candidate this close to a kept contact is a duplicate of it.
DUPLICATE_RADIUS = 10


def reviewed_frame(contact: Mapping[str, Any]) -> int:
    """Where the human placed the contact; the model's frame when unmoved."""
    override = contact["review"].get("frame")
    return int(contact["frame"] if override is None else override)


def load_review(
    review_path: str | Path, duplicate_radius: int = DUPLICATE_RADIUS,
) -> dict[str, Any]:
    """Human-verified contacts per video, with an audit of what was dropped.

    Returns ``{"videos", "contacts": {stem: [...]}, "masked": {stem: [...]},
    "audit": {stem: {...}}, "review_sha256"}``. Contacts carry the reviewed
    frame, the raw type (``None`` when merged candidates disagree), the core
    type and the review ids, in the shape ``dense_contacts.build_targets``
    takes.
    """
    path = Path(review_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"{path}: expected schema {REVIEW_SCHEMA}")

    contacts: dict[str, dict[int, dict[str, Any]]] = {}
    masked: dict[str, set[int]] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Counter] = {}
    for contact in raw["contacts"]:
        stem = contact["video_id"]
        if stem not in raw["videos"]:
            raise ValueError(f"{contact['id']}: unknown video {stem}")
        contacts.setdefault(stem, {})
        masked.setdefault(stem, set())
        rejected.setdefault(stem, [])
        counts = audit.setdefault(stem, Counter())
        review = contact.get("review")
        verified = (
            bool(contact.get("verified"))
            and contact.get("label_source") == "human"
            and review is not None
        )
        if not verified:
            # Unanswered or unsure: a real touch may be here, at an unknown
            # frame, with an unknown type. Masked, never background.
            counts["unverified_masked"] += 1
            masked[stem].add(int(contact["frame"]))
            continue
        status = review.get("status")
        if status == "rejected":
            counts["rejected"] += 1
            rejected[stem].append(contact)
            continue
        if status != "kept":
            raise ValueError(f"{contact['id']}: unexpected review status {status}")
        contact_type = review.get("type")
        if contact_type not in dc.CONTACT_TYPES:
            raise ValueError(f"{contact['id']}: kept with type {contact_type!r}")
        frame = reviewed_frame(contact)
        counts["kept"] += 1
        if review.get("frame") is not None:
            counts["frame_corrected"] += 1
        if contact.get("predicted_type") != contact_type:
            counts["type_corrected"] += 1
        existing = contacts[stem].get(frame)
        if existing is None:
            contacts[stem][frame] = {
                "frame": frame,
                "contact_type": contact_type,
                "core_type": dc.to_core(contact_type),
                "team": review.get("team"),
                "review_ids": [contact["id"]],
                # Only gold sources carry full-video annotation ids.
                "annotation_ids": [],
                "origin": contact.get("origin"),
            }
            continue
        counts["merged_into_one_frame"] += 1
        existing["review_ids"].append(contact["id"])
        if existing["contact_type"] != contact_type:
            if existing["contact_type"] is not None:
                counts["merged_type_conflicts"] += 1
            existing["contact_type"] = None
        if existing["core_type"] != dc.to_core(contact_type):
            existing["core_type"] = None
        if existing["team"] != review.get("team"):
            existing["team"] = None

    for stem, rows in rejected.items():
        frames = np.array(sorted(contacts[stem]), dtype=np.int64)
        for contact in rows:
            frame = int(contact["frame"])
            distance = int(np.abs(frames - frame).min()) if frames.size else None
            if distance is not None and distance <= duplicate_radius:
                audit[stem]["rejected_duplicate_of_kept"] += 1
            else:
                audit[stem]["rejected_isolated"] += 1

    return {
        "schema": SCHEMA,
        "review_path": str(path.resolve()),
        "review_sha256": dc.file_sha256(path),
        "duplicate_radius": int(duplicate_radius),
        "videos": raw["videos"],
        "contacts": {
            stem: [rows[frame] for frame in sorted(rows)]
            for stem, rows in sorted(contacts.items())
        },
        "masked": {stem: sorted(frames) for stem, frames in sorted(masked.items())},
        "audit": {stem: dict(counts) for stem, counts in sorted(audit.items())},
    }


def load_reviewed_source(
    stem: str,
    review: Mapping[str, Any],
    contact_radius: int = 2,
    guard: int = 3,
    annotations_path: str | Path | None = None,
    verify_fingerprint: bool = True,
) -> dict[str, Any]:
    """Features and sparse targets for one reviewed video.

    The analysis must be the one the review was made on: the contacts were
    chosen on its frames, and a rerun moves the trajectory under them. The
    fingerprint stored at review time is checked unless told otherwise.

    The returned dict has the shape of ``spotter_dataset.load_source`` with
    ``sparse=True`` and ``structure=None``: there is no phase track to count
    decode constraints from.
    """
    from action_recognition.rally_spotting.prepare import build_continuous_features
    from action_recognition.rally_spotting.spotter_dataset import (
        POSE_CHANNELS, POSE_OFFSET,
    )

    video = review["videos"][stem]
    path = Path(annotations_path or video["annotations"])
    sha = dc.file_sha256(path)
    if verify_fingerprint and sha != video["annotations_sha256"]:
        raise ValueError(
            f"{stem}: {path} changed since it was reviewed "
            f"({sha[:12]} != {video['annotations_sha256'][:12]})"
        )
    annotations = json.loads(path.read_text(encoding="utf-8"))
    frame_numbers, features = build_continuous_features(annotations)
    if int(frame_numbers[0]) != 0:
        raise ValueError(f"{stem}: analyses must start at frame 0")
    length = int(features.shape[0])

    contacts = review["contacts"].get(stem, [])
    outside = [c["frame"] for c in contacts if not 0 <= c["frame"] < length]
    if outside:
        raise ValueError(f"{stem}: reviewed contacts outside the analysis: {outside}")
    # Index 0 of the exported phase vocabulary is ``unlabeled``, which
    # ``build_targets`` masks: no phase is invented for an unannotated video.
    unlabelled = np.zeros(length, dtype=np.int64)
    targets, audit = dc.build_targets(
        length, unlabelled, contacts,
        masked_frames=review["masked"].get(stem, []),
        contact_radius=contact_radius, guard=guard,
    )
    pose_block = features[:, POSE_OFFSET:POSE_OFFSET + POSE_CHANNELS]
    return {
        "stem": stem,
        "sparse": True,
        "frames": length,
        "features": features,
        "targets": targets,
        "structure": None,
        "truth": dc.truth_events(contacts, raw_only=False),
        "verified_contacts": len(contacts),
        "uncovered_human_frames": 0,
        "pose_feature_frames": int(np.count_nonzero(np.any(pose_block != 0.0, axis=1))),
        "pose_channels_zeroed": False,
        "net_y": float(annotations.get("net_y", 0.0)),
        "target_audit": audit,
        "review_audit": review["audit"].get(stem, {}),
        "annotation_path": str(path.resolve()),
        "annotation_sha256": sha,
        "dense_targets_sha256": None,
        "label_source": "human_reviewed_model_candidates",
    }
