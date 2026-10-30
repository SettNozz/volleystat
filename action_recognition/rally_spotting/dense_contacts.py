"""Dense per-frame segmentation targets over human phases and typed contacts.

The label space is single-label and covers every frame of a source video::

    0 between        - human ``between_rallies`` or ``non_game``
    1 serve_flight   - human ``serve_flight`` phase
    2 rally_live     - human ``rally_live`` phase
    3..8             - typed contact, one short segment per reviewed contact

Contacts come from the audited spatial review (``contact_spatial_gold_v1``),
which supplies a *verified* contact frame; the original full-video annotation
frame is only a proposal and differs for 155 of 410 reviewed events. Phases come
from the exported ``dense_targets.npz`` phase channel, which is derived from the
human segment labels.

Three rules keep weak evidence out of the targets:

* a contact whose time the reviewer left uncertain is masked, never background;
* a human contact that the review set does not cover (non-core touches such as
  ``free_ball``) is masked, never background;
* a guard ring around every contact segment is masked, so a one-frame timing
  disagreement is not trained against.

Nothing here reads model output, FSM state or reconstructed ball coordinates.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

IGNORE_INDEX = -100

PHASE_TYPES: tuple[str, ...] = ("between", "serve_flight", "rally_live")
CONTACT_TYPES: tuple[str, ...] = (
    "serve", "reception", "set", "dig", "attack", "block",
)
LABEL_TYPES: tuple[str, ...] = PHASE_TYPES + CONTACT_TYPES
NUM_LABELS = len(LABEL_TYPES)
CONTACT_OFFSET = len(PHASE_TYPES)

# Exported phase vocabulary -> our three phase labels. ``unlabeled`` stays
# masked: the exporter emits it where no human segment covers the frame.
EXPORTED_PHASES: tuple[str, ...] = (
    "unlabeled", "non_game", "between_rallies", "serve_flight", "rally_live",
)
_PHASE_MAP: dict[str, int | None] = {
    "unlabeled": None,
    "non_game": 0,
    "between_rallies": 0,
    "serve_flight": 1,
    "rally_live": 2,
}

# Four-class projection used only to compare against existing checkpoints.
CORE_TYPES: tuple[str, ...] = ("serve", "reception", "pass", "attack")
CONTACT_TO_CORE: dict[str, str] = {
    "serve": "serve",
    "reception": "reception",
    "set": "pass",
    "dig": "pass",
    "attack": "attack",
    "block": "attack",
    # The FSM already emits the collapsed name for a rally touch.
    "pass": "pass",
}

SCHEMA = "dense_phase_contact_targets_v1"

# One reviewed frame carries two events whose four-class labels disagree. The
# resolution is the one already authorized in
# ``action_recognition/rally_spotting/spatial_labels.py``: the first touch after
# the reviewed serve at 3684, with a separate pass at 3794, so the frame is a
# reception. Applied here verbatim rather than resolved by a new rule.
AUTHORIZED_CONFLICTS: dict[tuple[str, int], dict[str, Any]] = {
    ("Khe4P8QcJkg_5_4k", 3761): {
        "review_ids": {
            "Khe4P8QcJkg_5_4k:3759:reception", "Khe4P8QcJkg_5_4k:3765:pass",
        },
        "contact_type": "reception",
        "core_type": "reception",
        "resolution": (
            "Authorized resolution: first touch after reviewed serve at 3684; "
            "next separate pass at 3794; identical reviewed ball centre."
        ),
    },
}


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_index(contact_type: str) -> int:
    """Label index for a raw contact type."""
    return CONTACT_OFFSET + CONTACT_TYPES.index(contact_type)


def is_contact_label(index: int) -> bool:
    return CONTACT_OFFSET <= index < NUM_LABELS


def label_to_contact(index: int) -> str:
    if not is_contact_label(index):
        raise ValueError(f"label {index} is not a contact")
    return CONTACT_TYPES[index - CONTACT_OFFSET]


def _unique_raw_type(raw_labels: Sequence[str]) -> str | None:
    """The single raw contact type of an event, or ``None`` when ambiguous.

    Duplicate annotations of one touch repeat the same type; genuinely
    conflicting types (``block`` plus ``attack``) are not resolved here.
    """
    distinct = {
        label for label in raw_labels if label in CONTACT_TYPES
    }
    unresolved = {label for label in raw_labels if label not in CONTACT_TYPES}
    if len(distinct) != 1 or unresolved:
        return None
    return next(iter(distinct))


def reviewed_contacts(review_path: str | Path) -> dict[str, Any]:
    """Verified contacts per source video, plus frames that must stay masked.

    Returns ``{"videos": {...}, "contacts": {stem: [...]}, "masked": {stem:
    [frames]}, "audit": {...}}``. A contact carries the reviewed frame, the raw
    type when unambiguous, the four-class reviewed label, and the review ids.
    """
    path = Path(review_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = {event["id"]: event for event in raw["events"]}
    reviews = raw["reviews"]
    if set(events) != set(reviews):
        raise ValueError("every reviewed event must appear exactly once")

    contacts: dict[str, dict[int, dict[str, Any]]] = {}
    masked: dict[str, set[int]] = {}
    audit = Counter()
    for identity, event in sorted(events.items()):
        review = reviews[identity]
        stem = event["video_id"]
        contacts.setdefault(stem, {})
        masked.setdefault(stem, set())
        if not review.get("verified"):
            raise ValueError(f"unverified review: {identity}")
        if not review.get("temporal_valid"):
            audit["excluded_" + str(review.get("status"))] += 1
            if review.get("status") != "wrong_event":
                # Unknown timing is unknown, not a negative example.
                masked[stem].update(
                    range(
                        min(event["original_frame"], review["frame"]),
                        max(event["original_frame"], review["frame"]) + 1,
                    )
                )
                masked[stem].add(review["frame"])
            continue
        frame = int(review["frame"])
        raw_type = _unique_raw_type(event.get("raw_labels") or [])
        existing = contacts[stem].get(frame)
        record = {
            "frame": frame,
            "contact_type": raw_type,
            "core_type": review["label"],
            "review_ids": [identity],
            "annotation_ids": list(event.get("annotation_ids") or []),
            "spatial_valid": bool(review.get("spatial_valid")),
            "xy_normalized": review.get("xy_normalized"),
        }
        if existing is None:
            contacts[stem][frame] = record
            audit["contacts"] += 1
            if raw_type is None:
                audit["ambiguous_raw_type"] += 1
            continue
        # Two reviewed events on one frame: keep the frame, keep both ids, and
        # only keep a raw type both agree on.
        audit["merged_duplicate_frames"] += 1
        existing["review_ids"].extend(record["review_ids"])
        existing["annotation_ids"] = sorted(
            set(existing["annotation_ids"]) | set(record["annotation_ids"])
        )
        if existing["contact_type"] != raw_type:
            existing["contact_type"] = None
        if existing["core_type"] != record["core_type"]:
            authorized = AUTHORIZED_CONFLICTS.get((stem, frame))
            if (
                authorized is None
                or set(existing["review_ids"]) != authorized["review_ids"]
            ):
                raise ValueError(
                    f"unresolved four-class conflict on {stem}:{frame}"
                )
            existing["contact_type"] = authorized["contact_type"]
            existing["core_type"] = authorized["core_type"]
            existing["resolution"] = authorized["resolution"]
            audit["authorized_conflict_resolved"] += 1

    return {
        "videos": raw["videos"],
        "sources": raw["sources"],
        "manifest_sha256": raw["manifest_sha256"],
        "review_sha256": file_sha256(path),
        "contacts": {
            stem: [rows[frame] for frame in sorted(rows)]
            for stem, rows in sorted(contacts.items())
        },
        "masked": {stem: sorted(frames) for stem, frames in sorted(masked.items())},
        "audit": dict(audit),
    }


def uncovered_human_frames(
    annotation_path: str | Path,
    stem: str,
    covered_annotation_ids: Iterable[str],
) -> list[int]:
    """Human contact frames the review set does not cover, for masking.

    These are real touches (``free_ball``, ``other``, and contacts dropped by
    the four-class collapse). They must not become background targets.
    """
    raw = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    covered = set(covered_annotation_ids)
    return sorted(
        int(row["frame"])
        for row in raw["annotations"]
        if row["video_id"] == stem and row["annotation_id"] not in covered
    )


def phase_labels(phase_targets: np.ndarray) -> np.ndarray:
    """Exported phase indices -> our phase labels, with ``unlabeled`` masked."""
    values = np.asarray(phase_targets)
    if values.ndim != 1:
        raise ValueError("phase_targets must be one-dimensional")
    if values.size and (values.min() < 0 or values.max() >= len(EXPORTED_PHASES)):
        raise ValueError("phase index outside the exported vocabulary")
    table = np.array(
        [IGNORE_INDEX if _PHASE_MAP[name] is None else _PHASE_MAP[name]
         for name in EXPORTED_PHASES],
        dtype=np.int64,
    )
    return table[values.astype(np.int64)]


def build_targets(
    length: int,
    phase_targets: np.ndarray,
    contacts: Sequence[Mapping[str, Any]],
    masked_frames: Iterable[int] = (),
    contact_radius: int = 2,
    guard: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dense single-label targets for one source video.

    ``contact_radius`` widens each verified contact into a short segment so a
    0.68%-positive frame target is learnable; ``guard`` masks the ring around
    that segment so a one-frame timing disagreement is not penalised.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    if contact_radius < 0 or guard < 0:
        raise ValueError("contact_radius and guard must be non-negative")
    phases = phase_labels(phase_targets)
    if phases.shape[0] < length:
        raise ValueError("phase targets shorter than the decoded video")
    targets = phases[:length].astype(np.int64, copy=True)

    for frame in masked_frames:
        left = max(0, int(frame) - guard)
        right = min(length, int(frame) + guard + 1)
        targets[left:right] = IGNORE_INDEX

    # Guard rings first, so a neighbouring contact segment still wins below.
    for contact in contacts:
        frame = int(contact["frame"])
        if not 0 <= frame < length:
            raise ValueError(f"verified contact {frame} outside decoded video")
        left = max(0, frame - contact_radius - guard)
        right = min(length, frame + contact_radius + guard + 1)
        targets[left:right] = IGNORE_INDEX

    audit = Counter({"segments": 0, "masked_ambiguous_type": 0})
    for contact in contacts:
        frame = int(contact["frame"])
        contact_type = contact.get("contact_type")
        if contact_type is None:
            audit["masked_ambiguous_type"] += 1
            continue
        index = contact_index(contact_type)
        left = max(0, frame - contact_radius)
        right = min(length, frame + contact_radius + 1)
        targets[left:right] = index
        audit["segments"] += 1
        audit["type_" + contact_type] += 1

    audit["frames"] = int(length)
    audit["masked_frames"] = int((targets == IGNORE_INDEX).sum())
    for index, name in enumerate(LABEL_TYPES):
        audit["label_" + name] = int((targets == index).sum())
    return targets, dict(audit)


def truth_events(
    contacts: Sequence[Mapping[str, Any]], raw_only: bool = True,
) -> list[dict[str, Any]]:
    """Verified contacts as spotting events for evaluation."""
    events = []
    for contact in contacts:
        contact_type = contact.get("contact_type")
        if raw_only and contact_type is None:
            continue
        events.append({
            "frame": int(contact["frame"]),
            "type": contact_type,
            "core_type": contact["core_type"],
        })
    return events


def to_core(event_type: str | None) -> str | None:
    """Project a raw contact type onto the four-class taxonomy."""
    if event_type is None:
        return None
    return CONTACT_TO_CORE.get(event_type)
