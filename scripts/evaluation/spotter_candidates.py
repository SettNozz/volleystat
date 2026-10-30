"""Widen a spotter run into review candidates.

The constrained decoder answers "what is the most likely sequence". A reviewer
needs a different question answered: "what might be a contact". Those are not
the same, and on footage from a camera the model has not seen the gap is large
-- on four new videos the decoder emitted 366 contacts where peak picking finds
590, and a 12-minute match holds closer to the latter.

For labelling, recall is worth far more than precision. Rejecting a candidate
costs one keystroke; finding a contact the queue never offered costs scrubbing
the video. So the review queue is built from peaks over the contact channels,
with the decoder's own answer marked on the ones it also chose.

Produces ``<stem>_candidates.json`` beside each analysis. Changes nothing else.

    python scripts/evaluation/spotter_candidates.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action_recognition.rally_spotting import constrained_decode as cd
from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.contact_spotter import ContactSpotter
from action_recognition.rally_spotting.prepare import build_continuous_features

SCHEMA = "spotter_candidates_v1"


def log(message: str) -> None:
    print(f"[candidates] {message}", flush=True)


def build_candidates(
    probabilities: np.ndarray, frame_numbers: Sequence[int], threshold: float,
    nms_radius: int, constraints: dict[str, Any], transition_weight: float,
    match_tolerance: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Peaks over the contact channels, marked where the decoder agrees."""
    peaks = cd.argmax_events(probabilities, threshold, nms_radius)
    # One peak per frame: several channels can fire on one touch, and a
    # reviewer should see one card for it, not three.
    best: dict[int, dict[str, Any]] = {}
    for event in peaks:
        frame = int(event["frame"])
        current = best.get(frame)
        if current is None or event["confidence"] > current["confidence"]:
            best[frame] = event

    log_probabilities = np.log(np.clip(probabilities, 1e-12, None))
    decode_status = "ok"
    try:
        labels = cd.constrained_viterbi(
            log_probabilities, constraints, transition_weight=transition_weight,
        )
        decoded = cd.events_from_labels(labels, probabilities)
    except cd.InfeasibleConstraints as error:
        decode_status = f"infeasible: {error}"
        decoded = []
    decoded_frames = sorted(int(event["frame"]) for event in decoded)
    decoded_by_frame = {int(event["frame"]): event for event in decoded}

    candidates = []
    for frame in sorted(best):
        event = best[frame]
        near = [
            value for value in decoded_frames
            if abs(value - frame) <= match_tolerance
        ]
        agreed = decoded_by_frame[near[0]] if near else None
        candidates.append({
            "frame": int(frame_numbers[frame]),
            # The decoder's label wins where it has one: it is the sequence
            # answer, and the peak is only a candidate generator.
            "type": (agreed or event)["type"],
            "core_type": dc.to_core((agreed or event)["type"]),
            "confidence": float(event["confidence"]),
            "peak_type": event["type"],
            "decoded": agreed is not None,
            "source": "dense_contact_spotter_peaks",
        })

    # Contacts the decoder placed where no peak survived NMS.
    peak_frames = set(best)
    extra = 0
    for frame, event in sorted(decoded_by_frame.items()):
        if any(abs(frame - value) <= match_tolerance for value in peak_frames):
            continue
        extra += 1
        candidates.append({
            "frame": int(frame_numbers[frame]),
            "type": event["type"],
            "core_type": dc.to_core(event["type"]),
            "confidence": event.get("confidence"),
            "peak_type": None,
            "decoded": True,
            "source": "dense_contact_spotter_decode",
        })
    candidates.sort(key=lambda row: row["frame"])
    return candidates, {
        "peaks": len(best),
        "decoded": len(decoded),
        "decoded_only": extra,
        "agreed": sum(1 for row in candidates if row["decoded"]),
        "decode_status": decode_status,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "models/contact_spotter/spotter_20261028_v1.pt",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="peak threshold; lower surfaces more candidates to reject",
    )
    parser.add_argument("--nms-radius", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    spotter = ContactSpotter.load(args.checkpoint)
    files = sorted(args.annotations_dir.glob("*_annotations.json"))
    if not files:
        parser.error(f"no analyses under {args.annotations_dir}")

    for path in files:
        stem = path.name[: -len("_annotations.json")]
        output = path.with_name(f"{stem}_candidates.json")
        if output.exists() and not args.force:
            log(f"{stem}: {output.name} exists, skipping")
            continue
        annotations = json.loads(path.read_text(encoding="utf-8"))
        frame_numbers, features = build_continuous_features(annotations)
        probabilities = spotter.probabilities(features)
        candidates, stats = build_candidates(
            probabilities, frame_numbers, args.threshold, args.nms_radius,
            spotter.constraints, spotter.transition_weight,
        )
        output.write_text(json.dumps({
            "schema": SCHEMA,
            "stem": stem,
            "annotations": str(path.resolve()),
            "annotations_sha256": dc.file_sha256(path),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_provenance": spotter.provenance.get("created_at"),
            "threshold": args.threshold,
            "nms_radius": args.nms_radius,
            "note": (
                "Review candidates, not a prediction of the sequence. Built "
                "for recall: rejecting one costs a keystroke, a miss costs a "
                "hunt. 'decoded' marks the ones the constrained decoder also "
                "chose."
            ),
            "stats": stats,
            "candidates": candidates,
        }, indent=1), encoding="utf-8")
        log(
            f"{stem}: {len(candidates)} candidates "
            f"({stats['agreed']} also decoded, {stats['decoded_only']} "
            f"decode-only) -> {output.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
