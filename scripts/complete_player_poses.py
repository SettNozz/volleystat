"""Create a fresh saved-result copy with bounded offline pose-estimate overlays.

Reads only saved JSON; no video decoding or model inference. Observed poses,
IDs, ball, contacts, features and analytics remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.player_contacts import source_fingerprint
from src.tracking.pose_completion import CompletionConfig, complete_player_poses


def rebuild(annotations_path, output, max_gap_seconds=.30):
    path, output = Path(annotations_path).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; existing results are never overwritten")
    if not path.name.endswith("_annotations.json"):
        raise ValueError("Expected <stem>_annotations.json")
    stem = path.name.removesuffix("_annotations.json")
    stats_path = path.with_name(f"{stem}_stats.json")
    ann_bytes, stats_bytes = path.read_bytes(), stats_path.read_bytes()
    payload, stats = json.loads(ann_bytes), json.loads(stats_bytes)
    if payload.get("video") != stats.get("video") or payload["fps"] != stats["fps"]:
        raise ValueError("Stats and annotations must belong to the same analysis")
    if payload.get("pose_completion") or any(a.get("pose_estimates") for a in payload["frames"]):
        raise ValueError("Use the original observed-pose analysis, not a previous completion")
    config = CompletionConfig(max_gap_seconds=max_gap_seconds,
                              max_arm_gap_seconds=min(.15, max_gap_seconds))
    overlays, report = complete_player_poses(
        payload["frames"], payload["fps"], (payload["frame_width"], payload["frame_height"]),
        payload.get("player_identity"), config,
    )
    report["rebuilt_from"] = {"annotations": str(path), "stats": str(stats_path),
                              **source_fingerprint(stats_bytes, ann_bytes)}
    for ann in payload["frames"]:
        if ann["f"] in overlays:
            ann["pose_estimates"] = overlays[ann["f"]]
    payload["pose_completion"] = stats["pose_completion"] = report
    output.mkdir(parents=True, exist_ok=False)
    for kind, data in (("annotations", payload), ("stats", stats)):
        with (output / f"{stem}_{kind}.json").open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"output": str(output.resolve()), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap-seconds", type=float, default=.30,
                        help="Bound for missing torso/leg joints; arms are capped at 0.15 s")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.annotations, args.output, args.max_gap_seconds), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
