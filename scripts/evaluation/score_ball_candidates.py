#!/usr/bin/env python3
"""Score dumped ball candidates against human ground truth.

Two numbers matter and they must be reported together:

* **recall** — of the frames where a human could see the ball, how many did the
  detector put a point on (within ``--tolerance`` px of the true centre);
* **false positives per minute** — points placed anywhere else. A detector that
  fires ten times a frame can reach any recall you like.

``--gate`` replays :class:`BallTrackGate` over the same candidates, which is
what the pipeline actually consumes: raw detector recall is an upper bound, the
gated number is the one the viewer draws.

Usage:
    python scripts/evaluation/score_ball_candidates.py \
        --gt results/ball_eval/3_6000_6897_gt.json \
        --candidates results/ball_eval/3_6000_6900_candidates.json --gate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tracking.ball_track_gate import BallCandidate, BallTrackGate  # noqa: E402


def load_gt(path: Path) -> tuple[dict[int, tuple[float, float] | None], list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gt: dict[int, tuple[float, float] | None] = {}
    for key, label in data["labels"].items():
        state = label.get("state")
        if state == "unsure":
            continue  # excluded from both numerator and denominator
        gt[int(key)] = (label["x"], label["y"]) if state == "visible" else None
    return gt, data["frames"]


def score(
    dets_by_frame: dict[int, list[dict]],
    gt: dict[int, tuple[float, float] | None],
    tolerance: float,
    fps: float,
    stride: int,
) -> dict:
    hit = miss = fp = 0
    visible = absent = 0
    dist_sum = 0.0
    for frame_idx, truth in gt.items():
        dets = dets_by_frame.get(frame_idx, [])
        if truth is None:
            absent += 1
            fp += len(dets)
            continue
        visible += 1
        near = [
            d for d in dets
            if math.dist((d["x"], d["y"]), truth) <= tolerance
        ]
        if near:
            hit += 1
            dist_sum += min(math.dist((d["x"], d["y"]), truth) for d in near)
            fp += len(dets) - 1  # one detection explains the ball; the rest do not
        else:
            miss += 1
            fp += len(dets)
    labeled = visible + absent
    minutes = (labeled * stride) / fps / 60 if labeled else 0.0
    return {
        "visible_frames": visible,
        "absent_frames": absent,
        "recall": hit / visible if visible else 0.0,
        "hits": hit,
        "misses": miss,
        "false_positives": fp,
        "fp_per_min": fp / minutes if minutes else 0.0,
        "fp_per_frame": fp / labeled if labeled else 0.0,
        "mean_hit_dist_px": dist_sum / hit if hit else 0.0,
    }


def run_gate(frame_indices: list[int], dets_by_frame: dict[int, list[dict]]) -> dict[int, list[dict]]:
    """Replay the production selector: at most one detection survives per frame."""
    gate = BallTrackGate()
    out: dict[int, list[dict]] = {}
    for frame_idx in frame_indices:
        cands = [
            BallCandidate(x=d["x"], y=d["y"], confidence=d["conf"], size=d["size"])
            for d in dets_by_frame.get(frame_idx, [])
        ]
        chosen = gate.update(frame_idx, cands)
        out[frame_idx] = (
            [{"x": chosen.x, "y": chosen.y, "conf": chosen.confidence, "size": chosen.size}]
            if chosen else []
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--candidates", required=True, nargs="+")
    ap.add_argument("--tolerance", type=float, default=20.0, help="px from the clicked centre")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--fps", type=float, default=29.97)
    ap.add_argument("--gate", action="store_true", help="Also score after BallTrackGate")
    ap.add_argument(
        "--assume-unlabeled-absent",
        action="store_true",
        help="Count skipped frames as ball-not-visible (they carry the FP rate when nothing is in play)",
    )
    args = ap.parse_args()

    gt, all_frames = load_gt(Path(args.gt))
    if args.assume_unlabeled_absent:
        for f in all_frames:
            gt.setdefault(f, None)
    visible = sum(1 for v in gt.values() if v is not None)
    # GT is usually a strided sample of the dumped window; each labeled frame
    # therefore stands for ``gt_stride`` frames of wall-clock time, which is what
    # turns a false-positive count into a per-minute rate.
    keys = sorted(gt)
    gt_stride = min((b - a) for a, b in zip(keys, keys[1:])) if len(keys) > 1 else 1
    print(
        f"GT: {len(gt)} labeled frames ({visible} with ball, {len(gt) - visible} without), "
        f"sampled every {gt_stride} frame(s)\n"
    )

    header = f"{'source':>26} {'conf':>5} {'recall':>7} {'hits':>5} {'miss':>5} {'FP':>6} {'FP/min':>7} {'px':>5}"
    print(header)
    print("-" * len(header))

    for cand_path in args.candidates:
        data = json.loads(Path(cand_path).read_text(encoding="utf-8"))
        frame_indices = data["frame_indices"]
        for imgsz, run in sorted(data["runs"].items()):
            per_frame_all: dict[int, list[dict]] = {}
            for fi, dets in zip(frame_indices, run["detections"]):
                per_frame_all[fi] = [
                    {
                        "x": (d["box"][0] + d["box"][2]) / 2,
                        "y": (d["box"][1] + d["box"][3]) / 2,
                        "conf": d["conf"],
                        "size": max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]),
                    }
                    for d in dets
                ]
            for conf in args.conf:
                filtered = {
                    fi: [d for d in dets if d["conf"] >= conf]
                    for fi, dets in per_frame_all.items()
                }
                variants = [(f"{imgsz}", filtered)]
                if args.gate:
                    variants.append((f"{imgsz}+gate", run_gate(frame_indices, filtered)))
                for label, dets_by_frame in variants:
                    # GT frames may be a strided subset; score only what is labeled.
                    m = score(dets_by_frame, gt, args.tolerance, args.fps, gt_stride)
                    print(
                        f"{label:>26} {conf:>5.2f} {m['recall']:>7.3f} {m['hits']:>5d} "
                        f"{m['misses']:>5d} {m['false_positives']:>6d} {m['fp_per_min']:>7.1f} "
                        f"{m['mean_hit_dist_px']:>5.1f}"
                    )
        print()


if __name__ == "__main__":
    main()
