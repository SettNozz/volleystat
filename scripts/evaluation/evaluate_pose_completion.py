"""Deterministic synthetic pose-dropout check on frozen observed model poses.

Masks complete saved skeletons in real overlap intervals, then compares
bidirectional estimates with last-pose hold and forward constant velocity.
Saved detector poses are proxy references, NOT human keypoint ground truth.
No model, video, training or label refresh runs here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.pose_completion import (
    ARMS, CompletionConfig, _box, _overlap, _quality, complete_player_poses,
)


def metrics(values):
    errors = np.asarray(values, dtype=float)
    return {"joint_frames": len(values),
            "median_error_body_heights": float(np.median(errors)) if len(errors) else None,
            "p90_error_body_heights": float(np.percentile(errors, 90)) if len(errors) else None,
            "pck_at_0_10_body_heights": float(np.mean(errors <= .10)) if len(errors) else None}


def evaluate(payload, samples_per_gap=40):
    fps = float(payload["fps"])
    identity = payload.get("player_identity") or {}
    side = identity.get("side", "b")
    config = CompletionConfig()
    source_frames = {a["f"]: a for a in payload["frames"]}
    observations = defaultdict(dict)
    for ann in payload["frames"]:
        for index, pose in enumerate(ann.get("poses") or []):
            if pose.get("player_id") and pose.get("team") in (side, f"team_{side}"):
                observations[pose["player_id"]][ann["f"]] = (index, pose)
    groups, samples = [], []
    for gap in (1, 3, 6, 9):
        candidates = []
        for pid, rows in sorted(observations.items()):
            # One possible start per second keeps neighbouring masks apart.
            for left in range(min(rows)+1, max(rows)-gap-2, max(1, round(fps))):
                right = left+gap+1
                if not all(f in rows and _quality(rows[f][1], config) for f in range(left-1, right+2)):
                    continue
                covered = 0
                for f in range(left+1, right):
                    index, p = rows[f]
                    occlusion = max((_overlap(_box(p), b) for i, q in enumerate(source_frames[f]["poses"])
                                     if i != index and (b := _box(q)) is not None), default=0.)
                    covered += occlusion >= config.min_overlap
                if covered / gap >= config.min_occluded_fraction:
                    candidates.append((left, right, pid))
        candidates.sort()
        chosen = [candidates[i] for i in np.linspace(0, len(candidates)-1,
                  min(samples_per_gap, len(candidates)), dtype=int)] if candidates else []
        errors = {kind: {method: [] for method in ("bidirectional", "last_pose", "forward_velocity")}
                  for kind in ("all", "arms", "other_joints")}
        available = accepted = accepted_samples = 0
        for left, right, pid in chosen:
            rows = observations[pid]
            clip = [deepcopy(source_frames[f]) for f in range(left-1, right+2)]
            for ann in clip:
                if left < ann["f"] < right:
                    ann["poses"] = [p for p in ann["poses"] if p.get("player_id") != pid]
            overlays, _ = complete_player_poses(
                clip, fps, (payload["frame_width"], payload["frame_height"]), identity, config)
            previous = np.asarray(rows[left-1][1]["kp"], dtype=float)
            last = np.asarray(rows[left][1]["kp"], dtype=float)
            count = 0
            for f in range(left+1, right):
                truth = np.asarray(rows[f][1]["kp"], dtype=float)
                height = rows[f][1]["bb"][3] - rows[f][1]["bb"][1]
                available += int(np.count_nonzero(truth[:, 2] >= config.anchor_confidence))
                entry = next((e for e in overlays.get(f, []) if e["player_id"] == pid), None)
                if entry is None:
                    continue
                for joint in entry["estimated_joints"]:
                    if truth[joint, 2] < config.anchor_confidence:
                        continue
                    estimates = {"bidirectional": np.asarray(entry["keypoints"][joint]),
                                 "last_pose": last[joint, :2],
                                 "forward_velocity": last[joint, :2] + (f-left)*(last[joint, :2]-previous[joint, :2])}
                    kind = "arms" if joint in ARMS else "other_joints"
                    for method, point in estimates.items():
                        error = float(np.linalg.norm(point-truth[joint, :2]) / height)
                        errors["all"][method].append(error)
                        errors[kind][method].append(error)
                    count += 1
            accepted += count
            accepted_samples += bool(count)
            samples.append({"player_id": pid, "left_frame": left, "right_frame": right,
                            "gap_frames": gap, "evaluated_joint_frames": count})
        groups.append({"gap_frames": gap, "gap_seconds": gap/fps,
                       "candidate_intervals": len(candidates), "masked_intervals": len(chosen),
                       "intervals_with_estimates": accepted_samples,
                       "available_reference_joint_frames": available, "evaluated_joint_frames": accepted,
                       "joint_coverage": accepted/available if available else None,
                       "errors": {kind: {method: metrics(values) for method, values in methods.items()}
                                  for kind, methods in errors.items()}})
    return {"schema": "synthetic_pose_dropout_eval_v1", "parameters": vars(config),
            "reference": "saved YOLO model keypoints; not human gold", "selection": "real overlaps; deterministic whole-skeleton masks",
            "comparison": "All methods scored on exactly the same accepted joint-frames; rejected joints remain in coverage denominator",
            "limitations": "Synthetic dropout is not a held-out test of real hidden anatomy, identity accuracy, or contact attribution.",
            "groups": groups, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-gap", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh evaluation output directory")
    if args.samples_per_gap < 1:
        parser.error("--samples-per-gap must be positive")
    raw = args.annotations.read_bytes()
    report = evaluate(json.loads(raw), args.samples_per_gap)
    report["source"] = {"annotations": str(args.annotations.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "groups": report["groups"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
