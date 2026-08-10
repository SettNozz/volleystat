"""
Recompute features for an EXISTING manifest from fresh analysis results.

Use case: the merged datasets were built before pose estimation existed, so
their Parquet files have no pose channels. After re-running analyze_video.py
on the source videos (which now writes "pose_f" per frame), this script
rebuilds every clip's feature table (18 columns) from the new annotations —
clip boundaries, labels and manifest stay untouched.

Usage:
    # 1) re-analyze each source video (stats only is enough):
    python analyze_video.py --video path/to/3.mp4 --output results/refresh/3 --no-video
    #    (repeat per source video; dir name doesn't matter, stem does)

    # 2) refresh features for the whole merged dataset:
    python -m action_recognition.event_clips.refresh_features \
        --manifest data/event_clips_merged_v5/manifest.json \
        --results-root results/refresh \
        --output-features-dir data/event_clips_merged_v5/features_pose

    # 3) train on the pose-aware features:
    ... train_event_sequence --features-dir data/event_clips_merged_v5/features_pose ...
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.event_clips.compute_features import compute_clip_features
from action_recognition.event_clips.feature_schema import RESOLUTION_NORMALIZED_SCHEMA


def find_annotations(results_root: Path):
    """stem -> annotations path, searched recursively under results_root."""
    out = {}
    for p in results_root.rglob("*_annotations.json"):
        stem = p.name[: -len("_annotations.json")]
        out.setdefault(stem, p)
    return out


def main():
    ap = argparse.ArgumentParser(description="Refresh clip features from new annotations")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--results-root", required=True,
                    help="Root containing the base annotations (poses/events are preserved)")
    ap.add_argument(
        "--ball-results-root",
        default=None,
        help=(
            "Optional root containing fresh ball-only annotations. Ball fields "
            "are overlaid by frame while pose_f and labels stay from --results-root."
        ),
    )
    ap.add_argument("--output-features-dir", required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        clips = json.load(f)["clips"]

    ann_paths = find_annotations(Path(args.results_root))
    print(f"Found annotations for stems: {sorted(ann_paths)}")
    ball_ann_paths = (
        find_annotations(Path(args.ball_results_root))
        if args.ball_results_root else {}
    )
    if ball_ann_paths:
        print(f"Fresh ball annotations for stems: {sorted(ball_ann_paths)}")

    out_dir = Path(args.output_features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = {}
    refreshed, missing_stems, missing_frames = 0, set(), 0
    pose_frames = total_frames = 0

    for clip in clips:
        stem = clip.get("source_stem", "")
        if stem not in ann_paths:
            missing_stems.add(stem)
            continue
        if stem not in cache:
            with open(ann_paths[stem]) as f:
                ann = json.load(f)
            fresh_ball_by_idx = {}
            if stem in ball_ann_paths:
                with open(ball_ann_paths[stem]) as f:
                    ball_ann = json.load(f)
                fresh_ball_by_idx = {
                    int(fr["f"]): fr for fr in ball_ann.get("frames", [])
                }
            cache[stem] = {
                "by_idx": {fr["f"]: fr for fr in ann["frames"]},
                "fresh_ball_by_idx": fresh_ball_by_idx,
                "net_y": float(ann.get("net_tape_top") or ann.get("net_y", 0.0)),
                "fps": float(ann.get("fps", 30.0)),
                "frame_width": int(ann.get("frame_width") or 0),
                "frame_height": int(ann.get("frame_height") or 0),
            }
        c = cache[stem]
        if c["frame_width"] <= 0 or c["frame_height"] <= 0:
            raise SystemExit(
                f"{ann_paths[stem]} has no valid frame_width/frame_height; "
                "cannot generate resolution_normalized_v2 features"
            )
        frames = []
        for fi in range(clip["start_frame"], clip["end_frame"] + 1):
            if fi not in c["by_idx"]:
                continue
            frame = c["by_idx"][fi]
            fresh = c["fresh_ball_by_idx"].get(fi)
            if fresh is not None:
                frame = dict(frame)
                ball_fields = (
                    "ball", "ball_raw", "ball_raw_confidence",
                    "ball_source", "ball_confidence", "ball_interp",
                    "ball_new", "ball_out_of_frame",
                    "ball_contact_reconstructed",
                    "ball_conservative", "ball_conservative_source",
                    "ball_conservative_confidence", "ball_contact_aware",
                    "ball_contact_aware_source",
                    "ball_contact_aware_confidence",
                )
                for key in ball_fields:
                    if key in fresh:
                        frame[key] = fresh.get(key)
                    else:
                        frame.pop(key, None)
            frames.append(frame)
        if len(frames) < 5:
            missing_frames += 1
            continue

        total_frames += len(frames)
        pose_frames += sum(1 for fr in frames if fr.get("pose_f"))

        df = compute_clip_features(
            frames,
            net_y=c["net_y"],
            fps=c["fps"],
            frame_width=c["frame_width"],
            frame_height=c["frame_height"],
            feature_schema=RESOLUTION_NORMALIZED_SCHEMA,
        )
        df.to_parquet(out_dir / f"{clip['clip_id']}.parquet", index=False)
        refreshed += 1

    with open(out_dir / "feature_schema.json", "w") as f:
        json.dump(
            {
                "feature_schema": RESOLUTION_NORMALIZED_SCHEMA,
                "feature_files": refreshed,
            },
            f,
            indent=2,
        )

    print(f"\nRefreshed {refreshed}/{len(clips)} clips -> {out_dir}")
    if total_frames:
        print(f"Pose coverage: {pose_frames}/{total_frames} frames "
              f"({100.0 * pose_frames / total_frames:.0f}%) have pose_f")
    if missing_stems:
        print(f"NO annotations for stems: {sorted(missing_stems)} — "
              f"re-run analyze_video.py for those source videos")
    if missing_frames:
        print(f"{missing_frames} clips skipped (frames outside annotation range)")


if __name__ == "__main__":
    main()
