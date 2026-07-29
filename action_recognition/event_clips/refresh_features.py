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
                    help="Root dir containing <stem>_annotations.json (searched recursively)")
    ap.add_argument("--output-features-dir", required=True)
    args = ap.parse_args()

    with open(args.manifest) as f:
        clips = json.load(f)["clips"]

    ann_paths = find_annotations(Path(args.results_root))
    print(f"Found annotations for stems: {sorted(ann_paths)}")

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
            cache[stem] = {
                "by_idx": {fr["f"]: fr for fr in ann["frames"]},
                "net_y": float(ann.get("net_tape_top") or ann.get("net_y", 0.0)),
                "fps": float(ann.get("fps", 30.0)),
            }
        c = cache[stem]
        frames = [c["by_idx"][fi]
                  for fi in range(clip["start_frame"], clip["end_frame"] + 1)
                  if fi in c["by_idx"]]
        if len(frames) < 5:
            missing_frames += 1
            continue

        total_frames += len(frames)
        pose_frames += sum(1 for fr in frames if fr.get("pose_f"))

        df = compute_clip_features(frames, net_y=c["net_y"], fps=c["fps"])
        df.to_parquet(out_dir / f"{clip['clip_id']}.parquet", index=False)
        refreshed += 1

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
