"""
Prepare event clips + features from GameAnalyzer results.

One clip per FSM contact: segment spans from the previous contact (or rally
start) to the current contact frame. For every clip we write an 18-column
feature Parquet (trajectory + pose) and a manifest entry compatible with the
legacy datasets (event_clips_merged_v*).

Label seeding:
    - serve / reception clips get event_types pre-filled from the
      serve-centric rules (rally structure — highly reliable)
    - other clips get event_types=[] and carry fsm_event_type as a weak hint

Usage:
    python -m action_recognition.event_clips.prepare \
        --results-dir results/part_2 \
        --output-dir data/event_clips_part2 \
        --original-video "C:/path/to/video.mp4"   # if analysis ran --no-video

    # Features/manifest only (no mp4 cutting):
    python -m action_recognition.event_clips.prepare --results-dir results/X \
        --output-dir data/X --no-clips
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.event_clips.compute_features import compute_clip_features
from src.tracking.serve_reception import annotate_rallies

# FSM type -> v3 taxonomy label used as primary_event_type hint
_FSM_TO_PRIMARY = {
    "serve": "serve", "reception": "reception", "dig": "pass",
    "pass": "pass", "attack": "attack", "block": "attack",
}


def load_results(results_dir: Path):
    stats_files = sorted(results_dir.glob("*_stats.json"))
    ann_files = sorted(results_dir.glob("*_annotations.json"))
    if not stats_files or not ann_files:
        raise SystemExit(f"Missing *_stats.json / *_annotations.json in {results_dir}")
    with open(stats_files[0]) as f:
        stats = json.load(f)
    with open(ann_files[0]) as f:
        ann = json.load(f)
    return stats, ann


def build_segments(stats: dict, fps: float):
    """Yield one segment dict per FSM contact, with rule-based label seeds."""
    rallies = stats.get("rallies", [])
    if rallies and "serve_reception" not in rallies[0]:
        annotate_rallies(rallies, fps)

    for rally in rallies:
        events = rally.get("events", [])
        sr = rally.get("serve_reception") or {}
        serve_frame = (sr.get("serve") or {}).get("frame")
        reception_frame = (sr.get("reception") or {}).get("frame")

        prev_frame = rally.get("start_frame", 0)
        for ev in events:
            contact = ev["frame"]
            if contact <= prev_frame:
                prev_frame = min(prev_frame, contact)
                continue
            if ev["frame"] == serve_frame:
                seed, seed_src = ["serve"], "fsm_rule"
            elif ev["frame"] == reception_frame:
                seed, seed_src = ["reception"], "fsm_rule"
            else:
                seed, seed_src = [], None
            yield {
                "start_frame": prev_frame,
                "end_frame": contact,
                "contact_frame": contact,
                "fsm_event_type": ev["type"],
                "primary_event_type": _FSM_TO_PRIMARY.get(ev["type"], "other"),
                "event_types": seed,
                "label_source": seed_src,
                "team": ev.get("team"),
            }
            prev_frame = contact


def cut_clip(video_path: str, start: int, end: int, out_path: Path, fps: float) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ok = False
    for _ in range(end - start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        ok = True
    cap.release()
    writer.release()
    return ok


def main():
    ap = argparse.ArgumentParser(description="Prepare event clips + features")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--original-video", default=None,
                    help="Source video for clip cutting (required if analysis ran --no-video)")
    ap.add_argument("--no-clips", action="store_true",
                    help="Skip mp4 cutting; write features + manifest only")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    features_dir = out_dir / "features"
    clips_dir = out_dir / "clips"
    features_dir.mkdir(parents=True, exist_ok=True)

    stats, ann = load_results(results_dir)
    frames = ann["frames"]
    frame_by_idx = {fr["f"]: fr for fr in frames}
    fps = float(ann.get("fps", 30.0))
    net_y = float(ann.get("net_tape_top") or ann.get("net_y", 0.0))
    stem = Path(stats.get("video", "video")).stem

    video_path = args.original_video or ann.get("original_video_path", "")
    if not args.no_clips:
        if not video_path or not Path(video_path).exists():
            raise SystemExit(
                "Source video not found — pass --original-video or use --no-clips")
        clips_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"clips": []}
    n_rule = 0
    for i, seg in enumerate(build_segments(stats, fps)):
        clip_id = f"{stem}_e{i:04d}_f{seg['contact_frame']}"
        seg_frames = [frame_by_idx[fi]
                      for fi in range(seg["start_frame"], seg["end_frame"] + 1)
                      if fi in frame_by_idx]
        if len(seg_frames) < 5:
            continue

        df = compute_clip_features(seg_frames, net_y=net_y, fps=fps)
        df.to_parquet(features_dir / f"{clip_id}.parquet", index=False)

        clip_path = ""
        if not args.no_clips:
            cp = clips_dir / f"{clip_id}.mp4"
            if cut_clip(video_path, seg["start_frame"], seg["end_frame"], cp, fps):
                clip_path = str(cp.resolve())

        if seg["label_source"] == "fsm_rule":
            n_rule += 1

        manifest["clips"].append({
            "clip_id": clip_id,
            "path": clip_path,
            "source_video": video_path,
            "source_stem": stem,
            "start_frame": seg["start_frame"],
            "end_frame": seg["end_frame"],
            "contact_frame": seg["contact_frame"],
            "num_frames": seg["end_frame"] - seg["start_frame"] + 1,
            "primary_event_type": seg["primary_event_type"],
            "fsm_event_type": seg["fsm_event_type"],
            "event_types": seg["event_types"],
            "label_source": seg["label_source"],
            "team": seg["team"],
            "fps": fps,
        })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(manifest['clips'])} clips → {out_dir}")
    print(f"  rule-labeled serve/reception: {n_rule}")
    print(f"  features: 18 columns (10 trajectory + 8 pose)")


if __name__ == "__main__":
    main()
