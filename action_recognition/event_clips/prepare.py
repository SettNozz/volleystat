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
import math
import sys
from pathlib import Path

import cv2

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.event_clips.compute_features import compute_clip_features
from action_recognition.event_clips.feature_schema import RESOLUTION_NORMALIZED_SCHEMA
from action_recognition.event_clips.manifest_schema import (
    MANIFEST_SCHEMA_VERSION,
    manifest_taxonomies,
    upgrade_clip,
)
from src.tracking.match_analytics import enrich_match_analytics
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
    rallies = stats.get("service_attempts") or stats.get("rallies", [])
    if rallies:
        annotate_rallies(rallies, fps)
    if rallies and "rally_id" not in rallies[0]:
        enrich_match_analytics(rallies)

    for rally in rallies:
        events = rally.get("events", [])
        sr = rally.get("serve_reception") or {}
        serve_frame = (sr.get("serve") or {}).get("frame")
        reception_frame = (sr.get("reception") or {}).get("frame")

        prev_frame = rally.get("start_frame", 0)
        for event_index, ev in enumerate(events):
            contact = ev["frame"]
            if event_index == 0 and contact <= prev_frame:
                # Rally start is normally the serve-contact frame itself. Keep
                # two seconds of pre-contact context so the annotator/model can
                # see the approach, toss and arm swing instead of dropping the
                # serve clip entirely.
                segment_start = max(0, contact - int(round(2.0 * fps)))
            elif contact <= prev_frame:
                prev_frame = min(prev_frame, contact)
                continue
            else:
                segment_start = prev_frame
            if ev["frame"] == serve_frame:
                seed, seed_src = ["serve"], "fsm_rule"
            elif ev["frame"] == reception_frame:
                seed, seed_src = ["reception"], "fsm_rule"
            else:
                seed, seed_src = [], None
            yield {
                "start_frame": segment_start,
                "end_frame": contact,
                "contact_frame": contact,
                "fsm_event_type": ev["type"],
                "primary_event_type": _FSM_TO_PRIMARY.get(ev["type"], "other"),
                "event_types": seed,
                "label_source": seed_src,
                "team": ev.get("team"),
                "rally_id": rally.get("rally_id"),
                "possession_id": ev.get("possession_id"),
                "touch_index": ev.get("touch_index"),
                "court_side": ev.get("court_side") or ev.get("team"),
                "stable_team_id": ev.get("stable_team_id"),
                "match_phase": ev.get("match_phase"),
                "physical_action": ev.get("physical_action"),
                "context_role": ev.get("context_role"),
                "is_service_reception": ev.get("context_role") == "serve_reception",
                "serve_outcome": (rally.get("lifecycle") or {}).get("serve_outcome"),
                "reception_quality": (
                    (rally.get("lifecycle") or {}).get("reception_quality")
                    if ev.get("context_role") == "serve_reception"
                    else "not_applicable"
                ),
                "point_winner_team_id": (rally.get("point") or {}).get("winner_team_id"),
                "point_status": (rally.get("point") or {}).get("status", "unresolved"),
                "context_label_source": ev.get("context_label_source"),
                "context_label_confidence": ev.get("context_label_confidence"),
                "rally_context_start_frame": rally.get("start_frame"),
                "rally_context_end_frame": rally.get("end_frame"),
            }
            prev_frame = contact


def build_additional_segments(
    stats: dict,
    fps: float,
    excluded_frames=(),
    exclusion_tolerance: int = 5,
    context_seconds: float = 2.0,
):
    """Mine contacts from ``all_events`` that were absent from a prior queue.

    These events did not necessarily form a confirmed/playable rally, so their
    FSM type is only a UI hint. They deliberately remain unlabeled until a
    human reviews them.
    """
    excluded = sorted({int(frame) for frame in excluded_frames})
    seen: set[int] = set()
    context_frames = max(1, int(round(context_seconds * fps)))
    for ev in sorted(stats.get("all_events", []), key=lambda item: item["frame"]):
        contact = int(ev["frame"])
        if contact in seen:
            continue
        seen.add(contact)
        if any(abs(contact - frame) <= exclusion_tolerance for frame in excluded):
            continue
        event_type = str(ev.get("type") or "other")
        yield {
            "start_frame": max(0, contact - context_frames),
            "end_frame": contact,
            "contact_frame": contact,
            "fsm_event_type": event_type,
            "primary_event_type": _FSM_TO_PRIMARY.get(event_type, "other"),
            "event_types": [],
            "label_source": None,
            "team": ev.get("team"),
            "candidate_source": "all_events",
        }


def _mean_velocity(points):
    vectors = [
        (cur[0] - prev[0], cur[1] - prev[1])
        for prev, cur in zip(points, points[1:])
    ]
    if not vectors:
        return (0.0, 0.0), 0.0, 0.0
    vx = sum(v[0] for v in vectors) / len(vectors)
    vy = sum(v[1] for v in vectors) / len(vectors)
    speed = math.hypot(vx, vy)
    mean_step = sum(math.hypot(*v) for v in vectors) / len(vectors)
    return (vx, vy), speed, speed / max(mean_step, 1e-9)


def build_trajectory_segments(
    frames,
    fps: float,
    frame_width: int,
    frame_height: int,
    excluded_frames=(),
    exclusion_tolerance: int = 15,
    context_seconds: float = 2.0,
    max_candidates: int = 50,
    nms_frames: int = 30,
):
    """Mine coherent, unexplained velocity changes from the full ball track."""
    excluded = sorted({int(frame) for frame in excluded_frames})
    candidates = []
    window = 3
    min_speed = 0.0025  # normalized image units per frame (~9.6 px/frame at 4K)
    for index in range(window, len(frames) - window):
        segment = frames[index - window:index + window + 1]
        frame_ids = [int(frame["f"]) for frame in segment]
        if any(b != a + 1 for a, b in zip(frame_ids, frame_ids[1:])):
            continue
        if any(frame.get("ball") is None for frame in segment):
            continue
        if any(bool(frame.get("ball_interp")) or bool(frame.get("ball_new"))
               for frame in segment):
            continue
        contact = frame_ids[window]
        if any(abs(contact - frame) <= exclusion_tolerance for frame in excluded):
            continue

        points = [
            (float(frame["ball"][0]) / frame_width,
             float(frame["ball"][1]) / frame_height)
            for frame in segment
        ]
        incoming, speed_before, consistency_before = _mean_velocity(points[:window + 1])
        outgoing, speed_after, consistency_after = _mean_velocity(points[window:])
        if min(consistency_before, consistency_after) < 0.72:
            continue
        if min(speed_before, speed_after) < min_speed:
            continue
        cosine = (
            incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
        ) / max(speed_before * speed_after, 1e-9)
        turn_degrees = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        relative_speed_change = abs(speed_after - speed_before) / max(
            speed_before, speed_after, 1e-9
        )
        if turn_degrees < 25.0 and relative_speed_change < 0.55:
            continue
        motion = min(min(speed_before, speed_after) / min_speed, 2.0) / 2.0
        score = (
            0.55 * min(turn_degrees / 180.0, 1.0)
            + 0.20 * min(relative_speed_change, 1.0)
            + 0.15 * min(consistency_before, consistency_after)
            + 0.10 * motion
        )
        candidates.append({
            "frame": contact,
            "score": score,
            "turn_degrees": turn_degrees,
            "relative_speed_change": relative_speed_change,
        })

    selected = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(abs(candidate["frame"] - kept["frame"]) <= nms_frames
               for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max(0, max_candidates):
            break

    context_frames = max(1, int(round(context_seconds * fps)))
    for candidate in sorted(selected, key=lambda item: item["frame"]):
        contact = candidate["frame"]
        yield {
            "start_frame": max(0, contact - context_frames),
            "end_frame": contact,
            "contact_frame": contact,
            "fsm_event_type": None,
            "primary_event_type": None,
            "event_types": [],
            "label_source": None,
            "team": None,
            "candidate_source": "trajectory_change",
            "candidate_score": round(candidate["score"], 4),
            "direction_change_deg": round(candidate["turn_degrees"], 1),
            "relative_speed_change": round(candidate["relative_speed_change"], 3),
        }


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
    mining = ap.add_mutually_exclusive_group()
    mining.add_argument(
        "--additional-events", action="store_true",
        help="Mine all_events contacts instead of confirmed-rally contacts",
    )
    mining.add_argument(
        "--trajectory-candidates", action="store_true",
        help="Mine coherent trajectory changes outside previously labeled contacts",
    )
    ap.add_argument(
        "--exclude-manifest",
        help="When mining, exclude contact frames already present in this manifest",
    )
    ap.add_argument("--exclude-tolerance", type=int, default=5)
    ap.add_argument("--context-seconds", type=float, default=2.0)
    ap.add_argument("--max-candidates", type=int, default=50)
    ap.add_argument("--nms-frames", type=int, default=30)
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
    frame_width = int(ann.get("frame_width") or 0)
    frame_height = int(ann.get("frame_height") or 0)
    if frame_width <= 0 or frame_height <= 0:
        raise SystemExit(
            "Annotations are missing valid frame_width/frame_height; "
            "resolution_normalized_v2 features cannot be generated"
        )
    stem = Path(stats.get("video", "video")).stem

    video_path = args.original_video or ann.get("original_video_path", "")
    if not args.no_clips:
        if not video_path or not Path(video_path).exists():
            raise SystemExit(
                "Source video not found — pass --original-video or use --no-clips")
        clips_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "feature_schema": RESOLUTION_NORMALIZED_SCHEMA,
        "taxonomies": manifest_taxonomies(),
        "clips": [],
    }
    excluded_frames = []
    if args.exclude_manifest:
        previous = json.loads(Path(args.exclude_manifest).read_text(encoding="utf-8"))
        excluded_frames = [
            int(clip["contact_frame"])
            for clip in previous.get("clips", [])
            if clip.get("source_stem") == stem and clip.get("contact_frame") is not None
        ]
    if args.trajectory_candidates:
        segments = build_trajectory_segments(
            frames,
            fps,
            frame_width,
            frame_height,
            excluded_frames=excluded_frames,
            exclusion_tolerance=max(0, args.exclude_tolerance),
            context_seconds=max(0.1, args.context_seconds),
            max_candidates=max(0, args.max_candidates),
            nms_frames=max(1, args.nms_frames),
        )
    elif args.additional_events:
        segments = build_additional_segments(
            stats,
            fps,
            excluded_frames=excluded_frames,
            exclusion_tolerance=max(0, args.exclude_tolerance),
            context_seconds=max(0.1, args.context_seconds),
        )
    else:
        segments = build_segments(stats, fps)

    n_rule = 0
    clip_kind = "t" if args.trajectory_candidates else "m" if args.additional_events else "e"
    for i, seg in enumerate(segments):
        clip_id = f"{stem}_{clip_kind}{i:04d}_f{seg['contact_frame']}"
        seg_frames = [frame_by_idx[fi]
                      for fi in range(seg["start_frame"], seg["end_frame"] + 1)
                      if fi in frame_by_idx]
        if len(seg_frames) < 5:
            continue

        df = compute_clip_features(
            seg_frames,
            net_y=net_y,
            fps=fps,
            frame_width=frame_width,
            frame_height=frame_height,
            feature_schema=RESOLUTION_NORMALIZED_SCHEMA,
        )
        df.to_parquet(features_dir / f"{clip_id}.parquet", index=False)

        clip_path = ""
        if not args.no_clips:
            cp = clips_dir / f"{clip_id}.mp4"
            if cut_clip(video_path, seg["start_frame"], seg["end_frame"], cp, fps):
                clip_path = str(cp.resolve())

        if seg["label_source"] == "fsm_rule":
            n_rule += 1

        manifest["clips"].append(upgrade_clip({
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
            "candidate_source": seg.get("candidate_source", "rally_events"),
            "candidate_score": seg.get("candidate_score"),
            "direction_change_deg": seg.get("direction_change_deg"),
            "relative_speed_change": seg.get("relative_speed_change"),
            "fps": fps,
            "feature_schema": RESOLUTION_NORMALIZED_SCHEMA,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "rally_id": seg.get("rally_id"),
            "possession_id": seg.get("possession_id"),
            "touch_index": seg.get("touch_index"),
            "court_side": seg.get("court_side"),
            "stable_team_id": seg.get("stable_team_id"),
            "match_phase": seg.get("match_phase"),
            "physical_action": seg.get("physical_action"),
            "context_role": seg.get("context_role"),
            "is_service_reception": seg.get("is_service_reception", False),
            "serve_outcome": seg.get("serve_outcome"),
            "reception_quality": seg.get("reception_quality"),
            "point_winner_team_id": seg.get("point_winner_team_id"),
            "point_status": seg.get("point_status", "unresolved"),
            "context_label_source": seg.get("context_label_source"),
            "context_label_confidence": seg.get("context_label_confidence"),
            "rally_context_start_frame": seg.get("rally_context_start_frame"),
            "rally_context_end_frame": seg.get("rally_context_end_frame"),
        }))

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(manifest['clips'])} clips -> {out_dir}")
    print(f"  rule-labeled serve/reception: {n_rule}")
    print(f"  features: 18 columns (10 trajectory + 8 pose), "
          f"schema={RESOLUTION_NORMALIZED_SCHEMA}")


if __name__ == "__main__":
    main()
