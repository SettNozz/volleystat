#!/usr/bin/env python3
"""Apply offline ball-gap reconstruction to an annotations JSON.

Original file is never overwritten unless ``--output`` explicitly points to it
and ``--force`` is supplied.  A gold manifest may provide human contact frames;
labels and clip boundaries remain untouched.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tracking.ball_gap_reconstructor import BallGapReconstructor, ContactHint


def _review_segments(
    frames: Sequence[dict],
    context: int,
    *,
    selection: str,
    max_segments: int,
) -> list[tuple[int, int]]:
    maximum = max(int(frame["f"]) for frame in frames)
    if selection == "full":
        return [(0, maximum)]

    def selected(frame: dict) -> bool:
        conservative_source = str(frame.get("ball_conservative_source") or "missing")
        aware_source = str(frame.get("ball_contact_aware_source") or "missing")
        if selection == "differences":
            return frame.get("ball_conservative") != frame.get("ball_contact_aware")
        if selection == "short":
            return conservative_source == "interpolated_short"
        if selection == "conservative":
            return conservative_source == "reconstructed_conservative"
        if selection == "contact-aware":
            return aware_source == "reconstructed_contact_aware"
        if selection == "occlusion":
            return aware_source == "reconstructed_occlusion"
        return any(
            source not in {"detected", "missing", "out_of_frame"}
            for source in (conservative_source, aware_source)
        )

    selected_frames = [int(frame["f"]) for frame in frames if selected(frame)]
    if not selected_frames:
        return []
    intervals = [
        (max(0, frame - context), min(maximum, frame + context))
        for frame in selected_frames
    ]
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    result = [(start, end) for start, end in merged]
    if max_segments > 0 and len(result) > max_segments:
        if max_segments == 1:
            return [result[len(result) // 2]]
        indices = [
            round(index * (len(result) - 1) / (max_segments - 1))
            for index in range(max_segments)
        ]
        result = [result[index] for index in dict.fromkeys(indices)]
    return result


def render_comparison_video(
    payload: dict,
    video_path: Path,
    output_path: Path,
    *,
    context: int,
    scale: float,
    selection: str,
    max_segments: int,
) -> None:
    """Render compact clips around frames where comparison variants differ."""
    import cv2

    frames = payload["frames"]
    by_frame = {int(frame["f"]): frame for frame in frames}
    segments = _review_segments(
        frames, context,
        selection=selection,
        max_segments=max_segments,
    )
    if not segments:
        raise RuntimeError("comparison variants are identical; no review video to render")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open review source video: {video_path}")
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or payload.get("fps") or 30.0)
    width = max(2, int(round(source_width * scale / 2.0) * 2))
    height = max(2, int(round(source_height * scale / 2.0) * 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open review video writer: {output_path}")

    colors = {
        "raw": (80, 255, 80),
        "conservative": (255, 255, 0),
        "contact_aware": (255, 0, 255),
    }
    radii = {"raw": 7, "conservative": 12, "contact_aware": 17}
    letters = {"raw": "R", "conservative": "C", "contact_aware": "A"}
    trails = {name: deque(maxlen=35) for name in colors}

    def scaled_point(value: object) -> tuple[int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        return int(round(float(value[0]) * scale)), int(round(float(value[1]) * scale))

    try:
        for segment_index, (start, end) in enumerate(segments, 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for trail in trails.values():
                trail.clear()
            for frame_index in range(start, end + 1):
                ok, image = cap.read()
                if not ok:
                    break
                if scale != 1.0:
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                ann = by_frame.get(frame_index, {})
                if ann.get("ball_new"):
                    for trail in trails.values():
                        trail.clear()

                for pose in ann.get("poses") or []:
                    bbox = pose.get("bb") if isinstance(pose, dict) else None
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                        continue
                    top_left = scaled_point(bbox[:2])
                    bottom_right = scaled_point(bbox[2:4])
                    if top_left is None or bottom_right is None:
                        continue
                    x1, y1 = top_left
                    x2, y2 = bottom_right
                    team_color = (80, 190, 255) if pose.get("team") == "b" else (100, 190, 100)
                    cv2.rectangle(image, (x1, y1), (x2, y2), team_color, 1)

                labels = []
                for name, color in colors.items():
                    key = "ball_raw" if name == "raw" else f"ball_{name}"
                    point = scaled_point(ann.get(key))
                    trails[name].append(point)
                    previous = None
                    for trail_point in trails[name]:
                        if trail_point is not None and previous is not None:
                            cv2.line(image, previous, trail_point, color, 2)
                        previous = trail_point
                    if name == "raw":
                        source = "detected" if point is not None else "missing"
                        confidence = float(ann.get("ball_raw_confidence") or 0.0)
                    else:
                        source = str(ann.get(f"{key}_source") or "missing")
                        confidence = float(ann.get(f"{key}_confidence") or 0.0)
                    labels.append(f"{name}: {source} {confidence:.2f}")
                    if point is not None:
                        cv2.circle(image, point, radii[name], color, 3)
                        cv2.putText(
                            image, letters[name],
                            (point[0] + radii[name] + 3, point[1] - radii[name]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                        )

                cv2.rectangle(image, (0, 0), (width, 138), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    f"{selection} {segment_index}/{len(segments)} | original frame {frame_index} | state {ann.get('state') or 'unknown'}",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                cv2.putText(image, labels[0], (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors["raw"], 2)
                cv2.putText(image, labels[1], (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors["conservative"], 2)
                cv2.putText(image, labels[2], (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors["contact_aware"], 2)
                if ann.get("ball_out_of_frame"):
                    cv2.putText(
                        image, "BALL OUT OF FRAME", (width - 330, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                    )
                writer.write(image)
    finally:
        cap.release()
        writer.release()
    print(f"Rendered {len(segments)} review segments to {output_path}")


def contact_hints(payload: object, source_stem: str) -> Iterable[ContactHint]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = (
            payload.get("events")
            or payload.get("annotations")
            or payload.get("clips")
            or []
        )
    else:
        records = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        stem = str(record.get("source_stem") or "")
        if stem and stem != source_stem:
            continue
        frame = record.get("frame", record.get("contact_frame"))
        if frame is None or int(frame) in seen:
            continue
        seen.add(int(frame))
        yield ContactHint(
            int(frame),
            str(
                record.get("type")
                or record.get("primary_event_type")
                or record.get("event_type")
                or "other"
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", choices=("conservative", "contact-aware", "compare"),
        default="compare",
    )
    parser.add_argument(
        "--contacts",
        default=None,
        help="Optional full-video labels or gold manifest containing contact frames",
    )
    parser.add_argument("--source-stem", default=None)
    parser.add_argument(
        "--comparison-video",
        default=None,
        help="Optional compact MP4 containing only frames where variants differ",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Original video for --comparison-video; defaults to annotations header",
    )
    parser.add_argument("--review-context", type=int, default=30)
    parser.add_argument("--review-scale", type=float, default=0.5)
    parser.add_argument(
        "--review-selection",
        choices=(
            "differences", "all-reconstructed", "short", "conservative",
            "contact-aware", "occlusion", "full",
        ),
        default="differences",
    )
    parser.add_argument("--review-max-segments", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.annotations).resolve()
    output = Path(args.output).resolve()
    if output == source and not args.force:
        parser.error("refusing to overwrite input; use a different --output or --force")
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        parser.error("annotations JSON has no frames list")

    stem = args.source_stem or source.name.removesuffix("_annotations.json")
    hints = []
    if args.contacts:
        contacts_payload = json.loads(Path(args.contacts).read_text(encoding="utf-8"))
        hints = list(contact_hints(contacts_payload, stem))

    active_mode = "conservative" if args.mode == "compare" else args.mode.replace("-", "_")
    reconstructor = BallGapReconstructor(
        int(payload.get("frame_width") or 1280),
        int(payload.get("frame_height") or 720),
        net_tape_top=payload.get("net_tape_top"),
        net_tape_bottom=payload.get("net_tape_bottom"),
        net_floor_left=payload.get("net_floor_left"),
        net_floor_right=payload.get("net_floor_right"),
    )
    summaries = reconstructor.apply(
        frames,
        active_mode=active_mode,
        contact_hints=hints,
        keep_variants=True,
    )
    payload["ball_reconstruction"] = {
        "requested_mode": args.mode,
        "active_mode": active_mode,
        "affects_fsm": False,
        "contact_hints": len(hints),
        "variants": {name: vars(summary) for name, summary in summaries.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload["ball_reconstruction"], indent=2))
    print(f"Wrote {output}")
    if args.comparison_video:
        video = Path(args.video or payload.get("original_video_path") or "").resolve()
        if not video.is_file():
            parser.error("--comparison-video needs --video or valid original_video_path")
        if not (0.1 <= args.review_scale <= 1.0):
            parser.error("--review-scale must be between 0.1 and 1.0")
        render_comparison_video(
            payload,
            video,
            Path(args.comparison_video).resolve(),
            context=max(0, args.review_context),
            scale=args.review_scale,
            selection=args.review_selection,
            max_segments=max(0, args.review_max_segments),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
