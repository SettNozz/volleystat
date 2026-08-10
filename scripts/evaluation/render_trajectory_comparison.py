"""Render a short clip with the ball trail before and after the clean-up.

A table of removed points is an argument; a clip is evidence. This draws both
trajectories over the real frames — the one the pipeline used to produce
(magenta) and the one it produces now (green) — plus a marker on every point
that was thrown away, so a claim about a specific second can be watched rather
than believed.

Both trails break where a step is physically impossible, exactly as the viewer
does, so a teleport shows up as what it is: two disconnected pieces, not a line
across the gym.

    python scripts/evaluation/render_trajectory_comparison.py \
        --results-dir results/clean3 --video results/3/3_analyzed.mp4 \
        --start 5540 --end 5740 --output results/trajectory_check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.ball_static_sources import clean_ball_annotations  # noqa: E402
from src.tracking.ball_trajectory_smoother import smooth_ball_annotations  # noqa: E402

BEFORE_COLOR = (200, 60, 220)   # BGR magenta
AFTER_COLOR = (95, 200, 60)     # BGR green
DROPPED_COLOR = (170, 170, 170)
MAX_STEP_PX_PER_FRAME = 90.0
TRAIL_FRAMES = 45


def _points(frames: List[Dict[str, Any]]) -> Dict[int, Tuple[float, float]]:
    return {
        int(a.get("f", i)): (float(a["ball"][0]), float(a["ball"][1]))
        for i, a in enumerate(frames) if a.get("ball")
    }


def _trail(
    points: Dict[int, Tuple[float, float]], frame: int, px_scale: float,
    split_impossible: bool = True,
) -> List[List[Tuple[float, float]]]:
    """Recent positions, split wherever no ball could have travelled that far.

    The "before" trail is drawn *without* that split on purpose: joining those
    points is exactly what produced the scribble on screen, and hiding the join
    would hide the failure being demonstrated.
    """
    limit = (MAX_STEP_PX_PER_FRAME if split_impossible else 1e9) * px_scale
    window = sorted(f for f in points if frame - TRAIL_FRAMES <= f <= frame)
    pieces: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    previous: Optional[int] = None
    for f in window:
        if previous is not None:
            dt = f - previous
            if dt > 5 or math.dist(points[f], points[previous]) > limit * dt:
                pieces.append(current)
                current = []
        current.append(points[f])
        previous = f
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) > 1]


def _draw_trail(img, pieces, color, thickness, scale=1.0) -> None:
    for piece in pieces:
        for a, b in zip(piece, piece[1:]):
            cv2.line(img, (int(a[0] * scale), int(a[1] * scale)),
                     (int(b[0] * scale), int(b[1] * scale)),
                     color, thickness, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path,
                    help="Any video whose frame numbering matches the annotations")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--output", type=Path, default=Path("results/trajectory_check"))
    ap.add_argument("--smooth", default="moderate")
    ap.add_argument("--out-width", type=int, default=1280,
                    help="Downscale the output (4K frames are unwatchable inline "
                         "and slow to encode); coordinates are scaled with it")
    ap.add_argument("--slow", type=float, default=2.0,
                    help="Play this many times slower than real time (default: 2)")
    args = ap.parse_args()

    ann_path = sorted(args.results_dir.glob("*_annotations.json"))[0]
    stem = ann_path.name.removesuffix("_annotations.json")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    fps = float(data.get("fps") or 30.0)
    px_scale = float(data.get("frame_width") or 1280) / 1280.0

    before = _points(data["frames"])
    cleaned = json.loads(json.dumps(data["frames"]))
    report = clean_ball_annotations(cleaned, fps=fps, px_scale=px_scale)
    if str(args.smooth).lower() != "off":
        smooth_ball_annotations(cleaned, window=args.smooth, px_scale=px_scale)
    after = _points(cleaned)
    dropped = {
        int(a.get("f", i)): tuple(a["ball_static_point"])
        for i, a in enumerate(cleaned) if a.get("ball_static")
    }

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Annotations are in the *analysed* frame's pixel space; if that differs
    # from this file's, every drawn point would land in the wrong place.
    ann_width = float(data.get("frame_width") or src_width)
    coord_scale = src_width / ann_width
    view = min(1.0, args.out_width / src_width) if args.out_width else 1.0
    width, height = int(src_width * view), int(src_height * view)
    draw_scale = coord_scale * view
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / f"{stem}_compare_{args.start}_{args.end}.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps / max(args.slow, 0.1)), (width, height),
    )

    removed_here = 0
    for frame_idx in range(args.start, args.end + 1):
        ok, img = cap.read()
        if not ok:
            break
        if view != 1.0:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        _draw_trail(
            img, _trail(before, frame_idx, px_scale, split_impossible=False),
            BEFORE_COLOR, 5, draw_scale,
        )
        _draw_trail(img, _trail(after, frame_idx, px_scale), AFTER_COLOR, 3, draw_scale)

        for f, (x, y) in dropped.items():
            if frame_idx - TRAIL_FRAMES <= f <= frame_idx:
                sx, sy = int(x * draw_scale), int(y * draw_scale)
                cv2.circle(img, (sx, sy), 9, DROPPED_COLOR, 2, cv2.LINE_AA)
                if f == frame_idx:
                    removed_here += 1
                    cv2.putText(img, "dropped", (sx + 12, sy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, DROPPED_COLOR, 2)

        seconds = frame_idx / fps
        cv2.rectangle(img, (0, 0), (width, 46), (25, 25, 25), -1)
        cv2.putText(img, f"f{frame_idx}  {int(seconds // 60):02d}:{seconds % 60:05.2f}",
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
        cv2.putText(img, "before", (330, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    BEFORE_COLOR, 2)
        cv2.putText(img, "after", (470, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    AFTER_COLOR, 2)
        cv2.putText(img, "dropped", (590, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    DROPPED_COLOR, 2)
        writer.write(img)

    cap.release()
    writer.release()
    print(f"{out_path}  ({args.end - args.start + 1} frames, "
          f"{report['removed_points']} points removed in the whole file, "
          f"{removed_here} inside this window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
