"""Find — and optionally remove — the places where the drawn trail goes wrong.

"The trajectory looks like a scribble" is not something to fix by eye. This
reports, with timecodes, the three mechanisms that produce it, so a claim about
a video can be checked against a specific second of it:

* **fixed object** — the detector fires on a spare ball, a bag or a lamp that
  never moves. A confident candidate is *cheap* for the trajectory solver, so a
  long stationary run can outscore a real flight;
* **jump island** — the path teleports onto something (a step no thrown object
  could make), sits there without moving, and teleports back;
* **teleport** — a single impossible step, usually a detector gap the selector
  bridged across two different objects.

Usage:
    python scripts/evaluation/find_trajectory_glitches.py --results-dir results/3
    python scripts/evaluation/find_trajectory_glitches.py --results-dir results/3 --fix
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.ball_static_sources import (  # noqa: E402
    clean_ball_annotations,
    find_static_sources,
    suppress_jump_islands,
)
from src.tracking.ball_trajectory_smoother import smooth_ball_annotations  # noqa: E402

MAX_STEP_PX_PER_FRAME = 90.0


def timecode(frame: int, fps: float) -> str:
    seconds = frame / max(fps, 1e-6)
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def teleports(frames: List[Dict[str, Any]], px_scale: float) -> List[Dict[str, Any]]:
    limit = MAX_STEP_PX_PER_FRAME * px_scale
    points = [
        (int(a.get("f", i)), float(a["ball"][0]), float(a["ball"][1]))
        for i, a in enumerate(frames) if a.get("ball")
    ]
    out = []
    for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:]):
        dt = max(1, f1 - f0)
        step = math.dist((x0, y0), (x1, y1))
        if step > limit * dt:
            out.append({"frame": f1, "step_px": round(step), "gap_frames": dt})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--fix", action="store_true",
                    help="Write a cleaned copy next to the input (…_clean)")
    ap.add_argument("--smooth", default="moderate", metavar="N|PRESET",
                    help="Also apply trajectory smoothing to the cleaned copy, "
                         "as the pipeline does (default: moderate; 'off' to skip)")
    ap.add_argument("--copy-video", action="store_true",
                    help="Copy the annotated mp4 into the cleaned dir. Off by "
                         "default — the viewer plays the original video, whose "
                         "path is already recorded in the annotations")
    ap.add_argument("--around", type=float, default=None, metavar="SEC",
                    help="Only report what happens within +-5 s of this timestamp")
    args = ap.parse_args()

    ann_files = sorted(args.results_dir.glob("*_annotations.json"))
    if not ann_files:
        raise SystemExit(f"no *_annotations.json in {args.results_dir}")
    ann_path = ann_files[0]
    stem = ann_path.name.removesuffix("_annotations.json")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    frames = data["frames"]
    fps = float(data.get("fps") or 30.0)
    px_scale = float(data.get("frame_width") or 1280) / 1280.0

    points = [
        (int(a.get("f", i)), float(a["ball"][0]), float(a["ball"][1]))
        for i, a in enumerate(frames) if a.get("ball")
    ]
    print(f"{stem}: {len(frames)} frames, {len(points)} with a ball, "
          f"{fps:.2f} fps, width {data.get('frame_width')}")

    def in_window(frame: int) -> bool:
        return args.around is None or abs(frame / fps - args.around) <= 5.0

    sources = [s for s in find_static_sources(points, fps=fps, px_scale=px_scale)
               if in_window(s.first_frame)]
    print(f"\nFixed objects: {len(sources)}")
    for s in sorted(sources, key=lambda s: -s.seconds)[:12]:
        print(f"  {timecode(s.first_frame, fps)}–{timecode(s.last_frame, fps)} "
              f"({s.seconds:5.1f} s) at ({s.x:.0f},{s.y:.0f}), spread {s.spread_px:.1f} px, "
              f"{s.frames} points")

    jumps = [t for t in teleports(frames, px_scale) if in_window(t["frame"])]
    print(f"\nImpossible steps: {len(jumps)}")
    for t in jumps[:12]:
        print(f"  {timecode(t['frame'], fps)} — {t['step_px']} px "
              f"across {t['gap_frames']} frame(s)")

    probe = json.loads(json.dumps(frames))
    islands = suppress_jump_islands(probe, fps=fps, px_scale=px_scale)
    shown = [s for s in islands.sources if in_window(int(s["first_frame"]))]
    print(f"\nJump islands: {len(islands.sources)}"
          + (f" ({len(shown)} in window)" if args.around is not None else ""))
    for s in shown[:12]:
        print(f"  {timecode(int(s['first_frame']), fps)}–{timecode(int(s['last_frame']), fps)} "
              f"({s['seconds']:.2f} s) at ({s['x']:.0f},{s['y']:.0f}), "
              f"travelled {s['travel_px']} px, {s['frames']} points")

    if args.fix:
        report = clean_ball_annotations(frames, fps=fps, px_scale=px_scale)
        if str(args.smooth).lower() != "off":
            report["smoothing"] = smooth_ball_annotations(
                frames, window=args.smooth, px_scale=px_scale,
            )
        out_dir = args.results_dir.parent / f"{args.results_dir.name}_clean"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ann_path.name).write_text(json.dumps(data), encoding="utf-8")
        stats_path = args.results_dir / f"{stem}_stats.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["ball_static_suppression"] = report
            (out_dir / stats_path.name).write_text(json.dumps(stats, indent=2),
                                                   encoding="utf-8")
        video = args.results_dir / f"{stem}_analyzed.mp4"
        if args.copy_video and video.exists():
            shutil.copy2(video, out_dir / video.name)
        left = sum(1 for a in frames if a.get("ball"))
        print(f"\nCleaned: {report['removed_points']} points cleared "
              f"({len(points)} -> {left}) -> {out_dir}")
        print(f"  viewer: /load?video={stem}&results_dir={out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
