"""Re-smooth an already analysed result without re-running detection.

Choosing a smoothing window is a visual judgement, and re-analysing a video to
try another one costs minutes of GPU per attempt. Nothing about the smoother
depends on the frames, though — it only touches the ball positions that are
already saved in ``*_annotations.json`` — so every candidate window can be
produced in about a second and compared in the viewer side by side.

    python scripts/smooth_existing_results.py --results-dir results/3 \
        --windows 5 7 13

writes ``results/3_smooth7`` and friends (annotations + stats, and a symlink or
copy of the analysed video so the viewer can play them), then prints the jitter
each window achieved.

The ball positions in the input are used as-is: if that run was already
smoothed, ``ball_presmooth`` is restored first, so windows are compared against
the same original trajectory rather than stacking on each other.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tracking.ball_trajectory_smoother import (  # noqa: E402
    jitter_metric,
    resolve_window,
    smooth_ball_annotations,
)


def _find_stem(results_dir: Path) -> str:
    files = sorted(results_dir.glob("*_annotations.json"))
    if not files:
        raise SystemExit(f"no *_annotations.json in {results_dir}")
    return files[0].name.removesuffix("_annotations.json")


def _restore_original(frames: List[Dict[str, Any]]) -> int:
    """Undo a previous smoothing pass so windows are compared fairly."""
    restored = 0
    for ann in frames:
        original = ann.pop("ball_presmooth", None)
        ann.pop("ball_smoothed", None)
        if original:
            ann["ball"] = list(original)
            restored += 1
    return restored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="Existing analysis output (contains *_annotations.json)")
    ap.add_argument("--windows", nargs="+", default=["5", "7", "13"],
                    help="Windows to try: frame counts or light/moderate/strong")
    ap.add_argument("--output-root", type=Path, default=None,
                    help="Where the variants are written (default: alongside the input)")
    ap.add_argument("--link-video", action="store_true",
                    help="Copy the analysed video into each variant so the viewer "
                         "can play it (off by default: the file is large)")
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    stem = _find_stem(results_dir)
    ann_path = results_dir / f"{stem}_annotations.json"
    stats_path = results_dir / f"{stem}_stats.json"
    output_root = args.output_root or results_dir.parent

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    frames = data["frames"]
    restored = _restore_original(frames)
    if restored:
        print(f"Restored {restored} points from a previous smoothing pass")

    px_scale = float(data.get("frame_width") or 1280) / 1280.0
    baseline = jitter_metric(frames)
    print(f"{stem}: {len(frames)} frames, width {data.get('frame_width')}, "
          f"px_scale {px_scale:.2f}")
    print(f"  raw jitter: mean {baseline['mean']} px/frame, p95 {baseline['p95']}")

    for raw_window in args.windows:
        window = resolve_window(raw_window)
        variant = json.loads(json.dumps(data))  # deep copy, plain JSON
        summary = smooth_ball_annotations(
            variant["frames"], window=window, px_scale=px_scale,
        )
        after = jitter_metric(variant["frames"])
        out_dir = output_root / f"{results_dir.name}_smooth{window or 'off'}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}_annotations.json").write_text(
            json.dumps(variant), encoding="utf-8",
        )
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            stats["ball_smoothing"] = {**summary, "jitter_before": baseline,
                                       "jitter_after": after}
            (out_dir / f"{stem}_stats.json").write_text(
                json.dumps(stats, indent=2), encoding="utf-8",
            )
        video = results_dir / f"{stem}_analyzed.mp4"
        if args.link_video and video.exists():
            shutil.copy2(video, out_dir / video.name)

        print(
            f"  window {window:>2}: moved {summary['moved']}/{summary['points']} "
            f"(mean {summary['mean_shift_px']} px, max {summary['max_shift_px']} px), "
            f"{summary['contacts_preserved']} contacts kept, "
            f"jitter {baseline['mean']} -> {after['mean']} px/frame  →  {out_dir}"
        )

    print(
        "\nCompare in the viewer:\n"
        f"  python viewer.py   then  /load?video={stem}&results_dir="
        f"{(output_root / (results_dir.name + '_smooth7')).as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
