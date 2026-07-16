#!/usr/bin/env python3
"""
Run SAM vit_h net detection on a video and save results to calibration JSON.

Usage:
    python scripts/run_sam_net_detection.py \
        --video path/to/video.mp4 \
        --calibration path/to/court_calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.detection.net_detector import NetDetector, save_sam_net_to_calibration
from src.utils.logger import get_logger

log = get_logger("SAMRun")

_SAM_CHECKPOINT = str(Path.home() / "Downloads" / "sam_vit_h_4b8939.pth")
_DEFAULT_CALIB = str(_ROOT / "court_calibration.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", default=_DEFAULT_CALIB)
    parser.add_argument("--sam-checkpoint", default=_SAM_CHECKPOINT)
    parser.add_argument("--validate", action="store_true",
                        help="Open validation page after detection")
    args = parser.parse_args()

    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"ERROR: calibration not found: {calib_path}")
        sys.exit(1)

    with open(calib_path) as f:
        calib = json.load(f)

    # --- Read net hints from calibration ---
    net_pts = calib.get("net_line_points", [])
    if not net_pts:
        print("ERROR: no net_line_points in calibration")
        sys.exit(1)

    frame_w = float(calib.get("frame_width", 3840))
    frame_h = float(calib.get("frame_height", 2160))
    net_y = float(sum(p[1] for p in net_pts) / len(net_pts))
    net_x_left = float(min(p[0] for p in net_pts))
    net_x_right = float(max(p[0] for p in net_pts))

    print(f"Video       : {args.video}")
    print(f"Calibration : {calib_path}")
    print(f"SAM model   : {args.sam_checkpoint}")
    print(f"net_y hint  : {net_y:.0f}px  x=[{net_x_left:.0f}, {net_x_right:.0f}]  "
          f"(original {frame_w:.0f}x{frame_h:.0f})")

    # --- Clear old SAM data ---
    old = calib.get("net_detection_sam", {})
    if old:
        print(f"Clearing old SAM data: {old}")
        calib.pop("net_detection_sam", None)
        with open(calib_path, "w") as f:
            json.dump(calib, f, indent=2)

    # --- Run SAM ---
    detector = NetDetector(sam_checkpoint=args.sam_checkpoint)
    top, bot = detector.detect_from_video(
        video_path=args.video,
        net_y_hint=net_y,
        net_x_left=net_x_left,
        net_x_right=net_x_right,
        frame_h=frame_h,
        frame_w=frame_w,
    )

    if top is None:
        print("SAM detection failed — no net tape found")
        sys.exit(1)

    # --- Save ---
    save_sam_net_to_calibration(str(calib_path), top, bot)
    print(f"\nSaved to calibration:")
    print(f"  net_tape_top  = {top:.1f} px (720p)")
    print(f"  net_tape_bot  = {bot:.1f} px (720p)")
    print(f"  tape height   = {bot - top:.1f} px (720p)")

    # --- Optionally open validation page ---
    if args.validate:
        import subprocess
        subprocess.Popen([
            sys.executable,
            str(_ROOT / "scripts" / "validate_net_detection.py"),
            "--video", args.video,
            "--calibration", str(calib_path),
            "--output", "results/net_validation",
        ])


if __name__ == "__main__":
    main()
