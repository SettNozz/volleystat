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
    parser.add_argument("--sam-checkpoint", default=None,
                        help="Explicit SAM weights; by default the local MLflow "
                             "registry decides (configs/model_registry.json)")
    parser.add_argument("--net-height-m", type=float, default=2.34,
                        help="Net-top height in meters (mixed=2.34, men=2.43)")
    parser.add_argument(
        "--posts-only",
        action="store_true",
        help="Reuse saved tape bounds and refresh only posts/team divider",
    )
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
    print(f"SAM model   : {args.sam_checkpoint or '<from model registry>'}")
    print(f"net_y hint  : {net_y:.0f}px  x=[{net_x_left:.0f}, {net_x_right:.0f}]  "
          f"(original {frame_w:.0f}x{frame_h:.0f})")

    old = calib.get("net_detection_sam", {})
    detector = NetDetector(
        sam_checkpoint=args.sam_checkpoint,
        net_height_m=args.net_height_m,
    )
    if detector.model_provenance:
        print(f"SAM resolved: {detector.model_provenance}")
    if args.posts_only:
        top = old.get("net_top_y_720p")
        bot = old.get("net_bottom_y_720p")
        if top is None or bot is None:
            print("ERROR: --posts-only needs saved tape bounds")
            sys.exit(1)
        top, bot = float(top), float(bot)
        print(f"Reusing tape bounds: top={top:.1f}, bottom={bot:.1f} (720p)")
    else:
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

    floor = detector.detect_floor_line_sam3(
        args.video,
        str(calib_path),
        net_tape_top=top,
        net_tape_bottom=bot,
    )
    floor_method = "sam3_post_x_physical_height" if floor else None
    floor_spacing_m = detector.estimate_post_spacing_m(
        floor[0], floor[1], calib.get("perspective_matrix"), frame_w, frame_h,
    ) if floor else None
    if floor and not detector.post_spacing_is_plausible(floor_spacing_m):
        print(
            f"SAM3 spacing {floor_spacing_m:.2f} m is outside 10-11 m; "
            "trying Hough fallback"
        )
        floor = None
        floor_method = None
    if floor is None:
        floor_priors = detector.project_net_floor_x_priors(
            calib.get("perspective_matrix"), frame_w, frame_h,
        )
        floor = detector.detect_floor_line_from_video(
            video_path=args.video,
            net_x_left=net_x_left,
            net_x_right=net_x_right,
            frame_h=frame_h,
            frame_w=frame_w,
            net_tape_top=top * frame_h / 720.0,
            net_tape_bottom=bot * frame_h / 720.0,
            floor_x_priors=floor_priors,
        )
        floor_method = "hough_post_x_physical_height" if floor else None
        floor_spacing_m = detector.estimate_post_spacing_m(
            floor[0], floor[1], calib.get("perspective_matrix"), frame_w, frame_h,
        ) if floor else None

    # Keep the previous calibration intact until replacement values have all
    # been computed.
    save_sam_net_to_calibration(
        str(calib_path), top, bot, floor, floor_method,
        net_height_m=args.net_height_m,
        net_post_spacing_m=floor_spacing_m,
    )
    print(f"\nSaved to calibration:")
    print(f"  net_tape_top  = {top:.1f} px (720p)")
    print(f"  net_tape_bot  = {bot:.1f} px (720p)")
    print(f"  tape height   = {bot - top:.1f} px (720p)")
    if floor:
        print(f"  floor A/B     = {floor[0]} -> {floor[1]} (720p)")
        print(f"  floor method  = {floor_method}")
        if floor_spacing_m is not None:
            print(f"  post spacing  = {floor_spacing_m:.2f} m (expected 10-11 m)")

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
