#!/usr/bin/env python3
"""
Volleyball Game Analyzer — CLI entry point.

Usage:
    # Analyze a video file:
    python analyze_video.py --video path/to/game.mp4

    # Analyze a directory of frames (for testing):
    python analyze_video.py --frames data/dataset_30_sec/frames/

    # With options:
    python analyze_video.py --video game.mp4 --output results/ --duration 60
    python analyze_video.py --video game.mp4 --skip-frames 1 --no-video
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.game_analyzer import GameAnalyzer
from src.utils.path_utils import local_path_from_uri


def main():
    parser = argparse.ArgumentParser(
        description="Analyze volleyball game video and extract statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input source (mutually exclusive)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", metavar="PATH", help="Input video file")
    src.add_argument("--frames", metavar="DIR", help="Directory of frame images (JPG/PNG)")

    # Output
    parser.add_argument("--output", default="results", metavar="DIR",
                        help="Output directory (default: results/)")

    # Calibration
    parser.add_argument("--calibration", default="court_calibration.json", metavar="PATH",
                        help="Court calibration JSON (default: court_calibration.json)")
    parser.add_argument("--net-y", type=float, default=None, metavar="PX",
                        help="Override net Y pixel position (useful when calibration is from different camera)")
    parser.add_argument(
        "--net-profile",
        choices=("mixed", "men"),
        default="mixed",
        help="Net height profile: mixed=2.34 m (default), men=2.43 m",
    )
    parser.add_argument(
        "--net-height-m",
        type=float,
        default=None,
        metavar="M",
        help="Explicit net height in meters; overrides --net-profile",
    )

    # Processing options
    parser.add_argument("--duration", type=float, default=None, metavar="SEC",
                        help="Process only first N seconds of video (default: full)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="FPS for frames mode (default: 30.0)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process in frames mode")
    parser.add_argument("--skip-frames", type=int, default=0, metavar="N",
                        help="Skip N frames between detections for speed (default: 0)")

    # Model overrides
    parser.add_argument("--ball-model", default=None, metavar="PATH",
                        help="Override ball detection model path")
    parser.add_argument("--pose-model", default=None, metavar="PATH",
                        help="Override pose estimation model path")
    parser.add_argument("--ball-ensemble", action="store_true",
                        help="Collect ball candidates from a second, tiled detector "
                             "(union recall 0.93 vs 0.72, roughly half the speed)")
    parser.add_argument("--no-ball-dp", action="store_true",
                        help="Keep the live per-frame gate instead of re-picking the "
                             "ball trajectory globally after the video is read")
    parser.add_argument(
        "--ball-smooth",
        default="moderate",
        metavar="N|PRESET",
        help=(
            "Smooth the chosen ball trajectory with a local quadratic fit: a "
            "frame count (odd) or light/moderate/strong/off. The fit never "
            "crosses a detected contact, so hits stay angular "
            "(default: moderate = 7 frames)"
        ),
    )
    parser.add_argument(
        "--no-model-registry",
        action="store_true",
        help="Skip the local MLflow registry and load the configured local files "
             "directly (configs/model_registry.json). Explicit --ball-model / "
             "--pose-model / --event-checkpoint bypass the registry anyway",
    )
    parser.add_argument(
        "--no-static-suppression",
        action="store_true",
        help="Keep detections that sit motionless for seconds (a spare ball on "
             "the floor, a bag, a lamp). On by default: those are what drag the "
             "trajectory into a scribble",
    )
    parser.add_argument("--ball-batch-size", type=int, default=8, metavar="N",
                        help="Ball YOLO GPU batch size (default: 8; use 4 on low VRAM)")

    # Output flags
    parser.add_argument("--no-video", action="store_true",
                        help="Skip annotated output (faster, stats only)")
    parser.add_argument("--event-checkpoint", default=None, metavar="PATH",
                        help="Path to event sequence model for overlay (default: Transformer v3)")
    parser.add_argument("--no-event-overlay", action="store_true",
                        help="Disable event model overlay on video")

    args = parser.parse_args()
    if args.video:
        args.video = str(local_path_from_uri(args.video))
        if not Path(args.video).is_file():
            parser.error(f"video file not found: {args.video}")

    print("\nVolleyball Game Analyzer")
    print("=" * 45)
    if args.video:
        print(f"Input video  : {args.video}")
    else:
        print(f"Input frames : {args.frames}")
    print(f"Output dir   : {args.output}")
    print(f"Calibration  : {args.calibration}")
    net_height_m = args.net_height_m
    if net_height_m is None:
        net_height_m = 2.34 if args.net_profile == "mixed" else 2.43
    print(f"Net height   : {net_height_m:.2f} m ({args.net_profile})")
    if args.skip_frames:
        print(f"Skip frames  : every {args.skip_frames+1}th frame processed")
    if args.duration:
        print(f"Duration     : first {args.duration}s only")
    print("=" * 45)

    event_ckpt = None if args.no_event_overlay else args.event_checkpoint
    analyzer = GameAnalyzer(
        calibration_path=args.calibration,
        ball_model_path=args.ball_model,
        pose_model_path=args.pose_model,
        output_video=not args.no_video,
        skip_frames=args.skip_frames,
        net_y_override=args.net_y,
        event_checkpoint_path=event_ckpt,
        ball_ensemble=args.ball_ensemble,
        ball_trajectory_dp=not args.no_ball_dp,
        ball_batch_size=args.ball_batch_size,
        ball_smooth_window=args.ball_smooth,
        suppress_static_ball_sources=not args.no_static_suppression,
        use_model_registry=not args.no_model_registry,
        net_height_m=net_height_m,
    )

    if args.video:
        stats = analyzer.process_video(
            video_path=args.video,
            output_dir=args.output,
            max_duration_sec=args.duration,
        )
    else:
        stats = analyzer.process_frames(
            frames_dir=args.frames,
            output_dir=args.output,
            fps=args.fps,
            max_frames=args.max_frames,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
