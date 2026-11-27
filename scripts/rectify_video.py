"""Prepare a separate Wide/fisheye video before court calibration and analysis."""

from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.path_utils import local_path_from_uri
from src.video.lens_correction import DEFAULT_DIRECTORY, LensCorrection, prepare_video, rectify_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Original video; it is never overwritten")
    parser.add_argument("--profile", required=True, help="JSON with enabled, horizontal_fov_deg and balance")
    parser.add_argument("--output-dir", default=str(DEFAULT_DIRECTORY), help="Directory for reusable corrected videos")
    parser.add_argument("--encoder", choices=("auto", "h264_nvenc", "libx264"), default="auto")
    parser.add_argument("--preview", metavar="PNG", help="Write one corrected preview image instead of a full video")
    parser.add_argument("--seconds", type=float, default=0, help="Preview time in seconds")
    args = parser.parse_args()
    try:
        profile = LensCorrection.from_dict(json.loads(Path(args.profile).read_text(encoding="utf-8")))
        source = local_path_from_uri(args.video)
        if args.preview:
            import cv2
            from src.viewer_app.calibration_preview import _calibration_frame
            import math

            if not math.isfinite(args.seconds) or args.seconds < 0:
                raise ValueError("Preview time must be finite and nonnegative")
            destination = Path(args.preview)
            if destination.exists():
                raise FileExistsError(f"Preview already exists: {destination}")
            frame, _ = _calibration_frame(source, args.seconds)
            success, encoded = cv2.imencode(destination.suffix, rectify_frame(frame, profile))
            if not success:
                raise ValueError("Could not encode the corrected preview")
            with destination.open("xb") as stream:
                stream.write(encoded.tobytes())
            print(destination.resolve())
        else:
            last_report = 0.0

            def report(update):
                nonlocal last_report
                now = time.monotonic()
                if now - last_report >= 10:
                    rate = update.get("processing_fps")
                    eta = update.get("eta_sec")
                    details = f" | {rate:.1f} fps | {eta / 60:.1f} min left" if rate and eta else ""
                    print(f"{update['message']} {update.get('progress', 0):.1f}%{details}", flush=True)
                    last_report = now

            video, record = prepare_video(source, profile, directory=args.output_dir, encoder=args.encoder,
                                          progress=report)
            print(json.dumps({"video": str(video), "provenance": record}, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
