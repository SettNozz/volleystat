"""Write a pipeline calibration file from a named viewer court setup.

The root ``court_calibration.json`` was measured on a different camera
(``data/labeling_batch/6_tiPcRYGcBtc.mp4``) and does not describe the 4K
YouTube sources: its quad covers a thin band and its stored net span reads
14.08 m. The viewer setup ``setup_1`` is the geometry that matches that fixed
camera, and it is what the completed offline pose extraction used.

This converts that setup into the schema ``GameAnalyzer._load_calibration``
reads, so a full analysis can be run with ``--no-net-detection`` instead of
falling back to Hough on every run.

The floor divider is written as the calibrated court midline at the setup's
post X coordinates, with ``net_floor_method`` naming it for what it is. It is
model geometry, not a measured or SAM-detected ground truth.

    python scripts/evaluation/build_setup_calibration.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Full-court quad in metres, in the frame ``CourtProjector`` documents:
# x in [-4.5, 4.5] across the court, y in [-9, 9] along it, y = 0 under the
# net. The four setup corners are therefore the two end lines, not one half.
# This is the convention the existing calibrations and the completed offline
# pose extraction both use.
WORLD_COURT = np.array(
    [[-4.5, -9.0], [4.5, -9.0], [4.5, 9.0], [-4.5, 9.0]], dtype=np.float64,
)


def build(setup: dict, width: float, height: float, net_height_m: float) -> dict:
    scale_x = width / float(setup["frame_width"])
    scale_y = height / float(setup["frame_height"])
    corners = np.asarray(setup["court_corners"], dtype=np.float64) * [scale_x, scale_y]
    net_y = float(setup["net_y"]) * scale_y
    net_x_left = float(setup["net_x_left"]) * scale_x
    net_x_right = float(setup["net_x_right"]) * scale_x

    import cv2

    # World metres -> image pixels, the direction CourtDetector uses.
    homography = cv2.getPerspectiveTransform(
        WORLD_COURT.astype(np.float32), corners.astype(np.float32),
    )
    floor = cv2.perspectiveTransform(
        np.array([[[-4.5, 0.0], [4.5, 0.0]]], dtype=np.float32), homography,
    )[0]
    slope = (
        (float(floor[1][1]) - float(floor[0][1]))
        / (float(floor[1][0]) - float(floor[0][0]))
    )
    floor_left_y = float(floor[0][1]) + slope * (net_x_left - float(floor[0][0]))
    floor_right_y = float(floor[0][1]) + slope * (net_x_right - float(floor[0][0]))

    # Corner order as stored by the viewer editor: far end line, then near.
    far_left, far_right, near_right, near_left = (
        [float(point[0]), float(point[1])] for point in corners
    )
    return {
        "court_corners": [far_left, far_right, near_right, near_left],
        "front_line_points": [far_left, far_right],
        "back_line_points": [near_left, near_right],
        "left_sideline_points": [far_left, near_left],
        "right_sideline_points": [far_right, near_right],
        "net_line_points": [[net_x_left, net_y], [net_x_right, net_y]],
        "perspective_matrix": homography.tolist(),
        "frame_width": int(width),
        "frame_height": int(height),
        "net_detection_sam": {
            "net_top_y_720p": net_y * 720.0 / height,
            "net_height_m": net_height_m,
            "net_floor_left_720p": [
                net_x_left * 1280.0 / width, floor_left_y * 720.0 / height,
            ],
            "net_floor_right_720p": [
                net_x_right * 1280.0 / width, floor_right_y * 720.0 / height,
            ],
            "net_floor_method": "calibrated_court_midline_fallback",
        },
        "metadata": {
            "calibration_completed": True,
            "input_source": f"configs/court_setups.json:{setup['name']}",
            "provenance": (
                "named viewer court setup scaled to native resolution; "
                "model geometry, not measured or SAM-detected ground truth"
            ),
            "court_geometry": {
                "length": 18.0, "width": 9.0, "half_length": 9.0,
                "attack_line_distance": 3.0, "net_height": net_height_m,
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setups", type=Path, default=ROOT / "configs/court_setups.json")
    parser.add_argument("--setup-name", default="setup_1")
    parser.add_argument("--width", type=float, default=3840.0)
    parser.add_argument("--height", type=float, default=2160.0)
    parser.add_argument("--net-height-m", type=float, default=2.34)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"refusing to overwrite existing {args.output}")
    payload = json.loads(args.setups.read_text(encoding="utf-8"))
    setup = next(
        (row for row in payload["setups"] if row["name"] == args.setup_name), None,
    )
    if setup is None:
        parser.error(f"no setup named {args.setup_name}")
    calibration = build(setup, args.width, args.height, args.net_height_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=1), encoding="utf-8")
    sam = calibration["net_detection_sam"]
    print(f"wrote {args.output}")
    print(f"  corners {calibration['court_corners']}")
    print(f"  net line {calibration['net_line_points']}")
    print(
        "  floor divider native "
        f"{[sam['net_floor_left_720p'][0] * args.width / 1280.0, sam['net_floor_left_720p'][1] * args.height / 720.0]}"
        " -> "
        f"{[sam['net_floor_right_720p'][0] * args.width / 1280.0, sam['net_floor_right_720p'][1] * args.height / 720.0]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
