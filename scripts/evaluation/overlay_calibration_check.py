"""Draw a calibration onto a frame so a human can say whether it fits.

A calibration measured for one camera placement produces plausible-looking
numbers on footage from another, and the damage is silent: the team split and
``net_y`` both come from it, and both feed the spotter's features. Before
running a new video through the pipeline with a borrowed calibration, look at
it.

    python scripts/evaluation/overlay_calibration_check.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COURT_COLOUR = (0, 255, 255)
NET_COLOUR = (0, 128, 255)
FLOOR_COLOUR = (255, 0, 255)


def draw(
    frame: np.ndarray, calibration: dict[str, Any], scale_to_frame: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the annotated frame and what was drawn, in frame pixels."""
    import cv2

    height, width = frame.shape[:2]
    saved_width = float(calibration.get("frame_width") or width)
    saved_height = float(calibration.get("frame_height") or height)
    scale_x = width / saved_width if scale_to_frame else 1.0
    scale_y = height / saved_height if scale_to_frame else 1.0

    def point(values: Sequence[float]) -> tuple[int, int]:
        return (int(round(values[0] * scale_x)), int(round(values[1] * scale_y)))

    drawn: dict[str, Any] = {
        "frame_size": [width, height],
        "calibration_size": [saved_width, saved_height],
        "scale": [scale_x, scale_y],
    }
    canvas = frame.copy()

    corners = calibration.get("court_corners")
    if corners:
        pixels = [point(corner) for corner in corners]
        cv2.polylines(
            canvas, [np.array(pixels, dtype=np.int32)], True, COURT_COLOUR, 4,
        )
        for index, pixel in enumerate(pixels):
            cv2.circle(canvas, pixel, 12, COURT_COLOUR, -1)
            cv2.putText(
                canvas, str(index), (pixel[0] + 16, pixel[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, COURT_COLOUR, 3,
            )
        drawn["court_corners"] = pixels

    net = calibration.get("net_line_points")
    if net:
        start, end = point(net[0]), point(net[1])
        cv2.line(canvas, start, end, NET_COLOUR, 4)
        cv2.putText(
            canvas, "net tape", (start[0], start[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, NET_COLOUR, 3,
        )
        drawn["net_line"] = [start, end]

    sam = calibration.get("net_detection_sam") or {}
    left = sam.get("net_floor_left_720p")
    right = sam.get("net_floor_right_720p")
    if left and right:
        # Stored against a 1280x720 reference, independent of the saved size.
        start = (
            int(round(left[0] * width / 1280.0)),
            int(round(left[1] * height / 720.0)),
        )
        end = (
            int(round(right[0] * width / 1280.0)),
            int(round(right[1] * height / 720.0)),
        )
        cv2.line(canvas, start, end, FLOOR_COLOUR, 4)
        cv2.putText(
            canvas, f"team divider ({sam.get('net_floor_method')})",
            (start[0], start[1] + 48),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, FLOOR_COLOUR, 3,
        )
        drawn["net_floor"] = [start, end]
    return canvas, drawn


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True, help="an image file")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-scale", dest="scale", action="store_false",
        help="draw the stored pixels as they are instead of scaling them to "
             "this frame's size",
    )
    parser.set_defaults(scale=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"refusing to overwrite existing {args.output}")
    import cv2

    frame = cv2.imread(str(args.frame))
    if frame is None:
        parser.error(f"could not read {args.frame}")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    canvas, drawn = draw(frame, calibration, scale_to_frame=args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        parser.error(f"could not write {args.output}")
    print(f"wrote {args.output}")
    print(json.dumps(drawn, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
