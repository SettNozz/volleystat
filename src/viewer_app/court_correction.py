"""Annotation I/O and court-corner validation for the court editor.

Moved verbatim from viewer.py, which re-exports these names.
"""

import json
import math
from pathlib import Path


from src.tracking.court_setups import homography_from_corners
from src.viewer_app.calibration_preview import _manual_calibration


def _sync_manual_net_geometry(payload: dict, previous_net_y: float) -> None:
    """Save a manual correction as one coherent tape/mesh/divider geometry."""
    if payload.get("net_floor_method") != "manual_court_midline":
        return
    width, height = payload["frame_width"], payload["frame_height"]
    net_height = float(payload.get("net_height_m") or 2.34)
    shift = payload["net_y"] - previous_net_y
    endpoint_y = payload.get("net_tape_endpoint_y") or [previous_net_y, previous_net_y]
    tape = [[payload["net_x_left"], endpoint_y[0] + shift],
            [payload["net_x_right"], endpoint_y[1] + shift]]
    manual = _manual_calibration(
        {"court_corners": payload["court_corners"], "net_tape": tape},
        width, height, net_height,
    )["net_detection_sam"]
    floor = [[point[0] * width / 1280, point[1] * height / 720] for point in (
        manual["net_floor_left_720p"], manual["net_floor_right_720p"],
    )]
    top = (tape[0][1] + tape[1][1]) / 2
    payload.update(
        net_tape_endpoint_y=[tape[0][1], tape[1][1]], net_tape_top=top,
        net_tape_bottom=top + ((floor[0][1] + floor[1][1]) / 2 - top) / net_height,
        net_floor_left=floor[0], net_floor_right=floor[1],
    )


def _read_annotations(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_annotations(path: str, payload: dict) -> None:
    """Atomically persist viewer-only calibration corrections."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(target)


def _validated_corners(value, width: float, height: float) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("court_corners must contain exactly four [x, y] points")
    corners: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each court corner must be an [x, y] pair")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("court corners must be finite numbers")
        # A wide lens routinely pushes a real court corner outside the frame,
        # so only obviously broken coordinates are rejected here.
        if not (-width <= x <= width * 2 and -height <= y <= height * 2):
            raise ValueError("court corner lies implausibly far outside the video")
        corners.append([round(x, 2), round(y, 2)])
    return corners


def _projector_for(payload: dict, corners: list):
    """Court projector for a result from its (possibly corrected) corners."""
    from src.detection.player_pose import CourtProjector

    homography = homography_from_corners(corners)
    floor_left = payload.get("net_floor_left")
    floor_right = payload.get("net_floor_right")
    floor_line = (
        (tuple(floor_left), tuple(floor_right))
        if floor_left and floor_right else None
    )
    tape_band = None
    if floor_line and payload.get("net_tape_top") and payload.get("net_tape_bottom"):
        x1, x2 = sorted((floor_left[0], floor_right[0]))
        tape_band = (x1, x2, payload["net_tape_top"], payload["net_tape_bottom"])
    return CourtProjector(
        homography, net_floor_line=floor_line, net_tape_band=tape_band,
    ), homography
