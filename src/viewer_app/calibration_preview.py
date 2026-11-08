"""Calibration preview frame and manual court/net calibration for new analyses.

Moved verbatim from viewer.py, which re-exports these names.
"""

import math
from pathlib import Path


from src.tracking.court_setups import court_guide_lines, homography_from_corners, validate_corners


def _manual_calibration(edit: dict, width: int, height: int, net_height: float) -> dict:
    corners = validate_corners(edit["court_corners"])
    matrix = homography_from_corners(corners)
    tape = edit["net_tape"]
    if len(tape) == 3:  # Older preview clients submitted one shared height.
        left, right, top = map(float, tape)
        left_y = right_y = top
    else:
        (left, left_y), (right, right_y) = [tuple(map(float, point)) for point in tape]
    if not (0 <= left < right < width and 0 < left_y < height and 0 < right_y < height):
        raise ValueError("Net tape must have two ordered endpoints inside the frame")
    center_left, center_right = sorted(court_guide_lines(matrix)["center_line"])
    if center_right[0] - center_left[0] < 1e-6:
        raise ValueError("Court centre line must span the frame horizontally")
    slope = (center_right[1] - center_left[1]) / (center_right[0] - center_left[0])
    floor = [(x, center_left[1] + (x - center_left[0]) * slope) for x in (left, right)]
    if not all(math.isfinite(v) for point in floor for v in point) or any(
        tape_y >= floor_point[1] for tape_y, floor_point in zip((left_y, right_y), floor)
    ):
        raise ValueError("Net tape must be above the court centre line")
    top = (left_y + right_y) / 2
    return {
        "frame_width": width, "frame_height": height,
        "court_corners": corners, "perspective_matrix": matrix,
        "back_line_points": max((corners[:2], corners[2:]), key=lambda p: sum(y for _, y in p)),
        "left_sideline_points": [corners[0], corners[3]],
        "right_sideline_points": [corners[1], corners[2]],
        "net_line_points": [[left, left_y], [right, right_y]],
        "net_detection_sam": {
            "net_top_y_720p": top * 720 / height,
            "net_floor_left_720p": [floor[0][0] * 1280 / width, floor[0][1] * 720 / height],
            "net_floor_right_720p": [floor[1][0] * 1280 / width, floor[1][1] * 720 / height],
            "net_floor_method": "manual_court_midline",
            "net_height_m": net_height,
        },
    }


def _calibration_frame(video_path: Path, seconds: float = 0):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        last_second = None
        if math.isfinite(fps) and fps > 0 and math.isfinite(count) and count >= 1:
            last_second = (count - 1) / fps
        if seconds:
            if last_second is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(min(seconds, last_second) * fps))
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Could not read preview frame. Try an earlier time.")
        return frame, last_second
    finally:
        cap.release()
