"""Two first-party tools disagreed on corner order; this pins the fix.

``interactive_court_calibration.py`` writes the court starting from the line
nearest the camera. ``GameAnalyzer._load_calibration`` reads
``back_line_points`` as the bottom of the court and expects the far line first.
The homography is valid under either reading, so the overlay looks right while
``court_bottom`` lands on the wrong line — the failure is silent, which is why
it needs a test rather than a convention written down somewhere.
"""

import json

import numpy as np
import pytest

from scripts.normalize_calibration import WORLD_COURT, normalize, order_corners

# As the interactive tool emits them: near line first, going clockwise.
TOOL_CORNERS = [
    [210.0, 1611.0],   # near left
    [4043.0, 1558.0],  # near right
    [2334.0, 1122.0],  # far right
    [1476.0, 1119.0],  # far left
]


def _calibration(corners=None):
    return {
        "court_corners": [list(point) for point in (corners or TOOL_CORNERS)],
        "back_line_points": [[1476.0, 1119.0], [2334.0, 1122.0]],
        "front_line_points": [[210.0, 1611.0], [4043.0, 1558.0]],
        "net_line_points": [[1232.0, 826.0], [2620.0, 848.0]],
        "frame_width": 3840,
        "frame_height": 2160,
        "metadata": {"calibration_completed": True},
    }


def test_corners_are_ordered_far_first_then_left_to_right():
    assert order_corners(TOOL_CORNERS) == [
        [1476.0, 1119.0],  # far left
        [2334.0, 1122.0],  # far right
        [4043.0, 1558.0],  # near right
        [210.0, 1611.0],   # near left
    ]


def test_ordering_is_idempotent():
    once = order_corners(TOOL_CORNERS)
    assert order_corners(once) == once


def test_four_corners_are_required():
    with pytest.raises(ValueError, match="four corners"):
        order_corners(TOOL_CORNERS[:3])


def test_the_world_axis_points_from_far_to_near():
    result, changed = normalize(_calibration())
    assert changed["corner_order_changed"] is True
    assert changed["ordering_ok"] is True
    # Far end line above the net, net above the near end line, in image y.
    assert changed["far_line_y"] < changed["net_line_y"] < changed["near_line_y"]
    assert changed["near_line_y"] == pytest.approx(1584, abs=3)


def test_back_line_becomes_the_near_line():
    result, _ = normalize(_calibration())
    back = np.mean([p[1] for p in result["back_line_points"]])
    front = np.mean([p[1] for p in result["front_line_points"]])
    # The loader reads back_line_points as the bottom of the court.
    assert back > front


def test_sidelines_run_from_far_to_near():
    result, _ = normalize(_calibration())
    for side in ("left_sideline_points", "right_sideline_points"):
        far, near = result[side]
        assert far[1] < near[1]


def test_the_homography_maps_the_world_quad_onto_the_ordered_corners():
    result, _ = normalize(_calibration())
    homography = np.asarray(result["perspective_matrix"], dtype=np.float64)
    for world, corner in zip(WORLD_COURT, result["court_corners"]):
        values = homography @ np.array([world[0], world[1], 1.0])
        projected = values[:2] / values[2]
        assert projected[0] == pytest.approx(corner[0], abs=1e-3)
        assert projected[1] == pytest.approx(corner[1], abs=1e-3)


def test_the_original_corners_are_recorded_not_discarded():
    result, _ = normalize(_calibration())
    kept = result["metadata"]["normalized_from"]["court_corners"]
    assert kept == TOOL_CORNERS


def test_a_calibration_already_in_order_is_left_alone():
    ordered = order_corners(TOOL_CORNERS)
    result, changed = normalize(_calibration(ordered))
    assert changed["corner_order_changed"] is False
    assert result["court_corners"] == ordered


def test_a_quad_without_corners_is_refused():
    with pytest.raises(ValueError, match="four-corner"):
        normalize({"court_corners": None})
