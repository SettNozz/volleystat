"""A calibration built from a named setup must use the pipeline's world frame.

The world frame is the one ``CourtProjector`` documents: x in [-4.5, 4.5]
across the court, y in [-9, 9] along it, y = 0 under the net. Getting this
wrong puts the team divider on an end line instead of the net, which silently
breaks the team split.
"""

import json

import numpy as np
import pytest

from scripts.evaluation.build_setup_calibration import build, main

SETUP = {
    "name": "setup_1",
    "court_corners": [
        [504.29, 375.11], [811.44, 375.05], [1391.68, 526.73], [113.47, 528.72],
    ],
    "frame_width": 1280.0,
    "frame_height": 720.0,
    "net_y": 283.0,
    "net_x_left": 400.0,
    "net_x_right": 944.0,
}


def _calibration():
    return build(SETUP, 3840.0, 2160.0, 2.34)


def test_corners_scale_to_native_resolution():
    corners = _calibration()["court_corners"]
    assert corners[0] == pytest.approx([1512.87, 1125.33])
    assert corners[2] == pytest.approx([4175.04, 1580.19])


def test_net_line_comes_from_the_setup():
    calibration = _calibration()
    assert calibration["net_line_points"] == [[1200.0, 849.0], [2832.0, 849.0]]


def test_world_origin_lands_under_the_net_not_on_an_end_line():
    calibration = _calibration()
    homography = np.asarray(calibration["perspective_matrix"], dtype=np.float64)

    def project(point):
        values = homography @ np.array([point[0], point[1], 1.0])
        return values[:2] / values[2]

    far = project((0.0, -9.0))[1]
    near = project((0.0, 9.0))[1]
    origin = project((0.0, 0.0))[1]
    corners = np.asarray(calibration["court_corners"], dtype=np.float64)
    assert far == pytest.approx(corners[:2, 1].mean(), abs=1.0)
    assert near == pytest.approx(corners[2:, 1].mean(), abs=1.0)
    # The divider sits between the end lines, which a half-court quad would
    # have placed on the far line itself.
    assert far < origin < near
    assert origin == pytest.approx(1213.9, abs=1.0)


def test_floor_divider_is_the_projected_midline():
    calibration = _calibration()
    sam = calibration["net_detection_sam"]
    left = sam["net_floor_left_720p"]
    right = sam["net_floor_right_720p"]
    assert left[0] * 3840.0 / 1280.0 == pytest.approx(1200.0)
    assert right[0] * 3840.0 / 1280.0 == pytest.approx(2832.0)
    # Matches the geometry the completed offline pose extraction recorded.
    assert left[1] * 2160.0 / 720.0 == pytest.approx(1214.2785, abs=1e-3)
    assert right[1] * 2160.0 / 720.0 == pytest.approx(1213.5298, abs=1e-3)
    assert sam["net_floor_method"] == "calibrated_court_midline_fallback"


def test_loader_facing_lines_bracket_the_court():
    calibration = _calibration()
    assert calibration["back_line_points"][0][1] > calibration["front_line_points"][0][1]
    assert min(p[0] for p in calibration["left_sideline_points"]) == pytest.approx(340.41)
    assert max(p[0] for p in calibration["right_sideline_points"]) == pytest.approx(4175.04)


def test_provenance_does_not_claim_a_measurement():
    metadata = _calibration()["metadata"]
    assert "court_setups.json:setup_1" in metadata["input_source"]
    assert "not measured" in metadata["provenance"]


def test_existing_output_is_refused(tmp_path):
    setups = tmp_path / "setups.json"
    setups.write_text(json.dumps({"version": 1, "setups": [SETUP]}), encoding="utf-8")
    output = tmp_path / "court_calibration_test.json"
    assert main([
        "--setups", str(setups), "--output", str(output),
    ]) == 0
    with pytest.raises(SystemExit):
        main(["--setups", str(setups), "--output", str(output)])


def test_unknown_setup_name_is_refused(tmp_path):
    setups = tmp_path / "setups.json"
    setups.write_text(json.dumps({"version": 1, "setups": [SETUP]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main([
            "--setups", str(setups), "--setup-name", "missing",
            "--output", str(tmp_path / "court_calibration_other.json"),
        ])
