"""Regression for the empty-floor skeleton at f4752 (2:38.56)."""

import numpy as np
import pytest

from src.detection.player_pose import (
    CourtPlayerGate,
    CourtProjector,
    PlayerPose,
    filter_perspective_pose_outliers,
)


# Saved manual court calibration for 7YnJlvChSzw_4k, in native 3840x2160.
H = np.array([
    [152.59991869581395, -137.01846253426748, 2035.185439859067],
    [3.4203971823944306, -72.5576528930461, 1195.898793570434],
    [0.00216103157746528, -0.07300050640644824, 1.0],
])


def player(track_id, box, ankles, *, complete=True):
    kp = np.zeros((17, 3), dtype=np.float32)
    # Only the confidence of these torso/leg points enters this scale check.
    kp[[5, 6, 11, 12, 13, 14], 2] = float(complete)
    kp[15:] = [[ankles[0], ankles[2], 0.98], [ankles[1], ankles[2], 0.98]]
    return PlayerPose('team_b', ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                      tuple(box), kp, track_id=track_id)


def scene():
    # Actual saved boxes/ankles at f4752: different depths, similar normalized size.
    real = [
        player(644, [1597.3, 892.7, 1754.0, 1281.9], [1617, 1723.5, 1251]),
        player(798, [1739.3, 926.7, 1849.6, 1215.3], [1768.5, 1804.5, 1191]),
        player(826, [2061.7, 862.1, 2275.1, 1423.9], [2184, 2149.5, 1390.5]),
    ]
    ghost = player(851, [1343.4, 1784.4, 1425.6, 2025.6], [1374, 1402.5, 1993.5])
    return real, ghost


@pytest.mark.parametrize('resolution', [1 / 3, 1, 2])
@pytest.mark.parametrize('homogeneous_scale', [1, -7])
def test_empty_floor_pose_rejected_before_it_can_earn_court_membership(resolution, homogeneous_scale):
    real, ghost = scene()
    for pose in real + [ghost]:
        pose.bbox = tuple(v * resolution for v in pose.bbox)
        pose.center = tuple(v * resolution for v in pose.center)
        pose.keypoints[:, :2] *= resolution
    projector = CourtProjector(np.diag([resolution, resolution, 1]) @ H * homogeneous_scale)
    assert projector.entry_score(ghost.foot_point()) > 0  # Old gate admitted it.
    gate = CourtPlayerGate(projector, max_per_team=None)
    gate._last_inside[851] = 0  # Existing grace must not bypass the check either.
    assert gate.filter(real + [ghost]) == real
    assert gate._last_inside[851] == 0
    assert gate.filter(real + [ghost]) == real
    assert gate._last_inside[851] == 0


def test_not_enough_court_references_abstains():
    real, ghost = scene()
    poses = real[:1] + [ghost]
    assert filter_perspective_pose_outliers(poses, CourtProjector(H)) == poses


def test_partial_occluded_pose_abstains():
    real, ghost = scene()
    ghost.keypoints[13, 2] = 0.1
    poses = real + [ghost]
    assert filter_perspective_pose_outliers(poses, CourtProjector(H)) == poses


def test_compact_player_and_wide_diving_player_survive():
    real, compact = scene()
    # More than twice the false pose's extent still permits a much shorter player.
    x1, y1, x2, y2 = compact.bbox
    compact.bbox = (x1, y2 - 600, x2, y2)
    diving = player(852, [900, 1784.4, 1900, 2025.6], [1374, 1402.5, 1993.5])
    poses = real + [compact, diving]
    assert filter_perspective_pose_outliers(poses, CourtProjector(H)) == poses


def test_off_court_detections_do_not_supply_reference_scale():
    real, ghost = scene()
    for pose in real:
        pose.keypoints[15:, 0] = -1000
    poses = real + [ghost]
    assert filter_perspective_pose_outliers(poses, CourtProjector(H)) == poses
