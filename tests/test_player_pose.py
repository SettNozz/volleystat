import unittest

import numpy as np

from src.detection.player_pose import (
    CourtPlayerGate,
    CourtProjector,
    PlayerPose,
    estimate_back_facing_score,
)


class _Projector:
    """Minimal projector: image y is already world y for this unit test."""

    def inside_score(self, foot):
        return 1.0

    def entry_score(self, foot):
        return 1.0

    has_net_floor_line = False

    def image_to_world(self, foot):
        return 0.0, foot[1]

    def team_of_world_y(self, world_y):
        return "team_b" if world_y > 0 else "team_a"


def _pose(track_id, world_y):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[15] = (100.0, world_y, 1.0)
    keypoints[16] = (110.0, world_y, 1.0)
    return PlayerPose(
        team="team_a",
        center=(105.0, world_y - 40.0),
        bbox=(80.0, world_y - 160.0, 130.0, world_y),
        keypoints=keypoints,
        track_id=track_id,
    )


class TestCourtPlayerGate(unittest.TestCase):
    def test_sideline_spectator_cannot_earn_membership_from_grace_margin(self):
        """Feet 1.5 m beyond a sideline are bench area, not a team member."""
        projector = CourtProjector(np.eye(3))
        gate = CourtPlayerGate(projector)

        spectator = _pose(99, 4.0)
        spectator.keypoints[15, 0] = -6.0
        spectator.keypoints[16, 0] = -6.0
        on_court = _pose(100, 4.0)
        on_court.keypoints[15, 0] = -4.3
        on_court.keypoints[16, 0] = -4.3

        kept = gate.filter([spectator, on_court])

        self.assertEqual([pose.track_id for pose in kept], [100])
        self.assertNotIn(99, gate._last_inside)

    def test_late_visible_player_recovers_from_wrong_first_side(self):
        """An occlusion-era bad sample must not permanently pick the far team."""
        gate = CourtPlayerGate(_Projector())

        # First ankle estimate is bad (behind net).  Subsequent reliable poses
        # put the same track on the observed near side.
        teams = []
        for world_y in (-2.0, 2.0, 2.0, 2.0):
            teams.append(gate.filter([_pose(42, world_y)])[0].team)

        self.assertEqual(teams[-1], "team_b")

    def test_team_is_not_locked_by_single_first_frame_measurement(self):
        gate = CourtPlayerGate(_Projector())
        gate.filter([_pose(7, -2.0)])

        self.assertNotIn(7, gate._team_of)


class TestNetFloorLine(unittest.TestCase):
    def test_near_side_tolerance_only_covers_ankle_noise(self):
        projector = CourtProjector(
            np.eye(3), net_floor_line=((0.0, 100.0), (200.0, 100.0)),
        )

        self.assertEqual(projector.team_of((100.0, 101.0)), "team_b")
        self.assertEqual(projector.team_of((100.0, 94.0), near_tolerance_px=8.0), "team_b")
        self.assertEqual(projector.team_of((100.0, 91.0), near_tolerance_px=8.0), "team_a")

    def test_back_facing_net_occluder_is_near_team_during_serve_setup(self):
        homography = np.array([
            [10.0, 0.0, 100.0],
            [0.0, 10.0, 100.0],
            [0.0, 0.0, 1.0],
        ])
        projector = CourtProjector(
            homography,
            net_floor_line=((40.0, 100.0), (160.0, 100.0)),
            net_tape_band=(40.0, 160.0, 45.0, 80.0),
        )
        gate = CourtPlayerGate(projector)
        pose = _pose(5, 94.0)
        pose.bbox = (80.0, 35.0, 130.0, 94.0)
        pose.keypoints[5] = (115.0, 55.0, 1.0)   # anatomical left on image-right
        pose.keypoints[6] = (90.0, 55.0, 1.0)
        pose.keypoints[11] = (112.0, 75.0, 1.0)
        pose.keypoints[12] = (93.0, 75.0, 1.0)
        pose.back_facing_score = estimate_back_facing_score(pose)

        kept = gate.filter([pose], serve_setup=True)

        self.assertGreater(pose.back_facing_score, 0.45)
        self.assertEqual(kept[0].team, "team_b")
        self.assertEqual(kept[0].team_prior, "serve_back_facing_net_override")

    def test_back_facing_prior_does_not_move_clear_far_player(self):
        homography = np.array([
            [10.0, 0.0, 100.0],
            [0.0, 10.0, 100.0],
            [0.0, 0.0, 1.0],
        ])
        projector = CourtProjector(
            homography,
            net_floor_line=((40.0, 100.0), (160.0, 100.0)),
            net_tape_band=(40.0, 160.0, 45.0, 80.0),
        )
        gate = CourtPlayerGate(projector)
        pose = _pose(6, 65.0)
        pose.bbox = (80.0, 30.0, 130.0, 80.0)
        pose.keypoints[5] = (115.0, 50.0, 1.0)
        pose.keypoints[6] = (90.0, 50.0, 1.0)
        pose.keypoints[11] = (112.0, 65.0, 1.0)
        pose.keypoints[12] = (93.0, 65.0, 1.0)
        pose.back_facing_score = estimate_back_facing_score(pose)

        kept = gate.filter([pose], serve_setup=True)

        self.assertEqual(kept[0].team, "team_a")


if __name__ == "__main__":
    unittest.main()
