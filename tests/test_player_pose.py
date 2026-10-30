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




class TestRejectedNetDivider(unittest.TestCase):
    def calibration(self, spacing=14.08):
        from src.pipeline.game_analyzer import CalibrationData
        return CalibrationData(
            net_y=200,net_x_left=0,net_x_right=1000,court_top=0,
            court_bottom=1000,court_left=0,court_right=1000,
            frame_width=1000,frame_height=1000,
            homography=[[100,0,500],[0,100,500],[0,0,1]],
            net_floor_left=(0,200),net_floor_right=(1000,200),
            net_floor_method='hough_post_x_physical_height',
            net_post_spacing_m=spacing,
        )

    def test_bad_floor_cannot_collapse_both_teams_and_trigger_six_cap(self):
        from src.pipeline.game_analyzer import GameAnalyzer
        original=self.calibration()
        corrected=GameAnalyzer._validated_net_floor(original)
        def players():
            result=[]
            for side,y in enumerate((400.,600.)):
                for i in range(6):
                    p=_pose(side*6+i,y)
                    x=200.+i*100
                    p.keypoints[15]=(x,y,1.)
                    p.keypoints[16]=(x+5,y,1.)
                    p.bbox=(x-20,y-80,x+20,y)
                    result.append(p)
            return result
        def gate(calib):
            proj=CourtProjector(calib.homography,net_floor_line=(calib.net_floor_left,calib.net_floor_right))
            return CourtPlayerGate(proj,6).filter(players())
        before,after=gate(original),gate(corrected)
        self.assertEqual(len(before),6)
        self.assertEqual(sum(p.team=='team_a' for p in after),6)
        self.assertEqual(sum(p.team=='team_b' for p in after),6)
        self.assertEqual(original.net_floor_left,(0,200))
        self.assertEqual(corrected.net_floor_method,'calibrated_court_midline_fallback')
        self.assertIsNone(corrected.net_post_spacing_m)

    def test_plausible_divider_is_preserved(self):
        from src.pipeline.game_analyzer import GameAnalyzer
        c=self.calibration(10.5)
        self.assertIs(GameAnalyzer._validated_net_floor(c),c)

    def test_missing_spacing_is_measured_before_trusting_divider(self):
        from src.pipeline.game_analyzer import GameAnalyzer
        c=self.calibration(None)
        # At constant affine scale these posts really are 10m apart.
        self.assertIs(GameAnalyzer._validated_net_floor(c),c)
        c.net_floor_right=(1500,200)
        self.assertEqual(GameAnalyzer._validated_net_floor(c).net_floor_method,'calibrated_court_midline_fallback')

    def test_fallback_retains_calibrated_slope(self):
        from src.pipeline.game_analyzer import GameAnalyzer
        c=self.calibration()
        c.homography=[[100,0,500],[10,100,500],[0,0,1]]
        fixed=GameAnalyzer._validated_net_floor(c)
        self.assertAlmostEqual(fixed.net_floor_left[1],450)
        self.assertAlmostEqual(fixed.net_floor_right[1],550)

    def test_unusable_homography_does_not_invent_geometry(self):
        from src.pipeline.game_analyzer import GameAnalyzer
        c=self.calibration()
        c.homography=None
        self.assertIs(GameAnalyzer._validated_net_floor(c),c)

    def test_fallback_survives_resolution_scaling(self):
        from src.pipeline.game_analyzer import GameAnalyzer
        a=GameAnalyzer.__new__(GameAnalyzer)
        a.calibration=self.calibration();a.net_y_override=None
        scaled=a._scale_calibration(2000,2000)
        self.assertEqual(scaled.net_floor_left,(0.,1000.))
        self.assertEqual(scaled.net_floor_right,(2000.,1000.))
        self.assertEqual(scaled.net_floor_method,'calibrated_court_midline_fallback')
        self.assertEqual(a.calibration.net_floor_left,(0,200))

    def test_six_is_a_cap_not_a_fabricated_count(self):
        from src.pipeline.game_analyzer import GameAnalyzer
        c=GameAnalyzer._validated_net_floor(self.calibration())
        proj=CourtProjector(c.homography,net_floor_line=(c.net_floor_left,c.net_floor_right))
        poses=[_pose(i,600) for i in range(3)]
        for p in poses:
            p.keypoints[15,0]=300
            p.keypoints[16,0]=310
        self.assertEqual(len(CourtPlayerGate(proj,6).filter(poses)),3)


if __name__ == "__main__":
    unittest.main()
