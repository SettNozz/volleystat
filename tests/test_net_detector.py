import unittest

import cv2
import numpy as np

from src.detection.net_detector import (
    NetDetector,
    anchor_net_top_to_calibration,
    infer_floor_y_from_net_bounds,
    infer_net_bottom_y_from_floor,
)


class TestNetPostFallback(unittest.TestCase):
    def test_floor_height_uses_regulation_net_proportions(self):
        self.assertAlmostEqual(
            infer_floor_y_from_net_bounds(279.0, 331.0),
            400.68,
            places=2,
        )
        self.assertAlmostEqual(
            infer_floor_y_from_net_bounds(279.0, 331.0, net_top_height_m=2.43),
            405.36,
            places=2,
        )

    def test_mesh_bottom_uses_mixed_net_proportions(self):
        self.assertAlmostEqual(
            infer_net_bottom_y_from_floor(276.0, 380.84),
            320.80,
            places=2,
        )
        self.assertAlmostEqual(
            infer_net_bottom_y_from_floor(
                276.0,
                380.84,
                net_top_height_m=2.43,
            ),
            319.14,
            places=2,
        )

    def test_sam_top_is_anchored_when_wall_mask_wins(self):
        self.assertEqual(
            anchor_net_top_to_calibration(203.0, 276.0, net_height_hint=60.0),
            276.0,
        )
        self.assertEqual(
            anchor_net_top_to_calibration(271.0, 276.0, net_height_hint=60.0),
            271.0,
        )

    def test_homography_projects_net_floor_priors_to_720p(self):
        homography = np.array([
            [100.0, 0.0, 640.0],
            [0.0, 100.0, 360.0],
            [0.0, 0.0, 1.0],
        ])

        priors = NetDetector.project_net_floor_x_priors(
            homography, frame_w=1280, frame_h=720,
        )

        self.assertIsNotNone(priors)
        self.assertAlmostEqual(priors[0], 190.0)
        self.assertAlmostEqual(priors[1], 1090.0)

    def test_post_spacing_is_measured_in_court_meters(self):
        homography = np.array([
            [100.0, 0.0, 640.0],
            [0.0, 100.0, 360.0],
            [0.0, 0.0, 1.0],
        ])

        spacing = NetDetector.estimate_post_spacing_m(
            (140.0, 360.0),
            (1140.0, 360.0),
            homography,
            frame_w=1280,
            frame_h=720,
        )

        self.assertAlmostEqual(spacing, 10.0)
        self.assertTrue(NetDetector.post_spacing_is_plausible(spacing))
        self.assertFalse(NetDetector.post_spacing_is_plausible(9.5))

    def test_finds_post_when_stale_prior_is_on_the_wrong_side(self):
        """A new camera can put the real left post right of the old prior."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (408, 250), (408, 430), (255, 255, 255), 4)

        point = NetDetector._find_post_base(
            frame, expected_x=372.0, net_top_y=279.0, outer_direction=-1,
        )

        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 408.0, delta=6.0)
        self.assertAlmostEqual(point[1], 430.0, delta=6.0)

    def test_tight_homography_search_rejects_distant_player_leg(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(frame, (896, 260), (896, 405), (255, 255, 255), 4)
        cv2.line(frame, (1035, 300), (1035, 500), (255, 255, 255), 7)

        point = NetDetector._find_post_base(
            frame,
            expected_x=888.0,
            net_top_y=279.0,
            outer_direction=1,
            search_half_width=70,
        )

        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 896.0, delta=6.0)


if __name__ == "__main__":
    unittest.main()
