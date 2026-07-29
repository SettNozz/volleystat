import unittest

import cv2
import numpy as np

from src.detection.net_detector import NetDetector


class TestNetPostFallback(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
