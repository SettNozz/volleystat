import unittest
from unittest import mock

import numpy as np

from scripts.interactive_court_calibration import (
    CalibrationStage,
    InteractiveCourtCalibration,
)


class InteractiveCourtCalibrationTests(unittest.TestCase):
    def test_final_enter_exits_gui_loop_calculates_and_saves(self):
        calibration = InteractiveCourtCalibration("test.mp4")
        calibration.current_stage = CalibrationStage.FRONT_LINE
        calibration.calibration_data.front_line_points = [(100, 100), (200, 100)]

        capture = mock.Mock()
        capture.read.return_value = (
            True,
            np.zeros((240, 320, 3), dtype=np.uint8),
        )

        with (
            mock.patch.object(
                calibration, "_open_video_source", return_value=capture
            ),
            mock.patch.object(calibration, "_update_display"),
            mock.patch.object(calibration, "_calculate_final_calibration") as calculate,
            mock.patch.object(calibration, "_save_calibration") as save,
            mock.patch(
                "scripts.interactive_court_calibration.cv2.namedWindow"
            ),
            mock.patch(
                "scripts.interactive_court_calibration.cv2.resizeWindow"
            ),
            mock.patch(
                "scripts.interactive_court_calibration.cv2.setMouseCallback"
            ),
            mock.patch("scripts.interactive_court_calibration.cv2.imshow"),
            mock.patch(
                "scripts.interactive_court_calibration.cv2.waitKey",
                return_value=13,
            ) as wait_key,
            mock.patch(
                "scripts.interactive_court_calibration.cv2.destroyAllWindows"
            ),
            mock.patch("builtins.print"),
        ):
            calibration.run()

        self.assertEqual(calibration.current_stage, CalibrationStage.COMPLETED)
        wait_key.assert_called_once()
        capture.release.assert_called_once()
        calculate.assert_called_once()
        save.assert_called_once()

    def test_reported_4k_points_produce_a_finite_homography(self):
        calibration = InteractiveCourtCalibration("test_4k.mp4")
        data = calibration.calibration_data
        data.frame_width = 3840
        data.frame_height = 2160
        data.back_line_points = [(3731, 1439), (0, 1426)]
        data.left_sideline_points = [(0, 1390), (1434, 1134)]
        data.right_sideline_points = [(3731, 1439), (2384, 1129)]
        data.net_line_points = [(1171, 826), (2570, 832)]
        data.front_line_points = [(1437, 1134), (2384, 1129)]

        with mock.patch("builtins.print"):
            calibration._calculate_final_calibration()

        self.assertEqual(len(data.court_corners), 4)
        self.assertEqual(np.asarray(data.perspective_matrix).shape, (3, 3))
        self.assertTrue(np.isfinite(data.court_corners).all())
        self.assertTrue(np.isfinite(data.perspective_matrix).all())


if __name__ == "__main__":
    unittest.main()
