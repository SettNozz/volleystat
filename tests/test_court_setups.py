import json
import tempfile
import unittest
from pathlib import Path

from src.tracking.court_setups import (
    CourtSetup,
    CourtSetupError,
    CourtSetupStore,
    corners_from_homography,
    court_guide_lines,
    homography_from_corners,
    scale_corners,
    validate_corners,
    world_x_screen_sign,
)

# A wide lens shot from behind the near baseline: the left near corner is
# outside the picture, which is the case the editor has to survive.
WIDE_LENS_CORNERS = [[1433, 1134], [2383, 1129], [3730, 1439], [-197, 1425]]


class TestCourtGeometry(unittest.TestCase):
    def test_homography_round_trips_the_corners(self):
        homography = homography_from_corners(WIDE_LENS_CORNERS)
        recovered = corners_from_homography(homography)

        for expected, actual in zip(WIDE_LENS_CORNERS, recovered):
            self.assertAlmostEqual(expected[0], actual[0], places=6)
            self.assertAlmostEqual(expected[1], actual[1], places=6)

    def test_corner_outside_the_frame_is_accepted(self):
        # -197 px is a real court corner the camera simply never saw.
        self.assertEqual(len(validate_corners(WIDE_LENS_CORNERS)), 4)

    def test_self_intersecting_quad_is_rejected(self):
        with self.assertRaises(CourtSetupError):
            homography_from_corners([[0, 0], [10, 0], [0, 10], [10, 10]])

    def test_collinear_corners_are_rejected(self):
        with self.assertRaises(CourtSetupError):
            homography_from_corners([[0, 0], [5, 0], [10, 0], [0, 10]])

    def test_guides_include_the_lines_a_user_can_actually_see(self):
        guides = court_guide_lines(homography_from_corners(WIDE_LENS_CORNERS))

        self.assertIn("center_line", guides)
        self.assertIn("attack_line_near", guides)
        self.assertIn("attack_line_far", guides)
        # The centre line sits between the two baselines in image space.
        centre_y = guides["center_line"][0][1]
        self.assertLess(WIDE_LENS_CORNERS[0][1], centre_y)
        self.assertGreater(WIDE_LENS_CORNERS[3][1], centre_y)

    def test_screen_sign_follows_the_calibration_orientation(self):
        upright = homography_from_corners(WIDE_LENS_CORNERS)
        # Same court, clicked with front and back swapped.
        flipped = homography_from_corners([
            WIDE_LENS_CORNERS[2], WIDE_LENS_CORNERS[3],
            WIDE_LENS_CORNERS[0], WIDE_LENS_CORNERS[1],
        ])

        self.assertEqual(world_x_screen_sign(upright), 1.0)
        self.assertEqual(world_x_screen_sign(flipped), -1.0)

    def test_corners_rescale_between_resolutions(self):
        scaled = scale_corners(WIDE_LENS_CORNERS, (3840, 2160), (1280, 720))

        self.assertAlmostEqual(scaled[0][0], 1433 / 3.0)
        self.assertAlmostEqual(scaled[3][1], 1425 / 3.0)


class TestCourtSetupStore(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = CourtSetupStore(Path(self.directory.name) / "setups.json")

    def _setup(self, name="Gym A", corners=None):
        return CourtSetup(
            name=name,
            court_corners=corners or WIDE_LENS_CORNERS,
            frame_width=3840,
            frame_height=2160,
        )

    def test_missing_file_is_an_empty_collection(self):
        self.assertEqual(self.store.load(), [])

    def test_saving_the_same_name_updates_in_place(self):
        first = self.store.save(self._setup())
        moved = [[x + 5, y] for x, y in WIDE_LENS_CORNERS]
        second = self.store.save(self._setup(corners=moved))

        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.store.load()), 1)
        self.assertEqual(self.store.load()[0].court_corners[0][0], 1438.0)

    def test_setups_are_independent_per_camera_placement(self):
        self.store.save(self._setup("Gym A"))
        self.store.save(self._setup("Gym B - camera moved"))

        self.assertEqual(
            sorted(setup.name for setup in self.store.load()),
            ["Gym A", "Gym B - camera moved"],
        )

    def test_delete_reports_whether_anything_was_removed(self):
        setup = self.store.save(self._setup())

        self.assertTrue(self.store.delete(setup.id))
        self.assertFalse(self.store.delete(setup.id))

    def test_a_corrupt_entry_does_not_hide_the_healthy_ones(self):
        self.store.save(self._setup("Gym A"))
        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        payload["setups"].append({"name": "broken", "court_corners": [[0, 0]]})
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")

        self.assertEqual([setup.name for setup in self.store.load()], ["Gym A"])

    def test_a_setup_rescales_to_another_recording_resolution(self):
        setup = self._setup()

        corners = setup.corners_for(1280, 720)

        self.assertAlmostEqual(corners[0][0], 1433 / 3.0)
        self.assertEqual(len(setup.homography_for(1280, 720)), 3)

    def test_a_setup_needs_a_name(self):
        with self.assertRaises(CourtSetupError):
            self._setup(name="   ")


if __name__ == "__main__":
    unittest.main()
