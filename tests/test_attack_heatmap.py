import unittest

from src.tracking.attack_heatmap import AttackHeatmap, AttackHeatmapConfig


class _Projector:
    def image_to_world(self, pt):
        return (0.0, 2.0)

    def team_of_world_y(self, world_y):
        return "team_b" if world_y > 0 else "team_a"


def _frames(points):
    return [{"f": i, "t": i / 30.0, "ball": list(point)} for i, point in enumerate(points)]


class TestAttackHeatmap(unittest.TestCase):
    def test_detects_one_sharp_turn_and_assigns_zone(self):
        # Ball rises up-left, is hit at f=3, then travels sharply down-right.
        points = [(100, 500), (120, 480), (140, 460), (160, 440),
                  (190, 490), (220, 540), (250, 590)]
        heatmap = AttackHeatmap(30, AttackHeatmapConfig(
            min_speed_px_per_sec=50.0, min_direction_change_deg=40.0,
        )).analyze(_frames(points), _Projector())

        self.assertEqual(heatmap["total_hits"], 1)
        self.assertEqual(heatmap["hits"][0]["frame"], 3)
        self.assertEqual(heatmap["hits"][0]["zone"], "team_b_front_middle")

    def test_smooth_motion_is_not_a_hit(self):
        points = [(i * 20, 500 - i * 10) for i in range(9)]
        heatmap = AttackHeatmap(30, AttackHeatmapConfig(min_speed_px_per_sec=50.0)).analyze(_frames(points))

        self.assertEqual(heatmap["total_hits"], 0)
