import unittest

from src.tracking.attack_heatmap import AttackHeatmap


class _Projector:
    def image_to_world(self, pt):
        # Image x=450 is court centre; every 100 px is one metre.
        return ((float(pt[0]) - 450.0) / 100.0, 2.0)


def _pose(team, foot_x):
    return {
        "team": team,
        "bb": [foot_x - 20, 200, foot_x + 20, 500],
        "kp": [],
    }


def _annotation(frame, ball_x, poses, **extra):
    return {
        "f": frame,
        "t": frame / 30.0,
        "ball": [ball_x, 180],
        "poses": poses,
        **extra,
    }


class TestAttackHeatmap(unittest.TestCase):
    def test_counts_only_confirmed_attacks_in_three_net_zones(self):
        annotations = [
            _annotation(10, 120, [_pose("b", 120)]),
            _annotation(30, 450, [_pose("b", 450)]),
            _annotation(50, 780, [_pose("a", 780)]),
            _annotation(70, 450, [_pose("b", 450)]),
        ]
        events = [
            {"frame": 10, "time": 10 / 30, "type": "attack", "team": "team_b"},
            {"frame": 30, "time": 1.0, "type": "attack", "team": "team_b"},
            {"frame": 50, "time": 50 / 30, "type": "attack", "team": "team_a"},
            {"frame": 70, "time": 70 / 30, "type": "pass", "team": "team_b"},
        ]

        heatmap = AttackHeatmap(30).analyze(
            annotations, _Projector(), events=events,
        )

        self.assertEqual(heatmap["algorithm"], "fsm_attack_origin")
        self.assertEqual(heatmap["total_attacks"], 3)
        self.assertEqual(
            heatmap["zones"],
            [
                {"id": "left", "count": 1},
                {"id": "center", "count": 1},
                {"id": "right", "count": 1},
            ],
        )

    def test_opposing_team_pose_cannot_define_attack_origin(self):
        annotation = _annotation(
            20,
            720,
            [
                _pose("a", 720),  # closer to ball, but not the attacking team
                _pose("b", 120),
            ],
        )
        event = {
            "frame": 20,
            "time": 20 / 30,
            "type": "attack",
            "team": "team_b",
        }

        heatmap = AttackHeatmap(30).analyze(
            [annotation], _Projector(), events=[event],
        )

        self.assertEqual(heatmap["hits"][0]["zone"], "left")
        self.assertEqual(heatmap["hits"][0]["zone_source"], "attacker_foot")

    def test_restart_or_interpolation_cannot_create_attack_zone(self):
        annotations = [
            _annotation(10, 120, [_pose("b", 120)], ball_new=True),
            _annotation(30, 450, [_pose("b", 450)], ball_interp=True),
        ]
        events = [
            {"frame": 10, "type": "attack", "team": "team_b"},
            {"frame": 30, "type": "attack", "team": "team_b"},
        ]

        heatmap = AttackHeatmap(30).analyze(
            annotations, _Projector(), events=events,
        )

        self.assertEqual(heatmap["total_attacks"], 0)
        self.assertEqual(heatmap["rejected_attacks"], 2)

    def test_annotation_events_are_supported_without_stats_event_list(self):
        annotation = _annotation(10, 450, [_pose("b", 450)], ev=["attack"])

        heatmap = AttackHeatmap(30).analyze([annotation], _Projector())

        self.assertEqual(heatmap["total_attacks"], 1)
        self.assertEqual(heatmap["hits"][0]["zone"], "center")


if __name__ == "__main__":
    unittest.main()
