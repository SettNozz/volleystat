import unittest

from src.tracking.attack_heatmap import AttackHeatmap


class _Projector:
    """Image x=450 is the court centre (100 px per metre across).

    Image y=500 is the line under the net; 50 px per metre in depth, with the
    near half positive.
    """

    def image_to_world(self, pt):
        return ((float(pt[0]) - 450.0) / 100.0, (float(pt[1]) - 500.0) / 50.0)

    def world_to_image(self, wpt):
        return (float(wpt[0]) * 100.0 + 450.0, float(wpt[1]) * 50.0 + 500.0)


class _MirroredProjector(_Projector):
    """Same court, calibrated with front/back swapped: world x runs left."""

    def image_to_world(self, pt):
        wx, wy = super().image_to_world(pt)
        return (-wx, -wy)

    def world_to_image(self, wpt):
        return super().world_to_image((-wpt[0], -wpt[1]))


def _pose(team, foot_x, foot_y=600):
    return {
        "team": team,
        "bb": [foot_x - 20, foot_y - 300, foot_x + 20, foot_y],
        "kp": [],
    }


def _annotation(frame, ball_x, poses, ball_y=180, **extra):
    return {
        "f": frame,
        "t": frame / 30.0,
        "ball": [ball_x, ball_y],
        "poses": poses,
        **extra,
    }


class TestAttackHeatmap(unittest.TestCase):
    def test_counts_only_confirmed_attacks_in_three_net_zones(self):
        annotations = [
            _annotation(10, 120, [_pose("b", 120)]),
            _annotation(30, 450, [_pose("b", 450)]),
            _annotation(50, 780, [_pose("a", 780, foot_y=400)]),
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

        self.assertEqual(heatmap["version"], 4)
        self.assertEqual(
            heatmap["algorithm"], "fsm_attack_origin_with_measured_target"
        )
        self.assertEqual(heatmap["total_attacks"], 3)
        self.assertEqual(
            [(zone["id"], zone["count"]) for zone in heatmap["zones"]],
            [("left", 1), ("center", 1), ("right", 1)],
        )

    def test_front_and_back_row_split_at_the_attack_line(self):
        annotations = [
            # 2 m from the net: front row. 5 m: back row.
            _annotation(10, 450, [_pose("b", 450, foot_y=600)]),
            _annotation(40, 450, [_pose("b", 450, foot_y=750)]),
        ]
        events = [
            {"frame": 10, "type": "attack", "team": "team_b"},
            {"frame": 40, "type": "attack", "team": "team_b"},
        ]

        heatmap = AttackHeatmap(30).analyze(
            annotations, _Projector(), events=events,
        )

        self.assertEqual([hit["row"] for hit in heatmap["hits"]], ["front", "back"])
        centre = heatmap["zones"][1]
        self.assertEqual((centre["front"], centre["back"]), (1, 1))

    def test_opposing_team_pose_cannot_define_attack_origin(self):
        annotation = _annotation(
            20,
            720,
            [
                _pose("a", 720, foot_y=400),  # closer to ball, wrong team
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

    def test_direction_is_measured_to_the_defender_that_touched_next(self):
        annotations = [
            _annotation(10, 450, [_pose("b", 450)]),
            _annotation(40, 450, [_pose("a", 450, foot_y=400)]),
            _annotation(100, 450, [_pose("b", 450)]),
            _annotation(130, 750, [_pose("a", 750, foot_y=400)]),
        ]
        events = [
            {"frame": 10, "type": "attack", "team": "team_b"},
            {"frame": 40, "type": "reception", "team": "team_a"},
            {"frame": 100, "type": "attack", "team": "team_b"},
            {"frame": 130, "type": "dig", "team": "team_a"},
        ]

        heatmap = AttackHeatmap(30).analyze(
            annotations, _Projector(), events=events,
        )

        down_the_line, cross_court = heatmap["hits"]
        self.assertEqual(down_the_line["origin_image"], [450.0, 600.0])
        self.assertEqual(down_the_line["origin_court"], [0.0, 2.0])
        self.assertEqual(down_the_line["target"]["source"], "defender_contact")
        self.assertEqual(down_the_line["target"]["court"], [0.0, -2.0])
        self.assertEqual(down_the_line["direction"]["kind"], "line")
        self.assertEqual(cross_court["target"]["court"], [3.0, -2.0])
        self.assertEqual(cross_court["direction"]["kind"], "diagonal")
        self.assertEqual(
            heatmap["directions"], {"line": 1, "diagonal": 1, "unknown": 0},
        )

    def test_far_side_attack_is_mirrored_onto_the_same_half(self):
        annotations = [
            _annotation(10, 450, [_pose("a", 450, foot_y=400)]),
            _annotation(40, 450, [_pose("b", 450, foot_y=600)]),
        ]
        events = [
            {"frame": 10, "type": "attack", "team": "team_a"},
            {"frame": 40, "type": "reception", "team": "team_b"},
        ]

        hit = AttackHeatmap(30).analyze(
            annotations, _Projector(), events=events,
        )["hits"][0]

        # The far-side attacker sits on negative world y, but the drawn court
        # always puts the attacking half above the net.
        self.assertEqual(hit["origin_world"], [0.0, -2.0])
        self.assertEqual(hit["origin_court"], [0.0, 2.0])
        self.assertEqual(hit["target"]["court"], [0.0, -2.0])

    def test_zone_names_follow_the_screen_not_the_world_sign(self):
        annotations = [_annotation(10, 120, [_pose("b", 120, foot_y=400)])]
        events = [{"frame": 10, "type": "attack", "team": "team_b"}]

        upright = AttackHeatmap(30).analyze(
            annotations, _Projector(), events=events,
        )
        mirrored = AttackHeatmap(30).analyze(
            annotations, _MirroredProjector(), events=events,
        )

        # Same pixel, same side of the screen: the calibration sign convention
        # must not rename the zone.
        self.assertEqual(upright["hits"][0]["zone"], "left")
        self.assertEqual(mirrored["hits"][0]["zone"], "left")

    def test_kill_without_a_defender_touch_falls_back_to_the_net_crossing(self):
        annotations = [
            _annotation(10, 450, [_pose("b", 450)], ball_y=560),
            _annotation(14, 300, [_pose("b", 450)], ball_y=440),
        ]
        events = [{"frame": 10, "type": "attack", "team": "team_b"}]

        hit = AttackHeatmap(30).analyze(
            annotations,
            _Projector(),
            events=events,
            net_y=500.0,
            net_x_left=0.0,
            net_x_right=900.0,
        )["hits"][0]

        target = hit["target"]
        self.assertEqual(target["source"], "net_crossing_estimate")
        self.assertEqual(target["court"], [-0.75, 0.0])
        self.assertEqual(hit["direction"]["target_source"], "net_crossing_estimate")

    def test_attacking_team_touching_again_is_not_a_target(self):
        annotations = [
            _annotation(10, 450, [_pose("b", 450)]),
            _annotation(40, 450, [_pose("b", 450)]),
        ]
        events = [
            {"frame": 10, "type": "attack", "team": "team_b"},
            {"frame": 40, "type": "pass", "team": "team_b"},
        ]

        hit = AttackHeatmap(30).analyze(
            annotations, _Projector(), events=events,
        )["hits"][0]

        self.assertIsNone(hit["target"])
        self.assertIsNone(hit["direction"])


if __name__ == "__main__":
    unittest.main()
