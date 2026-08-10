import unittest

from src.tracking.enhanced_event_tracker import BallContact, ContactType
from src.tracking.integrated_volleyball_fsm import (
    CourtBounds,
    GameState,
    IntegratedVolleyballFSM,
)
from src.tracking.serve_reception import (
    annotate_rallies,
    derive_rally_lifecycle,
)


def contact(frame: int, y: float, speed: float = 500.0) -> BallContact:
    return BallContact(
        frame_idx=frame,
        timestamp=frame / 30.0,
        ball_position=(500.0, y),
        contact_type=ContactType.PASS,
        player_id=1,
        team="team_a" if y < 500.0 else "team_b",
        contact_sequence=1,
        ball_speed_before=200.0,
        ball_speed_after=speed,
        direction_change=80.0,
        speed_change=300.0,
        contact_confidence=0.9,
    )


class IntegratedRallyLifecycleTests(unittest.TestCase):
    def make_fsm(self) -> IntegratedVolleyballFSM:
        fsm = IntegratedVolleyballFSM(
            CourtBounds(left=0, right=1000, top=0, bottom=1000, net_y=500),
            fps=30.0,
        )
        fsm.current_state = GameState.SERVE_READY
        return fsm

    def confirm_serve(self, fsm: IntegratedVolleyballFSM):
        self.assertIsNone(fsm._process_ball_contact(contact(20, 650), 20, 2 / 3))
        self.assertEqual(fsm.get_rally_lifecycle()["status"], "serve_candidate")

        fsm.ball_trajectory.extend([(20, 500.0, 650.0), (21, 500.0, 400.0)])
        fsm._handle_serving(21, (500.0, 400.0))

        self.assertEqual(len(fsm._deferred_events), 1)
        serve = fsm._deferred_events.pop()
        self.assertEqual(serve.event_type, "serve")
        self.assertEqual(serve.frame_idx, 20)
        self.assertEqual(serve.lifecycle["net_cross_frame"], 21)
        self.assertTrue(serve.lifecycle["serve_confirmed"])
        return serve

    def test_rally_only_becomes_active_after_controlled_followup(self):
        fsm = self.make_fsm()
        self.confirm_serve(fsm)

        reception = fsm._process_ball_contact(contact(30, 400), 30, 1.0)
        self.assertEqual(reception.event_type, "reception")
        self.assertEqual(fsm.current_state, GameState.SERVING)
        self.assertFalse(fsm.get_rally_lifecycle()["is_playable"])
        self.assertIsNone(fsm.get_rally_lifecycle()["start_frame"])

        followup = fsm._process_ball_contact(contact(36, 420), 36, 1.2)
        self.assertEqual(followup.event_type, "pass")
        self.assertEqual(fsm.current_state, GameState.RALLY_ACTIVE)
        lifecycle = fsm.get_rally_lifecycle()
        self.assertEqual(lifecycle["status"], "playable_rally")
        self.assertTrue(lifecycle["is_playable"])
        self.assertEqual(lifecycle["start_frame"], 20)
        self.assertEqual(lifecycle["reception_frame"], 30)
        self.assertEqual(lifecycle["followup_frame"], 36)

    def test_candidate_without_net_crossing_is_discarded(self):
        fsm = self.make_fsm()
        self.assertIsNone(fsm._process_ball_contact(contact(20, 650), 20, 2 / 3))

        fsm._handle_serving(80, (500.0, 650.0))

        self.assertEqual(fsm.current_state, GameState.SERVE_READY)
        lifecycle = fsm.get_rally_lifecycle()
        self.assertEqual(lifecycle["status"], "unconfirmed")
        self.assertFalse(lifecycle["serve_confirmed"])
        self.assertIsNone(lifecycle["start_frame"])
        self.assertEqual(fsm._deferred_events, [])

    def test_reception_is_ignored_before_net_crossing(self):
        fsm = self.make_fsm()
        fsm._process_ball_contact(contact(20, 650), 20, 2 / 3)

        event = fsm._process_ball_contact(contact(25, 400), 25, 5 / 6)

        self.assertIsNone(event)
        self.assertEqual(fsm.current_state, GameState.SERVING)
        self.assertIsNone(fsm.get_rally_lifecycle()["reception_frame"])

    def test_external_timeout_drops_unconfirmed_candidate(self):
        fsm = self.make_fsm()
        fsm._process_ball_contact(contact(20, 650), 20, 2 / 3)
        fsm._end_rally(40, "no_ball_timeout")

        events = fsm.update(
            frame_idx=41,
            ball_pos=None,
            player_positions={"team_a": [], "team_b": []},
            timestamp=41 / 30,
        )

        self.assertEqual(events, [])
        self.assertEqual(fsm.get_rally_lifecycle()["status"], "unconfirmed")

    def test_explicit_serve_out_is_a_serve_fault(self):
        fsm = self.make_fsm()
        fsm._process_ball_contact(contact(20, 650), 20, 2 / 3)
        fsm._end_rally(25, "serve_out")

        events = fsm.update(
            frame_idx=26,
            ball_pos=None,
            player_positions={"team_a": [], "team_b": []},
            timestamp=26 / 30,
        )

        self.assertEqual([event.event_type for event in events], ["serve_fault"])
        self.assertEqual(events[0].lifecycle["status"], "serve_fault")

    def test_crossed_serve_without_reception_is_ace(self):
        fsm = self.make_fsm()
        self.confirm_serve(fsm)

        fsm._end_rally(100, "serve_timeout")

        lifecycle = fsm.get_rally_lifecycle()
        self.assertEqual(lifecycle["status"], "ace_no_reception")
        self.assertTrue(lifecycle["serve_confirmed"])
        self.assertFalse(lifecycle["is_playable"])

    def test_reception_without_followup_is_failed(self):
        fsm = self.make_fsm()
        self.confirm_serve(fsm)
        fsm._process_ball_contact(contact(30, 400), 30, 1.0)

        fsm._end_rally(50, "ball_out")

        self.assertEqual(
            fsm.get_rally_lifecycle()["status"], "reception_failed"
        )

    def test_reception_return_across_net_is_overpass_and_live_rally(self):
        fsm = self.make_fsm()
        self.confirm_serve(fsm)
        fsm._process_ball_contact(contact(30, 400), 30, 1.0)
        fsm.ball_trajectory.extend([(30, 500.0, 400.0), (40, 500.0, 650.0)])

        fsm._handle_serving(40, (500.0, 650.0))

        lifecycle = fsm.get_rally_lifecycle()
        self.assertEqual(fsm.current_state, GameState.RALLY_ACTIVE)
        self.assertEqual(lifecycle["status"], "playable_rally")
        self.assertEqual(lifecycle["reception_quality"], "overpass")
        self.assertEqual(lifecycle["reception_return_frame"], 40)


class ServeReceptionDerivationTests(unittest.TestCase):
    def test_canonical_outcomes_and_legacy_counters(self):
        attempts = [
            {
                "end_frame": 30,
                "events": [
                    {"frame": 10, "type": "serve_fault", "team": "team_b"},
                ],
            },
            {
                "end_frame": 60,
                "events": [
                    {"frame": 40, "type": "serve", "team": "team_b"},
                ],
            },
            {
                "end_frame": 100,
                "events": [
                    {"frame": 70, "type": "serve", "team": "team_b"},
                    {"frame": 80, "type": "reception", "team": "team_a"},
                ],
            },
            {
                "end_frame": 150,
                "lifecycle": {
                    "status": "playable_rally",
                    "is_playable": True,
                    "serve_confirmed": True,
                    "start_frame": 110,
                    "serve_frame": 110,
                    "net_cross_frame": 115,
                    "reception_frame": 120,
                    "followup_frame": 130,
                    "serving_team": "team_b",
                    "receiving_team": "team_a",
                    "end_reason": "ball_out",
                },
                "events": [
                    {"frame": 110, "type": "serve", "team": "team_b"},
                    {"frame": 120, "type": "reception", "team": "team_a"},
                    {"frame": 130, "type": "pass", "team": "team_a"},
                ],
            },
        ]

        counts = annotate_rallies(attempts)

        self.assertEqual(
            [attempt["lifecycle"]["status"] for attempt in attempts],
            [
                "serve_fault",
                "ace_no_reception",
                "reception_failed",
                "playable_rally",
            ],
        )
        self.assertIsNone(attempts[2]["lifecycle"]["start_frame"])
        self.assertEqual(attempts[3]["lifecycle"]["start_frame"], 110)
        self.assertEqual(attempts[3]["lifecycle"]["net_cross_frame"], 115)
        self.assertEqual(attempts[3]["lifecycle"]["end_reason"], "ball_out")
        self.assertEqual(counts["service_attempts"], 4)
        self.assertEqual(counts["confirmed_serves"], 3)
        self.assertEqual(counts["playable_rallies"], 1)
        self.assertEqual(counts["aces_no_reception"], 1)
        self.assertEqual(counts["reception_failed"], 1)
        self.assertEqual(counts["serve_faults"], 1)
        self.assertEqual(counts["success"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["no_reception"], 2)

    def test_opponent_touch_after_reception_marks_overpass(self):
        lifecycle = derive_rally_lifecycle(
            [
                {"frame": 10, "type": "serve", "team": "team_b"},
                {"frame": 20, "type": "reception", "team": "team_a"},
                {"frame": 40, "type": "dig", "team": "team_b"},
            ],
            rally_end_frame=100,
        )

        self.assertEqual(lifecycle["status"], "playable_rally")
        self.assertTrue(lifecycle["is_playable"])
        self.assertEqual(lifecycle["start_frame"], 10)
        self.assertEqual(lifecycle["reception_quality"], "overpass")


if __name__ == "__main__":
    unittest.main()
