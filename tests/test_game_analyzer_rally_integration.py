from types import SimpleNamespace

from src.pipeline.game_analyzer import GameAnalyzer


def _event(frame, event_type, team, lifecycle):
    return SimpleNamespace(
        frame_idx=frame,
        timestamp=frame / 30.0,
        event_type=event_type,
        game_state=SimpleNamespace(value="serving"),
        rally_phase=None,
        team_controlling=team,
        contacts_count=0,
        lifecycle=lifecycle,
    )


class _ScriptedFSM:
    def __init__(self, script):
        self.script = script
        self.index = -1
        self.status = {
            "game_state": "serve_ready",
            "rally_phase": None,
            "lifecycle": {"status": "unconfirmed", "serve_frame": None},
        }

    def get_current_status(self):
        return self.status

    def update(self, **_kwargs):
        self.index += 1
        item = self.script[self.index]
        self.status = item["status"]
        return item.get("events", [])

    def _end_rally(self, _frame_idx, _reason):
        self.status = {
            **self.status,
            "game_state": "rally_end",
        }


def _analyzer_with(fsm):
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer._make_fsm = lambda _fps: fsm
    return analyzer


def test_clean_pass_keeps_only_confirmed_playable_rally():
    serve_lc = {
        "status": "serve_confirmed",
        "is_playable": False,
        "serve_confirmed": True,
        "serve_frame": 0,
    }
    reception_lc = {
        **serve_lc,
        "status": "reception_pending",
        "reception_frame": 1,
    }
    playable_lc = {
        **reception_lc,
        "status": "playable_rally",
        "is_playable": True,
        "start_frame": 0,
        "followup_frame": 2,
    }
    script = [
        {
            "status": {"game_state": "serving", "rally_phase": None,
                       "lifecycle": serve_lc},
            "events": [_event(0, "serve", "team_a", serve_lc)],
        },
        {
            "status": {"game_state": "serving", "rally_phase": "serve_reception",
                       "lifecycle": reception_lc},
            "events": [_event(1, "reception", "team_b", reception_lc)],
        },
        {
            "status": {"game_state": "rally_active", "rally_phase": "second_contact",
                       "lifecycle": playable_lc},
            "events": [_event(2, "pass", "team_b", playable_lc)],
        },
        {
            "status": {"game_state": "rally_end", "rally_phase": None,
                       "lifecycle": {**playable_lc, "end_reason": "ball_out"}},
        },
    ]
    annotations = [{"f": frame, "ball": None, "poses": []} for frame in range(4)]

    events, rallies, attempts = _analyzer_with(
        _ScriptedFSM(script)
    )._recompute_events_on_clean_trajectory(annotations, 30.0)

    assert [event["type"] for event in events] == ["serve", "reception", "pass"]
    assert len(attempts) == 1
    assert len(rallies) == 1
    assert rallies[0]["start_frame"] == 0
    assert rallies[0]["lifecycle"]["status"] == "playable_rally"


def test_clean_pass_keeps_serve_fault_out_of_rallies():
    fault_lc = {
        "status": "serve_fault",
        "is_playable": False,
        "serve_confirmed": False,
        "start_frame": None,
        "serve_frame": 4,
        "end_reason": "serve_no_crossing",
    }
    script = [{
        "status": {
            "game_state": "rally_end",
            "rally_phase": None,
            "lifecycle": fault_lc,
        },
        "events": [_event(4, "serve_fault", "team_a", fault_lc)],
    }]
    annotations = [{"f": 4, "ball": None, "poses": []}]

    _, rallies, attempts = _analyzer_with(
        _ScriptedFSM(script)
    )._recompute_events_on_clean_trajectory(annotations, 30.0)

    assert rallies == []
    assert len(attempts) == 1
    assert attempts[0]["lifecycle"]["status"] == "serve_fault"
