"""A rally is a serve that someone then played.

These pin the definition: every rally opens on its serve, a serve nobody
touched is a service attempt and not a rally, a serve the decode missed is
recovered only from the spotter's own serve probability, and contacts with no
serve evidence join no rally.
"""

from types import SimpleNamespace

import numpy as np

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.contact_spotter import SpottedEvents
from action_recognition.rally_spotting.rally_segmentation import (
    INFERRED_SERVE_SOURCE, RallyRules, SPOTTER_SOURCE, segment_rallies,
)
from src.pipeline.game_analyzer import GameAnalyzer, GameStats

FPS = 30.0
BETWEEN = dc.LABEL_TYPES.index("between")
LIVE = dc.LABEL_TYPES.index("rally_live")


def _event(frame: int, kind: str) -> dict:
    return {
        "frame": frame, "time": round(frame / FPS, 2), "type": kind,
        "core_type": dc.to_core(kind), "confidence": 0.9,
        "source": SPOTTER_SOURCE, "team": None,
    }


def _frames(length: int) -> np.ndarray:
    return np.arange(length)


def test_a_served_and_played_point_is_one_rally_starting_on_the_serve():
    events = [_event(100, "serve"), _event(135, "reception"), _event(180, "set")]
    result = segment_rallies(events, FPS)

    assert len(result["rallies"]) == 1
    rally = result["rallies"][0]
    assert rally["start_frame"] == 100
    assert [e["type"] for e in rally["events"]] == ["serve", "reception", "set"]
    assert rally["serve_source"] == SPOTTER_SOURCE
    assert rally["lifecycle"]["status"] == "playable_rally"
    assert rally["lifecycle"]["reception_frame"] == 135
    # No side attribution exists, so none is invented.
    assert rally["contacts_team_a"] is None


def test_a_serve_nobody_touched_is_an_attempt_not_a_rally():
    events = [
        _event(100, "serve"),
        _event(700, "serve"), _event(735, "reception"),
    ]
    result = segment_rallies(events, FPS)

    assert [a["lifecycle"]["status"] for a in result["service_attempts"]] == [
        "serve_not_played", "playable_rally",
    ]
    assert [r["start_frame"] for r in result["rallies"]] == [700]
    lifecycle = result["rally_lifecycle"]
    assert lifecycle["service_attempts"] == 2
    assert lifecycle["playable_rallies"] == 1
    assert lifecycle["serves_not_played"] == 1
    # Out, net and an untouched ace look the same; none is claimed.
    assert "aces_no_reception" not in lifecycle
    assert "serve_faults" not in lifecycle


def test_a_serve_starts_a_new_point_even_right_after_contacts():
    events = [
        _event(100, "serve"), _event(130, "reception"), _event(160, "attack"),
        _event(200, "serve"), _event(230, "reception"),
    ]
    result = segment_rallies(events, FPS)

    assert [r["start_frame"] for r in result["rallies"]] == [100, 200]


def test_a_missed_serve_is_recovered_from_the_serve_probability():
    events = [_event(300, "reception"), _event(340, "set")]
    serve = np.zeros(600)
    serve[268] = 0.4
    result = segment_rallies(
        events, FPS, serve_probability=serve, frame_numbers=_frames(600),
    )

    assert len(result["rallies"]) == 1
    rally = result["rallies"][0]
    assert rally["start_frame"] == 268
    assert rally["events"][0]["type"] == "serve"
    assert rally["events"][0]["inferred"] is True
    assert rally["serve_source"] == INFERRED_SERVE_SOURCE
    assert rally["lifecycle"]["serve_confirmed"] is False
    assert [e["frame"] for e in result["inferred_serves"]] == [268]
    assert result["rally_lifecycle"]["inferred_serves"] == 1
    assert result["rally_lifecycle"]["confirmed_serves"] == 0


def test_contacts_without_serve_evidence_join_no_rally():
    events = [_event(300, "reception")]
    serve = np.full(600, 0.05)
    result = segment_rallies(
        events, FPS, serve_probability=serve, frame_numbers=_frames(600),
    )

    assert result["rallies"] == []
    assert result["service_attempts"] == []
    assert result["unanchored_groups"][0]["contacts"] == [
        {"frame": 300, "type": "reception"},
    ]


def test_without_probabilities_nothing_is_inferred():
    result = segment_rallies([_event(300, "reception"), _event(330, "set")], FPS)

    assert result["rallies"] == []
    assert result["inferred_serves"] == []


def test_the_serve_search_never_reaches_into_the_previous_point():
    # The previous point ended at frame 160; the strongest serve frame inside
    # the next group's search window lies before that and must not be taken.
    events = [
        _event(100, "serve"), _event(130, "reception"), _event(160, "attack"),
        _event(250, "reception"),
    ]
    serve = np.zeros(600)
    serve[150] = 0.9
    serve[220] = 0.2
    rules = RallyRules(max_contact_gap_sec=2.0, serve_search_sec=3.5)
    result = segment_rallies(
        events, FPS, serve_probability=serve, frame_numbers=_frames(600), rules=rules,
    )

    assert [r["start_frame"] for r in result["rallies"]] == [100, 220]


def test_a_peak_on_the_first_contacts_shoulder_is_not_a_serve():
    events = [_event(300, "reception")]
    serve = np.zeros(600)
    serve[297] = 0.9  # 0.1 s before the contact: its own ramp, not a serve
    result = segment_rallies(
        events, FPS, serve_probability=serve, frame_numbers=_frames(600),
    )

    assert result["rallies"] == []


def test_the_rally_ends_when_the_decoded_track_goes_dead():
    events = [_event(100, "serve"), _event(130, "reception")]
    labels = np.full(600, BETWEEN)
    labels[100:161] = LIVE
    result = segment_rallies(
        events, FPS, phase_labels=labels, frame_numbers=_frames(600),
    )
    assert result["rallies"][0]["end_frame"] == 160

    labels[100:] = LIVE  # never goes dead: capped at max_tail_sec
    result = segment_rallies(
        events, FPS, phase_labels=labels, frame_numbers=_frames(600),
    )
    assert result["rallies"][0]["end_frame"] == 130 + 90


def test_frame_numbers_map_rows_to_the_videos_own_frames():
    # A spotter row is not a frame number when the analysis did not start at 0.
    offset = 1000
    events = [_event(offset + 300, "reception")]
    serve = np.zeros(600)
    serve[268] = 0.5
    result = segment_rallies(
        events, FPS, serve_probability=serve,
        frame_numbers=np.arange(offset, offset + 600),
    )

    assert result["rallies"][0]["start_frame"] == offset + 268


class _FakeSpotter:
    members = [object()]

    def __init__(self, spotted: SpottedEvents):
        self.spotted = spotted

    def run(self, payload):
        return self.spotted


def test_analyzer_replaces_fsm_rallies_and_keeps_them():
    length = 400
    probabilities = np.zeros((length, dc.NUM_LABELS))
    probabilities[:, BETWEEN] = 1.0
    probabilities[140, dc.LABEL_TYPES.index("serve")] = 0.5
    labels = np.full(length, BETWEEN)
    labels[170:260] = LIVE
    spotted = SpottedEvents(
        events=[
            {**_event(172, "reception"), "phase_before": "serve_flight",
             "phase_after": "rally_live"},
            {**_event(210, "set"), "phase_before": "rally_live",
             "phase_after": "rally_live"},
        ],
        phase_labels=labels,
        frame_numbers=np.arange(length),
        decode_status="ok",
        diagnostics={"per_type": {"reception": 1, "set": 1}},
        probabilities=probabilities,
    )
    fsm_rally = {"start_frame": 210, "lifecycle": {"status": "playable_rally"}}
    stats = GameStats(
        fps=FPS,
        frame_annotations=[{"f": f, "ev": []} for f in range(length)],
        all_events=[{"frame": 210, "type": "serve"}],
        rallies=[fsm_rally],
        service_attempts=[fsm_rally],
        rally_lifecycle={"service_attempts": 1, "playable_rallies": 1},
    )
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.contact_spotter_path = "unused.pt"
    analyzer._contact_spotter = _FakeSpotter(spotted)
    analyzer.match_segments = []
    analyzer.calibration = SimpleNamespace(frame_width=3840, frame_height=2160, net_y=800.0)

    analyzer._spot_contacts(stats, FPS)

    assert stats.rallies_fsm == [fsm_rally]
    assert stats.rally_lifecycle_fsm["playable_rallies"] == 1
    assert len(stats.rallies) == 1
    assert stats.rallies[0]["start_frame"] == 140
    assert [e["type"] for e in stats.all_events] == ["serve", "reception", "set"]
    assert stats.all_events[0]["inferred"] is True
    # The inferred serve shows up on its frame like any other event.
    assert stats.frame_annotations[140]["ev"] == ["serve"]
    assert stats.rally_lifecycle["source"] == SPOTTER_SOURCE
    exported = stats.to_dict()
    assert exported["total_rallies"] == 1
    assert exported["rallies_fsm"] == [fsm_rally]
