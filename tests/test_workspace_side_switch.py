"""A deciding-set side switch must preserve stable point winners in every view."""

from copy import deepcopy

import pytest

from src.tracking.statistics_reconciliation import reconcile_statistics
from src.viewer_app.match_workspace import project_context, score_projection
from src.viewer_app.tactics import build_tactics


def switched_workspace(point_reviews=()):
    events = [
        {"frame": 0, "type": "serve", "team": "team_a"},
        {"frame": 10, "type": "reception", "team": "team_b"},
        {"frame": 20, "type": "set", "team": "team_b"},
        {"frame": 30, "type": "attack", "team": "team_b"},
        {"frame": 60, "type": "serve", "team": "team_b"},
    ]
    attempts = [
        {"rally_id": "fsm_before_switch", "start_frame": 0, "end_frame": 40,
         "lifecycle": {"serve_frame": 0, "serving_team": "team_a", "receiving_team": "team_b",
                       "serve_confirmed": True, "reception_frame": 10,
                       "reception_quality": "controlled"}},
        {"rally_id": "fsm_after_switch", "start_frame": 60, "end_frame": 99,
         "lifecycle": {"serve_frame": 60, "serving_team": "team_b", "receiving_team": "team_a",
                       "serve_confirmed": True}},
    ]
    current = deepcopy(attempts)
    for index, attempt in enumerate(current):
        attempt["rally_id"] = f"spotter_{index}"
    raw = {"fps": 30, "all_events": events, "service_attempts_fsm": attempts,
           "service_attempts": current, "rallies": deepcopy(current)}
    config = {
        "teams": {"away": {"name": "Away"}, "home": {"name": "Home"}},
        "initial_score": {"away": 8, "home": 7},
        "team_identity_map": {"team_a": "away", "team_b": "home"},
        "segments": [
            {"start_frame": 0, "end_frame": 49, "set_id": "set_5"},
            {"start_frame": 50, "end_frame": 99, "set_id": "set_5",
             "team_identity_map": {"team_a": "home", "team_b": "away"}},
        ],
        "rotations": [
            {"start_frame": 0, "end_frame": 99, "team_id": "home", "rotation": 2},
            {"start_frame": 0, "end_frame": 99, "team_id": "away", "rotation": 4},
        ],
    }
    report = {"roster_ids": ["P1"], "side": "b", "contacts": [
        {"event_index": index, "frame": event["frame"], "type": event["type"],
         "status": "assigned", "player_id": "P1", "label_source": "derived", "verified": False}
        for index, event in enumerate(events) if index in (1, 2, 3)
    ]}
    annotations = {"net_floor_left": [0, 0], "net_floor_right": [300, 0],
                   "frames": [{"f": frame, "ball": [75, 80], "ball_source": "detected"}
                              for frame in range(100)]}
    before = deepcopy(raw)
    stats = project_context(raw, config)
    assert raw == before
    return {"stats": stats, "report": report, "config": config, "context_saved": True,
            "point_reviews": list(point_reviews), "annotations": annotations}


def test_same_set_switch_normalizes_automatic_winner_to_original_rally_side():
    state = switched_workspace()
    point = state["stats"]["match_analytics"]["points"][0]
    assert point["winner_team_id"] == "away"
    assert point["winner_court_side"] == "team_a"
    assert point["evidence_frame"] == 60
    attempt_point = state["stats"]["service_attempts_fsm"][0]["point"]
    assert attempt_point["winner_team_id"] == "away"
    assert attempt_point["winner_court_side"] == "team_a"
    assert attempt_point["evidence_frame"] == 60
    assert attempt_point["score_after"] == point["score_after"]
    assert attempt_point["set_score_after"] == point["set_score_after"]
    projected = score_projection(state)
    assert projected["points"][0]["winner_team_id"] == "away"
    assert projected["points"][0]["frame"] == 60


@pytest.mark.parametrize("reviewed", [False, True])
def test_same_set_switch_rotation_rates_keep_manual_winner_in_original_rally_sides(reviewed):
    reviews = [{"attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}] if reviewed else []
    state = switched_workspace(reviews)
    result = build_tactics(state["stats"], state["report"], state["annotations"], state["config"], reviews)
    first = [row for row in result["rotation_rallies"] if row["attempt_ref"] == "service_attempts_fsm:0"]
    by_team = {row["team_id"]: row for row in first}
    assert by_team["away"]["role"] == "serve"
    assert by_team["home"]["role"] == "receive"
    assert by_team["away"]["won"] is (not reviewed)
    assert by_team["home"]["won"] is reviewed
    assert by_team["home"]["point_verified"] is reviewed
    assert score_projection(state)["points"][0]["winner_team_id"] == ("home" if reviewed else "away")


@pytest.mark.parametrize("reviewed", [False, True])
def test_same_set_switch_terminal_attack_outcome_matches_stable_winner(reviewed):
    reviews = [{"attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}] if reviewed else []
    state = switched_workspace(reviews)
    reconciliation = reconcile_statistics(state["stats"], state["report"], reviews)
    attack = next(contact for contact in reconciliation["contacts"] if contact["event_index"] == 3)
    assert attack["outcome"] == ("point" if reviewed else "lost")
    assert attack["issues"] == []
    assert attack["point_reviewed"] is reviewed
    assert attack["event_verified"] is False
    assert attack["author_source"] == "automatic"
