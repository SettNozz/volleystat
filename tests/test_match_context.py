"""Manual match-context validation and read-only projection contracts."""

from copy import deepcopy

import pytest

from src.viewer_app.match_workspace import project_context, validate_context


def context():
    return {
        "team_identity_map": {"team_a": "red", "team_b": "blue"},
        "teams": {"red": {"name": "Reds"}, "blue": {"name": "Blues"}},
        "segments": [
            {"start_frame": 0, "end_frame": 9, "match_phase": "non_game"},
            {"start_frame": 10, "end_frame": 79, "set_id": "set_1"},
            {"start_frame": 80, "end_frame": 159, "set_id": "set_2",
             "team_identity_map": {"team_a": "blue", "team_b": "red"}},
        ],
        "initial_score": {"red": 3, "blue": 4},
        "roster": [{"player_id": "P1", "team_id": None, "confirmed": False}],
        "rotations": [],
    }


def attempt(start, end, server, rally_id):
    return {
        "start_frame": start, "end_frame": end, "rally_id": rally_id,
        "lifecycle": {"serve_frame": start, "serving_team": server,
                      "receiving_team": "team_b" if server == "team_a" else "team_a",
                      "serve_confirmed": True, "status": "playable_rally"},
        "events": [{"frame": start, "type": "serve", "team": server,
                    "label_source": "model", "verified": False}],
    }


def saved_stats():
    # Deliberately unsorted source windows: the projection must preserve indexes.
    fsm = [attempt(110, 125, "team_a", "late"), attempt(0, 8, "team_a", "warmup"),
           attempt(50, 65, "team_a", "middle"), attempt(10, 25, "team_b", "first"),
           attempt(90, 105, "team_b", "new_set")]
    return {
        "video": "synthetic", "fps": 30, "total_frames": 160,
        "all_events": [
            {"frame": 3, "type": "serve", "team": "team_a", "label_source": "human", "verified": True},
            {"frame": 20, "type": "attack", "team": "team_b", "source": "saved_spotter"},
            {"frame": 95, "type": "attack", "team": "team_b", "source": "saved_spotter"},
            {"frame": 115, "type": "attack", "team": None, "source": "saved_spotter"},
        ],
        "service_attempts_fsm": fsm,
        "service_attempts": [attempt(10, 25, None, "spotter_first"), attempt(0, 8, None, "spotter_warmup")],
        "rallies": [attempt(10, 25, None, "spotter_first")],
        "rallies_fsm": [deepcopy(fsm[0]), deepcopy(fsm[3])],
        "match_analytics": {"score": {"red": 99}, "schema_version": "original"},
        "rally_lifecycle": {"service_attempts": 2, "playable_rallies": 1},
    }


def test_valid_context_keeps_nullable_roster_and_normalizes_a_copy():
    config = context()
    config["teams"]["red"]["name"] = "  Reds  "
    config["segments"].reverse()
    config["rotations"] = [
        {"start_frame": 10, "end_frame": 79, "team_id": "red", "rotation": 1},
        {"start_frame": 10, "end_frame": 79, "team_id": "blue", "rotation": 6},
    ]
    before = deepcopy(config)
    validated = validate_context(config, 160, ["P1"])
    assert validated["schema"] == "viewer_match_context_v1"
    assert validated["teams"]["red"]["name"] == "Reds"
    assert [segment["start_frame"] for segment in validated["segments"]] == [0, 10, 80]
    assert validated["roster"] == [{"player_id": "P1", "team_id": None, "confirmed": False}]
    assert len(validated["rotations"]) == 2  # Opposing teams may overlap in time.
    assert config == before


def test_projection_excludes_non_game_and_preserves_source_arrays_and_human_fields():
    stats, config = saved_stats(), context()
    before_stats, before_config = deepcopy(stats), deepcopy(config)
    result = project_context(stats, config)
    assert stats == before_stats and config == before_config
    assert len(result["all_events"]) == len(stats["all_events"])
    assert [event["frame"] for event in result["all_events"]] == [event["frame"] for event in stats["all_events"]]
    assert result["all_events"][0]["excluded_from_match_stats"] is True
    assert result["all_events"][0]["label_source"] == "human"
    assert result["all_events"][0]["verified"] is True
    assert result["all_events"][1]["excluded_from_match_stats"] is False
    for key in ("service_attempts_fsm", "service_attempts", "rallies", "rallies_fsm"):
        assert len(result[key]) == len(stats[key])
        assert [item["start_frame"] for item in result[key]] == [item["start_frame"] for item in stats[key]]
        assert [item["rally_id"] for item in result[key]] == [item["rally_id"] for item in stats[key]]
    assert result["service_attempts_fsm"][1]["excluded_from_match_stats"] is True
    assert result["service_attempts"][1]["excluded_from_match_stats"] is True
    assert "warmup" not in {point["rally_id"] for point in result["match_analytics"]["points"]}
    assert result["match_analytics_original"] == stats["match_analytics"]
    assert result["rally_lifecycle"] == {"service_attempts": 1, "playable_rallies": 1}


def test_set_side_switches_map_stable_teams_without_filling_unknown_event_sides():
    result = project_context(saved_stats(), context())
    assert result["all_events"][1]["stable_team_id"] == "blue"
    assert result["all_events"][2]["stable_team_id"] == "red"
    assert result["all_events"][3]["stable_team_id"] is None
    assert result["all_events"][3]["team"] is None
    points = {point["rally_id"]: point for point in result["match_analytics"]["points"]}
    assert points["first"]["winner_team_id"] == "red"
    assert points["new_set"]["winner_team_id"] == "blue"
    assert points["middle"]["status"] == "unresolved"
    assert points["middle"]["evidence"] == "set_boundary"


def test_starting_score_is_consistent_and_applies_to_first_set_only():
    result = project_context(saved_stats(), context())
    analytics = result["match_analytics"]
    points = {point["rally_id"]: point for point in analytics["points"]}
    assert analytics["score"] == {"red": 4, "blue": 5}
    assert points["first"]["score_after"] == {"red": 4, "blue": 4}
    assert points["first"]["set_score_after"] == {"red": 4, "blue": 4}
    assert analytics["score_by_set"]["set_1"] == {"red": 4, "blue": 4}
    assert points["new_set"]["set_score_after"] == {"red": 0, "blue": 1}
    assert analytics["score_by_set"]["set_2"] == {"red": 0, "blue": 1}


def test_starting_score_survives_a_first_set_with_no_confirmed_points():
    stats = saved_stats()
    stats["service_attempts_fsm"] = [attempt(10, 25, "team_b", "unfinished")]
    result = project_context(stats, context())
    analytics = result["match_analytics"]
    assert analytics["confirmed_points"] == 0
    assert analytics["score"] == {"red": 3, "blue": 4}
    assert analytics["points"][0]["set_score_after"] == {"red": 3, "blue": 4}
    assert analytics["score_by_set"]["set_1"] == {"red": 3, "blue": 4}


def invalid_context(case):
    config = context()
    if case == "overlap":
        config["segments"][1]["start_frame"] = 9
    elif case == "range_bool":
        config["segments"][0]["start_frame"] = False
    elif case == "range_float":
        config["segments"][0]["start_frame"] = 0.0
    elif case == "outside_video":
        config["segments"][-1]["end_frame"] = 160
    elif case == "same_side_team":
        config["segments"][1]["team_identity_map"] = {"team_a": "blue", "team_b": "blue"}
    elif case == "score_bool":
        config["initial_score"]["red"] = True
    elif case == "negative_score":
        config["initial_score"]["red"] = -1
    elif case == "unknown_player":
        config["roster"][0]["player_id"] = "P99"
    elif case == "duplicate_player":
        config["roster"].append(deepcopy(config["roster"][0]))
    elif case == "roster_bool":
        config["roster"][0]["confirmed"] = 1
    elif case == "unknown_confirmed_team":
        config["roster"][0]["confirmed"] = True
    elif case == "rotation_overlap":
        config["rotations"] = [{"start_frame": 10, "end_frame": 79, "team_id": "blue", "rotation": 1},
                               {"start_frame": 79, "end_frame": 99, "team_id": "blue", "rotation": 2}]
    elif case == "rotation_bool":
        config["rotations"] = [{"start_frame": 10, "end_frame": 79, "team_id": "blue", "rotation": True}]
    elif case == "rotation_outside_video":
        config["rotations"] = [{"start_frame": 10, "end_frame": 200, "team_id": "blue", "rotation": 2}]
    elif case == "non_finite_metadata":
        config["metadata"] = {"invalid": float("nan")}
    return config


@pytest.mark.parametrize("case", ["overlap", "range_bool", "range_float", "outside_video", "same_side_team",
                                 "score_bool", "negative_score", "unknown_player", "duplicate_player", "roster_bool",
                                 "unknown_confirmed_team", "rotation_overlap", "rotation_bool", "rotation_outside_video",
                                 "non_finite_metadata"])
def test_invalid_context_rejects_whole_payload_without_mutation(case):
    config = invalid_context(case)
    before = deepcopy(config)
    with pytest.raises(ValueError):
        validate_context(config, 160, ["P1"])
    if case != "non_finite_metadata":
        assert config == before
