from src.tracking.match_analytics import (
    annotate_match_phases,
    annotate_score_timeline,
    enrich_match_analytics,
    filter_non_game_records,
)


def _attempt(frame, team, status="playable_rally", events=()):
    return {
        "start_frame": frame,
        "end_frame": frame + 20,
        "lifecycle": {
            "status": status,
            "is_playable": status == "playable_rally",
            "serve_confirmed": status != "serve_fault",
            "serve_frame": frame,
            "serving_team": team,
            "receiving_team": "team_b" if team == "team_a" else "team_a",
        },
        "events": list(events),
    }


def test_next_confirmed_server_wins_previous_point():
    attempts = [
        _attempt(
            10,
            "team_b",
            events=(
                {"frame": 10, "type": "serve", "team": "team_b"},
                {"frame": 20, "type": "reception", "team": "team_a"},
                {"frame": 25, "type": "pass", "team": "team_a"},
            ),
        ),
        _attempt(100, "team_b"),
        _attempt(200, "team_a"),
    ]

    result = enrich_match_analytics(attempts)

    assert result["score"] == {"opponent_team": 1, "observed_team": 1}
    assert result["points"][0]["winner_team_id"] == "observed_team"
    assert result["points"][1]["winner_team_id"] == "opponent_team"
    assert result["points"][2]["status"] == "unresolved"
    assert attempts[0]["rally_id"] == "rally_0001"
    assert attempts[0]["events"][1]["context_role"] == "serve_reception"
    assert attempts[0]["events"][1]["touch_index"] == 1
    assert attempts[0]["events"][2]["context_role"] == "set"
    assert attempts[0]["events"][2]["touch_index"] == 2


def test_explicit_final_serve_fault_awards_receiver():
    attempts = [_attempt(10, "team_b", status="serve_fault")]

    result = enrich_match_analytics(attempts)

    assert result["score"]["opponent_team"] == 1
    assert result["points"][0]["evidence"] == "serve_fault"


def test_manual_non_game_segment_overrides_fsm_phase():
    frames = [
        {"f": 1, "state": "serve_ready"},
        {"f": 2, "state": "serving"},
        {"f": 3, "state": "rally_active"},
    ]

    annotate_match_phases(
        frames,
        [{"start_frame": 1, "end_frame": 2, "match_phase": "non_game"}],
    )

    assert [frame["match_phase"] for frame in frames] == [
        "non_game",
        "non_game",
        "rally_live",
    ]


def test_set_boundary_does_not_award_previous_point_and_side_map_can_flip():
    attempts = [
        _attempt(10, "team_b"),
        _attempt(100, "team_a"),
    ]
    segments = [
        {
            "start_frame": 0,
            "end_frame": 50,
            "set_id": "set_1",
            "team_identity_map": {
                "team_a": "opponent_team",
                "team_b": "observed_team",
            },
        },
        {
            "start_frame": 51,
            "end_frame": 150,
            "set_id": "set_2",
            "team_identity_map": {
                "team_a": "observed_team",
                "team_b": "opponent_team",
            },
        },
    ]

    result = enrich_match_analytics(attempts, match_segments=segments)

    assert attempts[0]["serving_team_id"] == "observed_team"
    assert attempts[1]["serving_team_id"] == "observed_team"
    assert result["points"][0]["status"] == "unresolved"
    assert result["points"][0]["evidence"] == "set_boundary"


def test_non_game_service_attempt_is_excluded_from_match_stats():
    warmup = _attempt(10, "team_b")
    game = _attempt(100, "team_a")

    kept = filter_non_game_records(
        [warmup, game],
        [{"start_frame": 0, "end_frame": 50, "match_phase": "non_game"}],
        frame_field="start_frame",
    )

    assert kept == [game]
    assert warmup["excluded_from_match_stats"] is True


def test_score_timeline_changes_on_next_serve_evidence_frame():
    attempts = [_attempt(10, "team_b"), _attempt(100, "team_b")]
    analytics = enrich_match_analytics(attempts)
    frames = [{"f": 99}, {"f": 100}, {"f": 101}]

    annotate_score_timeline(frames, analytics)

    assert frames[0]["score"]["observed_team"] == 0
    assert frames[1]["score"]["observed_team"] == 1
    assert frames[2]["last_confirmed_point_id"] == "point_0001"
