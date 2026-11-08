from src.tracking.player_match_statistics import build_player_match_statistics


def test_player_statistics_uses_score_attempts_and_current_author_assignments():
    stats = {
        "all_events": [
            {"frame": 12, "type": "reception"},
            {"frame": 16, "type": "attack"},
            {"frame": 21, "type": "attack"},
            {"frame": 34, "type": "attack"},
            {"frame": 46, "type": "attack"},
            {"frame": 51, "type": "attack"},
        ],
        "service_attempts": [{"rally_id": "independent", "start_frame": 0, "end_frame": 99}],
        "service_attempts_fsm": [
            {"rally_id": "win", "start_frame": 10, "end_frame": 22, "lifecycle": {
                "reception_frame": 12, "receiving_team": "team_b", "reception_quality": "controlled",
            }},
            {"rally_id": "loss", "start_frame": 31, "end_frame": 37, "lifecycle": {
                "end_reason": "attack_into_net",
            }},
            {"rally_id": "ambiguous", "start_frame": 44, "end_frame": 49, "lifecycle": {
                "end_reason": "ball_out",
            }},
        ],
        "match_analytics": {"points": [
            {"rally_id": "win", "status": "confirmed", "winner_court_side": "team_b"},
            {"rally_id": "loss", "status": "confirmed", "winner_court_side": "team_a"},
            {"rally_id": "ambiguous", "status": "confirmed", "winner_court_side": "team_a"},
        ]},
    }
    report = {
        "side": "b", "roster_ids": ["P1", "P2", "P3"],
        "contacts": [
            {"event_index": 0, "player_id": "P1", "status": "assigned"},
            {"event_index": 1, "player_id": "P1", "status": "assigned"},
            {"event_index": 2, "player_id": "P2", "status": "assigned", "verified": True},
            {"event_index": 3, "player_id": "P1", "status": "assigned"},
            {"event_index": 4, "player_id": "P1", "status": "assigned"},
            {"event_index": 5, "status": "unresolved"},
        ],
    }
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P1"]["reception"] == {"total": 1, "controlled": 1}
    assert summary["players"]["P1"]["attack"] == {"total": 3, "continued": 1, "net": 1, "lost": 1}
    assert summary["players"]["P2"]["attack"] == {"total": 1, "point": 1}
    assert summary["players"]["P3"]["attack"] == {}
    assert summary["unassigned"] == {"attack": 1}
