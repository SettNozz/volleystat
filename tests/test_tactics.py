from copy import deepcopy

import pytest

from src.viewer_app.tactics import build_tactics, summarize_rotation_rows


def analysis():
    events = [{"frame": frame, "type": kind, "team": team} for frame, kind, team in
              [(0, "serve", "team_a"), (10, "reception", "team_b"),
               (20, "setting_pass", "team_b"), (30, "attack", "team_b")]]
    attempt = {"rally_id": "r1", "start_frame": 0, "end_frame": 99, "set_id": "set_1",
               "lifecycle": {"serve_frame": 0, "reception_frame": 10, "serving_team": "team_a",
                             "receiving_team": "team_b", "reception_quality": "controlled"}}
    stats = {"all_events": events, "service_attempts": [attempt], "match_analytics": {"points": [
        {"rally_id": "r1", "status": "confirmed", "winner_court_side": "team_b"}]}}
    report = {"roster_ids": ["P1", "P2"], "side": "b", "contacts": [
        {"event_index": i, "frame": event["frame"], "type": event["type"], "status": "assigned",
         "player_id": "P1" if i == 2 else "P2", "label_source": "derived", "verified": False}
        for i, event in enumerate(events) if i > 0]}
    annotations = {"net_floor_left": [0, 0], "net_floor_right": [300, 0],
                   "frames": [{"f": frame, "ball": [75, 80], "ball_source": "detected"} for frame in range(100)]}
    context = {"team_identity_map": {"team_a": "away", "team_b": "home"}, "rotations": [
        {"start_frame": 0, "end_frame": 99, "team_id": "home", "rotation": 1},
        {"start_frame": 0, "end_frame": 99, "team_id": "away", "rotation": 4}]}
    return stats, report, annotations, context


def test_tactics_preserves_source_indexes_provenance_and_inputs():
    args = analysis()
    before = deepcopy(args)
    result = build_tactics(*args)
    pair, = result["pairs"]
    assert (pair["set_event_index"], pair["attack_event_index"]) == (2, 3)
    assert (pair["setter"], pair["team_id"], pair["zone"], pair["rotation"]) == ("P1", "home", "left", 1)
    assert pair["reception_quality"] == "controlled"
    assert pair["score_attempt_ref"] == "service_attempts:0"
    assert pair["setter_source"] == "automatic" and pair["ball_provenance"] == "observed"
    assert args == before
    home = next(row for row in result["rotations"] if row["team_id"] == "home")
    assert home["side_out"] == {"attempts": 1, "classified": 1, "wins": 1, "unknown": 0, "rate": 1.0}
    away = next(row for row in result["rotations"] if row["team_id"] == "away")
    assert away["break_point"]["rate"] == 0


@pytest.mark.parametrize("failure", ["missing_frame", "off_camera", "no_ball", "restart", "overlap", "intervening", "opponent", "side_conflict", "non_game"])
def test_pairs_never_cross_gaps_other_contacts_or_ambiguous_rallies(failure):
    stats, report, annotations, context = analysis()
    if failure == "missing_frame":
        annotations["frames"] = [row for row in annotations["frames"] if row["f"] != 25]
    elif failure == "off_camera":
        annotations["frames"][25]["ball_out_of_frame"] = True
    elif failure == "no_ball":
        annotations["frames"][25]["ball"] = None
    elif failure == "restart":
        annotations["frames"][25]["ball_new"] = True
    elif failure == "overlap":
        stats["service_attempts"].append(deepcopy(stats["service_attempts"][0]))
    elif failure == "intervening":
        stats["all_events"].append({"frame": 25, "type": "block", "team": "team_a"})
    elif failure == "opponent":
        stats["all_events"][3]["team"] = "team_a"
    elif failure == "side_conflict":
        stats["all_events"][2]["court_side"] = "team_a"
    else:
        context["segments"] = [{"start_frame": 25, "end_frame": 26, "match_phase": "non_game"}]
    assert build_tactics(stats, report, annotations, context)["pairs"] == []


def test_derived_ball_stays_derived_and_missing_geometry_leaves_zone_unknown():
    stats, report, annotations, context = analysis()
    annotations["frames"][25]["ball_source"] = "contact_aware_reconstruction"
    annotations.pop("net_floor_left")
    result = build_tactics(stats, report, annotations, context)
    pair, = result["pairs"]
    assert pair["ball_provenance"] == "derived" and pair["derived_ball_frames"] == 1
    assert pair["zone"] == "unknown"


def test_unknown_setter_team_and_rotation_are_not_filled_from_roster_or_geometry():
    stats, report, annotations, _ = analysis()
    report["contacts"][1].update(status="unresolved", proposed_player_id="P1", player_id=None)
    result = build_tactics(stats, report, annotations, {})
    pair, = result["pairs"]
    assert pair["setter"] is None and pair["team_id"] is None and pair["rotation"] is None
    assert result["rotation_rallies"] == []
    assert result["summary"]["unidentified_setters"] == 1


def test_stale_contact_author_cannot_name_the_setter():
    stats, report, annotations, context = analysis()
    report["contacts"][1]["frame"] = 19
    assert build_tactics(stats, report, annotations, context)["pairs"][0]["setter"] is None


def test_confirmed_roster_team_conflict_abstains_without_filling_unknown_team():
    stats, report, annotations, context = analysis()
    context["roster"] = [{"player_id": "P1", "team_id": "away", "confirmed": True}]
    pair, = build_tactics(stats, report, annotations, context)["pairs"]
    assert pair["setter"] is None and pair["setter_reason"] == "confirmed_roster_team_conflict"
    context["roster"][0]["confirmed"] = False
    assert build_tactics(stats, report, annotations, context)["pairs"][0]["setter"] == "P1"


def test_context_side_switch_keeps_stable_team_and_manual_rotation():
    stats, report, annotations, context = analysis()
    context["segments"] = [{"start_frame": 0, "end_frame": 99, "set_id": "set_2",
                            "team_identity_map": {"team_a": "home", "team_b": "away"}}]
    result = build_tactics(stats, report, annotations, context)
    pair, = result["pairs"]
    assert pair["team_id"] == "away" and pair["set_id"] == "set_2" and pair["rotation"] == 4
    away = next(row for row in result["rotations"] if row["team_id"] == "away")
    assert away["side_out"]["rate"] == 1


def test_no_reception_quality_is_borrowed_from_later_rally_possessions():
    stats, report, annotations, context = analysis()
    stats["all_events"][1]["type"] = "dig"
    pair, = build_tactics(stats, report, annotations, context)["pairs"]
    assert pair["reception_quality"] == "unknown"


def test_independent_score_rally_ids_never_match_by_ordinal():
    stats, report, annotations, context = analysis()
    stats["service_attempts_fsm"] = [deepcopy(stats["service_attempts"][0])]
    stats["service_attempts_fsm"][0]["rally_id"] = "different_score_id"
    result = build_tactics(stats, report, annotations, context)
    assert result["pairs"][0]["attack_outcome"] == "unknown"
    assert all(row["won"] is None for row in result["rotation_rallies"])


def test_excluding_a_score_window_does_not_renumber_review_targets():
    stats, report, annotations, context = analysis()
    stats["service_attempts"][0]["excluded_from_match_stats"] = True
    second = deepcopy(stats["service_attempts"][0])
    second.update(rally_id="r2", start_frame=50, end_frame=99, excluded_from_match_stats=False)
    second["lifecycle"]["serve_frame"] = 50
    stats["service_attempts"].append(second)
    stats["match_analytics"]["points"].append({"rally_id": "r2", "status": "unresolved"})
    result = build_tactics(stats, report, annotations, context, [{"attempt_ref": "service_attempts:1", "winner_side": "a"}])
    assert result["rotation_rallies"]
    assert {row["attempt_ref"] for row in result["rotation_rallies"]} == {"service_attempts:1"}
    assert all(row["point_verified"] for row in result["rotation_rallies"])
    assert next(row for row in result["rotation_rallies"] if row["team_id"] == "away")["won"] is True


def test_reviewed_point_is_reflected_without_verifying_setter_or_attack():
    stats, report, annotations, context = analysis()
    stats["match_analytics"]["points"][0]["status"] = "unresolved"
    result = build_tactics(stats, report, annotations, context, [{"attempt_ref": "service_attempts:0", "winner_side": "b"}])
    assert next(row for row in result["rotation_rallies"] if row["team_id"] == "home")["point_verified"]
    assert result["pairs"][0]["setter_source"] == "automatic"


def test_unknown_point_and_service_role_do_not_dilute_classified_denominators():
    rows = [{"team_id": "home", "set_id": "s1", "rotation": 1, "role": role, "won": won}
            for role, won in [("receive", True), ("receive", False), ("receive", None),
                              ("serve", True), ("unknown", True)]]
    group, = summarize_rotation_rows(rows)
    assert group["side_out"] == {"attempts": 3, "classified": 2, "wins": 1, "unknown": 1, "rate": .5}
    assert group["break_point"]["rate"] == 1 and group["unknown_roles"] == 1


def test_overlapping_score_windows_keep_point_evidence_but_abstain_from_rates():
    stats, report, annotations, context = analysis()
    second = deepcopy(stats["service_attempts"][0])
    second["rally_id"] = "r2"
    second["start_frame"] = second["lifecycle"]["serve_frame"] = 50
    second["end_frame"] = 120
    stats["service_attempts"].append(second)
    stats["match_analytics"]["points"].append({"rally_id": "r2", "status": "confirmed", "winner_court_side": "team_b"})
    result = build_tactics(stats, report, annotations, context)
    home, = [group for group in result["rotations"] if group["team_id"] == "home"]
    assert home["side_out"]["classified"] == 0 and home["side_out"]["unknown"] == 2
    assert all(row["evidence_issue"] == "overlapping_score_windows" for row in result["rotation_rallies"])
    assert all(row["winner_team_id"] == "home" for row in result["rotation_rallies"])


def test_missing_score_window_end_cannot_enter_a_rotation_rate():
    stats, report, annotations, context = analysis()
    stats["service_attempts"][0].pop("end_frame")
    result = build_tactics(stats, report, annotations, context)
    assert result["rotation_rallies"]
    assert all(row["won"] is None and row["evidence_issue"] == "invalid_score_window" for row in result["rotation_rallies"])


def test_touching_rally_boundary_does_not_make_two_distinct_windows_overlap():
    stats, report, annotations, context = analysis()
    stats["service_attempts"][0]["end_frame"] = 50
    second = deepcopy(stats["service_attempts"][0])
    second["rally_id"] = "r2"
    second["start_frame"] = second["lifecycle"]["serve_frame"] = 50
    second["end_frame"] = 99
    stats["service_attempts"].append(second)
    stats["match_analytics"]["points"].append({"rally_id": "r2", "status": "confirmed", "winner_court_side": "team_a"})
    rows = build_tactics(stats, report, annotations, context)["rotation_rallies"]
    assert all(row["evidence_issue"] is None and row["won"] is not None for row in rows)


def test_overlapping_manual_rotation_labels_do_not_force_a_rotation():
    stats, report, annotations, context = analysis()
    context["rotations"].append({"start_frame": 0, "end_frame": 50, "team_id": "home", "rotation": 2})
    assert build_tactics(stats, report, annotations, context)["pairs"][0]["rotation"] is None


def test_review_excluded_and_non_game_contacts_are_not_tactical_evidence():
    for flag in ("review_excluded", "excluded_from_match_stats"):
        stats, report, annotations, context = analysis()
        stats["all_events"][2][flag] = True
        assert build_tactics(stats, report, annotations, context)["pairs"] == []
