from copy import deepcopy

import pytest

from src.tracking.player_match_statistics import build_player_match_statistics
from src.tracking.player_statistics_excel import player_statistic_value, STATISTICS_VIEWS
from src.tracking.statistics_reconciliation import statistic_evidence


def analysis():
    stats = {
        "all_events": [{"frame": 12, "type": "reception"}, {"frame": 20, "type": "attack"}],
        "service_attempts": [{"rally_id": "rally_1", "start_frame": 10, "end_frame": 24,
                              "lifecycle": {"serve_frame": 10}}],
        "service_attempts_fsm": [{"rally_id": "score_9", "start_frame": 10, "end_frame": 24,
                                  "lifecycle": {"serve_frame": 10, "reception_frame": 12,
                                                "receiving_team": "team_b", "reception_quality": "controlled"}}],
        "match_analytics": {"points": [{"rally_id": "score_9", "status": "confirmed", "winner_court_side": "team_b"}]},
    }
    report = {"side": "b", "roster_ids": ["P1"], "contacts": [
        {"event_index": i, "status": "assigned", "player_id": "P1", "label_source": "derived", "verified": False}
        for i in range(2)
    ]}
    return stats, report


def test_source_namespaces_and_evidence_share_the_same_counts_without_mutation():
    stats, report = analysis()
    report["contacts"][0].update(label_source="human", verified=True)
    before = deepcopy((stats, report))
    summary = build_player_match_statistics(stats, report)
    attack = statistic_evidence(summary, "P1", "attack", "point")
    assert [c["event_index"] for c in attack["contacts"]] == [1]
    assert attack["contacts"][0]["attempt_ref"] == "service_attempts_fsm:0"
    assert attack["contacts"][0]["event_attempt_refs"] == ["service_attempts:0"]
    assert summary["coverage"]["P1"]["reception"] == {"total": 1, "human": 1, "classified": 1}
    assert not summary["reconciliation"]["contacts"][0]["event_verified"]
    assert (stats, report) == before
    for kind, (_, columns) in STATISTICS_VIEWS.items():
        for key, _ in columns:
            value = player_statistic_value(kind, summary["players"]["P1"][kind], key)
            evidence = statistic_evidence(summary, "P1", kind, key)
            if key == "rate":
                assert value == (evidence["numerator"] / evidence["total"] if evidence["total"] else None)
            else:
                assert evidence["total"] == value


@pytest.mark.parametrize("problem", ["overlap", "anchor", "duplicate_point", "set", "reused_id", "same_terminal_frame"])
def test_ambiguous_evidence_abstains_instead_of_crediting_a_point(problem):
    stats, report = analysis()
    if problem == "overlap":
        stats["service_attempts_fsm"].append(deepcopy(stats["service_attempts_fsm"][0]))
    elif problem == "anchor":
        stats["service_attempts"][0]["lifecycle"]["serve_frame"] = 11
    elif problem == "duplicate_point":
        stats["match_analytics"]["points"].append(deepcopy(stats["match_analytics"]["points"][0]))
    elif problem == "set":
        stats["service_attempts_fsm"][0]["set_id"] = "set_2"
        stats["match_analytics"]["points"][0]["set_id"] = "set_1"
    elif problem == "reused_id":
        stats["service_attempts_fsm"].append({"rally_id": "score_9", "start_frame": 100, "end_frame": 120})
    else:
        stats["all_events"].append({"frame": 20, "type": "block"})
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P1"]["attack"] == {"total": 1, "unknown": 1}
    assert summary["reconciliation"]["conflicts"] > 0


def test_empty_fsm_namespace_cannot_borrow_a_same_number_spotter_point():
    stats, report = analysis()
    stats["service_attempts_fsm"] = []
    stats["match_analytics"]["points"][0]["rally_id"] = "rally_1"
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P1"]["attack"]["unknown"] == 1
    assert summary["reconciliation"]["points"] == []


def test_rejected_confirming_serve_cannot_still_award_an_automatic_point():
    stats, report = analysis()
    stats["all_events"].append({"frame": 30, "type": "serve", "review_excluded": True})
    stats["match_analytics"]["points"][0].update(evidence="next_confirmed_serve", evidence_frame=30)
    summary = build_player_match_statistics(stats, report)
    point = summary["reconciliation"]["points"][0]
    assert point["status"] == "conflict" and point["issues"] == ["point_confirmation_rejected"]
    assert summary["players"]["P1"]["attack"]["unknown"] == 1
    confirmed = build_player_match_statistics(stats, report, [{"attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}])
    assert confirmed["players"]["P1"]["attack"]["point"] == 1


def test_legacy_rallies_remain_available_when_service_attempts_are_empty():
    stats, report = analysis()
    stats["rallies"] = stats.pop("service_attempts_fsm")
    stats["service_attempts"] = []
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P1"]["attack"]["point"] == 1
    assert summary["reconciliation"]["score_attempt_source"] == "rallies"


def test_point_winner_cannot_silently_override_a_contradictory_attack_fault():
    stats, report = analysis()
    stats["service_attempts_fsm"][0]["lifecycle"]["end_reason"] = "attack_into_net"
    summary = build_player_match_statistics(stats, report, [{"attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}])
    contact = statistic_evidence(summary, "P1", "attack", "unknown")["contacts"][0]
    assert contact["issues"] == ["attack_fault_point_mismatch"]
    assert contact["point_reviewed"] and not contact["outcome_verified"]


def test_ace_and_opponent_point_are_a_visible_conflict():
    stats = {"all_events": [{"frame": 10, "type": "serve"}],
             "service_attempts": [{"rally_id": "r1", "start_frame": 10, "end_frame": 20,
                "lifecycle": {"serve_frame": 10, "serving_team": "team_b", "serve_outcome": "ace"}}],
             "match_analytics": {"points": [{"rally_id": "r1", "status": "confirmed", "winner_court_side": "team_a"}]}}
    report = {"side": "b", "roster_ids": ["P1"], "contacts": [{"event_index": 0, "player_id": "P1", "status": "assigned"}]}
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P1"]["serve"] == {"total": 1, "unknown": 1}
    assert summary["reconciliation"]["conflicts"] == 1


def test_point_confirmation_and_reset_do_not_verify_an_attack():
    stats, report = analysis()
    stats["match_analytics"]["points"][0].update(status="unresolved", winner_court_side=None)
    review = {"attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}
    summary = build_player_match_statistics(stats, report, [review])
    contact = statistic_evidence(summary, "P1", "attack", "point")["contacts"][0]
    assert contact["point_reviewed"] and not contact["outcome_verified"]
    assert contact["author_source"] == "automatic" and not contact["event_verified"]
    reset = dict(review, reset=True)
    restored = build_player_match_statistics(stats, report, [review, reset])
    assert restored["players"]["P1"]["attack"]["unknown"] == 1
    assert not restored["reconciliation"]["points"][0]["reviewed"]
    with pytest.raises(ValueError, match="unknown attempt"):
        build_player_match_statistics(stats, report, [dict(review, attempt_ref="service_attempts:0")])


def test_percentage_denominator_excludes_unknown_and_unassigned_contacts():
    stats = {"all_events": [{"frame": n, "type": "serve"} for n in (10, 20, 30, 40)],
             "service_attempts": [{"lifecycle": {"serve_frame": n, "serve_outcome": outcome}}
                                  for n, outcome in [(10, "fault"), (20, "in_play")]]}
    report = {"side": "b", "roster_ids": ["P1"], "contacts": [
        {"event_index": i, "status": "assigned" if i < 3 else "unresolved", "player_id": "P1" if i < 3 else None}
        for i in range(4)]}
    summary = build_player_match_statistics(stats, report)
    evidence = statistic_evidence(summary, "P1", "serve", "rate")
    assert evidence["total"] == 2 and evidence["numerator"] == 1
    assert summary["players"]["P1"]["serve"]["unknown"] == 1
    assert summary["unassigned"]["serve"] == 1
    unknown = statistic_evidence(summary, "unassigned", "serve", "total")
    assert unknown["unassigned_authors"] == 1 and unknown["automatic_authors"] == 0


def test_duplicate_fault_report_does_not_double_count_or_override_human_author():
    stats = {"all_events": [{"frame": 30, "type": "serve_fault"}]}
    report = {"side": "b", "roster_ids": ["P1"], "contacts": [
        {"event_index": 0, "status": "assigned", "player_id": "P1", "label_source": "human", "verified": True},
        {"event_index": 0, "status": "unresolved", "player_id": None}]}
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P1"]["serve"] == {"total": 1, "error": 1}
    assert summary["coverage"]["P1"]["serve"]["human"] == 1


@pytest.mark.parametrize("kind,metric,player", [("fake", "total", "P1"), ("attack", "rate", "P1"), ("serve", "net", "P1"), ("attack", "total", "P99")])
def test_unknown_filters_are_rejected(kind, metric, player):
    stats, report = analysis()
    with pytest.raises(ValueError):
        statistic_evidence(build_player_match_statistics(stats, report), player, kind, metric)
