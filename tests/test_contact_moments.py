"""Repeated detector segments share a moment only with observed actor support."""
from copy import deepcopy

import pytest

from src.tracking.contact_moments import contact_moment_groups
from src.tracking.player_match_statistics import build_player_match_statistics
from src.viewer_app.player_highlights import player_highlights


def inputs(kind="attack", fps=30):
    events = [{"frame": 100, "type": kind, "confidence": .99},
              {"frame": 103, "type": kind, "confidence": .8}]
    actor = {"frame": 98, "pose_index": 2, "raw_track_id": 42,
             "distance_body_heights": .1, "bbox": [100, 0, 140, 100]}
    contacts = [{"event_index": i, "frame": e["frame"], "type": kind, "contact_frame": 98,
                 "status": "assigned", "player_id": "P3", "label_source": "derived",
                 "candidates": [deepcopy(actor)]} for i, e in enumerate(events)]
    return {"fps": fps, "all_events": events}, {"side": "b", "roster_ids": ["P3"], "contacts": contacts}


def test_highlights_and_statistics_share_one_moment_and_retain_event_indices():
    stats, report = inputs()
    before = deepcopy((stats, report))
    moments = player_highlights(report, stats["all_events"], stats["fps"], "P3")
    assert len(moments) == 1 and moments[0]["source_event_indices"] == [0, 1]
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P3"]["attack"]["total"] == 1
    assert summary["reconciliation"]["contacts"][0]["source_event_indices"] == [0, 1]
    assert (stats, report) == before


@pytest.mark.parametrize("fps", [25, 30, 60])
def test_stride_shifted_support_for_one_serve_is_counted_once(fps):
    stats, report = inputs("serve", fps)
    stats["all_events"][1]["frame"] = 100 + round(.23 * fps)
    report["contacts"][1]["candidates"][0]["frame"] += round(.067 * fps)
    assert len(contact_moment_groups(report["contacts"], stats["all_events"], fps)) == 1


@pytest.mark.parametrize("difference", ["human", "human_event", "player", "type", "track", "pose", "missing_ball",
                                        "no_geometry", "time", "support_time", "teleport", "intervening"])
def test_close_contacts_with_distinct_or_uncertain_evidence_survive(difference):
    stats, report = inputs()
    contact = report["contacts"][1]
    actor = contact["candidates"][0]
    if difference == "human": contact["label_source"] = "human"
    elif difference == "human_event": stats["all_events"][1]["label_source"] = "human"
    elif difference == "player": contact["player_id"] = "P4"
    elif difference == "type": stats["all_events"][1]["type"] = "block"
    elif difference == "track": actor.update(frame=99, raw_track_id=43)
    elif difference == "pose": actor.update(pose_index=3)
    elif difference == "missing_ball": actor["ball_out_of_frame"] = True
    elif difference == "no_geometry": actor.pop("distance_body_heights")
    elif difference == "time": stats["all_events"][1]["frame"] += 30
    elif difference == "support_time": actor["frame"] += 4
    elif difference == "teleport": actor.update(frame=99, bbox=[1000, 0, 1040, 100])
    elif difference == "intervening": stats["all_events"].append({"frame": 102, "type": "block"})
    assert len(contact_moment_groups(report["contacts"], stats["all_events"], 30)) == 2


def test_duplicate_windows_do_not_chain_across_multiple_contacts():
    stats, report = inputs()
    stats["all_events"][1]["frame"] = 108
    stats["all_events"].append({"frame": 116, "type": "attack"})
    report["contacts"].append(dict(deepcopy(report["contacts"][1]), event_index=2))
    assert [indices for _, indices in contact_moment_groups(report["contacts"], stats["all_events"], 30)] == [[0, 1], [2]]


def test_higher_confidence_representative_keeps_terminal_attack_outcome():
    stats, report = inputs()
    stats["all_events"][1]["confidence"] = 1.
    stats["rallies"] = [{"rally_id": 1, "start_frame": 90, "end_frame": 105}]
    stats["match_analytics"] = {"points": [{"rally_id": 1, "status": "confirmed", "winner_court_side": "b"}]}
    summary = build_player_match_statistics(stats, report)
    assert summary["players"]["P3"]["attack"] == {"total": 1, "point": 1}
    assert summary["reconciliation"]["contacts"][0]["event_index"] == 1
