from copy import deepcopy

import pytest

from src.viewer_app.developer_review import apply_contact_edits, build_review_queue, validate_contact_edit


def sample():
    stats = {"fps": 30, "total_frames": 300, "all_events": [
        {"frame": 30, "type": "set", "source": "dense_contact_spotter", "confidence": .8},
        {"frame": 60, "type": "attack", "source": "dense_contact_spotter", "confidence": .9},
    ], "service_attempts_fsm": [{"rally_id": "r1", "start_frame": 10, "end_frame": 90}]}
    report = {"roster_ids": ["P1", "P2"], "side": "b", "contacts": [
        {"event_index": 0, "frame": 30, "time": 1, "type": "set", "status": "assigned", "player_id": "P1",
         "label_source": "derived", "verified": False, "candidates": [{"frame": 30}], "contact_frame": 31},
        {"event_index": 1, "frame": 60, "time": 2, "type": "attack", "status": "unresolved", "player_id": None,
         "label_source": "derived", "verified": False, "candidates": []},
    ]}
    return stats, report


def edit(stats, report, body):
    return validate_contact_edit(body, stats, report, 300)


def test_author_only_review_never_verifies_detection_or_type_and_preserves_source():
    stats, report = sample()
    before = deepcopy((stats, report))
    correction = edit(stats, report, {"action": "update", "event_index": 1, "changes": {"player_id": "P2"}})
    effective, authors = apply_contact_edits(stats, report, [correction])
    assert (stats, report) == before
    event = effective["all_events"][1]
    assert event["source"] == "dense_contact_spotter" and event["confidence"] == .9
    assert not event.get("verified") and event.get("label_source") != "human"
    assert event["review_axes"] == {"author": {"label_source": "human", "verified": True}}
    assert authors["contacts"][1]["player_id"] == "P2" and authors["contacts"][1]["verified"]


def test_type_and_detection_confirmation_are_independent():
    stats, report = sample()
    typed = edit(stats, report, {"action": "update", "event_index": 0, "changes": {"type": "dig"}})
    detection = edit(stats, report, {"action": "update", "event_index": 0, "changes": {"detection_verified": True}})
    partial, _ = apply_contact_edits(stats, report, [typed])
    assert not partial["all_events"][0]["verified"]
    full, _ = apply_contact_edits(stats, report, [typed, detection])
    assert full["all_events"][0]["verified"] and full["all_events"][0]["label_source"] == "human"
    assert full["all_events"][0]["core_type"] == "pass"


def test_human_source_detection_is_not_lost_when_its_type_is_corrected():
    stats, report = sample()
    stats["all_events"][0].update(label_source="human", verified=True)
    correction = edit(stats, report, {"action": "update", "event_index": 0, "changes": {"type": "dig"}})
    effective, _ = apply_contact_edits(stats, report, [correction])
    assert effective["all_events"][0]["verified"]
    assert effective["all_events"][0]["review_original"]["type"] == "set"


def test_timing_change_drops_old_author_evidence_unless_explicitly_assigned():
    stats, report = sample()
    report["contacts"][0].update(contact_time=1.0, contact_frame=30, proposed_player_id="P1")
    shifted = edit(stats, report, {"action": "update", "event_index": 0, "changes": {"frame": 33}})
    effective, authors = apply_contact_edits(stats, report, [shifted])
    assert effective["all_events"][0]["frame"] == 33
    assert authors["contacts"][0]["status"] == "unresolved" and authors["contacts"][0]["candidates"] == []
    assert "contact_frame" not in authors["contacts"][0]
    assert "contact_time" not in authors["contacts"][0]
    shifted["changes"]["player_id"] = "P2"
    _, assigned = apply_contact_edits(stats, report, [shifted])
    assert assigned["contacts"][0]["player_id"] == "P2" and assigned["contacts"][0]["label_source"] == "human"
    assert assigned["contacts"][0]["candidates"] == []
    assert "contact_time" not in assigned["contacts"][0] and "contact_frame" not in assigned["contacts"][0]
    reset = {"action": "reset", "event_index": 0, "axes": ["author"]}
    _, reset_authors = apply_contact_edits(stats, report, [shifted, reset])
    assert "contact_time" not in reset_authors["contacts"][0]


def test_added_contacts_are_idempotent_and_reset_never_recycles_their_index():
    stats, report = sample()
    added = edit(stats, report, {"action": "add", "review_id": "human-1", "frame": 90, "type": "block", "player_id": "P1"})
    effective, authors = apply_contact_edits(stats, report, [added, added])
    assert len(effective["all_events"]) == 3
    assert effective["all_events"][2]["source"] == "human_review_added"
    assert effective["all_events"][2]["verified"] and authors["contacts"][-1]["event_index"] == 2
    reset = edit(effective, authors, {"action": "reset", "event_index": 2})
    after, after_authors = apply_contact_edits(stats, report, [added, reset])
    assert len(after["all_events"]) == 3 and after["all_events"][2]["review_excluded"]
    assert [c["event_index"] for c in after_authors["contacts"]] == [0, 1]
    next_added = edit(after, after_authors, {"action": "add", "review_id": "human-2", "frame": 91, "type": "dig"})
    final, _ = apply_contact_edits(stats, report, [added, reset, next_added])
    assert final["all_events"][3]["review_id"] == "human-2"


def test_added_uncertain_contact_can_leave_detection_unverified():
    stats, report = sample()
    added = edit(stats, report, {"action": "add", "review_id": "uncertain", "changes": {
        "frame": 90, "type": "dig", "detection_verified": False, "type_verified": False}})
    effective, authors = apply_contact_edits(stats, report, [added])
    assert not effective["all_events"][2]["verified"]
    assert authors["contacts"][-1]["player_id"] is None


def test_remove_and_reset_restore_saved_event_and_current_legacy_author_baseline():
    stats, report = sample()
    removed = edit(stats, report, {"action": "remove", "event_index": 0})
    reset = edit(stats, report, {"action": "reset", "event_index": 0})
    partial, authors = apply_contact_edits(stats, report, [removed])
    assert partial["all_events"][0]["review_excluded"] and authors["contacts"][0]["event_index"] == 1
    restored, restored_authors = apply_contact_edits(stats, report, [removed, reset])
    assert restored["all_events"] == stats["all_events"]
    assert restored_authors["contacts"] == report["contacts"]


def test_author_reset_keeps_type_and_timing_corrections_but_cannot_reuse_old_crop():
    stats, report = sample()
    change = edit(stats, report, {"action": "update", "event_index": 0, "changes": {"type": "dig", "frame": 35, "player_id": "P2"}})
    reset = edit(stats, report, {"action": "reset", "event_index": 0, "axes": ["author"]})
    effective, authors = apply_contact_edits(stats, report, [change, reset])
    assert effective["all_events"][0]["frame"] == 35 and effective["all_events"][0]["type"] == "dig"
    assert "author" not in effective["all_events"][0]["review_axes"]
    assert authors["contacts"][0]["status"] == "unresolved" and authors["contacts"][0]["candidates"] == []


def test_removed_contact_needs_explicit_detection_confirmation_to_return():
    stats, report = sample()
    removed = edit(stats, report, {"action": "remove", "event_index": 0})
    typed = edit(stats, report, {"action": "update", "event_index": 0, "changes": {"type": "dig", "player_id": "P2"}})
    partial, authors = apply_contact_edits(stats, report, [removed, typed])
    assert partial["all_events"][0]["review_excluded"] and not partial["all_events"][0]["verified"]
    assert 0 not in {c["event_index"] for c in authors["contacts"]}
    restored = edit(partial, authors, {"action": "update", "event_index": 0, "changes": {"detection_verified": True}})
    effective, restored_authors = apply_contact_edits(stats, report, [removed, typed, restored])
    assert not effective["all_events"][0]["review_excluded"]
    assert effective["all_events"][0]["review_axes"]["detection"]["value"] is True
    assert effective["all_events"][0]["verified"]
    assert restored_authors["contacts"][0]["event_index"] == 0


def test_queue_prioritizes_point_conflicts_and_keeps_dismissal_separate_from_labels():
    stats, report = sample()
    stats["all_events"][1]["confidence"] = .2
    reconciliation = {"contacts": [{"event_index": 1, "issues": ["serve_anchor_mismatch"], "outcome": "unknown"}],
                      "points": [{"attempt_ref": "service_attempts_fsm:0", "start_frame": 10, "status": "conflict", "issues": ["ambiguous_point_link"]}]}
    queue = build_review_queue(stats, report, reconciliation)
    assert queue["items"][0]["ref"] == "event:1"
    assert "low_event_confidence" in queue["items"][0]["reasons"]
    dismissal = edit(stats, report, {"action": "dismiss", "item_ref": "event:1"})
    effective, authors = apply_contact_edits(stats, report, [dismissal])
    after = build_review_queue(effective, authors, reconciliation)
    assert next(item for item in after["items"] if item["ref"] == "event:1")["status"] == "dismissed"
    assert not effective["all_events"][1].get("verified")


@pytest.mark.parametrize("body", [
    {"action": []}, {"action": "unknown"}, {"action": "update", "event_index": True, "changes": {"frame": 1}},
    {"action": "update", "event_index": 9, "changes": {"type": "attack"}},
    {"action": "add", "frame": 300, "type": "attack"}, {"action": "add", "frame": -1, "type": "attack"},
    {"action": "add", "frame": 1.5, "type": "attack"}, {"action": "add", "frame": 1, "type": "unknown"},
    {"action": "update", "event_index": 0, "changes": {"side": {}}},
    {"action": "update", "event_index": 0, "changes": {"player_id": "P99"}},
    {"action": "update", "event_index": 0, "changes": {"player_id": "P1", "other_side": True}},
    {"action": "update", "event_index": 0, "changes": {"detection_verified": "yes"}},
    {"action": "update", "event_index": 0, "changes": {"ball": [1, 2]}},
    {"action": "dismiss", "item_ref": "point:service_attempts_fsm:99"},
    {"action": "reset", "event_index": 0, "axes": ["type"]},
])
def test_invalid_edits_fail_before_projection(body):
    stats, report = sample()
    with pytest.raises(ValueError):
        edit(stats, report, body)
