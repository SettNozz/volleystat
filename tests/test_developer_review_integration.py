"""Cross-view regression tests for source-bound developer review integration."""

import json

import pytest

from src.tracking.player_contacts import build_player_contacts


@pytest.fixture
def client_workspace(tmp_path):
    import viewer
    events = [{"frame": 30, "type": "set", "source": "dense_contact_spotter"},
              {"frame": 60, "type": "attack", "source": "dense_contact_spotter"},
              {"frame": 90, "type": "serve", "source": "dense_contact_spotter"}]
    poses = [{"id": index, "player_id": player, "team": "b", "bb": [x - 20, 0, x + 20, 100],
              "kp": [[x, 50, .99] for _ in range(17)]}
             for index, (player, x) in enumerate((("P1", 100), ("P2", 300)), start=1)]
    frames = [{"f": frame, "ball": [300 if frame == 90 else 100, 50], "poses": poses} for frame in (30, 60, 90)]
    report = build_player_contacts(events, frames, 30)
    attempts = [{"rally_id": "r1", "start_frame": 10, "end_frame": 80, "lifecycle": {"serve_frame": 10}},
                {"rally_id": "r2", "start_frame": 90, "end_frame": 130, "lifecycle": {"serve_frame": 90}}]
    stats = {"video": "review", "fps": 30, "total_frames": 300, "all_events": events, "player_contacts": report,
             "service_attempts": attempts, "service_attempts_fsm": attempts,
             "match_analytics": {"points": [{"rally_id": "r1", "status": "confirmed", "winner_court_side": "team_b"}]}}
    ann = {"video": "review", "fps": 30, "total_frames": 300, "frames": frames, "player_contacts": report}
    stats_path, ann_path = tmp_path / "review_stats.json", tmp_path / "review_annotations.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    viewer.jobs["developer-integration"] = {"job_id": "developer-integration", "video_stem": "review", "status": "done",
        "stats_path": str(stats_path), "annotations_path": str(ann_path), "video_path": str(tmp_path / "review.mp4")}
    with viewer.app.test_client() as client:
        yield client, stats_path, ann_path
    viewer.jobs.pop("developer-integration", None)
    viewer._player_contact_inputs.cache_clear()


def workspace(client):
    result = client.get("/api/developer_review/developer-integration")
    assert result.status_code == 200, result.data
    return result.json


def post_edit(client, edit):
    state = workspace(client)
    response = client.post("/api/developer_review/developer-integration", json={
        "source": state["source"], "revision": state["revision"], "edit": edit})
    assert response.status_code == 200, response.data
    return response.json


def test_add_remove_and_author_edit_keep_indices_and_source_pair_stable(client_workspace):
    client, stats_path, ann_path = client_workspace
    before = stats_path.read_bytes(), ann_path.read_bytes()
    post_edit(client, {"action": "add", "review_id": "added", "frame": 150, "type": "attack", "player_id": "P2"})
    state = post_edit(client, {"action": "remove", "event_index": 0})
    assert [c["event_index"] for c in state["report"]["contacts"]] == [1, 2, 3]
    result = client.post("/api/player_contacts/developer-integration", json={"source": state["source"],
        "revision": state["revision"], "event_index": 3, "player_id": "P1"})
    assert result.status_code == 200, result.data
    by_index = {c["event_index"]: c for c in result.json["contacts"]}
    assert by_index[3]["frame"] == 150 and by_index[3]["player_id"] == "P1"
    assert by_index[1]["frame"] == 60
    summary = client.get("/api/statistics/developer-integration").json
    assert summary["touch_counts"]["P1"]["attack"] == 2
    assert summary["touch_counts"]["P1"].get("dig_pass", 0) == 0
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


def test_legacy_author_review_is_restored_by_full_reset_after_timing_change(client_workspace):
    client, _, _ = client_workspace
    state = workspace(client)
    response = client.post("/api/player_contacts/developer-integration", json={
        "source": state["source"], "event_index": 1, "player_id": "P2"})
    assert response.status_code == 200
    shifted = post_edit(client, {"action": "update", "event_index": 1, "changes": {"frame": 66}})
    contact = next(c for c in shifted["report"]["contacts"] if c["event_index"] == 1)
    assert contact["player_id"] is None and not contact["verified"]
    restored = post_edit(client, {"action": "reset", "event_index": 1})
    contact = next(c for c in restored["report"]["contacts"] if c["event_index"] == 1)
    assert contact["frame"] == 60 and contact["player_id"] == "P2" and contact["verified"]


@pytest.mark.parametrize("endpoint,body", [
    ("/api/player_contacts/developer-integration", {"event_index": 1, "player_id": "P2"}),
    ("/api/statistics/developer-integration", {"attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}),
])
def test_source_only_requests_cannot_override_existing_workspace_revision(client_workspace, endpoint, body):
    client, _, _ = client_workspace
    state = post_edit(client, {"action": "dismiss", "item_ref": "event:1"})
    result = client.post(endpoint, json=dict(body, source=state["source"]))
    assert result.status_code == 409


def test_author_only_edit_cannot_silently_restore_removed_detection(client_workspace):
    client, _, _ = client_workspace
    removed = post_edit(client, {"action": "remove", "event_index": 1})
    result = client.post("/api/player_contacts/developer-integration", json={
        "source": removed["source"], "revision": removed["revision"], "event_index": 1, "player_id": "P2"})
    assert result.status_code in {200, 400}
    contacts = workspace(client)["report"]["contacts"]
    assert 1 not in {c["event_index"] for c in contacts}
