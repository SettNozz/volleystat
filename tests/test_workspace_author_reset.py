"""Author-only reset returns to the proposal across the workspace transition."""

import json

import pytest

from src.tracking.player_contacts import build_player_contacts
from src.viewer_app.match_workspace import author_review_path


@pytest.fixture
def reset_client(tmp_path):
    import viewer

    events = [{"frame": 10, "type": "attack", "team": "team_b"}]
    poses = [{"id": index, "player_id": player, "team": "b", "bb": [x - 20, 0, x + 20, 100],
              "kp": [[x, 50, .99] for _ in range(17)]}
             for index, (player, x) in enumerate((("P1", 100), ("P2", 300)), start=1)]
    frames = [{"f": 10, "ball": [100, 50], "poses": poses}]
    identity = {"side": "b", "gallery_keys": {"P1": [], "P2": []}}
    report = build_player_contacts(events, frames, 30, identity)
    assert report["contacts"][0]["player_id"] == "P1"
    stats = {"video": "author-reset", "fps": 30, "total_frames": 120,
             "all_events": events, "player_contacts": report}
    annotations = {"video": "author-reset", "fps": 30, "total_frames": 120,
                   "frames": frames, "player_contacts": report, "player_identity": identity}
    stats_path = tmp_path / "author-reset_stats.json"
    ann_path = tmp_path / "author-reset_annotations.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    ann_path.write_text(json.dumps(annotations), encoding="utf-8")
    original = stats_path.read_bytes(), ann_path.read_bytes()
    job_id = "workspace-author-reset"
    viewer.jobs[job_id] = {"job_id": job_id, "video_stem": "author-reset", "status": "done",
                           "stats_path": str(stats_path), "annotations_path": str(ann_path),
                           "video_path": str(tmp_path / "author-reset.mp4")}
    viewer._player_contact_inputs.cache_clear()
    try:
        with viewer.app.test_client() as client:
            yield client, job_id, stats_path, ann_path, original
    finally:
        viewer.jobs.pop(job_id, None)
        viewer._player_contact_inputs.cache_clear()


def test_reset_to_proposal_after_setup_clears_legacy_author_review_and_preserves_event_axes(reset_client):
    client, job, stats_path, ann_path, original = reset_client
    state = client.get(f"/api/match_context/{job}").json
    legacy = client.post(f"/api/player_contacts/{job}", json={
        "source": state["source"], "revision": state["revision"], "event_index": 0, "player_id": "P2"})
    assert legacy.status_code == 200, legacy.data
    assert legacy.json["contacts"][0]["player_id"] == "P2"
    legacy_bytes = author_review_path(stats_path).read_bytes()

    state = client.get(f"/api/match_context/{job}").json
    saved = client.post(f"/api/match_context/{job}", json={
        "source": state["source"], "revision": state["revision"], "config": state["config"]})
    assert saved.status_code == 200, saved.data
    reviewed = client.post(f"/api/developer_review/{job}", json={
        "source": saved.json["source"], "revision": saved.json["revision"],
        "edit": {"action": "update", "event_index": 0,
                 "changes": {"type": "dig", "detection_verified": True, "type_verified": True}}})
    assert reviewed.status_code == 200, reviewed.data
    reset = client.post(f"/api/player_contacts/{job}", json={
        "source": reviewed.json["source"], "revision": reviewed.json["revision"], "event_index": 0, "reset": True})
    assert reset.status_code == 200, reset.data
    contact = reset.json["contacts"][0]
    assert contact["player_id"] == "P1" and contact["status"] == "assigned"
    assert contact["label_source"] != "human" and contact["verified"] is False
    assert contact["type"] == "dig"
    assert contact["review_axes"]["detection"]["verified"] is True
    assert contact["review_axes"]["type"]["verified"] is True
    assert "author" not in contact["review_axes"]
    assert author_review_path(stats_path).read_bytes() == legacy_bytes
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == original
