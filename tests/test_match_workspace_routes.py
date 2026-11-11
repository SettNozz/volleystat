"""Cross-workspace routes share reviewed projections and source-bound revisions."""

from copy import deepcopy
import json

import pytest

from src.tracking.player_contacts import build_player_contacts
from src.viewer_app.match_workspace import author_review_path, workspace_path
from src.viewer_app.statistics_reviews import review_path


JOB = "workspace-routes-test"


@pytest.fixture
def workspace_client(tmp_path):
    import viewer
    events = [
        {"frame": 10, "type": "serve", "team": "team_a", "source": "dense_contact_spotter"},
        {"frame": 20, "type": "reception", "team": "team_b", "source": "dense_contact_spotter"},
        {"frame": 30, "type": "set", "team": "team_b", "source": "dense_contact_spotter"},
        {"frame": 40, "type": "attack", "team": "team_b", "source": "dense_contact_spotter"},
        {"frame": 80, "type": "serve", "team": "team_b", "source": "dense_contact_spotter"},
    ]
    frames = []
    for f in range(120):
        player, team = ("P2", "b") if 37 <= f <= 43 else ("P1", "a" if f == 10 else "b")
        frames.append({"f": f, "ball": [100, 50], "ev": [deepcopy(event) for event in events if event["frame"] == f],
            "poses": [{"id": 2 if player == "P2" else 1,
            "player_id": player, "team": team, "bb": [80, 0, 120, 100],
            "kp": [[100, 50, .99] for _ in range(17)]}]})
    identity = {"side": "b", "gallery_keys": {"P1": [], "P2": []}}
    report = build_player_contacts(events, frames, 30, identity)
    first = {"rally_id": "fsm_first", "start_frame": 10, "end_frame": 60,
             "lifecycle": {"serve_frame": 10, "serving_team": "team_a", "receiving_team": "team_b",
                "serve_confirmed": True, "status": "playable_rally", "reception_frame": 20,
                "reception_quality": "controlled", "serve_outcome": "in_play"}}
    second = {"rally_id": "fsm_second", "start_frame": 80, "end_frame": 100,
              "lifecycle": {"serve_frame": 80, "serving_team": "team_b", "receiving_team": "team_a",
                 "serve_confirmed": True, "status": "playable_rally", "serve_outcome": "in_play"}}
    stats = {"video": "workspace", "fps": 30, "total_frames": 120, "all_events": events,
             "ball_detection_rate": 1.0, "processing_time_sec": 1.0,
             "player_contacts": report, "service_attempts_fsm": [first, second],
             "service_attempts": [dict(deepcopy(first), rally_id="spotter_first"), dict(deepcopy(second), rally_id="spotter_second")],
             "rallies": [dict(deepcopy(first), rally_id="spotter_first"), dict(deepcopy(second), rally_id="spotter_second")],
             "match_analytics": {"points": [
                 {"rally_id": "fsm_first", "status": "confirmed", "winner_court_side": "team_b"},
                 {"rally_id": "fsm_second", "status": "unresolved"}]}}
    annotations = {"video": "workspace", "fps": 30, "total_frames": 120, "frames": frames,
                   "player_identity": identity, "player_contacts": report}
    stats_path, ann_path = tmp_path / "workspace_stats.json", tmp_path / "workspace_annotations.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    ann_path.write_text(json.dumps(annotations), encoding="utf-8")
    before = stats_path.read_bytes(), ann_path.read_bytes()
    viewer.jobs[JOB] = {"job_id": JOB, "video_stem": "workspace", "status": "done",
        "stats_path": str(stats_path), "annotations_path": str(ann_path), "video_path": str(tmp_path / "workspace.mp4")}
    viewer._player_contact_inputs.cache_clear()
    with viewer.app.test_client() as client:
        yield client, stats_path, ann_path, before
    viewer.jobs.pop(JOB, None)
    viewer._player_contact_inputs.cache_clear()


def get_state(client):
    response = client.get(f"/api/match_context/{JOB}")
    assert response.status_code == 200, response.data
    return response.json


def save_context(client, state, config=None):
    return client.post(f"/api/match_context/{JOB}", json={"source": state["source"],
        "revision": state["revision"], "config": config if config is not None else state["config"]})


def save_edit(client, edit, state=None):
    state = state or get_state(client)
    return client.post(f"/api/developer_review/{JOB}", json={"source": state["source"],
        "revision": state["revision"], "edit": edit})


def moments(client, player):
    response = client.get(f"/player-highlights/{JOB}?player={player}")
    assert response.status_code == 200, response.data
    return json.loads(response.data.split(b"const MOMENTS = ", 1)[1].split(b";\n", 1)[0])


def test_pages_and_apis_share_the_same_source_revision_and_tactical_evidence(workspace_client):
    client, stats_path, ann_path, before = workspace_client
    for route in ("match-setup", "developer-review", "tactics"):
        response = client.get(f"/{route}/{JOB}")
        assert response.status_code == 200, response.data
    state = get_state(client)
    assert state["total_frames"] == 120 and state["roster_ids"] == ["P1", "P2"]
    assert state["history"] == []
    for route in ("developer_review", "tactics"):
        response = client.get(f"/api/{route}/{JOB}")
        assert response.status_code == 200, response.data
        assert response.json["source"] == state["source"]
        assert response.json["revision"] == state["revision"]
        assert response.json["config"] == state["config"]
    tactics = client.get(f"/api/tactics/{JOB}").json
    assert tactics["summary"]["accepted_pairs"] == 1
    assert [(pair["set_event_index"], pair["attack_event_index"], pair["setter"]) for pair in tactics["pairs"]] == [(2, 3, "P1")]
    assert not workspace_path(stats_path).exists()
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


def test_saved_non_game_context_changes_statistics_and_highlights_without_source_writes(workspace_client):
    client, stats_path, ann_path, before = workspace_client
    initial = get_state(client)
    summary = client.get(f"/api/statistics/{JOB}").json
    assert summary["players"]["P2"]["attack"]["total"] == 1
    config = deepcopy(initial["config"])
    config["segments"] = [{"start_frame": 0, "end_frame": 60, "match_phase": "non_game"},
                          {"start_frame": 61, "end_frame": 119, "set_id": "set_1"}]
    saved = save_context(client, initial, config)
    assert saved.status_code == 200, saved.data
    assert saved.json["revision"] != initial["revision"]
    assert saved.json["source"] == initial["source"]
    assert len(saved.json["history"]) == 1
    current = client.get(f"/api/statistics/{JOB}").json
    assert current["players"]["P2"]["attack"].get("total", 0) == 0
    assert current["players"]["P1"]["reception"].get("total", 0) == 0
    assert moments(client, "P2") == []
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


def test_contact_edits_propagate_to_statistics_and_highlights_with_stable_indexes(workspace_client):
    client, stats_path, ann_path, before = workspace_client
    shifted = save_edit(client, {"action": "update", "event_index": 3, "changes": {"frame": 45, "player_id": "P2"}})
    assert shifted.status_code == 200, shifted.data
    assert [(m["frame"], m["category"]) for m in moments(client, "P2")] == [(45, "attack")]
    added = save_edit(client, {"action": "add", "review_id": "manual-route-contact", "frame": 50,
                               "type": "reception", "side": "b", "player_id": "P2"})
    assert added.status_code == 200, added.data
    summary = client.get(f"/api/statistics/{JOB}").json
    assert summary["players"]["P2"]["attack"]["total"] == 1
    assert summary["players"]["P2"]["reception"]["total"] == 1
    assert [(m["frame"], m["event_index"]) for m in moments(client, "P2")] == [(45, 3), (50, 5)]
    removed = save_edit(client, {"action": "remove", "event_index": 3})
    assert removed.status_code == 200, removed.data
    assert client.get(f"/api/statistics/{JOB}").json["players"]["P2"]["attack"].get("total", 0) == 0
    assert [(m["frame"], m["event_index"]) for m in moments(client, "P2")] == [(50, 5)]
    restored = save_edit(client, {"action": "reset", "event_index": 3})
    assert restored.status_code == 200, restored.data
    assert [(m["frame"], m["event_index"]) for m in moments(client, "P2")] == [(40, 3), (50, 5)]
    reset_added = save_edit(client, {"action": "reset", "event_index": 5})
    assert reset_added.status_code == 200, reset_added.data
    assert [(m["frame"], m["event_index"]) for m in moments(client, "P2")] == [(40, 3)]
    assert len(workspace_path(stats_path).read_text().splitlines()) == 5
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


def boot_value(page, name):
    return json.loads(page.split(f"const {name} = ".encode(), 1)[1].split(b";\n", 1)[0])


def test_main_viewer_boots_corrected_events_without_reindexing_or_rewriting_source(workspace_client):
    client, stats_path, ann_path, before = workspace_client
    assert save_edit(client, {"action": "remove", "event_index": 1}).status_code == 200
    assert save_edit(client, {"action": "update", "event_index": 3,
        "changes": {"frame": 45, "type": "dig", "player_id": "P2"}}).status_code == 200
    assert save_edit(client, {"action": "add", "review_id": "earlier-contact", "frame": 25,
        "type": "block", "side": "b", "player_id": "P1"}).status_code == 200
    state = get_state(client)
    config = deepcopy(state["config"])
    config["segments"] = [{"start_frame": 0, "end_frame": 15, "match_phase": "non_game"},
                          {"start_frame": 16, "end_frame": 119, "set_id": "set_1"}]
    saved = save_context(client, state, config)
    assert saved.status_code == 200, saved.data
    response = client.get(f"/results/{JOB}")
    assert response.status_code == 200, response.data
    events = boot_value(response.data, "EVENTS")
    assert [event["frame"] for event in events] == [10, 20, 30, 45, 80, 25]
    assert events[0]["excluded_from_match_stats"] is True
    assert events[1]["review_excluded"] is True
    assert events[3]["type"] == "dig"
    assert events[5]["source"] == "human_review_added" and events[5]["verified"] is True
    assert boot_value(response.data, "WORKSPACE_CONTEXT")["config"] == saved.json["config"]
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


def test_annotation_projection_updates_review_report_and_context_but_preserves_observations(workspace_client):
    client, stats_path, ann_path, before = workspace_client
    original = json.loads(before[1])
    assert save_edit(client, {"action": "update", "event_index": 3,
        "changes": {"frame": 45, "player_id": "P2"}}).status_code == 200
    state = get_state(client)
    config = deepcopy(state["config"])
    config["segments"] = [{"start_frame": 0, "end_frame": 25, "match_phase": "non_game"},
                          {"start_frame": 26, "end_frame": 119, "set_id": "set_1"}]
    saved = save_context(client, state, config)
    assert saved.status_code == 200, saved.data
    response = client.get(f"/api/annotations/{JOB}")
    assert response.status_code == 200, response.data
    annotation = response.json
    assert annotation["workspace_revision"] == saved.json["revision"]
    assert annotation["match_context"] == saved.json["config"]
    assert [row["f"] for row in annotation["frames"]] == [row["f"] for row in original["frames"]]
    for projected, raw in zip(annotation["frames"], original["frames"]):
        assert projected["ball"] == raw["ball"]
        assert projected["poses"] == raw["poses"]
        assert projected["ev"] == raw["ev"]
    assert annotation["frames"][20]["excluded_from_match_stats"] is True
    assert annotation["frames"][30]["excluded_from_match_stats"] is False
    contacts = {contact["event_index"]: contact for contact in annotation["player_contacts"]["contacts"]}
    assert contacts[1]["excluded_from_match_stats"] is True
    assert contacts[3]["frame"] == 45 and contacts[3]["player_id"] == "P2"
    assert contacts[3]["label_source"] == "human" and contacts[3]["candidates"] == []
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


@pytest.mark.parametrize("changed", ["stats", "annotations"])
def test_changed_pair_rejects_saved_workspace_without_appending(workspace_client, changed):
    client, stats_path, ann_path, _ = workspace_client
    state = get_state(client)
    saved = save_context(client, state)
    assert saved.status_code == 200
    log_before = workspace_path(stats_path).read_bytes()
    path = stats_path if changed == "stats" else ann_path
    path.write_bytes(path.read_bytes() + b"\n")
    assert save_context(client, saved.json).status_code == 409
    assert save_edit(client, {"action": "remove", "event_index": 3}, saved.json).status_code == 409
    assert client.get(f"/api/tactics/{JOB}").status_code == 409
    assert workspace_path(stats_path).read_bytes() == log_before


def test_stale_workspace_revision_and_invalid_edits_never_append(workspace_client):
    client, stats_path, _, _ = workspace_client
    state = get_state(client)
    saved = save_context(client, state)
    assert saved.status_code == 200
    log_before = workspace_path(stats_path).read_bytes()
    assert save_context(client, state).status_code == 409
    assert save_edit(client, {"action": "remove", "event_index": 3}, state).status_code == 409
    current = saved.json
    invalid_context = deepcopy(current["config"])
    invalid_context["initial_score"][next(iter(invalid_context["teams"]))] = True
    assert save_context(client, current, invalid_context).status_code == 400
    assert save_edit(client, {"action": "update", "event_index": 3, "changes": {"frame": 120}}, current).status_code == 400
    assert workspace_path(stats_path).read_bytes() == log_before


@pytest.mark.parametrize("review_kind", ["author", "point"])
def test_author_and_point_reviews_invalidate_other_workspace_revisions(workspace_client, review_kind):
    client, stats_path, ann_path, before = workspace_client
    stale = get_state(client)
    if review_kind == "author":
        response = client.post(f"/api/player_contacts/{JOB}", json={"source": stale["source"],
            "revision": stale["revision"], "event_index": 3, "player_id": "P1"})
        log = author_review_path(stats_path)
    else:
        response = client.post(f"/api/statistics/{JOB}", json={"source": stale["source"],
            "revision": stale["revision"], "attempt_ref": "service_attempts_fsm:0", "winner_side": "a"})
        log = review_path(stats_path)
    assert response.status_code == 200, response.data
    log_before = log.read_bytes()
    fresh = get_state(client)
    assert fresh["revision"] != stale["revision"] and fresh["source"] == stale["source"]
    assert save_context(client, stale).status_code == 409
    assert save_edit(client, {"action": "remove", "event_index": 3}, stale).status_code == 409
    assert not workspace_path(stats_path).exists()
    assert log.read_bytes() == log_before
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


@pytest.mark.parametrize("route", ["match-setup", "developer-review", "tactics", "api/match_context", "api/developer_review", "api/tactics"])
def test_unknown_workspace_job_is_not_ready(workspace_client, route):
    client, _, _, _ = workspace_client
    assert client.get(f"/{route}/absent-workspace-job").status_code == 404
