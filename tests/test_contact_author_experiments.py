from copy import deepcopy
import json

import pytest

import viewer
from src.tracking.player_contacts import build_player_contacts, source_fingerprint
from src.tracking.contact_author_experiments import experiment_path


@pytest.fixture
def experiment(tmp_path):
    pose = {"player_id": "P1", "team": "b", "id": 7, "bb": [0, 0, 100, 100], "kp": [[50, 50, .9]]*17}
    annotations = {"video": "clip", "fps": 30, "frames": [{"f": 30, "ball": [50, 50], "poses": [pose]}]}
    events = [{"frame": 30, "type": "pass"}]
    stats = {"video": "clip", "fps": 30, "all_events": events}
    ann, stat = tmp_path/"clip_annotations.json", tmp_path/"clip_stats.json"
    ann.write_text(json.dumps(annotations), encoding="utf-8"); stat.write_text(json.dumps(stats), encoding="utf-8")
    report = build_player_contacts(events, annotations["frames"], 30)
    other = deepcopy(report)
    other["contacts"][0].update(player_id=None, status="other_side", reason="test_other")
    bundle = {"schema": "player_contact_experiment_v1", "source": source_fingerprint(stat.read_bytes(), ann.read_bytes()),
              "selected_method": "learned", "reports": {"learned": report, "control": other},
              "methods": {"learned": {"label": "Learned side"}, "control": {"label": "Control"}},
              "selection": {"person_accuracy": None}}
    path = experiment_path(stat)
    path.write_text(json.dumps(bundle), encoding="utf-8")
    viewer.jobs["author-experiment-test"] = {"status": "done", "stats_path": str(stat), "annotations_path": str(ann)}
    with viewer.app.test_client() as client:
        yield client, stat, ann, path, bundle
    viewer.jobs.pop("author-experiment-test", None)
    viewer._player_contact_inputs.cache_clear()


def test_default_method_control_and_human_reviews_share_event_binding(experiment):
    client, stat, ann, path, bundle = experiment
    url = "/api/player_contacts/author-experiment-test"
    source_before = stat.read_bytes(), ann.read_bytes()
    report = client.get(url).get_json()
    assert report["method_id"] == "learned" and report["contacts"][0]["player_id"] == "P1"
    control = client.get(url+"?method=control").get_json()
    assert control["contacts"][0]["status"] == "other_side"
    response = client.post(url, json={"source": report["source"], "method": "control", "event_index": 0, "player_id": "P1"})
    assert response.status_code == 200
    assert response.get_json()["contacts"][0]["label_source"] == "human"
    assert client.get(url).get_json()["contacts"][0]["verified"] is True
    assert client.post(url, json={"source": report["source"], "method": "control", "event_index": 0, "reset": True}).status_code == 200
    assert client.get(url+"?method=control").get_json()["contacts"][0]["status"] == "other_side"
    assert (stat.read_bytes(), ann.read_bytes()) == source_before
    assert path.read_text(encoding="utf-8") == json.dumps(bundle)


def test_source_changes_reject_bundle_and_invalid_methods_do_not_write(experiment):
    client, stat, ann, path, _ = experiment
    url = "/api/player_contacts/author-experiment-test"
    assert client.get(url+"?method=missing").status_code == 400
    assert not list(stat.parent.glob("*_reviews.jsonl"))
    ann.write_text(ann.read_text(encoding="utf-8")+" ", encoding="utf-8")
    assert client.get(url).status_code == 409


@pytest.mark.parametrize("change", ["frame", "verification", "index", "roster"])
def test_invalid_event_or_fake_ground_truth_is_rejected(experiment, change):
    client, _, _, path, bundle = experiment
    contact = bundle["reports"]["learned"]["contacts"][0]
    if change == "frame": contact["frame"] = 31
    if change == "verification": contact["verified"] = True
    if change == "index": contact["event_index"] = 99
    if change == "roster": contact["player_id"] = "P999"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    assert client.get("/api/player_contacts/author-experiment-test").status_code == 400
