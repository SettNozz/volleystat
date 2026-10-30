from copy import deepcopy
import json

import pytest

from scripts.apply_player_contacts import rebuild
from src.tracking.player_contacts import apply_reviews, build_player_contacts


def pose(x, player_id="P1", team="b", raw_id=1):
    return {"id": raw_id, "player_id": player_id, "team": team,
            "bb": [x-20, 0, x+20, 100],
            "kp": [[x, 50, .99] for _ in range(17)]}


def frame(f=30, poses=None, ball=None, **kwargs):
    return {"f": f, "ball": [100, 50] if ball is None else ball,
            "poses": poses if poses is not None else [pose(100)], **kwargs}


def event(f=30, kind="attack"):
    return {"frame": f, "type": kind, "confidence": .9, "source": "dense_contact_spotter"}


def report(frames=None, events=None, identity=None):
    return build_player_contacts(events or [event()], frames or [frame()], 30, identity)


def test_assigns_roster_not_raw_id_preserves_inputs_and_type():
    frames = [frame(poses=[pose(100, "P7", raw_id=123)])]
    events = [event(kind=k) for k in ("reception", "dig", "pass", "set", "attack", "block", "serve")]
    original = deepcopy((events, frames))
    result = report(frames, events)
    assert (events, frames) == original
    assert result["roster_ids"] == ["P7"]
    assert all(c["player_id"] == "P7" and not c["verified"] for c in result["contacts"])
    assert result["players"][0]["counts"] == {
        "reception": 1, "dig_pass": 3, "attack": 1, "block": 1, "serve": 1, "other": 0}
    assert result["players"][0]["raw_counts"]["dig"] == 1


@pytest.mark.parametrize("competitor", [pose(100, "P2", raw_id=2), pose(100, None, "a", 2), pose(100, None, "b", 2)])
def test_ambiguous_opponent_or_unknown_is_not_ignored(competitor):
    result = report([frame(poses=[pose(105), competitor])])
    assert result["contacts"][0]["reason"] == "ambiguous_players"
    assert result["contacts"][0]["player_id"] is None


def test_clear_opponent_and_unknown_identity_do_not_acquire_near_id():
    for team, reason in (("a", "other_side_candidate"), ("b", "unknown_identity")):
        result = report([frame(poses=[pose(100, None, team), pose(200, "P2", raw_id=2)])])
        assert result["contacts"][0]["reason"] == reason
        assert result["players"][0]["total"] == 0


@pytest.mark.parametrize("missing", [{"ball": None}, {"ball_out_of_frame": True}])
def test_no_ball_interpolation_through_contact(missing):
    anchor = frame()
    anchor.update(missing)
    result = report([frame(29), anchor, frame(31)])
    assert result["contacts"][0]["reason"] == "no_ball_or_pose"


def test_window_stops_at_gap_and_new_track():
    anchor = frame(poses=[])
    assert report([anchor, frame(32)])["contacts"][0]["player_id"] is None
    assert report([anchor, frame(31, ball_new=True)])["contacts"][0]["player_id"] is None
    assert report([anchor, frame(31)])["contacts"][0]["player_id"] == "P1"


def test_scale_invariance_and_full_body_contact():
    # Contact on a forearm segment, not at a foot/body centre.
    p = pose(200)
    p["kp"] = [[0, 0, 0] for _ in range(17)]
    p["kp"][7], p["kp"][9] = [80, 50, .9], [120, 50, .9]
    original = frame(poses=[p])
    scaled = deepcopy(original)
    scaled["ball"] = [v * 3 for v in scaled["ball"]]
    for p in scaled["poses"]:
        p["bb"] = [v * 3 for v in p["bb"]]
        p["kp"] = [[x*3, y*3, c] for x, y, c in p["kp"]]
    assert report([original])["contacts"][0]["player_id"] == "P1"
    assert report([scaled])["contacts"][0]["player_id"] == "P1"


def test_roster_includes_inactive_seed_people_and_substitutes():
    identity = {"side": "a", "gallery_keys": {"P1": [], "P7": []}}
    result = report([frame(poses=[pose(100, "P7", "a")])], identity=identity)
    assert result["roster_ids"] == ["P1", "P7"]
    assert result["players"][0]["total"] == 0
    assert result["players"][1]["total"] == 1


def test_review_changes_attribution_only_and_reset_restores_proposal():
    original = report(identity={"gallery_keys": {"P1": [], "P2": []}})
    result = apply_reviews(original, [{"event_index": 0, "player_id": "P2"}])
    assert original["contacts"][0]["player_id"] == "P1"
    assert result["contacts"][0]["verified"] is True
    assert result["contacts"][0]["event_confidence"] == .9
    assert result["players"][1]["counts"]["attack"] == 1
    reset = apply_reviews(original, [{"event_index": 0, "player_id": "P2"}, {"event_index": 0, "reset": True}])
    assert reset == original
    with pytest.raises(ValueError, match="roster"):
        apply_reviews(original, [{"event_index": 0, "player_id": "P99"}])


def test_export_is_fresh_model_free_and_preserves_all_source_fields(tmp_path):
    ann_path = tmp_path / "clip_annotations.json"
    stats_path = tmp_path / "clip_stats.json"
    annotations = {"video": "clip.webm", "fps": 30, "frames": [frame()], "human": {"verified": True}}
    stats = {"video": "clip.webm", "fps": 30, "all_events": [event()], "rallies": [{"x": 1}]}
    for path, payload in ((ann_path, annotations), (stats_path, stats)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    before = [ann_path.read_bytes(), stats_path.read_bytes()]
    output = tmp_path / "new"
    rebuild(ann_path, output)
    assert before == [ann_path.read_bytes(), stats_path.read_bytes()]
    for path, original in ((ann_path, annotations), (stats_path, stats)):
        exported = json.loads((output / path.name).read_text(encoding="utf-8"))
        assert exported.pop("player_contacts")["schema"] == "player_contacts_v1"
        assert exported == original
    assert not list(output.glob("*.mp4"))
    with pytest.raises(FileExistsError):
        rebuild(ann_path, output)


@pytest.fixture
def client_result(tmp_path):
    import viewer
    ann_path, stats_path = tmp_path / "clip_annotations.json", tmp_path / "clip_stats.json"
    ann_path.write_text(json.dumps({"video": "clip", "fps": 30, "frames": [frame()]}), encoding="utf-8")
    stats_path.write_text(json.dumps({"video": "clip", "fps": 30, "all_events": [event()]}), encoding="utf-8")
    viewer.jobs["contact-test"] = {"status": "done", "annotations_path": str(ann_path), "stats_path": str(stats_path)}
    with viewer.app.test_client() as client:
        yield client, ann_path, stats_path
    viewer.jobs.pop("contact-test", None)
    viewer._player_contact_inputs.cache_clear()


def test_viewer_review_persists_without_rewriting_analysis(client_result):
    client, ann, stats = client_result
    before = ann.read_bytes(), stats.read_bytes()
    url = "/api/player_contacts/contact-test"
    report = client.get(url).get_json()
    response = client.post(url, json={"source": report["source"], "event_index": 0, "other_side": True})
    assert response.status_code == 200
    assert response.get_json()["totals"] == {"other_side": 1}
    assert client.get(url).get_json()["contacts"][0]["label_source"] == "human"
    assert (ann.read_bytes(), stats.read_bytes()) == before
    assert len(list(stats.parent.glob("*_reviews.jsonl"))) == 1
    for body in ({"event_index": 0, "player_id": "P99"}, {"event_index": 9}, {"event_index": False}):
        assert client.post(url, json={"source": report["source"], **body}).status_code == 400
    assert len(next(stats.parent.glob("*_reviews.jsonl")).read_text().splitlines()) == 1


def test_viewer_rejects_stale_source_and_does_not_reuse_reviews(client_result):
    client, ann, stats = client_result
    url = "/api/player_contacts/contact-test"
    report = client.get(url).get_json()
    body = {"source": report["source"], "event_index": 0, "player_id": "P1"}
    stats.write_text(stats.read_text() + " ", encoding="utf-8")
    assert client.post(url, json=body).status_code == 409
    new = client.get(url).get_json()
    assert client.post(url, json={**body, "source": new["source"]}).status_code == 200
    ann.write_text(ann.read_text() + " ", encoding="utf-8")
    assert client.get(url).status_code == 409


def test_unknown_job_is_404(client_result):
    client, _, _ = client_result
    assert client.get("/api/player_contacts/absent").status_code == 404


def test_result_jobs_do_not_mix_two_hypotheses_of_same_video(tmp_path, monkeypatch):
    import viewer
    monkeypatch.setattr(viewer, "PROJECT_ROOT", tmp_path)
    ids = []
    with viewer.app.test_client() as client:
        for directory in ("first", "second"):
            result = tmp_path / directory
            result.mkdir()
            (result / "clip_stats.json").write_text("{}", encoding="utf-8")
            response = client.get("/load", query_string={"video": "clip", "results_dir": directory})
            ids.append(response.location.rsplit("/", 1)[-1])
    assert ids[0] != ids[1]
    assert viewer.jobs[ids[0]]["stats_path"] != viewer.jobs[ids[1]]["stats_path"]
    for job_id in ids:
        viewer.jobs.pop(job_id, None)
