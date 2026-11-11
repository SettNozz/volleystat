from io import BytesIO
import json
import zipfile

import pytest

from src.tracking.player_contacts import build_player_contacts
from src.viewer_app.statistics_reviews import review_path


@pytest.fixture
def evidence_client(tmp_path):
    import viewer
    events = [{"frame": 30, "type": "attack", "source": "dense_contact_spotter"}]
    frames = [{"f": 30, "ball": [100, 50], "poses": [{"id": 1, "player_id": "P1", "team": "b",
               "bb": [80, 0, 120, 100], "kp": [[100, 50, .99] for _ in range(17)]}]}]
    report = build_player_contacts(events, frames, 30)
    stats = {"video": "evidence", "fps": 30, "all_events": events, "player_contacts": report,
             "service_attempts": [{"start_frame": 15, "end_frame": 45, "lifecycle": {"serve_frame": 15}}],
             "service_attempts_fsm": [{"rally_id": "score_1", "start_frame": 15, "end_frame": 45,
                                       "lifecycle": {"serve_frame": 15}}],
             "match_analytics": {"points": [{"rally_id": "score_1", "status": "unresolved"}]}}
    ann = {"video": "evidence", "fps": 30, "frames": frames, "player_contacts": report}
    stats_path, ann_path = tmp_path / "evidence_stats.json", tmp_path / "evidence_annotations.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    viewer.jobs["evidence-test"] = {"job_id": "evidence-test", "video_stem": "evidence", "status": "done",
                                    "stats_path": str(stats_path), "annotations_path": str(ann_path),
                                    "video_path": str(tmp_path / "evidence.mp4")}
    with viewer.app.test_client() as client:
        yield client, stats_path, ann_path
    viewer.jobs.pop("evidence-test", None)
    viewer._player_contact_inputs.cache_clear()


def test_clickable_metrics_empty_categories_and_evidence_payload(evidence_client):
    client, _, _ = evidence_client
    page = client.get("/player-statistics/evidence-test?player=P1")
    assert page.status_code == 200
    assert b"metric=point" in page.data and b"Evidence coverage" in page.data
    for kind in ("attack", "serve", "reception", "dig_pass", "block"):
        evidence = client.get(f"/statistics-evidence/evidence-test?player=P1&kind={kind}&metric=total")
        assert evidence.status_code == 200, evidence.data
        assert b'"/api/video/evidence-test"' in evidence.data
        assert b'id="evidence-data"' in evidence.data
    assert client.get("/statistics-evidence/evidence-test?metric=bogus").status_code == 400
    assert client.get("/statistics-evidence/evidence-test?player=P99").status_code == 400
    assert client.get("/api/statistics/absent").status_code == 404


def test_point_review_updates_statistics_evidence_and_export_without_source_writes(evidence_client):
    client, stats_path, ann_path = evidence_client
    before = (stats_path.read_bytes(), ann_path.read_bytes())
    data = client.get("/api/statistics/evidence-test").json
    review = {"source": data["source"], "attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}
    saved = client.post("/api/statistics/evidence-test", json=review)
    assert saved.status_code == 200, saved.data
    assert saved.json["players"]["P1"]["attack"] == {"total": 1, "point": 1}
    evidence = client.get("/statistics-evidence/evidence-test?player=P1&metric=point")
    assert b'"point_reviewed": true' in evidence.data
    assert b'"outcome_verified": false' in evidence.data
    exported = client.post("/player-statistics/evidence-test/excel")
    assert exported.status_code == 200
    with zipfile.ZipFile(BytesIO(exported.data)) as workbook:
        assert b"Contact evidence" in workbook.read("xl/workbook.xml")
        assert b"Point evidence" in workbook.read("xl/workbook.xml")
        assert b"human_review" in workbook.read("xl/sharedStrings.xml")
    reset = client.post("/api/statistics/evidence-test", json=dict(review, reset=True))
    assert reset.json["players"]["P1"]["attack"] == {"total": 1, "unknown": 1}
    assert len(review_path(stats_path).read_text().splitlines()) == 2
    assert (stats_path.read_bytes(), ann_path.read_bytes()) == before


@pytest.mark.parametrize("extra,status", [({"attempt_ref": "bad"}, 400), ({"winner_side": "team_b"}, 400),
                                          ({"reset": "yes"}, 400), ({"source": {}}, 409)])
def test_invalid_review_never_creates_a_log(evidence_client, extra, status):
    client, stats_path, _ = evidence_client
    data = client.get("/api/statistics/evidence-test").json
    body = {"source": data["source"], "attempt_ref": "service_attempts_fsm:0", "winner_side": "b"}
    assert client.post("/api/statistics/evidence-test", json=dict(body, **extra)).status_code == status
    assert not review_path(stats_path).exists()


@pytest.mark.parametrize("changed", ["stats", "annotations"])
def test_reviews_reject_changed_source_pair(evidence_client, changed):
    client, stats_path, ann_path = evidence_client
    data = client.get("/api/statistics/evidence-test").json
    assert client.post("/api/statistics/evidence-test", json={"source": data["source"],
        "attempt_ref": "service_attempts_fsm:0", "winner_side": "a"}).status_code == 200
    path = stats_path if changed == "stats" else ann_path
    path.write_bytes(path.read_bytes() + b"\n")
    assert client.get("/api/statistics/evidence-test").status_code == 409
    assert client.get("/player-statistics/evidence-test").status_code == 409
