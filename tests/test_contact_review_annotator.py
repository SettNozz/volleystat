"""Model output under review must never pass for ground truth by accident.

The review queue starts as predictions. These tests pin the line between the
two: what a record looks like before a human touches it, what changes when one
does, and that the model's own guess survives next to the correction instead of
being overwritten by it.
"""

import json

import pytest

from action_recognition.annotator.contact_review_annotator import (
    SCHEMA, build_queue, create_app,
)


def _annotations(stem, frames=400, events=((100, "serve"), (140, "reception"))):
    return {
        "fps": 29.97,
        "video": stem,
        "original_video_path": f"/videos/{stem}.webm",
        "frame_width": 3840,
        "frame_height": 2160,
        "net_y": 849.0,
        "net_floor_left": [1200.0, 1214.0],
        "net_floor_right": [2832.0, 1214.0],
        "contact_spotter": {
            "enabled": True,
            "checkpoint": "models/contact_spotter/test.pt",
            "diagnostics": {"events": len(events), "members": 3},
        },
        "frames": [
            {"f": index, "ball": [1900.0, 900.0]} for index in range(frames)
        ],
    }


def _stats(events):
    return {
        "all_events": [
            {
                "frame": frame, "type": kind, "core_type": "serve",
                "confidence": 0.9, "source": "dense_contact_spotter",
            }
            for frame, kind in events
        ],
    }


def _write(directory, stem, events=((100, "serve"), (140, "reception"))):
    (directory / f"{stem}_annotations.json").write_text(
        json.dumps(_annotations(stem, events=events)), encoding="utf-8",
    )
    (directory / f"{stem}_stats.json").write_text(
        json.dumps(_stats(events)), encoding="utf-8",
    )


@pytest.fixture
def analyses(tmp_path):
    directory = tmp_path / "analyses"
    directory.mkdir()
    _write(directory, "video_one")
    return directory


@pytest.fixture
def client(analyses, tmp_path):
    app = create_app(analyses, tmp_path / "review" / "review.json")
    app.config.update(TESTING=True)
    return app.test_client()


def test_queue_starts_as_unverified_model_output(analyses):
    queue = build_queue(analyses)
    assert queue["schema"] == SCHEMA
    assert len(queue["contacts"]) == 2
    for contact in queue["contacts"]:
        assert contact["label_source"] == "model"
        assert contact["verified"] is False
        assert contact["review"] is None
        assert contact["origin"] == "model"
    assert "ground truth" in queue["provenance"]["note"]


def test_an_analysis_without_a_spotter_block_is_refused(tmp_path):
    directory = tmp_path / "bare"
    directory.mkdir()
    payload = _annotations("bare")
    payload.pop("contact_spotter")
    (directory / "bare_annotations.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no contact_spotter"):
        build_queue(directory)


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_queue(tmp_path)


def test_contacts_come_back_in_order(analyses):
    queue = build_queue(analyses)
    frames = [c["frame"] for c in queue["contacts"]]
    assert frames == sorted(frames)


def test_keeping_a_contact_makes_it_human_and_remembers_the_prediction(client):
    identity = "video_one:100"
    response = client.post("/api/review", json={
        "id": identity, "status": "kept", "type": "set", "team": "team_b",
    })
    contact = response.get_json()["contact"]
    assert contact["verified"] is True
    assert contact["label_source"] == "human"
    assert contact["review"]["type"] == "set"
    assert contact["review"]["core_type"] == "pass"
    assert contact["review"]["team"] == "team_b"
    # The model's own answer is still there to be compared against.
    assert contact["predicted_type"] == "serve"


def test_unsure_does_not_make_a_contact_verified(client):
    response = client.post("/api/review", json={
        "id": "video_one:100", "status": "unsure",
    })
    contact = response.get_json()["contact"]
    assert contact["verified"] is False
    assert contact["label_source"] == "model"


def test_rejecting_marks_it_reviewed_without_inventing_a_label(client):
    response = client.post("/api/review", json={
        "id": "video_one:140", "status": "rejected",
    })
    contact = response.get_json()["contact"]
    assert contact["verified"] is True
    assert contact["review"]["status"] == "rejected"
    assert contact["review"]["type"] is None
    assert contact["review"]["team"] is None


def test_keeping_requires_both_a_type_and_a_side(client):
    assert client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "set",
    }).status_code == 400
    assert client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "team": "team_a",
    }).status_code == 400
    assert client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "banana",
        "team": "team_a",
    }).status_code == 400


def test_an_inserted_contact_carries_no_prediction(client):
    response = client.post("/api/insert", json={
        "video_id": "video_one", "frame": 120, "type": "attack",
        "team": "team_a",
    })
    body = response.get_json()
    added = next(c for c in body["contacts"] if c["frame"] == 120)
    assert added["origin"] == "human"
    assert added["label_source"] == "human"
    assert added["verified"] is True
    assert added["predicted_type"] is None
    assert added["review"]["status"] == "added"
    # It lands in order, so the queue stays chronological.
    frames = [c["frame"] for c in body["contacts"]]
    assert frames == sorted(frames)


def test_two_contacts_cannot_share_a_frame(client):
    first = client.post("/api/insert", json={
        "video_id": "video_one", "frame": 120, "type": "attack", "team": "team_a",
    })
    assert first.status_code == 200
    second = client.post("/api/insert", json={
        "video_id": "video_one", "frame": 120, "type": "dig", "team": "team_b",
    })
    assert second.status_code == 409


def test_a_frame_outside_the_video_is_refused(client):
    assert client.post("/api/insert", json={
        "video_id": "video_one", "frame": 99999, "type": "dig", "team": "team_b",
    }).status_code == 400


def test_review_survives_a_restart(analyses, tmp_path):
    output = tmp_path / "review" / "review.json"
    first = create_app(analyses, output).test_client()
    first.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "dig", "team": "team_a",
    })
    reopened = create_app(analyses, output).test_client()
    contacts = reopened.get("/api/project").get_json()["contacts"]
    kept = next(c for c in contacts if c["id"] == "video_one:100")
    assert kept["verified"] is True
    assert kept["review"]["type"] == "dig"


def test_changed_analyses_are_refused_rather_than_mixed(analyses, tmp_path):
    output = tmp_path / "review" / "review.json"
    create_app(analyses, output).test_client().post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "dig", "team": "team_a",
    })
    # Re-running the analysis changes the file, so its hash no longer matches.
    _write(analyses, "video_one", events=((100, "serve"), (200, "attack")))
    with pytest.raises(ValueError, match="analyses changed"):
        create_app(analyses, output)


def test_summary_counts_what_changed(client):
    client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "serve", "team": "team_a",
    })
    client.post("/api/review", json={
        "id": "video_one:140", "status": "kept", "type": "attack", "team": "team_b",
    })
    summary = client.get("/api/summary").get_json()
    assert summary["type"]["type_kept"] == 1
    assert summary["type"]["type_changed"] == 1
    assert summary["per_video"]["video_one"]["verified"] == 2


def test_clearing_a_review_puts_it_back_to_model_output(client):
    client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "dig", "team": "team_a",
    })
    response = client.post("/api/review", json={"id": "video_one:100", "status": None})
    contact = response.get_json()["contact"]
    assert contact["verified"] is False
    assert contact["label_source"] == "model"
    assert contact["review"] is None


def _candidates(stem, rows):
    return {
        "schema": "spotter_candidates_v1",
        "stem": stem,
        "threshold": 0.2,
        "stats": {"peaks": len(rows), "decoded": sum(1 for r in rows if r[2])},
        "candidates": [
            {
                "frame": frame, "type": kind, "core_type": "pass",
                "confidence": 0.5, "peak_type": kind, "decoded": decoded,
                "source": "dense_contact_spotter_peaks",
            }
            for frame, kind, decoded in rows
        ],
    }


def test_a_candidates_file_is_preferred_over_the_decoder_events(tmp_path):
    directory = tmp_path / "analyses"
    directory.mkdir()
    _write(directory, "video_one")
    # The decoder found two; the candidate generator offers three.
    (directory / "video_one_candidates.json").write_text(
        json.dumps(_candidates("video_one", [
            (100, "serve", True), (120, "set", False), (140, "reception", True),
        ])),
        encoding="utf-8",
    )
    queue = build_queue(directory)
    assert [c["frame"] for c in queue["contacts"]] == [100, 120, 140]
    spare = next(c for c in queue["contacts"] if c["frame"] == 120)
    assert spare["decoded"] is False
    assert spare["label_source"] == "model"
    assert queue["videos"]["video_one"]["candidate_threshold"] == 0.2


def test_replacing_the_candidates_file_is_refused_mid_review(tmp_path):
    directory = tmp_path / "analyses"
    directory.mkdir()
    _write(directory, "video_one")
    path = directory / "video_one_candidates.json"
    path.write_text(json.dumps(_candidates("video_one", [(100, "serve", True)])),
                    encoding="utf-8")
    output = tmp_path / "review" / "review.json"
    create_app(directory, output).test_client().post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "serve", "team": "team_a",
    })
    path.write_text(json.dumps(_candidates("video_one", [(100, "serve", True),
                                                         (300, "dig", True)])),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="analyses changed"):
        create_app(directory, output)


def test_a_corrected_frame_is_stored_beside_the_model_frame(client):
    response = client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "attack",
        "team": "team_a", "frame": 104,
    })
    contact = response.get_json()["contact"]
    # The record keeps its identity and the model's frame; the correction
    # lives in the review, as the type correction does.
    assert contact["frame"] == 100
    assert contact["review"]["frame"] == 104


def test_no_frame_correction_is_recorded_as_none(client):
    response = client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "attack", "team": "team_a",
    })
    assert response.get_json()["contact"]["review"]["frame"] is None


def test_a_frame_correction_outside_the_video_is_refused(client):
    assert client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "attack",
        "team": "team_a", "frame": 99999,
    }).status_code == 400


def test_a_hand_placed_ball_is_stored(client):
    response = client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "attack",
        "team": "team_a", "ball": [1920.5, 800.25],
    })
    assert response.get_json()["contact"]["review"]["ball"] == [1920.5, 800.25]


def test_a_ball_outside_the_frame_is_refused(client):
    assert client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "attack",
        "team": "team_a", "ball": [9000, 100],
    }).status_code == 400
    assert client.post("/api/review", json={
        "id": "video_one:100", "status": "kept", "type": "attack",
        "team": "team_a", "ball": [100],
    }).status_code == 400


def test_the_ball_endpoint_reports_what_the_detector_had(client):
    body = client.get("/api/ball/video_one/10").get_json()
    assert body["ball"] == [1900.0, 900.0]
    assert client.get("/api/ball/missing/10").status_code == 404
