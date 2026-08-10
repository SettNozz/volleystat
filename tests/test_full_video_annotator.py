import json

from action_recognition.annotator.full_video_annotator import (
    STATE,
    app,
    build_project,
    upgrade_project,
)


def _configure(tmp_path):
    project = {
        "schema_version": 1,
        "videos": [{
            "video_id": "match_1",
            "name": "match_1",
            "path": str(tmp_path / "match.webm"),
            "fps": 25.0,
            "legacy_contact_count": 0,
        }],
        "annotations": [],
        "segments": [],
    }
    STATE.clear()
    STATE.update(
        project=project,
        videos={"match_1": project["videos"][0]},
        output=tmp_path / "full_video_annotations.json",
    )
    return project


def test_build_project_groups_source_clips_by_video():
    project = build_project({
        "clips": [
            {"source_stem": "game", "source_video": "game.webm", "fps": 30},
            {"source_stem": "game", "source_video": "game.webm", "fps": 30},
        ]
    })

    assert len(project["videos"]) == 1
    assert project["videos"][0]["legacy_contact_count"] == 2
    assert project["annotations"] == []


def test_serve_annotation_uses_automatic_context(tmp_path):
    project = _configure(tmp_path)

    response = app.test_client().post(
        "/api/annotations",
        json={
            "video_id": "match_1",
            "frame": 250,
            "team_side": "team_b",
            "event_type": "serve",
            "reception_quality": "perfect",
        },
    )

    assert response.status_code == 200
    event = project["annotations"][0]
    assert event["time_seconds"] == 10.0
    assert event["match_phase"] == "serve_flight"
    assert event["physical_action"] == "serve"
    assert event["context_role"] == "serve"
    assert event["reception_quality"] == "not_applicable"
    assert json.loads(STATE["output"].read_text(encoding="utf-8"))["annotations"]


def test_reception_preserves_human_quality(tmp_path):
    project = _configure(tmp_path)

    response = app.test_client().post(
        "/api/annotations",
        json={
            "video_id": "match_1",
            "frame": 251,
            "team_side": "team_a",
            "event_type": "reception",
            "reception_quality": "overpass",
        },
    )

    assert response.status_code == 200
    event = project["annotations"][0]
    assert event["context_role"] == "serve_reception"
    assert event["reception_quality"] == "overpass"
    assert event["is_service_reception"] is True


def test_non_game_range_has_unknown_contact_context(tmp_path):
    project = _configure(tmp_path)

    response = app.test_client().post(
        "/api/segments",
        json={"video_id": "match_1", "start_frame": 100, "end_frame": 200},
    )

    assert response.status_code == 200
    segment = project["segments"][0]
    assert segment["match_phase"] == "non_game"
    assert segment["physical_action"] == "unknown"
    assert segment["context_role"] == "unknown"
    assert segment["reception_quality"] == "unknown"
    assert segment["team_side"] is None


def test_between_rallies_range_is_saved_as_no_contact_phase(tmp_path):
    project = _configure(tmp_path)

    response = app.test_client().post(
        "/api/segments",
        json={
            "video_id": "match_1",
            "start_frame": 20,
            "end_frame": 40,
            "match_phase": "between_rallies",
        },
    )

    assert response.status_code == 200
    segment = project["segments"][0]
    assert segment["match_phase"] == "between_rallies"
    assert segment["team_side"] is None


def test_legacy_dead_ball_and_serve_setup_merge_without_losing_origin():
    project = {
        "segments": [
            {"segment_id": "a", "match_phase": "dead_ball"},
            {"segment_id": "b", "match_phase": "serve_setup"},
        ]
    }

    upgraded = upgrade_project(project)

    assert [s["match_phase"] for s in upgraded["segments"]] == [
        "between_rallies",
        "between_rallies",
    ]
    assert [s["legacy_match_phase"] for s in upgraded["segments"]] == [
        "dead_ball",
        "serve_setup",
    ]


def test_ball_contact_requires_a_team(tmp_path):
    _configure(tmp_path)

    response = app.test_client().post(
        "/api/annotations",
        json={
            "video_id": "match_1",
            "frame": 10,
            "team_side": None,
            "event_type": "serve",
        },
    )

    assert response.status_code == 400


def test_project_api_is_never_served_from_stale_cache(tmp_path):
    _configure(tmp_path)

    response = app.test_client().get("/api/project")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store, max-age=0"


def test_annotation_can_be_deleted(tmp_path):
    project = _configure(tmp_path)
    created = app.test_client().post(
        "/api/annotations",
        json={
            "video_id": "match_1",
            "frame": 10,
            "team_side": "team_b",
            "event_type": "attack",
        },
    ).get_json()["annotation"]

    response = app.test_client().delete(
        f"/api/annotations/{created['annotation_id']}"
    )

    assert response.status_code == 200
    assert project["annotations"] == []
