from action_recognition.annotator.event_clip_annotator import STATE, app
from action_recognition.event_clips.manifest_schema import upgrade_manifest


def _configure(tmp_path):
    manifest = upgrade_manifest({
        "clips": [{
            "clip_id": "clip_1",
            "contact_frame": 10,
            "fsm_event_type": "reception",
            "event_types": [],
            "team": "team_a",
        }],
    })
    clip = manifest["clips"][0]
    STATE.clear()
    STATE.update(
        manifest=manifest,
        clips=manifest["clips"],
        by_id={"clip_1": clip},
        output=tmp_path / "manifest_labeled.json",
    )
    return clip


def test_context_endpoint_saves_independent_axes(tmp_path):
    clip = _configure(tmp_path)

    response = app.test_client().post(
        "/api/label",
        json={
            "clip_id": "clip_1",
            "context": {
                "match_phase": "rally_live",
                "context_role": "serve_reception",
                "reception_quality": "overpass",
            },
        },
    )

    assert response.status_code == 200
    assert clip["reception_quality"] == "overpass"
    assert clip["context_label_source"] == "human"
    assert clip["is_service_reception"] is True
    assert STATE["output"].exists()


def test_invalid_context_does_not_partially_update_clip(tmp_path):
    clip = _configure(tmp_path)
    original_phase = clip["match_phase"]

    response = app.test_client().post(
        "/api/label",
        json={
            "clip_id": "clip_1",
            "context": {
                "match_phase": "non_game",
                "context_role": "not-a-volleyball-role",
            },
        },
    )

    assert response.status_code == 400
    assert clip["match_phase"] == original_phase


def test_draft_context_is_not_counted_as_human(tmp_path):
    clip = _configure(tmp_path)

    response = app.test_client().post(
        "/api/label",
        json={
            "clip_id": "clip_1",
            "context": {"reception_quality": "positive"},
            "draft_context": True,
        },
    )

    assert response.status_code == 200
    assert clip["reception_quality"] == "positive"
    assert clip["context_label_source"] != "human"


def test_accept_context_marks_inferred_values_as_human(tmp_path):
    clip = _configure(tmp_path)

    response = app.test_client().post(
        "/api/label",
        json={"clip_id": "clip_1", "accept_context": True},
    )

    assert response.status_code == 200
    assert clip["context_label_source"] == "human"


def test_task_exposes_english_context_help(tmp_path):
    _configure(tmp_path)

    payload = app.test_client().get("/api/task").get_json()

    assert payload["context_help"]["match_phase"]["label"] == "Match phase"
    assert "reception statistics" in payload["context_help"]["context_role"][
        "values"
    ]["serve_reception"]
    assert "otherwise select not_applicable" in payload["context_help"][
        "reception_quality"
    ]["description"]


def test_non_game_resets_all_action_context_fields(tmp_path):
    clip = _configure(tmp_path)

    response = app.test_client().post(
        "/api/label",
        json={
            "clip_id": "clip_1",
            "context": {
                "match_phase": "non_game",
                "physical_action": "forearm_pass",
                "context_role": "serve_reception",
                "reception_quality": "perfect",
            },
            "draft_context": True,
        },
    )

    assert response.status_code == 200
    assert clip["match_phase"] == "non_game"
    assert clip["physical_action"] == "unknown"
    assert clip["context_role"] == "unknown"
    assert clip["reception_quality"] == "unknown"
    assert clip["is_service_reception"] is False


def test_serve_flight_sets_serve_context_defaults(tmp_path):
    clip = _configure(tmp_path)

    response = app.test_client().post(
        "/api/label",
        json={
            "clip_id": "clip_1",
            "context": {
                "match_phase": "serve_flight",
                "physical_action": "attack_hit",
                "context_role": "attack",
                "reception_quality": "perfect",
            },
            "draft_context": True,
        },
    )

    assert response.status_code == 200
    assert clip["match_phase"] == "serve_flight"
    assert clip["physical_action"] == "serve"
    assert clip["context_role"] == "serve"
    assert clip["reception_quality"] == "not_applicable"
    assert clip["is_service_reception"] is False
