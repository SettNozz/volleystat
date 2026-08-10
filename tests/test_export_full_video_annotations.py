import json

import numpy as np

from action_recognition.annotator.export_full_video_annotations import (
    audit_annotations,
    build_clip_manifest,
    build_dense_targets,
    build_rallies,
    build_spotting_labels,
    export_clip_features_npz,
    deduplicate_exact_events,
)


def _event(frame, event_type, team, quality="not_applicable"):
    return {
        "annotation_id": f"a_{frame}_{event_type}", "frame": frame,
        "event_type": event_type, "team_side": team,
        "match_phase": "serve_flight" if event_type == "serve" else "rally_live",
        "physical_action": "serve" if event_type == "serve" else "forearm_pass",
        "context_role": "serve" if event_type == "serve" else "serve_reception",
        "reception_quality": quality,
        "is_service_reception": event_type == "reception", "label_source": "human",
    }


def test_next_serve_confirms_point_and_final_rally_stays_unresolved():
    events = [
        _event(10, "serve", "team_b"), _event(20, "reception", "team_a", "perfect"),
        _event(30, "attack", "team_a"), _event(100, "serve", "team_a"),
        _event(110, "reception", "team_b", "negative"),
    ]
    segments = [{"start_frame": 31, "end_frame": 99, "match_phase": "between_rallies"}]
    rallies, enriched = build_rallies(events, segments, 150)
    assert len(rallies) == 2
    assert rallies[0]["point"]["winner_team_id"] == "opponent_team"
    assert rallies[0]["point"]["evidence_frame"] == 100
    assert rallies[1]["point"]["status"] == "unresolved"
    assert enriched[1]["rally_id"] == "rally_0001"


def test_clip_manifest_groups_concurrent_events_and_keeps_future_context():
    events = [_event(10, "serve", "team_b"), _event(30, "attack", "team_a"), _event(30, "block", "team_b")]
    rallies, _ = build_rallies(events, [], 100)
    manifest = build_clip_manifest(
        {"video_id": "game", "path": "game.mp4", "fps": 30.0},
        rallies, pre_frames=10, post_frames=20,
    )
    assert len(manifest["clips"]) == 2
    contact = manifest["clips"][1]
    assert contact["full_event_types"] == ["attack", "block"]
    assert contact["event_types"] == ["attack"]
    assert contact["contact_offset"] == 10
    assert contact["post_contact_frames"] > 0


def test_spotting_collapses_same_coarse_label_but_preserves_rally_end():
    events = [_event(10, "serve", "team_b"), _event(30, "attack", "team_a"), _event(30, "block", "team_b")]
    rallies, _ = build_rallies(events, [], 100)
    labels = build_spotting_labels(rallies)["events"]
    assert sum(e["type"] == "followup_contact" for e in labels) == 1
    assert sum(e["type"] == "rally_end" for e in labels) == 1


def test_dense_targets_are_multilabel_and_contacts_override_phase_boundary():
    events = [_event(10, "serve", "team_b"), _event(20, "reception", "team_a", "perfect")]
    segments = [{"segment_id": "s", "start_frame": 0, "end_frame": 10, "match_phase": "non_game"}]
    rallies, enriched = build_rallies(events, segments, 50)
    target, phase, quality = build_dense_targets(enriched, rallies, segments, 51)
    assert target[10].sum() == 1
    assert phase[10] != phase[9]
    assert quality[20] >= 0
    assert audit_annotations(events, segments)[-1]["code"] == "contact_on_phase_boundary"


def test_clip_feature_export_uses_18_channel_event_contract(tmp_path, monkeypatch):
    fake = np.arange(5 * 22, dtype=np.float32).reshape(5, 22)
    monkeypatch.setattr(
        "action_recognition.rally_spotting.prepare.build_continuous_features",
        lambda annotations: (np.arange(5, dtype=np.int64), fake),
    )
    manifest = {"clips": [{"clip_id": "c1", "start_frame": 1, "end_frame": 3}]}
    export_clip_features_npz({"frames": [{}]}, manifest, tmp_path)
    with np.load(tmp_path / "event_clip_features" / "c1.npz") as payload:
        assert payload["features"].shape == (3, 18)
        np.testing.assert_array_equal(payload["features"][:, :10], fake[1:4, :10])
        np.testing.assert_array_equal(payload["features"][:, 10:], fake[1:4, 12:20])


def test_exact_duplicates_collapse_but_multilabel_contacts_remain():
    events = [
        _event(10, "serve", "team_b"),
        {**_event(10, "serve", "team_b"), "annotation_id": "duplicate"},
        _event(20, "attack", "team_a"),
        _event(20, "block", "team_b"),
    ]
    kept, duplicates = deduplicate_exact_events(events)
    assert len(kept) == 3
    assert len(duplicates) == 1
    assert {e["event_type"] for e in kept if e["frame"] == 20} == {"attack", "block"}
