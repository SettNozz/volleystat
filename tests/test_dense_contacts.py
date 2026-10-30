"""Dense phase/contact targets must never invent or erase human evidence."""

import json

import numpy as np
import pytest

from action_recognition.rally_spotting import dense_contacts as dc


def _review_file(tmp_path, reviews, events, num_frames=200):
    source = tmp_path / "annotations_v1.json"
    source.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    payload = {
        "schema": "contact_spatial_review_v1",
        "manifest_path": str(source),
        "manifest_sha256": dc.file_sha256(source),
        "sources": {"vid": {"path": str(source), "sha256": dc.file_sha256(source)}},
        "videos": {"vid": {
            "video_id": "vid", "fps": 29.97, "width": 3840, "height": 2160,
            "num_frames": num_frames,
        }},
        "events": events,
        "reviews": reviews,
    }
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _event(identity, frame, label, raw, original=None):
    return {
        "id": identity, "video_id": "vid", "original_frame": original or frame,
        "label": label, "annotation_ids": [f"ann_{identity}"], "raw_labels": raw,
    }


def _review(frame, label, **kwargs):
    base = {
        "frame": frame, "label": label, "status": "visible", "timing": "clear",
        "xy": [10.0, 20.0], "xy_normalized": [0.1, 0.2], "verified": True,
        "spatial_valid": True, "temporal_valid": True, "ball_out_of_frame": False,
    }
    base.update(kwargs)
    return base


def test_reviewed_frame_overrides_the_original_proposal(tmp_path):
    path = _review_file(
        tmp_path,
        {"a": _review(57, "serve")},
        [_event("a", 57, "serve", ["serve"], original=903)],
    )
    result = dc.reviewed_contacts(path)
    assert [c["frame"] for c in result["contacts"]["vid"]] == [57]
    assert result["contacts"]["vid"][0]["contact_type"] == "serve"


def test_uncertain_timing_is_masked_not_background(tmp_path):
    path = _review_file(
        tmp_path,
        {"a": _review(60, "pass", timing="uncertain", temporal_valid=False,
                      status="visible")},
        [_event("a", 60, "pass", ["set"], original=58)],
    )
    result = dc.reviewed_contacts(path)
    assert result["contacts"].get("vid", []) == []
    assert set(range(58, 61)) <= set(result["masked"]["vid"])


def test_wrong_event_is_excluded_without_masking(tmp_path):
    path = _review_file(
        tmp_path,
        {"a": _review(60, "pass", status="wrong_event", temporal_valid=False,
                      spatial_valid=False, xy=None, xy_normalized=None)},
        [_event("a", 60, "pass", ["set"])],
    )
    result = dc.reviewed_contacts(path)
    assert result["contacts"].get("vid", []) == []
    assert result["masked"]["vid"] == []


def test_occluded_contact_keeps_its_verified_time(tmp_path):
    path = _review_file(
        tmp_path,
        {"a": _review(70, "attack", status="occluded", spatial_valid=False,
                      xy=None, xy_normalized=None)},
        [_event("a", 70, "attack", ["attack"])],
    )
    result = dc.reviewed_contacts(path)
    contact = result["contacts"]["vid"][0]
    assert contact["frame"] == 70
    assert contact["spatial_valid"] is False


def test_conflicting_raw_types_lose_their_raw_label(tmp_path):
    path = _review_file(
        tmp_path,
        {"a": _review(80, "attack")},
        [_event("a", 80, "attack", ["block", "attack"])],
    )
    contact = dc.reviewed_contacts(path)["contacts"]["vid"][0]
    assert contact["contact_type"] is None
    assert contact["core_type"] == "attack"


def test_repeated_raw_type_is_not_ambiguous(tmp_path):
    path = _review_file(
        tmp_path,
        {"a": _review(80, "attack")},
        [_event("a", 80, "attack", ["block", "block"])],
    )
    assert dc.reviewed_contacts(path)["contacts"]["vid"][0]["contact_type"] == "block"


def test_phase_labels_mask_unlabeled_only():
    exported = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    assert dc.phase_labels(exported).tolist() == [dc.IGNORE_INDEX, 0, 0, 1, 2]


def test_targets_widen_contacts_and_guard_the_ring():
    phases = np.full(60, 4, dtype=np.int64)  # rally_live everywhere
    contacts = [{"frame": 30, "contact_type": "attack", "core_type": "attack"}]
    targets, audit = dc.build_targets(60, phases, contacts, contact_radius=2, guard=3)
    attack = dc.contact_index("attack")
    assert targets[28:33].tolist() == [attack] * 5
    assert targets[25:28].tolist() == [dc.IGNORE_INDEX] * 3
    assert targets[33:36].tolist() == [dc.IGNORE_INDEX] * 3
    assert targets[24] == 2 and targets[36] == 2
    assert audit["segments"] == 1
    assert audit["label_attack"] == 5


def test_adjacent_contact_survives_the_neighbour_guard():
    phases = np.full(60, 4, dtype=np.int64)
    contacts = [
        {"frame": 30, "contact_type": "set", "core_type": "pass"},
        {"frame": 34, "contact_type": "attack", "core_type": "attack"},
    ]
    targets, audit = dc.build_targets(60, phases, contacts, contact_radius=2, guard=3)
    assert targets[30] == dc.contact_index("set")
    assert targets[34] == dc.contact_index("attack")
    assert audit["segments"] == 2


def test_ambiguous_contact_is_masked_never_labelled():
    phases = np.full(40, 4, dtype=np.int64)
    contacts = [{"frame": 20, "contact_type": None, "core_type": "attack"}]
    targets, audit = dc.build_targets(40, phases, contacts, contact_radius=2, guard=3)
    assert targets[20] == dc.IGNORE_INDEX
    assert audit["masked_ambiguous_type"] == 1
    assert audit["segments"] == 0


def test_uncovered_human_contacts_are_masked(tmp_path):
    path = tmp_path / "annotations_v1.json"
    path.write_text(json.dumps({"annotations": [
        {"video_id": "vid", "annotation_id": "kept", "frame": 10},
        {"video_id": "vid", "annotation_id": "free", "frame": 50},
        {"video_id": "other", "annotation_id": "elsewhere", "frame": 70},
    ]}), encoding="utf-8")
    assert dc.uncovered_human_frames(path, "vid", ["kept"]) == [50]


def test_contact_outside_the_decoded_video_is_an_error():
    phases = np.full(10, 4, dtype=np.int64)
    with pytest.raises(ValueError, match="outside decoded video"):
        dc.build_targets(
            10, phases, [{"frame": 11, "contact_type": "serve", "core_type": "serve"}],
        )


def test_core_projection_matches_the_training_taxonomy():
    assert dc.to_core("set") == "pass" and dc.to_core("dig") == "pass"
    assert dc.to_core("block") == "attack" and dc.to_core("serve") == "serve"
    assert dc.to_core(None) is None
