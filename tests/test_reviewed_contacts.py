"""A reviewed queue may teach contacts and nothing it did not verify.

The reviewer saw only the candidates the spotter offered, so every other frame
is unknown rather than background, and a rejection next to a kept contact means
"duplicate", not "no touch here". These tests pin how those cases become
targets, and that the held-out experiment never scores an arm on a video it
trained on.
"""

import json
from argparse import Namespace

import numpy as np
import pytest

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting import reviewed_contacts as rc
from action_recognition.rally_spotting import spotter_dataset as sd
from scripts.evaluation import run_reviewed_expansion as experiment
from scripts.models import train_contact_spotter as trainer


def _contact(frame, status, contact_type=None, moved=None, verified=True,
             predicted="set", stem="v1"):
    review = None
    if verified:
        review = {
            "status": status,
            "type": contact_type if status == "kept" else None,
            "core_type": dc.to_core(contact_type) if status == "kept" else None,
            "team": "team_a" if status == "kept" else None,
            "frame": moved,
            "ball": None,
        }
    return {
        "id": f"{stem}:{frame}",
        "video_id": stem,
        "frame": frame,
        "predicted_type": predicted,
        "origin": "model",
        "label_source": "human" if verified else "model",
        "verified": verified,
        "review": review,
    }


def _review(tmp_path, contacts, stem="v1"):
    payload = {
        "schema": "contact_review_v1",
        "videos": {stem: {"annotations": "unused", "annotations_sha256": "0" * 64}},
        "contacts": contacts,
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_kept_contact_lands_on_the_frame_the_human_chose(tmp_path):
    path = _review(tmp_path, [_contact(100, "kept", "dig", moved=103)])
    review = rc.load_review(path)
    (contact,) = review["contacts"]["v1"]
    assert contact["frame"] == 103
    assert contact["contact_type"] == "dig"
    assert contact["core_type"] == "pass"
    assert review["audit"]["v1"]["frame_corrected"] == 1
    assert review["audit"]["v1"]["type_corrected"] == 1


def test_a_rejection_beside_a_kept_contact_is_a_duplicate_not_a_negative(tmp_path):
    path = _review(tmp_path, [
        _contact(100, "kept", "set"),
        _contact(106, "rejected"),
        _contact(500, "rejected"),
    ])
    review = rc.load_review(path)
    assert review["audit"]["v1"]["rejected_duplicate_of_kept"] == 1
    assert review["audit"]["v1"]["rejected_isolated"] == 1
    # Neither rejection becomes a target of any kind.
    assert [c["frame"] for c in review["contacts"]["v1"]] == [100]
    assert review["masked"]["v1"] == []


def test_candidates_moved_onto_one_frame_are_one_touch(tmp_path):
    path = _review(tmp_path, [
        _contact(100, "kept", "attack", moved=104),
        _contact(108, "kept", "attack", moved=104),
    ])
    review = rc.load_review(path)
    (contact,) = review["contacts"]["v1"]
    assert contact["contact_type"] == "attack"
    assert len(contact["review_ids"]) == 2
    assert review["audit"]["v1"]["merged_into_one_frame"] == 1


def test_merged_candidates_that_disagree_keep_the_frame_but_not_the_type(tmp_path):
    path = _review(tmp_path, [
        _contact(100, "kept", "set"),
        _contact(103, "kept", "reception", moved=100),
    ])
    review = rc.load_review(path)
    (contact,) = review["contacts"]["v1"]
    assert contact["contact_type"] is None
    # set -> pass and reception -> reception disagree in four classes too.
    assert contact["core_type"] is None
    assert review["audit"]["v1"]["merged_type_conflicts"] == 1


def test_merged_raw_conflict_with_one_core_type_keeps_the_core(tmp_path):
    path = _review(tmp_path, [
        _contact(100, "kept", "set"),
        _contact(102, "kept", "dig", moved=100),
    ])
    (contact,) = rc.load_review(path)["contacts"]["v1"]
    assert contact["contact_type"] is None
    assert contact["core_type"] == "pass"


def test_an_unverified_candidate_is_masked_never_background(tmp_path):
    path = _review(tmp_path, [_contact(300, "kept", "set", verified=False)])
    review = rc.load_review(path)
    assert review["contacts"]["v1"] == []
    assert review["masked"]["v1"] == [300]
    assert review["audit"]["v1"]["unverified_masked"] == 1


def test_a_kept_contact_without_a_valid_type_is_refused(tmp_path):
    path = _review(tmp_path, [_contact(100, "kept", "free_ball")])
    with pytest.raises(ValueError, match="kept with type"):
        rc.load_review(path)


def test_another_schema_is_refused(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"schema": "contact_spatial_review_v1"}), encoding="utf-8")
    with pytest.raises(ValueError, match="contact_review_v1"):
        rc.load_review(path)


def test_an_analysis_changed_since_review_is_refused(tmp_path):
    analysis = tmp_path / "a_annotations.json"
    analysis.write_text("{}", encoding="utf-8")
    review = {
        "videos": {"v1": {"annotations": str(analysis), "annotations_sha256": "0" * 64}},
        "contacts": {"v1": []}, "masked": {"v1": []}, "audit": {"v1": {}},
    }
    with pytest.raises(ValueError, match="changed since it was reviewed"):
        rc.load_reviewed_source("v1", review)


def test_sparse_targets_label_contacts_and_mask_everything_else():
    contacts = [{"frame": 50, "contact_type": "serve", "core_type": "serve"},
                {"frame": 120, "contact_type": None, "core_type": "pass"}]
    targets, audit = dc.build_targets(200, np.zeros(200, np.int64), contacts)
    assert set(np.unique(targets)) == {dc.IGNORE_INDEX, dc.contact_index("serve")}
    assert (targets[48:53] == dc.contact_index("serve")).all()
    # No phase frame is invented anywhere.
    assert not np.isin(targets, [0, 1, 2]).any()
    assert audit["masked_ambiguous_type"] == 1


def test_training_windows_skip_sparse_windows_without_a_contact():
    targets = np.full(1000, dc.IGNORE_INDEX, dtype=np.int64)
    targets[100:105] = dc.contact_index("set")
    sparse = {"frames": 1000, "targets": targets, "sparse": True}
    dense = {"frames": 1000, "targets": targets, "sparse": False}
    sparse_items = sd.training_windows([sparse], window=200, stride=100, contact_repeat=2)
    dense_items = sd.training_windows([dense], window=200, stride=100, contact_repeat=2)
    # Only windows 0 and 100 hold the contact; each repeated twice.
    assert sorted(set(start for _, start in sparse_items)) == [0, 100]
    assert len(sparse_items) == 4
    assert len(dense_items) > len(sparse_items)


def test_constraints_come_only_from_sources_with_a_phase_track():
    structure = np.array([0] * 10 + [dc.contact_index("serve")] * 3 + [1] * 10 + [0] * 5)
    args = Namespace(duration_quantile=0.0, duration_slack=1.5)
    constraints = trainer.count_constraints(
        [{"structure": structure}, {"structure": None}], args,
    )
    assert constraints["sequences"] == 1
    with pytest.raises(ValueError, match="phase track"):
        trainer.count_constraints([{"structure": None}], args)


def test_no_arm_is_scored_on_a_video_it_trained_on():
    reviewed = ["r1", "r2", "r3"]
    plan = experiment.arms(reviewed)
    for name, spec in plan.items():
        trained = set(spec["train_gold"]) | set(spec["train_reviewed"])
        assert not trained & set(spec["evaluate"]), name
    # Every fold has a matched baseline trained without any reviewed video.
    for spec in plan.values():
        if spec["condition"] == "base":
            assert spec["train_reviewed"] == []
    assert len(plan) == 2 * len(experiment.GOLD) + 1 + len(reviewed)


def test_plus_and_base_differ_only_by_the_reviewed_videos():
    plan = experiment.arms(["r1", "r2"])
    for held in experiment.GOLD:
        base, plus = plan[f"gold_{held}_base"], plan[f"gold_{held}_plus"]
        assert base["train_gold"] == plus["train_gold"]
        assert base["evaluate"] == plus["evaluate"]
        assert plus["train_reviewed"] == ["r1", "r2"]


def test_type_at_frames_reads_the_type_at_the_human_frame():
    probabilities = np.zeros((40, dc.NUM_LABELS))
    probabilities[:, 0] = 1.0
    probabilities[10, :] = 0.0
    probabilities[10, dc.contact_index("dig")] = 1.0
    probabilities[30, :] = 0.0
    probabilities[30, dc.contact_index("attack")] = 1.0
    truth = [
        {"frame": 11, "type": "dig", "core_type": "pass"},
        {"frame": 30, "type": "block", "core_type": "attack"},
        {"frame": 20, "type": None, "core_type": None},
    ]
    result = experiment.type_at_frames(probabilities, truth)
    assert result["raw_contacts"] == 2
    assert result["raw_accuracy"] == pytest.approx(0.5)
    assert result["core_accuracy"] == pytest.approx(1.0)
    assert result["confusion_truth_to_predicted"] == {"block->attack": 1}
