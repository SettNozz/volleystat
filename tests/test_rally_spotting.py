import json
from pathlib import Path

import numpy as np
import pytest
import torch

from action_recognition.rally_spotting.metrics import (
    match_events,
    nms_peaks,
    spotting_metrics,
)
from action_recognition.rally_spotting.model import RallySpottingTCN
from action_recognition.rally_spotting.prepare import (
    FEATURE_NAMES,
    build_continuous_features,
    events_to_targets,
    prepare_manifest,
    weak_events_from_stats,
)
from action_recognition.rally_spotting.taxonomy import EVENT_TYPES
from action_recognition.rally_spotting.train import (
    split_records_by_video,
    validate_label_policy,
)


def sample_annotations():
    return {
        "fps": 30,
        "frame_width": 3840,
        "frame_height": 2160,
        "net_y": 900,
        "frames": [
            {
                "f": 10,
                "ball": [1920, 1080],
                "p": [[960, 1080]],
                "pose_f": [1.0] * 8,
                "poses": [{
                    "team": "b",
                    "team_prior": "serve_back_facing_net_override",
                }],
            },
            {"f": 11, "ball": [2304, 1080], "p": [[960, 1080]]},
            {"f": 13, "ball": None, "p": []},
        ],
    }


def test_continuous_features_are_resolution_normalized_and_keep_gaps():
    frames, features = build_continuous_features(sample_annotations())
    assert frames.tolist() == [10, 11, 12, 13]
    assert features.shape == (4, len(FEATURE_NAMES))
    assert features[0, 0] == pytest.approx(0.5)
    assert features[0, 1] == pytest.approx(0.5)
    assert features[1, 0] == pytest.approx(0.6)
    assert features[1, 2] == pytest.approx(3.0)
    assert features[2, 10] == 0.0
    assert features[0, 12:20].tolist() == [1.0] * 8
    assert features[0, 20] == 1.0
    assert features[0, 21] == 1.0


def test_weak_lifecycle_taxonomy_and_hard_targets():
    stats = {
        "service_attempts": [{
            "end_frame": 50,
            "lifecycle": {
                "serve_frame": 10,
                "reception_frame": 20,
                "followup_frame": 25,
            },
        }],
    }
    events = weak_events_from_stats(stats)
    assert [event["type"] for event in events] == list(EVENT_TYPES)
    frames = np.arange(10, 51)
    targets, counts = events_to_targets(events, frames)
    assert targets.shape == (41, 4)
    assert targets[:, 0].argmax() == 0
    assert targets[:, 3].argmax() == 40
    assert counts == {name: 1 for name in EVENT_TYPES}


def test_prepare_marks_weak_source_and_human_requires_labels(tmp_path: Path):
    annotations = tmp_path / "annotations.json"
    stats = tmp_path / "stats.json"
    annotations.write_text(json.dumps(sample_annotations()), encoding="utf-8")
    stats.write_text(json.dumps({
        "service_attempts": [{
            "end_frame": 13,
            "lifecycle": {"serve_frame": 10},
        }],
    }), encoding="utf-8")
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"videos": [{
        "source_video": "game.mp4",
        "stats_path": "stats.json",
        "annotations_path": "annotations.json",
        "label_source": "weak_fsm",
    }]}), encoding="utf-8")

    result = prepare_manifest(manifest, tmp_path / "prepared")
    assert result["records"][0]["label_source"] == "weak_fsm"
    validate_label_policy(result["records"], allow_weak_labels=True)
    with pytest.raises(ValueError, match="weak_fsm"):
        validate_label_policy(result["records"])

    human_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    human_manifest["videos"][0]["label_source"] = "human"
    manifest.write_text(json.dumps(human_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="requires labels_path"):
        prepare_manifest(manifest, tmp_path / "human")


def test_video_group_split_has_no_source_leakage():
    records = [
        {"id": "a1", "source_video": "a.mp4", "source_key": "a"},
        {"id": "a2", "source_video": "a.mp4", "source_key": "a"},
        {"id": "b", "source_video": "b.mp4", "source_key": "b"},
        {"id": "c", "source_video": "c.mp4", "source_key": "c"},
        {"id": "d", "source_video": "d.mp4", "source_key": "d"},
        {"id": "e", "source_video": "e.mp4", "source_key": "e"},
    ]
    splits = split_records_by_video(records, 0.2, 0.2, seed=7)
    keys = {
        name: {record["source_key"] for record in split}
        for name, split in splits.items()
    }
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])
    assert sum(len(split) for split in splits.values()) == len(records)


def test_nms_matching_and_tolerance_metrics():
    scores = np.zeros((40, 4), dtype=np.float32)
    labels = np.zeros_like(scores)
    scores[9, 0] = 0.8
    scores[10, 0] = 0.9
    scores[30, 0] = 0.7
    labels[12, 0] = 1.0
    assert nms_peaks(scores[:, 0], threshold=0.5, radius=3) == [10, 30]
    assert match_events([10, 30], [12], tolerance=5) == (1, 1, 0)
    metrics = spotting_metrics(
        [scores], [labels], EVENT_TYPES,
        tolerances=(5,), threshold=0.5, nms_radius=3,
    )
    assert metrics["at_5"]["per_class"]["serve_contact"]["precision"] == 0.5
    assert metrics["at_5"]["per_class"]["serve_contact"]["recall"] == 1.0


def test_matching_is_maximum_cardinality_not_greedy_nearest():
    assert match_events([4, 6], [0, 5], tolerance=4) == (2, 0, 0)


def test_tcn_output_shape_and_config():
    model = RallySpottingTCN(input_dim=len(FEATURE_NAMES), hidden_dim=16)
    inputs = torch.zeros((2, 32, len(FEATURE_NAMES)))
    outputs = model(inputs)
    assert outputs.shape == (2, 32, len(EVENT_TYPES))
    assert model.config()["input_dim"] == len(FEATURE_NAMES)
