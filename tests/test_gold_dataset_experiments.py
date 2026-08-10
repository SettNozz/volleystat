import json

import numpy as np

from action_recognition.training.gold_dataset_experiments import (
    _aggregate,
    build_rally_holdout_manifest,
    build_spotting_temporal_holdout,
)


def test_rally_holdout_keeps_whole_rallies_together(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"clips": [
        {"clip_id": "legacy", "source_stem": "other_video"},
        {"clip_id": "a", "rally_id": "rally_0024", "source_stem": "video"},
        {"clip_id": "b", "rally_id": "rally_0025", "source_stem": "video"},
        {"clip_id": "c", "rally_id": "rally_0025", "source_stem": "video"},
    ]}), encoding="utf-8")
    output = tmp_path / "holdout.json"
    report = build_rally_holdout_manifest(source, output, 24)
    clips = json.loads(output.read_text(encoding="utf-8"))["clips"]
    assert report["counts"] == {"independent_train": 1, "gold_train": 1, "gold_val": 2}
    assert clips[0]["source_stem"] == "other_video"
    assert {clip["source_stem"] for clip in clips[2:]} == {"gold_val"}


def test_aggregate_has_five_statistics_per_channel():
    sequence = np.asarray([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32)
    result = _aggregate(sequence)
    assert result.shape == (10,)
    np.testing.assert_array_equal(result[-2:], [2.0, 4.0])


def test_spotting_holdout_splits_at_rally_boundary(tmp_path):
    source = tmp_path / "spot"
    (source / "videos").mkdir(parents=True)
    np.savez_compressed(
        source / "videos" / "all.npz", features=np.zeros((10, 2)),
        targets=np.eye(4, dtype=np.float32)[np.arange(10) % 4],
        frame_numbers=np.arange(10),
    )
    (source / "manifest.json").write_text(json.dumps({
        "event_types": ["a", "b", "c", "d"], "feature_names": ["x", "y"],
        "feature_schema": "test", "records": [{
            "id": "all", "source_key": "all", "npz": "videos/all.npz",
            "source_video": "v", "label_source": "human",
        }],
    }), encoding="utf-8")
    events = tmp_path / "events.json"
    events.write_text(json.dumps({"clips": [
        {"rally_id": "rally_0025", "start_frame": 6},
    ]}), encoding="utf-8")
    result = build_spotting_temporal_holdout(source, events, tmp_path / "out", 24)
    assert result["boundary_frame"] == 6
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert [r["num_frames"] for r in manifest["records"]] == [6, 4]
