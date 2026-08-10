import json

import numpy as np

from action_recognition.event_clips.replace_source_dataset import replace_source_dataset


def _manifest(path, clips):
    path.write_text(json.dumps({"clips": clips}), encoding="utf-8")


def test_replaces_all_legacy_clips_from_completed_source(tmp_path):
    base_features = tmp_path / "base_features"
    new_features = tmp_path / "new_features"
    base_features.mkdir()
    new_features.mkdir()
    base = [
        {"clip_id": "old_a", "source_stem": "done"},
        {"clip_id": "keep_b", "source_stem": "other"},
    ]
    new = [{"clip_id": "gold_a", "source_stem": "done"}]
    base_manifest = tmp_path / "base.json"
    new_manifest = tmp_path / "new.json"
    _manifest(base_manifest, base)
    _manifest(new_manifest, new)
    np.savez_compressed(base_features / "old_a.npz", features=np.zeros((2, 18)))
    np.savez_compressed(base_features / "keep_b.npz", features=np.zeros((2, 18)))
    np.savez_compressed(new_features / "gold_a.npz", features=np.zeros((2, 18)))

    report = replace_source_dataset(
        base_manifest, base_features, new_manifest, new_features, tmp_path / "out"
    )
    result = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert report == {
        "clips": 2, "removed": 1, "inserted": 1,
        "by_source": {"other": 1, "done": 1}, "feature_files": 2,
    }
    assert {clip["clip_id"] for clip in result["clips"]} == {"keep_b", "gold_a"}
    assert not (tmp_path / "out" / "features" / "old_a.npz").exists()
