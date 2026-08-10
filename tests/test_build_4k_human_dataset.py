import json
from pathlib import Path

import pytest

from action_recognition.event_clips.build_4k_human_dataset import (
    build_leakage_safe_dataset,
    combine_human_manifests,
)


def _write_manifest(root: Path, clips: list[dict]) -> Path:
    root.mkdir(parents=True)
    features = root / "features"
    features.mkdir()
    for clip in clips:
        (features / f"{clip['clip_id']}.parquet").write_bytes(clip["clip_id"].encode())
    path = root / "manifest_labeled.json"
    path.write_text(json.dumps({"clips": clips}), encoding="utf-8")
    return path


def _clip(clip_id: str, stem: str, label: str, source: str = "human") -> dict:
    return {
        "clip_id": clip_id,
        "source_stem": stem,
        "event_types": [label],
        "label_source": source,
    }


def test_combine_human_manifests_links_features_and_rejects_unlabelled(tmp_path):
    first = _write_manifest(tmp_path / "a", [_clip("a1", "new_3", "serve")])
    second = _write_manifest(tmp_path / "b", [_clip("b1", "new_4", "attack")])

    output = tmp_path / "human"
    manifest = combine_human_manifests([first, second], output)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert [clip["clip_id"] for clip in data["clips"]] == ["a1", "b1"]
    assert data["feature_schema"] == "resolution_normalized_v2"
    assert (output / "features" / "a1.parquet").read_bytes() == b"a1"

    incomplete = _write_manifest(
        tmp_path / "incomplete",
        [_clip("u1", "new_5", "serve", source="model")],
    )
    with pytest.raises(ValueError, match="without completed human labels"):
        combine_human_manifests([incomplete], tmp_path / "bad")


def test_build_replaces_legacy_sources_and_preserves_video_split(tmp_path):
    old_clips = [
        _clip("old_train", "1", "pass", source="legacy"),
        _clip("old_3", "3", "serve", source="legacy"),
        _clip("old_4", "4", "attack", source="legacy"),
        {"clip_id": "old_unlabelled", "source_stem": "2", "event_types": []},
    ]
    old_manifest = _write_manifest(tmp_path / "old", old_clips)
    human_manifest = _write_manifest(
        tmp_path / "human",
        [
            _clip("new_val", "youtube_new_3", "reception"),
            _clip("new_train", "youtube_new_4", "other"),
        ],
    )

    manifest, report = build_leakage_safe_dataset(
        old_manifest=old_manifest,
        old_features_dir=old_manifest.parent / "features",
        human_manifest=human_manifest,
        output_dir=tmp_path / "v6",
        canonical_map={"youtube_new_3": "3", "youtube_new_4": "4"},
        excluded_old_stems={"3", "4", "5"},
        val_stems={"3"},
    )
    clips = json.loads(manifest.read_text(encoding="utf-8"))["clips"]

    assert {clip["clip_id"] for clip in clips} == {
        "old_train", "new_val", "new_train"
    }
    assert {clip["clip_id"]: clip["source_stem"] for clip in clips} == {
        "old_train": "1",
        "new_val": "3",
        "new_train": "4",
    }
    assert report["old_replaced"] == 2
    assert report["split_counts"] == {"train": 2, "validation": 1}
    assert report["class_counts"] == {"pass": 1, "reception": 1}
    assert report["excluded_non_core"] == 1
    assert len(list((tmp_path / "v6" / "features").glob("*.parquet"))) == 3


def test_build_requires_mapping_for_every_human_source(tmp_path):
    old_manifest = _write_manifest(tmp_path / "old", [])
    human_manifest = _write_manifest(
        tmp_path / "human", [_clip("new", "unknown_4k", "serve")]
    )

    with pytest.raises(ValueError, match="No canonical source_stem mapping"):
        build_leakage_safe_dataset(
            old_manifest=old_manifest,
            old_features_dir=old_manifest.parent / "features",
            human_manifest=human_manifest,
            output_dir=tmp_path / "v6",
            canonical_map={},
            excluded_old_stems={"3", "4", "5"},
        )
