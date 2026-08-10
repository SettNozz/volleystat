from pathlib import Path

import numpy as np
import pytest
import torch

from action_recognition.data.event_clip_dataset import EventClipDataset
from action_recognition.event_clips.build_causal_window_dataset import (
    build_causal_dataset,
)
from action_recognition.event_clips.config import (
    EVENT_TYPES,
    LEGACY_EVENT_TYPES,
    NUM_CLASSES,
    collapse_labels,
    label_index,
    labels_to_vector,
)
from action_recognition.models.sequence_event_model import (
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def _feature(path: Path, clip_id: str) -> None:
    np.savez_compressed(path / f"{clip_id}.npz", features=np.ones((8, 18), np.float32))


def test_core_taxonomy_excludes_other_but_preserves_core_label() -> None:
    assert EVENT_TYPES == ["serve", "reception", "pass", "attack"]
    assert NUM_CLASSES == 4
    assert collapse_labels(["other"]) is None
    assert collapse_labels(["free_ball"]) is None
    assert collapse_labels(["pass", "other"]) == "pass"
    assert collapse_labels(["set", "free_ball"], "set") == "pass"
    assert collapse_labels(["pass", "free_ball"], "pass") == "pass"
    assert labels_to_vector(["other"]) == [0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="No core action label"):
        label_index(["other"])


def test_dataset_filters_pure_non_core_contacts(tmp_path: Path) -> None:
    _feature(tmp_path, "other")
    _feature(tmp_path, "pass")
    dataset = EventClipDataset(
        [
            {"clip_id": "other", "event_types": ["other"]},
            {"clip_id": "pass", "event_types": ["pass", "other"]},
        ],
        str(tmp_path),
    )
    assert len(dataset) == 1
    assert dataset.excluded_non_core == 1
    assert dataset.items[0][0]["clip_id"] == "pass"


def test_dataset_uses_requested_sequence_length(tmp_path: Path) -> None:
    _feature(tmp_path, "serve")
    dataset = EventClipDataset(
        [{"clip_id": "serve", "event_types": ["serve"]}],
        str(tmp_path),
        sequence_len=120,
    )
    features, _ = dataset[0]
    assert features.shape == (120, 18)


def test_causal_builder_ends_window_on_contact(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"feature_schema":"resolution_normalized_v2","clips":['
        '{"clip_id":"c1","source_stem":"video","contact_frame":7,'
        '"event_types":["serve"]}]}',
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        __import__("json").dumps({
            "frame_width": 100,
            "frame_height": 100,
            "fps": 30,
            "net_y": 50,
            "frames": [{"f": frame, "ball": [frame, frame]} for frame in range(10)],
        }),
        encoding="utf-8",
    )
    output = tmp_path / "causal"
    report = build_causal_dataset(
        manifest, {"video": annotations}, output, window_frames=5
    )
    payload = __import__("json").loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    clip = payload["clips"][0]
    assert report["full_windows"] == 1
    assert (clip["start_frame"], clip["end_frame"]) == (3, 7)
    assert clip["contact_offset"] == 4
    assert clip["post_contact_frames"] == 0


def test_causal_builder_combines_manifests_without_source_leakage(tmp_path: Path) -> None:
    annotation_paths = {}
    manifests = []
    for source_index, stem in enumerate(("a", "b")):
        manifest = tmp_path / f"{stem}.json"
        manifest.write_text(
            __import__("json").dumps({"feature_schema": "resolution_normalized_v2", "clips": [{
                "clip_id": f"{stem}_clip", "source_stem": stem,
                "contact_frame": 5, "event_types": ["serve"],
            }]}), encoding="utf-8",
        )
        annotations = tmp_path / f"{stem}_annotations.json"
        annotations.write_text(__import__("json").dumps({
            "frame_width": 100, "frame_height": 100, "fps": 30, "net_y": 50,
            "frames": [{"f": frame, "ball": [frame, frame]} for frame in range(8)],
        }), encoding="utf-8")
        manifests.append(manifest)
        annotation_paths[stem] = annotations
    output = tmp_path / "combined"
    report = build_causal_dataset(manifests, annotation_paths, output, 4)
    payload = __import__("json").loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert report["clips"] == 2
    assert {clip["source_stem"] for clip in payload["clips"]} == {"a", "b"}
    assert len(payload["source_manifests"]) == 2


def test_checkpoint_loader_uses_checkpoint_taxonomy(tmp_path: Path) -> None:
    new_path = tmp_path / "four.pt"
    new_model = build_model("lstm")
    save_checkpoint(new_path, new_model, "lstm", 18)
    loaded_new, new_payload = load_checkpoint(new_path)
    assert new_payload["event_types"] == EVENT_TYPES
    assert loaded_new(torch.zeros(1, 60, 18)).shape == (1, 4)

    old_path = tmp_path / "five.pt"
    old_model = build_model("lstm", num_classes=5)
    save_checkpoint(
        old_path,
        old_model,
        "lstm",
        18,
        event_types=LEGACY_EVENT_TYPES,
    )
    loaded_old, old_payload = load_checkpoint(old_path)
    assert old_payload["event_types"] == LEGACY_EVENT_TYPES
    assert loaded_old(torch.zeros(1, 60, 18)).shape == (1, 5)


def test_multiscale_model_and_checkpoint_round_trip(tmp_path: Path) -> None:
    model = build_model(
        "multiscale", sequence_len=120, short_sequence_len=60
    )
    assert model(torch.zeros(2, 120, 18)).shape == (2, 4)
    assert model.short_branch.pos.shape[1] == 60
    assert model.long_branch.pos.shape[1] >= 120

    path = tmp_path / "multiscale.pt"
    save_checkpoint(
        path, model, "multiscale", 18,
        extra={"sequence_len": 120, "short_sequence_len": 60},
    )
    loaded, payload = load_checkpoint(path)
    assert payload["sequence_len"] == 120
    assert payload["short_sequence_len"] == 60
    assert loaded(torch.zeros(1, 120, 18)).shape == (1, 4)
