from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

import action_recognition.event_clips.extract_video_features as extractor
from action_recognition.event_clips.extract_video_features import (
    extract_manifest,
    read_sampled_frames,
    resolve_clip_source,
)
from action_recognition.event_clips.config import VIDEO_FEATURE_DIM, VIDEO_FEATURE_FRAMES


class _FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return len(self.frames)

    def set(self, prop, value):
        self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def release(self):
        self.released = True


def test_resolve_source_video_interval_without_cut_clip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "game.webm"
    source.touch()
    clip = {
        "clip_id": "game_f120",
        "source_video": "game.webm",
        "start_frame": 60,
        "end_frame": 120,
    }

    assert resolve_clip_source(clip, tmp_path) == (source.resolve(), 60, 120)


def test_existing_cut_clip_takes_precedence(tmp_path):
    source = tmp_path / "source.webm"
    cut = tmp_path / "clip.mp4"
    source.touch()
    cut.touch()
    clip = {
        "clip_id": "x",
        "path": str(cut),
        "source_video": str(source),
        "start_frame": 1,
        "end_frame": 2,
    }

    assert resolve_clip_source(clip, tmp_path) == (cut.resolve(), None, None)


def test_source_video_requires_frame_interval(tmp_path):
    source = tmp_path / "source.webm"
    source.touch()
    with pytest.raises(ValueError, match="start_frame/end_frame"):
        resolve_clip_source({"clip_id": "x", "source_video": str(source)}, tmp_path)


def test_read_sampled_frames_uses_requested_source_interval(tmp_path):
    # BGR pixel values encode the frame number. RGB conversion preserves the
    # value because all three channels are equal.
    raw = [np.full((2, 3, 3), i, dtype=np.uint8) for i in range(10)]
    fake = _FakeCapture(raw)

    frames = read_sampled_frames(
        tmp_path / "unused.webm",
        start_frame=2,
        end_frame=8,
        count=4,
        capture_factory=lambda _: fake,
    )

    assert [int(frame[0, 0, 0]) for frame in frames] == [2, 4, 6, 8]
    assert fake.released


def test_extract_manifest_commits_parquet_atomically(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"clip_id": "clip_1"}]}))
    output = tmp_path / "video_features"

    monkeypatch.setattr(extractor, "build_encoder", lambda device: (object(), object()))
    monkeypatch.setattr(
        extractor, "resolve_clip_source", lambda clip, manifest_dir: (tmp_path / "x", 0, 1)
    )
    monkeypatch.setattr(
        extractor,
        "read_sampled_frames",
        lambda *args, **kwargs: [np.zeros((2, 2, 3), np.uint8)] * VIDEO_FEATURE_FRAMES,
    )
    monkeypatch.setattr(
        extractor,
        "encode_frames",
        lambda *args, **kwargs: np.ones(
            (VIDEO_FEATURE_FRAMES, VIDEO_FEATURE_DIM), dtype=np.float32
        ),
    )

    assert extract_manifest(manifest, output, "cpu") == (1, 0, 0)
    assert pd.read_parquet(output / "clip_1.parquet").shape == (
        VIDEO_FEATURE_FRAMES,
        VIDEO_FEATURE_DIM,
    )
    assert not list(output.glob(".*.tmp"))
