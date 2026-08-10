from pathlib import Path

import numpy as np
import pandas as pd

from action_recognition.data.event_clip_dataset import FusionClipDataset
from action_recognition.event_clips.config import (
    NUM_FEATURES,
    VIDEO_FEATURE_DIM,
    VIDEO_FEATURE_FRAMES,
)


def test_fusion_features_are_cached_in_memory(tmp_path: Path) -> None:
    clip_id = "sample_f10"
    trajectory_dir = tmp_path / "features"
    video_dir = tmp_path / "video_features"
    trajectory_dir.mkdir()
    video_dir.mkdir()

    pd.DataFrame(np.ones((10, NUM_FEATURES), dtype=np.float32)).to_parquet(
        trajectory_dir / f"{clip_id}.parquet", index=False
    )
    expected = np.arange(
        VIDEO_FEATURE_FRAMES * VIDEO_FEATURE_DIM, dtype=np.float32
    ).reshape(VIDEO_FEATURE_FRAMES, VIDEO_FEATURE_DIM)
    video_path = video_dir / f"{clip_id}.parquet"
    pd.DataFrame(expected).to_parquet(video_path, index=False)

    dataset = FusionClipDataset(
        [{"clip_id": clip_id, "event_types": ["serve"]}],
        str(trajectory_dir),
        str(video_dir),
    )
    video_path.unlink()

    _, video_features, _ = dataset[0]
    np.testing.assert_array_equal(video_features.numpy(), expected)
