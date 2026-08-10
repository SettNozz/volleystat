"""
Datasets for the event-sequence classifier.

EventClipDataset  — trajectory(+pose) features only: (SEQUENCE_LEN, 18)
FusionClipDataset — adds MobileNetV3 clip features: (VIDEO_FEATURE_FRAMES, 576)

Backward compatibility: legacy Parquet files have 10 columns (trajectory
only); the pose block is zero-filled for them, so old datasets
(event_clips_merged_v*) train without any conversion.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from action_recognition.event_clips.config import (
    NUM_FEATURES, NUM_TRAJ_FEATURES, SEQUENCE_LEN,
    VIDEO_FEATURE_DIM, VIDEO_FEATURE_FRAMES, collapse_labels, label_index,
)


def load_manifest(manifest_path: str, labeled_only: bool = True) -> List[Dict]:
    with open(manifest_path) as f:
        clips = json.load(f)["clips"]
    if labeled_only:
        clips = [c for c in clips if c.get("event_types")]
    return clips


def _load_features(features_dir: Path, clip_id: str) -> Optional[np.ndarray]:
    """Load Parquet or lightweight NPZ; legacy 10-col data gets pose padding."""
    p = features_dir / f"{clip_id}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        arr = df.to_numpy(dtype=np.float32)
    else:
        p = features_dir / f"{clip_id}.npz"
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as payload:
            if "features" not in payload:
                raise ValueError(f"{p}: expected a 'features' array")
            arr = np.asarray(payload["features"], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{p}: expected a 2-D feature array, got {arr.shape}")
    if arr.shape[1] < NUM_FEATURES:  # legacy trajectory-only file
        pad = np.zeros((arr.shape[0], NUM_FEATURES - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :NUM_FEATURES]


def _fix_length(arr: np.ndarray, length: int = SEQUENCE_LEN) -> np.ndarray:
    """Keep the LAST `length` frames (contact is at the end); left-pad zeros."""
    if arr.shape[0] >= length:
        return arr[-length:]
    pad = np.zeros((length - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([pad, arr], axis=0)


class EventClipDataset(Dataset):
    """Trajectory(+pose) sequences → single-label class indices."""

    def __init__(self, clips: List[Dict], features_dir: str,
                 augment_noise: float = 0.0,
                 sequence_len: int = SEQUENCE_LEN):
        self.features_dir = Path(features_dir)
        self.augment_noise = augment_noise
        self.sequence_len = int(sequence_len)
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self.items = []
        self.excluded_non_core = 0
        for c in clips:
            if collapse_labels(c.get("event_types"), c.get("fsm_event_type")) is None:
                self.excluded_non_core += 1
                continue
            feats = _load_features(self.features_dir, c["clip_id"])
            if feats is None or feats.shape[0] < 5:
                continue
            self.items.append((c, feats))

    def __len__(self):
        return len(self.items)

    # --- Per-channel normalization -----------------------------------
    # Raw channels mix pixels (0..4000), velocities (±600) and 0/1 flags,
    # and different source videos have different resolutions — without
    # z-scoring the model barely learns.

    def feature_stats(self):
        """(mean, std) per channel over all frames of this dataset."""
        X = np.concatenate([self._prepare_sequence(f) for _, f in self.items], axis=0)
        mean = X.mean(axis=0).astype(np.float32)
        std = X.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return mean, std

    def feature_active_mask(self, epsilon: float = 1e-6) -> np.ndarray:
        """Channels with real variation in this training split."""
        X = np.concatenate([self._prepare_sequence(f) for _, f in self.items], axis=0)
        return X.std(axis=0) >= float(epsilon)

    def _prepare_sequence(self, feats: np.ndarray) -> np.ndarray:
        """Create the same causal window representation used in production."""
        x = _fix_length(feats, self.sequence_len).copy()
        x[:, 7] = np.linspace(0.0, 1.0, self.sequence_len, dtype=np.float32)
        return x

    def set_normalization(self, mean, std) -> None:
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)

    def __getitem__(self, idx):
        clip, feats = self.items[idx]
        x = self._prepare_sequence(feats)
        if self.augment_noise > 0:
            x[:, :NUM_TRAJ_FEATURES] *= (
                1.0 + np.random.randn(*x[:, :NUM_TRAJ_FEATURES].shape).astype(np.float32)
                * self.augment_noise
            )
        if self._mean is not None:
            x = (x - self._mean) / self._std
        y = label_index(clip["event_types"], clip.get("fsm_event_type"))
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    def class_indices(self) -> np.ndarray:
        """Collapsed class index per item (for class weights / stratification)."""
        return np.asarray([label_index(c["event_types"], c.get("fsm_event_type"))
                           for c, _ in self.items])


class FusionClipDataset(EventClipDataset):
    """Adds per-clip CNN features (MobileNetV3-Small, 16×576)."""

    def __init__(self, clips: List[Dict], features_dir: str,
                 video_features_dir: str, augment_noise: float = 0.0,
                 sequence_len: int = SEQUENCE_LEN):
        super().__init__(clips, features_dir, augment_noise, sequence_len)
        self.video_features_dir = Path(video_features_dir)
        # The complete v6 image feature block is only ~38 MB. Reading one
        # Parquet per sample on every epoch leaves the GPU waiting on disk, so
        # normalize shapes once and retain the arrays in memory.
        self.video_features: List[np.ndarray] = []
        for clip, _ in self.items:
            vp = self.video_features_dir / f"{clip['clip_id']}.parquet"
            if vp.exists():
                values = pd.read_parquet(vp).to_numpy(dtype=np.float32)
            else:
                values = np.zeros(
                    (VIDEO_FEATURE_FRAMES, VIDEO_FEATURE_DIM), dtype=np.float32
                )
            if values.shape[0] != VIDEO_FEATURE_FRAMES:
                idxs = np.linspace(
                    0, values.shape[0] - 1, VIDEO_FEATURE_FRAMES
                ).astype(int)
                values = values[idxs]
            self.video_features.append(values)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x, torch.from_numpy(self.video_features[idx]), y


def split_by_stem(clips: List[Dict], val_stems=("3",)):
    """Legacy split: validation = whole source videos (no clip leakage)."""
    train = [c for c in clips if c.get("source_stem") not in val_stems]
    val = [c for c in clips if c.get("source_stem") in val_stems]
    return train, val


# (single-label v3 taxonomy: targets are class indices — see config.py)
