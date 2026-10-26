"""
Sliding-window event prediction for the live overlay.

API (used by GameAnalyzer):
    predictor = StreamingEventPredictor(checkpoint_path)
    label = predictor.add_frame(ball_pos, player_teams, net_y, fps,
                                fsm_state="rally_active",
                                pose_features=[...],   # optional, 8 values
                                frame_size=(width, height))

Keeps the last SEQUENCE_LEN frames of features, runs the model every
INFER_EVERY frames, smooths by mode over the last SMOOTH_WINDOW predictions,
suppresses "serve" during an active rally and low-confidence outputs.
"""

import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.event_clips.config import (
    EVENT_TYPES, NUM_FEATURES, NUM_POSE_FEATURES, NUM_TRAJ_FEATURES, SEQUENCE_LEN,
)
from action_recognition.event_clips.feature_schema import (
    LEGACY_PIXEL_SCHEMA,
    apply_checkpoint_feature_contract,
    build_trajectory_features,
    validate_feature_schema,
)
from action_recognition.models.sequence_event_model import load_checkpoint


def gate_event_probabilities(
    probabilities: np.ndarray,
    event_types: List[str],
    fsm_state: str,
) -> np.ndarray:
    """Condition predictions on the volleyball phase and renormalize them."""
    probs = np.asarray(probabilities, dtype=np.float32).copy()
    allowed_by_state = {
        "serving": {"serve", "reception"},
        "rally_active": {"pass", "attack"},
        "serve_prep": set(),
        "serve_ready": set(),
        "rally_end": set(),
    }
    allowed = allowed_by_state.get(fsm_state)
    if allowed is None:
        return probs
    for index, label in enumerate(event_types):
        if label not in allowed:
            probs[index] = 0.0
    total = float(probs.sum())
    if total > 0.0:
        probs /= total
    return probs


class StreamingEventPredictor:
    INFER_EVERY = 2
    SMOOTH_WINDOW = 5
    MIN_CONFIDENCE = 0.35

    def __init__(self, checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, ckpt = load_checkpoint(checkpoint_path, self.device)
        self.event_types = list(ckpt["event_types"])
        self.sequence_len = int(ckpt.get("sequence_len", SEQUENCE_LEN))
        if ckpt.get("model_type") == "dual":
            raise ValueError("Dual-stream checkpoints need video features — "
                             "use an lstm/transformer checkpoint for streaming")
        self.model.to(self.device)

        # Normalization stats stored by the training script
        self._mean = np.asarray(ckpt["feature_mean"], dtype=np.float32) \
            if ckpt.get("feature_mean") else None
        self._std = np.asarray(ckpt["feature_std"], dtype=np.float32) \
            if ckpt.get("feature_std") else None
        self._active_mask = ckpt.get("feature_active_mask")
        self.feature_schema = validate_feature_schema(
            ckpt.get("feature_schema", LEGACY_PIXEL_SCHEMA)
        )

        self._buf: deque = deque(maxlen=self.sequence_len)
        self._recent: deque = deque(maxlen=self.SMOOTH_WINDOW)
        self._frame_count = 0
        self._prev_ball: Optional[Tuple[float, float]] = None
        self._last_fsm_state = ""

    def reset(self):
        self._buf.clear()
        self._recent.clear()
        self._frame_count = 0
        self._prev_ball = None
        self._last_fsm_state = ""

    # ------------------------------------------------------------------

    def add_frame(
        self,
        ball_pos: Optional[Tuple[float, float]],
        player_teams: Dict[str, List[Tuple[float, float]]],
        net_y: float,
        fps: float,
        fsm_state: str = "",
        pose_features: Optional[List[float]] = None,
        frame_size: Optional[Tuple[float, float]] = None,
        ball_new: bool = False,
    ) -> Optional[str]:
        sequence_len = int(
            getattr(self, "sequence_len", self._buf.maxlen or SEQUENCE_LEN)
        )
        ball = ball_pos
        bx, by = ball if ball else (0.0, 0.0)
        if ball_new:
            # A new candidate identity is not a one-frame ball flight.
            self._prev_ball = None

        players = (player_teams.get("team_a", []) + player_teams.get("team_b", []))

        feats = np.zeros(NUM_FEATURES, dtype=np.float32)
        feats[:NUM_TRAJ_FEATURES] = build_trajectory_features(
            ball,
            self._prev_ball,
            players,
            net_y,
            fps,
            0.0,                                   # frame_norm — set per window below
            self.feature_schema,
            frame_size,
        )
        if pose_features and len(pose_features) == NUM_POSE_FEATURES:
            feats[NUM_TRAJ_FEATURES:] = pose_features

        self._buf.append(feats)
        if ball:
            self._prev_ball = (bx, by)
        else:
            self._prev_ball = None
        self._frame_count += 1

        if fsm_state != self._last_fsm_state:
            self._recent.clear()
            self._last_fsm_state = fsm_state

        if len(self._buf) < sequence_len // 2:
            return None
        if self._frame_count % self.INFER_EVERY != 0:
            return self._smoothed()

        x = np.stack(self._buf)
        if x.shape[0] < sequence_len:                     # left-pad zeros
            pad = np.zeros((sequence_len - x.shape[0], NUM_FEATURES), np.float32)
            x = np.concatenate([pad, x], axis=0)
        x[:, 7] = np.linspace(0.0, 1.0, sequence_len)     # frame_norm in window
        x = apply_checkpoint_feature_contract(
            x, self._mean, self._std, self._active_mask,
        )

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x[None]).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        probs = gate_event_probabilities(probs, self.event_types, fsm_state)

        k = int(np.argmax(probs))
        label = self.event_types[k]
        if probs[k] < self.MIN_CONFIDENCE or label == "other":
            self._recent.append(None)
        else:
            self._recent.append(label)
        return self._smoothed()

    def _smoothed(self) -> Optional[str]:
        votes = [v for v in self._recent if v is not None]
        if not votes:
            return None
        label, n = Counter(votes).most_common(1)[0]
        return label if n >= max(2, len(self._recent) // 2) else None
