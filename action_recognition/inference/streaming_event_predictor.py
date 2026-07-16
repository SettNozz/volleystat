"""
Sliding-window event prediction for the live overlay.

API (used by GameAnalyzer):
    predictor = StreamingEventPredictor(checkpoint_path)
    label = predictor.add_frame(ball_pos, player_teams, net_y, fps,
                                fsm_state="rally_active",
                                pose_features=[...])   # optional, 8 values

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
from action_recognition.models.sequence_event_model import load_checkpoint


class StreamingEventPredictor:
    INFER_EVERY = 2
    SMOOTH_WINDOW = 5
    MIN_CONFIDENCE = 0.35

    def __init__(self, checkpoint_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, ckpt = load_checkpoint(checkpoint_path, self.device)
        if ckpt.get("model_type") == "dual":
            raise ValueError("Dual-stream checkpoints need video features — "
                             "use an lstm/transformer checkpoint for streaming")
        self.model.to(self.device)

        # Normalization stats stored by the training script
        self._mean = np.asarray(ckpt["feature_mean"], dtype=np.float32) \
            if ckpt.get("feature_mean") else None
        self._std = np.asarray(ckpt["feature_std"], dtype=np.float32) \
            if ckpt.get("feature_std") else None

        self._buf: deque = deque(maxlen=SEQUENCE_LEN)
        self._recent: deque = deque(maxlen=self.SMOOTH_WINDOW)
        self._frame_count = 0
        self._prev_ball: Optional[Tuple[float, float]] = None
        self._last_ball: Optional[Tuple[float, float]] = None

    def reset(self):
        self._buf.clear()
        self._recent.clear()
        self._frame_count = 0
        self._prev_ball = self._last_ball = None

    # ------------------------------------------------------------------

    def add_frame(
        self,
        ball_pos: Optional[Tuple[float, float]],
        player_teams: Dict[str, List[Tuple[float, float]]],
        net_y: float,
        fps: float,
        fsm_state: str = "",
        pose_features: Optional[List[float]] = None,
    ) -> Optional[str]:
        ball = ball_pos or self._last_ball
        bx, by = ball if ball else (0.0, 0.0)

        if ball and self._prev_ball:
            dx, dy = bx - self._prev_ball[0], by - self._prev_ball[1]
            speed = float(np.hypot(dx, dy)) * fps
            direction = float(np.arctan2(dy, dx))
            vy = dy * fps
        else:
            speed = direction = vy = 0.0

        players = (player_teams.get("team_a", []) + player_teams.get("team_b", []))
        if players and ball:
            player_dist = min(float(np.hypot(bx - px, by - py)) for px, py in players)
        else:
            player_dist = 0.0

        feats = np.zeros(NUM_FEATURES, dtype=np.float32)
        feats[:NUM_TRAJ_FEATURES] = [
            bx, by, speed, direction,
            abs(by - net_y) if ball else 0.0,
            player_dist,
            0.0 if by < net_y else 1.0,
            0.0,                                   # frame_norm — set per window below
            vy,
            1.0 if (ball and by < net_y) else 0.0,
        ]
        if pose_features and len(pose_features) == NUM_POSE_FEATURES:
            feats[NUM_TRAJ_FEATURES:] = pose_features

        self._buf.append(feats)
        if ball:
            self._prev_ball = (bx, by)
            self._last_ball = (bx, by)
        self._frame_count += 1

        if len(self._buf) < SEQUENCE_LEN // 2:
            return None
        if self._frame_count % self.INFER_EVERY != 0:
            return self._smoothed()

        x = np.stack(self._buf)
        if x.shape[0] < SEQUENCE_LEN:                     # left-pad zeros
            pad = np.zeros((SEQUENCE_LEN - x.shape[0], NUM_FEATURES), np.float32)
            x = np.concatenate([pad, x], axis=0)
        x[:, 7] = np.linspace(0.0, 1.0, SEQUENCE_LEN)     # frame_norm in window
        if self._mean is not None:
            x = (x - self._mean) / self._std

        with torch.no_grad():
            logits = self.model(torch.from_numpy(x[None]).to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Exclude impossible classes given FSM state
        if fsm_state == "rally_active":
            probs[EVENT_TYPES.index("serve")] = 0.0

        k = int(np.argmax(probs))
        label = EVENT_TYPES[k]
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
