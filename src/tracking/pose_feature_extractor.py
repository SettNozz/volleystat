"""
Pose-derived per-frame features for the event model.

Combines player keypoints with the ball position to produce signals that
distinguish contact types (attack = jump + wrist above net near ball,
reception = low arms platform near ball, block = wrists above net at the
net line, serve = one player behind the court with a toss, ...).

Features are computed per frame and are designed to be appended to the
existing trajectory features (ball_x, ball_y, ball_speed, ...) at
dataset-preparation time. Order is fixed by POSE_FEATURE_NAMES.

Jump detection uses a per-track-ID ankle-height baseline (rolling median of
recent grounded positions), so ByteTrack IDs from the pose model are required
for accurate values; without IDs it degrades gracefully to zeros.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.player_pose import (
    PlayerPose, L_SHOULDER, R_SHOULDER,
)

# Fixed feature order — keep stable, dataset prep depends on it.
POSE_FEATURE_NAMES: List[str] = [
    "np_wrist_ball_dist",    # nearest player: min wrist↔ball distance / bbox_h (0 if no ball/wrist)
    "np_wrist_above_net",    # nearest player: (net_y - highest wrist y) / bbox_h, clipped to [-2, 2]
    "np_jump",               # nearest player: ankle rise vs baseline / bbox_h, [0, 1.5]
    "np_arm_raise",          # nearest player: (shoulder_y - highest wrist y) / bbox_h, [-1, 2]
    "np_team_side",          # nearest player team: 0 = team_a (far), 1 = team_b (near)
    "np_dist_to_net",        # nearest player: |center_y - net_y| / bbox_h, clipped to [0, 5]
    "b_max_jump",            # observed team (team_b): max jump over all players
    "b_max_wrist_above_net", # observed team (team_b): max wrist-above-net over all players
]

N_POSE_FEATURES = len(POSE_FEATURE_NAMES)


class PoseFeatureExtractor:
    """
    Stateful per-video extractor. Call ``update()`` once per processed frame.

    History: for each track ID keeps recent ankle heights; the baseline
    (standing level) is the rolling 80th percentile of ankle_y (screen y grows
    downward, so standing = larger y). Jump = (baseline - current) / bbox_h.
    """

    HISTORY_LEN = 90        # ~3 s at 30 fps
    MIN_HISTORY = 10        # frames needed before jump is trusted
    STALE_FRAMES = 150      # drop track history if unseen this long

    def __init__(self):
        self._ankle_hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.HISTORY_LEN))
        self._last_seen: Dict[int, int] = {}

    def reset(self) -> None:
        self._ankle_hist.clear()
        self._last_seen.clear()

    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        poses: List[PlayerPose],
        ball_pos: Optional[Tuple[float, float]],
        net_y: float,
    ) -> List[float]:
        """Returns feature values in POSE_FEATURE_NAMES order."""
        self._update_history(frame_idx, poses)

        feats = dict.fromkeys(POSE_FEATURE_NAMES, 0.0)

        if poses:
            # --- Nearest player to ball (falls back to nearest-to-net if no ball) ---
            if ball_pos is not None:
                nearest = min(poses, key=lambda p: _dist(p.center, ball_pos))
            else:
                nearest = min(poses, key=lambda p: abs(p.center[1] - net_y))

            h = nearest.bbox_h
            feats["np_team_side"] = 1.0 if nearest.team == "team_b" else 0.0
            feats["np_dist_to_net"] = _clip(abs(nearest.center[1] - net_y) / h, 0.0, 5.0)

            wrists = nearest.wrists()
            if wrists:
                top_wrist_y = min(w[1] for w in wrists)
                feats["np_wrist_above_net"] = _clip((net_y - top_wrist_y) / h, -2.0, 2.0)
                shoulder_y = _shoulder_y(nearest)
                if shoulder_y is not None:
                    feats["np_arm_raise"] = _clip((shoulder_y - top_wrist_y) / h, -1.0, 2.0)
                if ball_pos is not None:
                    d = min(_dist(w, ball_pos) for w in wrists)
                    feats["np_wrist_ball_dist"] = _clip(d / h, 0.0, 10.0)

            feats["np_jump"] = self._jump(nearest)

            # --- Observed team (team_b, near side) aggregates ---
            team_b = [p for p in poses if p.team == "team_b"]
            if team_b:
                feats["b_max_jump"] = max(self._jump(p) for p in team_b)
                wans = []
                for p in team_b:
                    ws = p.wrists()
                    if ws:
                        wans.append(_clip((net_y - min(w[1] for w in ws)) / p.bbox_h, -2.0, 2.0))
                if wans:
                    feats["b_max_wrist_above_net"] = max(wans)

        return [round(float(feats[k]), 4) for k in POSE_FEATURE_NAMES]

    # ------------------------------------------------------------------

    def _update_history(self, frame_idx: int, poses: List[PlayerPose]) -> None:
        for p in poses:
            if p.track_id is None:
                continue
            ankles = p.ankles()
            if not ankles:
                continue
            ankle_y = max(a[1] for a in ankles)  # lower ankle = ground reference
            self._ankle_hist[p.track_id].append(ankle_y)
            self._last_seen[p.track_id] = frame_idx

        # Purge stale tracks
        stale = [tid for tid, last in self._last_seen.items()
                 if frame_idx - last > self.STALE_FRAMES]
        for tid in stale:
            self._ankle_hist.pop(tid, None)
            self._last_seen.pop(tid, None)

    def _jump(self, pose: PlayerPose) -> float:
        """Ankle rise above rolling baseline, normalized by bbox height."""
        if pose.track_id is None:
            return 0.0
        hist = self._ankle_hist.get(pose.track_id)
        if hist is None or len(hist) < self.MIN_HISTORY:
            return 0.0
        ankles = pose.ankles()
        if not ankles:
            return 0.0
        baseline = float(np.percentile(list(hist), 80))  # standing level (largest y)
        current = max(a[1] for a in ankles)
        return _clip((baseline - current) / pose.bbox_h, 0.0, 1.5)


# ----------------------------------------------------------------------

def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _shoulder_y(pose: PlayerPose) -> Optional[float]:
    pts = [p for p in (pose.kp(L_SHOULDER), pose.kp(R_SHOULDER)) if p is not None]
    return min(p[1] for p in pts) if pts else None
