"""
Per-frame feature computation from *_annotations.json produced by GameAnalyzer.

Produces (T, 18) feature tables: 10 trajectory columns (legacy-compatible
names/semantics) + 8 pose columns (taken directly from the "pose_f" field
GameAnalyzer stores per frame; zeros when a frame has none — e.g. results
produced before the pose upgrade).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from action_recognition.event_clips.config import (
    ALL_FEATURE_NAMES, NUM_POSE_FEATURES, TRAJ_FEATURE_NAMES,
)


def compute_clip_features(
    frames: List[Dict],
    net_y: float,
    fps: float,
) -> pd.DataFrame:
    """
    Build the (T, 18) feature table for one clip segment.

    Args:
        frames: annotation dicts for the clip's frames, chronological
                (keys: "ball", "p", optionally "pose_f")
        net_y:  net reference in the same pixel space as the annotations
        fps:    source video fps (speeds are px/second)
    """
    T = len(frames)
    out = np.zeros((T, len(ALL_FEATURE_NAMES)), dtype=np.float32)

    last_ball: Optional[List[float]] = None
    prev_ball: Optional[List[float]] = None

    for i, ann in enumerate(frames):
        ball = ann.get("ball") or last_ball  # carry last known position
        players = ann.get("p") or []

        bx, by = (ball[0], ball[1]) if ball else (0.0, 0.0)
        if ball and prev_ball:
            dx, dy = bx - prev_ball[0], by - prev_ball[1]
            speed = float(np.hypot(dx, dy)) * fps
            direction = float(np.arctan2(dy, dx))
            vy = dy * fps
        else:
            speed = direction = vy = 0.0

        if players and ball:
            player_dist = min(float(np.hypot(bx - px, by - py)) for px, py in players)
        else:
            player_dist = 0.0

        traj = [
            bx,                                          # ball_x
            by,                                          # ball_y
            speed,                                       # ball_speed
            direction,                                   # ball_dir
            abs(by - net_y) if ball else 0.0,            # ball_net_dist
            player_dist,                                 # player_dist
            0.0 if by < net_y else 1.0,                  # team_side
            i / max(T - 1, 1),                           # frame_norm
            vy,                                          # ball_vy
            1.0 if (ball and by < net_y) else 0.0,       # ball_above_net
        ]

        pose = ann.get("pose_f")
        if not pose or len(pose) != NUM_POSE_FEATURES:
            pose = [0.0] * NUM_POSE_FEATURES

        out[i, :len(TRAJ_FEATURE_NAMES)] = traj
        out[i, len(TRAJ_FEATURE_NAMES):] = pose

        if ball:
            prev_ball = [bx, by]
            last_ball = [bx, by]

    return pd.DataFrame(out, columns=ALL_FEATURE_NAMES)
