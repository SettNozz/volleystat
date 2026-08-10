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
from action_recognition.event_clips.feature_schema import (
    LEGACY_PIXEL_SCHEMA,
    build_trajectory_features,
    validate_feature_schema,
)


def compute_clip_features(
    frames: List[Dict],
    net_y: float,
    fps: float,
    *,
    frame_width: Optional[float] = None,
    frame_height: Optional[float] = None,
    feature_schema: str = LEGACY_PIXEL_SCHEMA,
) -> pd.DataFrame:
    """
    Build the (T, 18) feature table for one clip segment.

    Args:
        frames: annotation dicts for the clip's frames, chronological
                (keys: "ball", "p", optionally "pose_f")
        net_y:  net reference in the same pixel space as the annotations
        fps:    source video fps
        frame_width/frame_height: required for resolution-normalized features
        feature_schema: versioned geometry semantics. The legacy default keeps
                        old callers and datasets compatible.
    """
    schema = validate_feature_schema(feature_schema)
    frame_size = (
        (frame_width, frame_height)
        if frame_width is not None and frame_height is not None
        else None
    )
    T = len(frames)
    out = np.zeros((T, len(ALL_FEATURE_NAMES)), dtype=np.float32)

    last_ball: Optional[List[float]] = None
    prev_ball: Optional[List[float]] = None

    for i, ann in enumerate(frames):
        ball = ann.get("ball") or last_ball  # carry last known position
        players = ann.get("p") or []

        bx, by = (ball[0], ball[1]) if ball else (0.0, 0.0)
        traj = build_trajectory_features(
            ball=ball,
            previous_ball=prev_ball,
            players=players,
            net_y=net_y,
            fps=fps,
            frame_norm=i / max(T - 1, 1),
            feature_schema=schema,
            frame_size=frame_size,
        )

        pose = ann.get("pose_f")
        if not pose or len(pose) != NUM_POSE_FEATURES:
            pose = [0.0] * NUM_POSE_FEATURES

        out[i, :len(TRAJ_FEATURE_NAMES)] = traj
        out[i, len(TRAJ_FEATURE_NAMES):] = pose

        if ball:
            prev_ball = [bx, by]
            last_ball = [bx, by]

    result = pd.DataFrame(out, columns=ALL_FEATURE_NAMES)
    result.attrs["feature_schema"] = schema
    if frame_size is not None:
        result.attrs["frame_width"] = float(frame_size[0])
        result.attrs["frame_height"] = float(frame_size[1])
    return result
