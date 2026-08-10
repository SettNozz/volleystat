"""
Shared configuration for the event-clip pipeline (rebuilt, pose-aware).

Feature schema
--------------
Trajectory features (10) — identical names/order to the legacy datasets
(event_clips_merged_v*), so old Parquet files remain loadable as-is:

    ball_x, ball_y, ball_speed, ball_dir, ball_net_dist,
    player_dist, team_side, frame_norm, ball_vy, ball_above_net

Pose features (8) — new channels from src/tracking/pose_feature_extractor
(same values GameAnalyzer writes to *_annotations.json as "pose_f"):

    np_wrist_ball_dist, np_wrist_above_net, np_jump, np_arm_raise,
    np_team_side, np_dist_to_net, b_max_jump, b_max_wrist_above_net

New Parquet files contain all 18 columns; the dataset loader zero-fills the
pose block for legacy 10-column files.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.tracking.pose_feature_extractor import POSE_FEATURE_NAMES  # noqa: E402

# Core single-label taxonomy v4 (order defines the class index).
# Raw ``other`` annotations remain available, but are not a model class.
EVENT_TYPES = ["serve", "reception", "pass", "attack"]
LEGACY_EVENT_TYPES = EVENT_TYPES + ["other"]
NUM_CLASSES = len(EVENT_TYPES)

# Legacy/raw labels -> v4 core-action collapse.
# dig = defensive first touch in rally -> pass; block = net action -> attack.
LEGACY_TO_V4 = {
    "serve": "serve",
    "reception": "reception",
    "pass": "pass",
    "set": "pass",
    "dig": "pass",
    "attack": "attack",
    "block": "attack",
    "free_ball": None,
    "other": None,
}
LEGACY_TO_V3 = LEGACY_TO_V4

# When an old multi-label set collapses to several classes, prefer the label
# of the clip's final contact (fsm hint), else this priority.
_PRIORITY = ["attack", "serve", "reception", "pass"]


def collapse_labels(event_types, fsm_event_type=None):
    """Old labels -> one core v4 label, or ``None`` for non-core contacts."""
    mapped = {LEGACY_TO_V4.get(l) for l in (event_types or []) if LEGACY_TO_V4.get(l)}
    if not mapped:
        return None
    if len(mapped) == 1:
        return next(iter(mapped))
    fsm_mapped = LEGACY_TO_V4.get(fsm_event_type or "")
    if fsm_mapped in mapped:
        return fsm_mapped
    for label in _PRIORITY:
        if label in mapped:
            return label
    return None


def label_index(event_types, fsm_event_type=None) -> int:
    """Class index for training (CrossEntropy target)."""
    label = collapse_labels(event_types, fsm_event_type)
    if label is None:
        raise ValueError(
            "No core action label (serve/reception/pass/attack) for this clip"
        )
    return EVENT_TYPES.index(label)

TRAJ_FEATURE_NAMES = [
    "ball_x", "ball_y", "ball_speed", "ball_dir", "ball_net_dist",
    "player_dist", "team_side", "frame_norm", "ball_vy", "ball_above_net",
]

ALL_FEATURE_NAMES = TRAJ_FEATURE_NAMES + list(POSE_FEATURE_NAMES)

NUM_TRAJ_FEATURES = len(TRAJ_FEATURE_NAMES)     # 10
NUM_POSE_FEATURES = len(POSE_FEATURE_NAMES)     # 8
NUM_FEATURES = len(ALL_FEATURE_NAMES)           # 18

# Fixed sequence length at train time (pad/trim clip features)
SEQUENCE_LEN = 60

# CNN video features (MobileNetV3-Small): frames x dims per clip
VIDEO_FEATURE_FRAMES = 16
VIDEO_FEATURE_DIM = 576

def labels_to_vector(event_types, fsm_event_type=None):
    """One-hot core-action vector; non-core contacts produce all zeros."""
    vec = [0.0] * NUM_CLASSES
    label = collapse_labels(event_types, fsm_event_type)
    if label is not None:
        vec[EVENT_TYPES.index(label)] = 1.0
    return vec
