"""Versioned trajectory-feature schemas shared by training and inference.

``legacy_pixels_v1`` preserves the original raw-pixel representation for old
datasets and checkpoints. ``resolution_normalized_v2`` removes the input
resolution from all geometric channels:

* positions are fractions of frame width/height;
* distances are measured in normalized image coordinates;
* velocity is normalized-image-units per second.

The feature order stays unchanged so the model input dimension remains 18.
Schema metadata, rather than column count, selects the semantics.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from action_recognition.event_clips.config import TRAJ_FEATURE_NAMES

LEGACY_PIXEL_SCHEMA = "legacy_pixels_v1"
RESOLUTION_NORMALIZED_SCHEMA = "resolution_normalized_v2"
FEATURE_SCHEMAS = (LEGACY_PIXEL_SCHEMA, RESOLUTION_NORMALIZED_SCHEMA)


def validate_feature_schema(feature_schema: Optional[str]) -> str:
    """Return a known schema; missing metadata means legacy compatibility."""
    schema = feature_schema or LEGACY_PIXEL_SCHEMA
    if schema not in FEATURE_SCHEMAS:
        raise ValueError(
            f"Unsupported feature schema {schema!r}; expected one of "
            f"{', '.join(FEATURE_SCHEMAS)}"
        )
    return schema


def validate_frame_size(
    frame_size: Optional[Tuple[float, float]],
    feature_schema: str,
) -> Optional[Tuple[float, float]]:
    """Validate ``(width, height)`` when normalized geometry is requested."""
    schema = validate_feature_schema(feature_schema)
    if schema == LEGACY_PIXEL_SCHEMA:
        return frame_size
    if frame_size is None or len(frame_size) != 2:
        raise ValueError(
            f"{RESOLUTION_NORMALIZED_SCHEMA} requires frame_size=(width, height)"
        )
    width, height = float(frame_size[0]), float(frame_size[1])
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame_size={frame_size!r}; dimensions must be positive")
    return width, height


def build_trajectory_features(
    ball: Optional[Sequence[float]],
    previous_ball: Optional[Sequence[float]],
    players: Sequence[Sequence[float]],
    net_y: float,
    fps: float,
    frame_norm: float,
    feature_schema: str = LEGACY_PIXEL_SCHEMA,
    frame_size: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Build one trajectory row in the fixed ``TRAJ_FEATURE_NAMES`` order."""
    schema = validate_feature_schema(feature_schema)
    size = validate_frame_size(frame_size, schema)

    bx, by = (float(ball[0]), float(ball[1])) if ball is not None else (0.0, 0.0)
    if schema == RESOLUTION_NORMALIZED_SCHEMA:
        width, height = size
        x_scale, y_scale = width, height
    else:
        x_scale = y_scale = 1.0

    if ball is not None and previous_ball is not None:
        dx = (bx - float(previous_ball[0])) / x_scale
        dy = (by - float(previous_ball[1])) / y_scale
        speed = float(np.hypot(dx, dy)) * float(fps)
        direction = float(np.arctan2(dy, dx))
        vy = dy * float(fps)
    else:
        speed = direction = vy = 0.0

    if players and ball is not None:
        player_dist = min(
            float(np.hypot((bx - float(px)) / x_scale, (by - float(py)) / y_scale))
            for px, py in players
        )
    else:
        player_dist = 0.0

    values = [
        bx / x_scale,
        by / y_scale,
        speed,
        direction,
        abs(by - float(net_y)) / y_scale if ball is not None else 0.0,
        player_dist,
        0.0 if by < float(net_y) else 1.0,
        float(frame_norm),
        vy,
        1.0 if (ball is not None and by < float(net_y)) else 0.0,
    ]
    return np.asarray(values, dtype=np.float32)


def apply_checkpoint_feature_contract(
    features: np.ndarray,
    feature_mean: Optional[Sequence[float]],
    feature_std: Optional[Sequence[float]],
    feature_active_mask: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    """Apply the exact feature contract stored by training.

    A channel that was constant in the training split is inactive. Replacing
    it with its training mean before z-scoring prevents live-only signals
    (for example pose features absent from older gold annotations) from
    becoming an out-of-distribution input to the model.
    """
    values = np.asarray(features, dtype=np.float32).copy()
    if feature_mean is None or feature_std is None:
        return values
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(mean) or mean.shape != std.shape:
        raise ValueError(
            "Feature/checkpoint shape mismatch: "
            f"features={values.shape}, mean={mean.shape}, std={std.shape}"
        )
    if np.any(std <= 0):
        raise ValueError("Checkpoint feature_std must be strictly positive")
    if feature_active_mask is not None:
        active = np.asarray(feature_active_mask, dtype=bool)
        if active.shape != mean.shape:
            raise ValueError(
                "feature_active_mask shape mismatch: "
                f"mask={active.shape}, expected={mean.shape}"
            )
        values[:, ~active] = mean[~active]
    return (values - mean) / std


assert len(TRAJ_FEATURE_NAMES) == 10
