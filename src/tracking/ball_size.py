"""Observed ball box size carried next to the ball centre.

``ball_size`` on a frame annotation is the longer side, in source pixels, of
the detector box that became the selected ball on that frame. It is a
measurement of the observed ball only. A bridged, interpolated or
reconstructed position has no box, so it never carries a size: a made-up
diameter would read as a depth measurement downstream.
"""

from typing import Any, Dict, Iterable, Optional


def observed_ball_size(value: Optional[float]) -> Optional[float]:
    """Rounded positive size, or None when nothing was measured."""
    if value is None:
        return None
    size = float(value)
    return round(size, 1) if size > 0.0 else None


def has_observed_ball(ann: Dict[str, Any]) -> bool:
    """True when this frame's ball is the detector's own observation."""
    if not ann.get("ball") or ann.get("ball_interp") or ann.get("ball_out_of_frame"):
        return False
    # Only the gap reconstructor writes ``ball_source``; without it every
    # remaining point is a detection.
    return ann.get("ball_source") in (None, "detected")


def drop_unobserved_ball_sizes(annotations: Iterable[Dict[str, Any]]) -> int:
    """Clear ``ball_size`` wherever the final ball is not an observed detection.

    Later trajectory passes remove points (static sources, jump islands) or add
    derived ones (gap reconstruction). The size must follow the observation it
    was measured on, never the frame number. Returns the sizes cleared.
    """
    cleared = 0
    for ann in annotations:
        if ann.get("ball_size") is not None and not has_observed_ball(ann):
            ann["ball_size"] = None
            cleared += 1
    return cleared
