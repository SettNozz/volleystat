"""
Ball Physics Classifier

Classifies volleyball shot types (serve / spike / set / pass / dig / free_ball)
from ball trajectory segments using physics-derived features.

Unlike existing per-frame trackers (speed + direction instant delta),
this works on FULL FLIGHT ARCS between contacts:
  - arc shape (peak height, arc ratio)
  - entry / exit angles
  - speed profile (accelerating vs decelerating)
  - spatial context (court zone, net crossing)

All distances normalised by net_width_px → resolution-independent.
Coordinates: pixel-space, Y increases downward, net at net_y.
"height above net" = net_y - ball_y  (positive = above net).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BallSegment:
    """One flight arc: ball positions between two contact moments (or gap)."""
    frames:    List[int]
    times:     List[float]
    positions: List[Tuple[float, float]]   # (x, y) in frame pixels
    fsm_label: Optional[str] = None        # FSM event label at start if known


@dataclass
class PhysicsFeatures:
    # Speed (normalised by net_width_px)
    speed_mean:   float = 0.0  # mean |velocity| per frame
    speed_max:    float = 0.0
    speed_entry:  float = 0.0  # mean over first 3 frames
    speed_exit:   float = 0.0  # mean over last 3 frames

    # Height (normalised by net_width_px; positive = above net_y)
    height_peak:  float = 0.0  # max height reached
    height_entry: float = 0.0  # height at segment start
    height_exit:  float = 0.0  # height at segment end

    # Angle (degrees; positive = going up, negative = going down)
    angle_entry_deg: float = 0.0   # avg over first 3 velocity vectors
    angle_exit_deg:  float = 0.0   # avg over last 3 velocity vectors

    # Trajectory shape
    horizontal_dist: float = 0.0   # total horizontal travel (normalised)
    arc_ratio:       float = 0.0   # height_peak / horizontal_dist
    duration_frames: int   = 0
    duration_sec:    float = 0.0

    # Court context
    crosses_net:          bool  = False
    start_side:           int   = 0     # +1 right of net centre, -1 left
    net_crossing_height:  float = 0.0   # height when crossing net (normalised)

    # Speed profile
    speed_change_ratio: float = 0.0     # exit_speed / entry_speed  (>1 = accel)


@dataclass
class BallEvent:
    start_frame: int
    end_frame:   int
    start_time:  float
    end_time:    float
    label:       str
    confidence:  float
    features:    PhysicsFeatures
    fsm_label:   Optional[str] = None

    def to_dict(self) -> dict:
        f = self.features
        return {
            "start_frame": self.start_frame,
            "end_frame":   self.end_frame,
            "start_time":  round(self.start_time, 3),
            "end_time":    round(self.end_time, 3),
            "label":       self.label,
            "confidence":  round(self.confidence, 3),
            "fsm_label":   self.fsm_label,
            "features": {
                "speed_mean":          round(f.speed_mean, 4),
                "speed_max":           round(f.speed_max, 4),
                "speed_entry":         round(f.speed_entry, 4),
                "speed_exit":          round(f.speed_exit, 4),
                "height_peak":         round(f.height_peak, 4),
                "height_entry":        round(f.height_entry, 4),
                "height_exit":         round(f.height_exit, 4),
                "angle_entry_deg":     round(f.angle_entry_deg, 1),
                "angle_exit_deg":      round(f.angle_exit_deg, 1),
                "horizontal_dist":     round(f.horizontal_dist, 4),
                "arc_ratio":           round(f.arc_ratio, 4),
                "duration_frames":     f.duration_frames,
                "duration_sec":        round(f.duration_sec, 3),
                "crosses_net":         f.crosses_net,
                "start_side":          f.start_side,
                "net_crossing_height": round(f.net_crossing_height, 4),
                "speed_change_ratio":  round(f.speed_change_ratio, 4),
            }
        }


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_trajectory(
    frames:     List[dict],
    event_frames: Optional[List[int]] = None,
    max_gap:    int = 8,
    min_len:    int = 4,
) -> List[BallSegment]:
    """
    Split ball trajectory into flight arcs.

    Cuts at:
      - Gaps in detection longer than max_gap frames
      - Known contact-event frames (from FSM)

    Segments shorter than min_len frames are discarded.
    """
    # Collect (frame_idx, time, x, y, event_label)
    pts = []
    ev_set = set(event_frames or [])
    for ann in frames:
        ball = ann.get("ball")
        if ball is None:
            continue
        ev = ann.get("ev", [])
        label = ev[0] if ev else None
        pts.append((ann["f"], ann.get("t", ann["f"] / 30.0),
                    ball[0], ball[1], label))

    if len(pts) < min_len:
        return []

    segments: List[BallSegment] = []
    cur_frames, cur_times, cur_pos, cur_label = [], [], [], None

    for i, (fi, t, x, y, label) in enumerate(pts):
        gap = fi - pts[i - 1][0] if i > 0 else 0
        cut = gap > max_gap or (fi in ev_set and i > 0)

        if cut and len(cur_frames) >= min_len:
            segments.append(BallSegment(
                frames=cur_frames, times=cur_times,
                positions=cur_pos, fsm_label=cur_label,
            ))
            cur_frames, cur_times, cur_pos, cur_label = [], [], [], None

        if label and cur_label is None:
            cur_label = label

        cur_frames.append(fi)
        cur_times.append(t)
        cur_pos.append((x, y))

    if len(cur_frames) >= min_len:
        segments.append(BallSegment(
            frames=cur_frames, times=cur_times,
            positions=cur_pos, fsm_label=cur_label,
        ))
    return segments


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    seg:          BallSegment,
    net_y:        float,
    net_x_left:   float,
    net_x_right:  float,
    fps:          float,
) -> PhysicsFeatures:
    """
    Compute physics features for one flight segment.
    All distances normalised by net_width_px.
    """
    pos = np.array(seg.positions, dtype=float)   # shape (N, 2)
    xs, ys = pos[:, 0], pos[:, 1]
    n = len(pos)

    net_width = max(net_x_right - net_x_left, 1.0)
    net_cx = (net_x_left + net_x_right) / 2.0

    # ----- Velocity vectors -----
    if n < 2:
        return PhysicsFeatures()

    dx = np.diff(xs)
    dy = np.diff(ys)
    speeds_raw = np.sqrt(dx ** 2 + dy ** 2)           # px/frame
    speeds = speeds_raw / net_width                    # normalised

    # Angle: positive = going UP (y decreasing in image)
    angles = np.degrees(np.arctan2(-dy, dx))           # (-dy because Y down)

    # ----- Speed stats -----
    w = min(3, len(speeds))
    speed_entry = float(np.mean(speeds[:w]))
    speed_exit  = float(np.mean(speeds[-w:]))
    speed_max   = float(np.max(speeds))
    speed_mean  = float(np.mean(speeds))
    speed_change_ratio = (speed_exit / speed_entry) if speed_entry > 1e-6 else 1.0

    # ----- Angle stats -----
    angle_entry = float(np.mean(angles[:w]))
    angle_exit  = float(np.mean(angles[-w:]))

    # ----- Height (above net_y, normalised) -----
    heights = (net_y - ys) / net_width                 # positive = above net
    height_peak  = float(np.max(heights))
    height_entry = float(heights[0])
    height_exit  = float(heights[-1])

    # ----- Horizontal distance -----
    h_dist = float(np.sum(np.abs(dx))) / net_width

    # ----- Arc ratio -----
    arc_ratio = (height_peak / h_dist) if h_dist > 1e-4 else 0.0

    # ----- Net crossing -----
    net_cx_norm = (net_cx - net_x_left) / net_width    # ≈ 0.5
    x_norm = (xs - net_x_left) / net_width
    start_side = int(np.sign(x_norm[0] - net_cx_norm)) or 1

    crosses = bool(np.any(np.diff(np.sign(x_norm - net_cx_norm)) != 0))

    # Height at crossing
    crossing_height = 0.0
    if crosses:
        for i in range(n - 1):
            s0 = x_norm[i] - net_cx_norm
            s1 = x_norm[i + 1] - net_cx_norm
            if s0 * s1 < 0:
                t = abs(s0) / (abs(s0) + abs(s1))
                y_cross = ys[i] + t * (ys[i + 1] - ys[i])
                crossing_height = float((net_y - y_cross) / net_width)
                break

    # ----- Duration -----
    dur_frames = seg.frames[-1] - seg.frames[0]
    dur_sec    = dur_frames / fps

    return PhysicsFeatures(
        speed_mean=speed_mean, speed_max=speed_max,
        speed_entry=speed_entry, speed_exit=speed_exit,
        height_peak=height_peak, height_entry=height_entry, height_exit=height_exit,
        angle_entry_deg=angle_entry, angle_exit_deg=angle_exit,
        horizontal_dist=h_dist, arc_ratio=arc_ratio,
        duration_frames=dur_frames, duration_sec=dur_sec,
        crosses_net=crosses, start_side=start_side,
        net_crossing_height=crossing_height,
        speed_change_ratio=speed_change_ratio,
    )


# ---------------------------------------------------------------------------
# Rule-based classifier
# ---------------------------------------------------------------------------

# All speed thresholds are in normalised units (fraction of net_width per frame)
# At 30fps, 4K (net_width≈1421px):
#   slow  (<0.011) ≈ <3 m/s    free ball / set
#   medium (0.011–0.025) ≈ 3–7 m/s  pass / reception
#   fast  (0.025–0.055) ≈ 7–15 m/s  serve
#   very fast (>0.055) ≈ >15 m/s    spike / smash

_SPEED_SLOW   = 0.012
_SPEED_MEDIUM = 0.025
_SPEED_FAST   = 0.050

# Arc ratio: height_peak / horizontal_dist
#   Set: high arc, short travel → large ratio (>0.5)
#   Serve/spike: lower arc relative to travel → smaller ratio

_ARC_SET     = 0.45
_ARC_FLAT    = 0.15

# Height thresholds (normalised by net_width)
_NET_CLEAR   = 0.02   # at least 2% net_width above net

# Duration
_DUR_SERVE   = 0.40   # seconds: serves are long arcs

# Descent angle (negative = going down)
_STEEP_EXIT  = -25.0


def classify_segment(
    seg:  BallSegment,
    feat: PhysicsFeatures,
) -> Tuple[str, float]:
    """
    Rule-based classification.  Returns (label, confidence ∈ [0,1]).

    Priority order (most specific first):
      spike → serve → free_ball → set → dig → pass → unknown
    """
    f = feat

    # ------------------------------------------------------------------
    # SPIKE / ATTACK
    # ball crosses net fast + steep exit angle (coming down)
    # ------------------------------------------------------------------
    if (f.crosses_net
            and f.speed_max > _SPEED_FAST
            and f.angle_exit_deg < _STEEP_EXIT
            and f.net_crossing_height > _NET_CLEAR):
        conf = min(1.0, 0.55
                   + 0.15 * min(1.0, (f.speed_max - _SPEED_FAST) / _SPEED_FAST)
                   + 0.10 * min(1.0, abs(f.angle_exit_deg) / 60.0)
                   + 0.10 * min(1.0, f.speed_change_ratio - 1.0 if f.speed_change_ratio > 1 else 0.0))
        return ("spike", round(conf, 3))

    # ------------------------------------------------------------------
    # SERVE
    # long flight, crosses net, starts far from net
    # ------------------------------------------------------------------
    if (f.crosses_net
            and f.duration_sec > _DUR_SERVE
            and f.height_peak > _NET_CLEAR
            and f.horizontal_dist > 0.4):
        conf = min(1.0, 0.55
                   + 0.15 * min(1.0, f.duration_sec / 1.0)
                   + 0.10 * min(1.0, f.horizontal_dist / 1.0)
                   + 0.10 * min(1.0, f.height_peak / 0.2))
        return ("serve", round(conf, 3))

    # ------------------------------------------------------------------
    # FREE BALL / OVER
    # crosses net but slow and gentle arc
    # ------------------------------------------------------------------
    if (f.crosses_net
            and f.speed_max < _SPEED_MEDIUM
            and f.height_peak > _NET_CLEAR):
        conf = min(1.0, 0.50
                   + 0.20 * min(1.0, 1.0 - f.speed_max / _SPEED_MEDIUM)
                   + 0.10 * min(1.0, f.arc_ratio / 0.5))
        return ("free_ball", round(conf, 3))

    # ------------------------------------------------------------------
    # SET
    # mostly vertical arc, short horizontal travel, moderate speed
    # ------------------------------------------------------------------
    if (not f.crosses_net
            and f.arc_ratio > _ARC_SET
            and f.horizontal_dist < 0.35):
        conf = min(1.0, 0.50
                   + 0.20 * min(1.0, f.arc_ratio / 1.0)
                   + 0.10 * min(1.0, 1.0 - f.horizontal_dist / 0.35))
        return ("set", round(conf, 3))

    # ------------------------------------------------------------------
    # DIG
    # low entry height, sharp direction reversal (down → up)
    # ------------------------------------------------------------------
    if (f.angle_entry_deg < -20.0
            and f.angle_exit_deg > 10.0
            and f.height_entry < 0.05):
        conf = min(1.0, 0.50
                   + 0.15 * min(1.0, abs(f.angle_entry_deg) / 60.0)
                   + 0.15 * min(1.0, f.angle_exit_deg / 45.0))
        return ("dig", round(conf, 3))

    # ------------------------------------------------------------------
    # PASS / RECEPTION
    # trajectory reversal (ball was going down, now up) any height
    # ------------------------------------------------------------------
    if f.angle_entry_deg < -10.0 and f.angle_exit_deg > 5.0:
        conf = min(1.0, 0.45
                   + 0.20 * min(1.0, abs(f.angle_entry_deg - f.angle_exit_deg) / 90.0)
                   + 0.10 * min(1.0, f.speed_mean / _SPEED_MEDIUM))
        return ("pass", round(conf, 3))

    # ------------------------------------------------------------------
    # Unknown
    # ------------------------------------------------------------------
    return ("unknown", 0.20)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

class BallPhysicsClassifier:
    """
    High-level API.  Feed annotations, get BallEvent list.

    Usage:
        clf = BallPhysicsClassifier()
        events = clf.run(annotations_dict)
    """

    def run(self, annotations: dict) -> List[BallEvent]:
        frames     = annotations.get("frames", [])
        net_y      = float(annotations.get("net_y", 897))
        net_x_left = float(annotations.get("net_x_left", 0))
        net_x_right= float(annotations.get("net_x_right", 1280))
        fps        = float(annotations.get("fps", 30))

        # Collect contact event frames from FSM for cut points
        ev_frames = [
            ann["f"] for ann in frames
            if ann.get("ev")
        ]

        segments = segment_trajectory(frames, event_frames=ev_frames)
        events: List[BallEvent] = []

        for seg in segments:
            feat = extract_features(seg, net_y, net_x_left, net_x_right, fps)
            label, conf = classify_segment(seg, feat)
            events.append(BallEvent(
                start_frame=seg.frames[0],
                end_frame=seg.frames[-1],
                start_time=seg.times[0],
                end_time=seg.times[-1],
                label=label,
                confidence=conf,
                features=feat,
                fsm_label=seg.fsm_label,
            ))
        return events
