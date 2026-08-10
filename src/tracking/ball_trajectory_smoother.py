"""Contact-aware smoothing of the chosen ball trajectory.

:class:`~src.tracking.ball_trajectory_dp.BallTrajectoryDP` decides *which*
candidate on each frame is the ball. It does not decide *where* the ball is:
the point it hands over is the centre of a YOLO box, and that box breathes by a
few pixels every frame even when the detection is perfectly correct. On a ball
that spans ~20 px, a 2–3 px wobble is 10-15% of its own size, which is exactly
what the drawn trail shows as a zig-zag along an otherwise clean flight.

The fix is a fit, not a filter: a ball in free flight is a parabola in image
space (gravity is linear in world space and the projection is smooth over the
tens of frames of one leg), so each point is replaced by the value of a local
least-squares **quadratic** through its neighbours — the Savitzky-Golay idea,
written out here so the module has no SciPy dependency.

What makes it safe is where the fit is *not* allowed to look:

* **across a contact.** A hit is a corner, and a quadratic through a corner
  rounds it off — it would move the apex by tens of pixels, shift the frame at
  which the FSM sees the turn, and blunt the very feature the contact detector
  keys on. Corners are found first (the same velocity-turn test the FSM uses)
  and the trajectory is cut there; the contact frame itself is an anchor and is
  never moved.
* **across a track restart or a detection gap.** Two sides of a restart are
  different hypotheses; fitting one curve through both invents motion that was
  never observed.

The result is a trajectory that is smooth *within* a flight and still angular
at every point where the ball was actually struck.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Frames each side of a candidate used to measure the velocity turn. Same value
# as ``EnhancedEventTracker.window_frames`` so both agree on what a hit is.
_TURN_WINDOW = 3
# Below this the two legs are noise, not flight (120 px/s at 30 fps).
_MIN_LEG_SPEED_PX_PER_FRAME = 4.0
# 1.0 = a perfectly straight leg. A leg that wanders cannot define a direction.
_MIN_LEG_CONSISTENCY = 0.72
# Turn that counts as a contact rather than the curve of a flight.
_MIN_TURN_DEG = 50.0
# Same physical speed ceiling as the offline DP solver, at 1280-wide framing.
_MAX_STEP_PX_PER_FRAME = 90.0

# Presets, so callers can talk about strength instead of window length.
WINDOW_PRESETS: Dict[str, int] = {"off": 0, "light": 5, "moderate": 7, "strong": 13}

DEFAULT_WINDOW = 7


def resolve_window(value: Any) -> int:
    """Accept either a preset name or a frame count; return the frame count."""
    if value is None:
        return DEFAULT_WINDOW
    if isinstance(value, str):
        key = value.strip().lower()
        if key in WINDOW_PRESETS:
            return WINDOW_PRESETS[key]
        try:
            value = int(key)
        except ValueError:
            raise ValueError(
                f"unknown smoothing preset {value!r}; "
                f"use one of {sorted(WINDOW_PRESETS)} or a frame count"
            )
    window = int(value)
    if window < 0:
        raise ValueError("smoothing window must be >= 0")
    if window and window < 3:
        # Two points define a line; there is nothing to fit.
        return 3
    if window and window % 2 == 0:
        # A symmetric window has an odd length, otherwise the fit is biased
        # towards one side and the whole trail drifts by half a frame.
        window += 1
    return window


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def _leg_velocity(
    points: Sequence[Tuple[int, float, float]]
) -> Tuple[Tuple[float, float], float, float]:
    """Mean per-frame velocity of a short leg, plus its directional coherence.

    Consistency is |mean velocity| / mean |velocity|: 1.0 for a straight leg,
    near 0 for a point that jitters in place. It is what stops detector noise
    from being read as a turn.
    """
    if len(points) < 2:
        return (0.0, 0.0), 0.0, 0.0
    vs: List[Tuple[float, float]] = []
    for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:]):
        dt = max(1, f1 - f0)
        vs.append(((x1 - x0) / dt, (y1 - y0) / dt))
    mx = sum(v[0] for v in vs) / len(vs)
    my = sum(v[1] for v in vs) / len(vs)
    speed = math.hypot(mx, my)
    mean_magnitude = sum(math.hypot(*v) for v in vs) / len(vs)
    return (mx, my), speed, speed / max(mean_magnitude, 1e-6)


def find_contact_indices(
    points: Sequence[Tuple[int, float, float]],
    px_scale: float = 1.0,
    window: int = _TURN_WINDOW,
    min_turn_deg: float = _MIN_TURN_DEG,
) -> List[int]:
    """Indices inside ``points`` where the velocity vector turns sharply.

    These are the frames the smoother must not fit across. The test is
    deliberately the same physics as the FSM contact detector, so the corner
    the FSM will later measure is the corner that is preserved here.
    """
    n = len(points)
    if n < 2 * window + 1:
        return []
    min_speed = _MIN_LEG_SPEED_PX_PER_FRAME * px_scale
    turns: List[Tuple[int, float]] = []
    for i in range(window, n - window):
        before = points[i - window: i + 1]
        after = points[i: i + window + 1]
        (bx, by), speed_before, cons_before = _leg_velocity(before)
        (ax, ay), speed_after, cons_after = _leg_velocity(after)
        if min(cons_before, cons_after) < _MIN_LEG_CONSISTENCY:
            continue
        if min(speed_before, speed_after) < min_speed:
            continue
        cosine = (bx * ax + by * ay) / (speed_before * speed_after)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if angle >= min_turn_deg:
            turns.append((i, angle))

    # Non-maximum suppression: one hit bends the path across several adjacent
    # windows, and cutting the track at every one of them would leave pieces
    # too short to fit.
    kept: List[int] = []
    for index, angle in sorted(turns, key=lambda t: -t[1]):
        if all(abs(index - other) > window for other in kept):
            kept.append(index)
    return sorted(kept)


def _fit_quadratic_at(
    points: Sequence[Tuple[int, float, float]], centre: int, half: int
) -> Tuple[float, float]:
    """Least-squares quadratic through a local window, evaluated at ``centre``.

    Written out with normal equations (a 3x3 solve) rather than a library call:
    the window is tiny, the matrix is well conditioned once ``t`` is centred on
    the sample being replaced, and the module stays dependency-free.
    """
    lo = max(0, centre - half)
    hi = min(len(points) - 1, centre + half)
    if hi - lo < 2:
        return points[centre][1], points[centre][2]

    t0 = points[centre][0]
    s = [0.0] * 5           # sums of t^0 .. t^4
    bx = [0.0] * 3          # sums of x * t^0..t^2
    by = [0.0] * 3
    for index in range(lo, hi + 1):
        f, x, y = points[index]
        t = float(f - t0)
        p = 1.0
        for k in range(5):
            s[k] += p
            p *= t
        p = 1.0
        for k in range(3):
            bx[k] += x * p
            by[k] += y * p
            p *= t

    # [[s0 s1 s2], [s1 s2 s3], [s2 s3 s4]] @ [a, b, c] = b*
    m = [
        [s[0], s[1], s[2]],
        [s[1], s[2], s[3]],
        [s[2], s[3], s[4]],
    ]
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if abs(det) < 1e-9:
        return points[centre][1], points[centre][2]

    def solve_constant(rhs: Sequence[float]) -> float:
        # Only the constant term is needed: t is centred on the sample, so
        # x(t=0) == a. Cramer's rule on the first unknown alone.
        d = (
            rhs[0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (rhs[1] * m[2][2] - m[1][2] * rhs[2])
            + m[0][2] * (rhs[1] * m[2][1] - m[1][1] * rhs[2])
        )
        return d / det

    return solve_constant(bx), solve_constant(by)


def smooth_points(
    points: Sequence[Tuple[int, float, float]],
    window: int = DEFAULT_WINDOW,
    px_scale: float = 1.0,
    max_shift_px: float = 40.0,
    anchors: Optional[Sequence[int]] = None,
) -> List[Tuple[float, float]]:
    """Smooth one physically continuous run of ``(frame, x, y)`` samples.

    ``anchors`` are indices that must not move and that the fit may not look
    across (contacts). When omitted they are detected from the points.
    """
    window = resolve_window(window)
    if window == 0 or len(points) < 3:
        return [(p[1], p[2]) for p in points]

    half = window // 2
    limit = max_shift_px * px_scale
    cuts = list(anchors) if anchors is not None else find_contact_indices(points, px_scale)
    bounds = [0] + [c for c in cuts if 0 < c < len(points) - 1] + [len(points) - 1]

    out = [(p[1], p[2]) for p in points]
    for start, end in zip(bounds, bounds[1:]):
        piece = points[start: end + 1]
        if len(piece) < 3:
            continue
        for offset in range(len(piece)):
            index = start + offset
            if index in cuts:
                continue  # the corner itself is measurement, not noise
            fx, fy = _fit_quadratic_at(piece, offset, half)
            ox, oy = points[index][1], points[index][2]
            shift = math.hypot(fx - ox, fy - oy)
            if shift > limit:
                # The fit disagrees with the sample by more than any plausible
                # detector wobble: that is a modelling failure (a corner the
                # turn test missed), not noise. Keep the observation.
                continue
            out[index] = (fx, fy)
    return out


# ----------------------------------------------------------------------
# Annotation-level entry point
# ----------------------------------------------------------------------


def annotation_stride(annotations: Sequence[Dict[str, Any]]) -> int:
    """Frames between consecutive annotations.

    ``--skip-frames`` makes the saved timeline sparse (every 2nd or 3rd frame),
    so "consecutive" cannot be hard-coded as a frame difference of one — that
    would cut every run into single points and disable smoothing entirely.
    """
    frames = [int(a["f"]) for a in annotations if "f" in a]
    if len(frames) < 2:
        return 1
    diffs = [b - a for a, b in zip(frames, frames[1:]) if b > a]
    return min(diffs) if diffs else 1


def _runs(
    annotations: Sequence[Dict[str, Any]],
    max_gap: int,
    stride: int = 1,
    px_scale: float = 1.0,
) -> List[List[int]]:
    """Split annotation indices into physically continuous ball runs.

    A run is cut on a solver restart, on a detection gap longer than
    ``max_gap`` samples, and on a step no ball could have made. The last test
    is what makes a gap tolerance above one sample safe on legacy annotations
    that predate ``ball_new``: without a restart marker, a jump across missing
    frames is only recognisable by its speed.
    """
    runs: List[List[int]] = []
    current: List[int] = []
    last_frame: Optional[int] = None
    last_point: Optional[Tuple[float, float]] = None
    span = max(1, max_gap) * max(1, stride)
    step_limit = _MAX_STEP_PX_PER_FRAME * px_scale
    for i, ann in enumerate(annotations):
        ball = ann.get("ball")
        if not ball:
            continue
        frame = int(ann.get("f", i))
        point = (float(ball[0]), float(ball[1]))
        restart = bool(ann.get("ball_new"))
        gap = None if last_frame is None else frame - last_frame
        too_fast = (
            last_point is not None
            and gap
            and math.dist(point, last_point) > step_limit * gap
        )
        if current and (restart or gap is None or gap > span or too_fast):
            runs.append(current)
            current = []
        current.append(i)
        last_frame = frame
        last_point = point
    if current:
        runs.append(current)
    return runs


def smooth_ball_annotations(
    annotations: Sequence[Dict[str, Any]],
    window: Any = DEFAULT_WINDOW,
    px_scale: float = 1.0,
    max_gap: int = 3,
    keep_raw: bool = True,
) -> Dict[str, Any]:
    """Smooth ``ann["ball"]`` in place; return a summary of what changed.

    ``keep_raw`` stores the untouched position in ``ann["ball_presmooth"]`` for
    every point that moved, so a viewer can draw both and the change stays
    auditable. The name is deliberately not ``ball_raw``: that key already
    means "before gap reconstruction" and overwriting it would erase which
    positions were observed at all.
    """
    window = resolve_window(window)
    summary = {
        "window": window,
        "runs": 0,
        "points": 0,
        "moved": 0,
        "contacts_preserved": 0,
        "mean_shift_px": 0.0,
        "max_shift_px": 0.0,
    }
    if window == 0:
        return summary

    total_shift = 0.0
    stride = annotation_stride(annotations)
    summary["stride"] = stride
    for run in _runs(annotations, max_gap=max_gap, stride=stride, px_scale=px_scale):
        pts = [
            (
                int(annotations[i].get("f", i)),
                float(annotations[i]["ball"][0]),
                float(annotations[i]["ball"][1]),
            )
            for i in run
        ]
        cuts = find_contact_indices(pts, px_scale)
        summary["runs"] += 1
        summary["points"] += len(pts)
        summary["contacts_preserved"] += len(cuts)
        smoothed = smooth_points(pts, window=window, px_scale=px_scale, anchors=cuts)
        for i, (sx, sy), (_, ox, oy) in zip(run, smoothed, pts):
            shift = math.hypot(sx - ox, sy - oy)
            if shift < 0.05:
                continue
            ann = annotations[i]
            if keep_raw and "ball_presmooth" not in ann:
                ann["ball_presmooth"] = [round(ox, 1), round(oy, 1)]
            ann["ball"] = [round(sx, 1), round(sy, 1)]
            ann["ball_smoothed"] = True
            summary["moved"] += 1
            total_shift += shift
            summary["max_shift_px"] = max(summary["max_shift_px"], shift)

    if summary["moved"]:
        summary["mean_shift_px"] = round(total_shift / summary["moved"], 2)
    summary["max_shift_px"] = round(summary["max_shift_px"], 2)
    return summary


def jitter_metric(
    annotations: Sequence[Dict[str, Any]], key: str = "ball", max_gap: int = 3,
    px_scale: float = 1.0,
) -> Dict[str, float]:
    """Mean absolute second difference — how much the path zig-zags per frame.

    A straight line and a clean parabola both score near zero; per-frame
    detector wobble is exactly what this measures, which makes it the number to
    compare before and after smoothing.
    """
    values: List[float] = []
    stride = annotation_stride(annotations)
    for run in _runs(annotations, max_gap=max_gap, stride=stride, px_scale=px_scale):
        pts = [annotations[i].get(key) or annotations[i]["ball"] for i in run]
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            values.append(
                math.hypot(
                    c[0] - 2 * b[0] + a[0],
                    c[1] - 2 * b[1] + a[1],
                )
            )
    if not values:
        return {"samples": 0, "mean": 0.0, "p95": 0.0}
    values.sort()
    return {
        "samples": len(values),
        "mean": round(sum(values) / len(values), 3),
        "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 3),
    }
