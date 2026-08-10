"""What the smoother must and must not do to a trajectory.

The interesting cases are not "does it average" — they are the three places
where averaging would destroy information the rest of the pipeline reads: a
contact corner, a track restart, and a frame where the ball was never seen.
"""

from __future__ import annotations

import math
import random

import pytest

from src.tracking.ball_trajectory_smoother import (
    find_contact_indices,
    jitter_metric,
    resolve_window,
    smooth_ball_annotations,
    smooth_points,
)


def _jitter(points):
    """Mean absolute second difference — the zig-zag the user sees."""
    values = [
        math.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1])
        for a, b, c in zip(points, points[1:], points[2:])
    ]
    return sum(values) / len(values)


def _parabola(n=40, noise=0.0, seed=1):
    """A flight: constant horizontal speed, gravity in y."""
    rng = random.Random(seed)
    truth = [(20.0 + 12.0 * t, 300.0 - 14.0 * t + 0.5 * t * t) for t in range(n)]
    observed = [
        (f, x + rng.gauss(0, noise), y + rng.gauss(0, noise))
        for f, (x, y) in enumerate(truth)
    ]
    return truth, observed


def test_smoothing_removes_detector_noise_and_moves_towards_truth():
    truth, observed = _parabola(noise=2.5)
    smoothed = smooth_points(observed, window=7)

    raw_error = sum(math.dist(o[1:], t) for o, t in zip(observed, truth)) / len(truth)
    fit_error = sum(math.dist(s, t) for s, t in zip(smoothed, truth)) / len(truth)
    assert fit_error < raw_error * 0.7

    assert _jitter(smoothed) < _jitter([o[1:] for o in observed]) * 0.4


def test_straight_flight_is_not_bent():
    """A quadratic fit reproduces a line exactly; nothing may drift."""
    points = [(f, 100.0 + 8.0 * f, 200.0 + 3.0 * f) for f in range(30)]
    smoothed = smooth_points(points, window=9)
    for (f, x, y), (sx, sy) in zip(points, smoothed):
        assert math.dist((x, y), (sx, sy)) < 1e-6


def test_contact_corner_survives_smoothing():
    """The hit is the signal. Rounding it off would move the event in time."""
    incoming = [(f, 100.0 + 10.0 * f, 400.0 - 10.0 * f) for f in range(15)]
    apex = incoming[-1]
    outgoing = [
        (apex[0] + k, apex[1] + 10.0 * k, apex[2] + 10.0 * k)
        for k in range(1, 16)
    ]
    points = incoming + outgoing

    cuts = find_contact_indices(points)
    assert len(incoming) - 1 in cuts

    smoothed = smooth_points(points, window=9)
    # The corner point itself is an anchor.
    assert smoothed[len(incoming) - 1] == (apex[1], apex[2])

    def angle(seq, i):
        ax, ay = seq[i][0] - seq[i - 3][0], seq[i][1] - seq[i - 3][1]
        bx, by = seq[i + 3][0] - seq[i][0], seq[i + 3][1] - seq[i][1]
        cos = (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    raw = [(p[1], p[2]) for p in points]
    i = len(incoming) - 1
    assert angle(smoothed, i) == pytest.approx(angle(raw, i), abs=2.0)


def test_gentle_flight_curvature_is_not_mistaken_for_a_contact():
    _, observed = _parabola(n=60, noise=1.0)
    assert find_contact_indices(observed) == []


def test_smoothing_never_crosses_a_track_restart():
    """Two hypotheses are not one flight; averaging across invents motion."""
    left = [
        {"f": f, "ball": [100.0 + 5.0 * f, 200.0]}
        for f in range(12)
    ]
    right = [
        {"f": 12 + k, "ball": [900.0 + 5.0 * k, 700.0], "ball_new": k == 0}
        for k in range(12)
    ]
    annotations = left + right
    smooth_ball_annotations(annotations, window=7)

    assert annotations[12]["ball"] == [900.0, 700.0]
    assert "ball_presmooth" not in annotations[12]
    # Neither side was pulled towards the other.
    assert annotations[11]["ball"][0] < 200
    assert annotations[13]["ball"][0] > 850


def test_impossible_step_breaks_a_run_even_without_a_restart_marker():
    """Legacy annotations predate ``ball_new``; speed still gives a jump away."""
    annotations = [{"f": f, "ball": [100.0 + 5.0 * f, 200.0]} for f in range(10)]
    annotations += [{"f": 10 + k, "ball": [1200.0 + 5.0 * k, 900.0]} for k in range(10)]
    smooth_ball_annotations(annotations, window=7)
    assert annotations[10]["ball"] == [1200.0, 900.0]


def test_frames_without_a_ball_are_left_alone():
    annotations = [{"f": f, "ball": None} for f in range(5)]
    summary = smooth_ball_annotations(annotations, window=7)
    assert summary["moved"] == 0
    assert all(a["ball"] is None for a in annotations)


def test_original_position_is_kept_for_audit():
    _, observed = _parabola(noise=3.0)
    annotations = [{"f": f, "ball": [x, y]} for f, x, y in observed]
    summary = smooth_ball_annotations(annotations, window=7)

    assert summary["moved"] > 0
    moved = [a for a in annotations if a.get("ball_smoothed")]
    assert moved and all("ball_presmooth" in a for a in moved)
    assert moved[0]["ball"] != moved[0]["ball_presmooth"]


def test_window_zero_is_a_no_op():
    _, observed = _parabola(noise=3.0)
    annotations = [{"f": f, "ball": [x, y]} for f, x, y in observed]
    before = [list(a["ball"]) for a in annotations]
    summary = smooth_ball_annotations(annotations, window="off")
    assert summary["moved"] == 0
    assert [a["ball"] for a in annotations] == before


def test_sparse_timelines_from_skip_frames_are_still_smoothed():
    """``--skip-frames`` writes every 2nd frame; that is still one flight."""
    _, observed = _parabola(noise=2.5)
    annotations = [{"f": 2 * f, "ball": [x, y]} for f, x, y in observed]
    summary = smooth_ball_annotations(annotations, window=7)
    assert summary["stride"] == 2
    assert summary["runs"] == 1
    assert summary["moved"] > len(annotations) * 0.5


def test_jitter_metric_reports_the_improvement():
    _, observed = _parabola(noise=3.0)
    annotations = [{"f": f, "ball": [x, y]} for f, x, y in observed]
    before = jitter_metric(annotations)
    smooth_ball_annotations(annotations, window=9)
    assert jitter_metric(annotations)["mean"] < before["mean"] * 0.5


@pytest.mark.parametrize(
    "value,expected",
    [(None, 7), ("off", 0), ("light", 5), ("strong", 13), (8, 9), ("11", 11), (1, 3)],
)
def test_resolve_window(value, expected):
    assert resolve_window(value) == expected


def test_resolve_window_rejects_nonsense():
    with pytest.raises(ValueError):
        resolve_window("very smooth please")
    with pytest.raises(ValueError):
        resolve_window(-1)
