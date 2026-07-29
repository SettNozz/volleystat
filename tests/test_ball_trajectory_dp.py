"""Global trajectory selection must prefer the smooth moving path.

The cases here are the ones a causal, one-frame-at-a-time rule cannot get
right: a distractor that outscores the ball on some frames, and a static object
whose path is perfectly smooth precisely because it never moves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.ball_track_gate import BallCandidate
from src.tracking.ball_trajectory_dp import BallTrajectoryDP, interpolate_gaps


def cand(x, y, conf=0.5, size=12.0):
    return BallCandidate(x=x, y=y, confidence=conf, size=size)


def flight(n=30, x0=100.0, y0=300.0, vx=20.0, conf=0.4, start=0):
    """A ball travelling at constant speed, one candidate per frame."""
    return {start + i: [cand(x0 + vx * i, y0, conf=conf)] for i in range(n)}


def test_follows_moving_ball_over_confident_distractor():
    """A brighter blob parked off the path must not win any frame."""
    candidates = flight()
    for frame in candidates:
        candidates[frame].append(cand(900.0, 50.0, conf=0.95))

    chosen = BallTrajectoryDP().solve(candidates)

    assert len(chosen) == len(candidates)
    for frame, c in chosen.items():
        assert c.x == 100.0 + 20.0 * frame
        assert c.y == 300.0


def test_static_high_confidence_blob_is_not_a_trajectory():
    """Standing still is free under a smoothness cost — the stillness penalty
    is what stops a scoreboard from being the cheapest path in the graph."""
    candidates = {f: [cand(566.0, 340.0, conf=0.9)] for f in range(40)}
    candidates.update(
        {f: candidates[f] + [cand(100.0 + 18.0 * f, 300.0, conf=0.3)] for f in range(40)}
    )

    chosen = BallTrajectoryDP().solve(candidates)

    moving = sum(1 for c in chosen.values() if c.y == 300.0)
    assert moving > 30, "the travelling candidate should own almost every frame"


def test_recovers_ball_on_frames_where_distractor_scores_higher():
    """The path through the neighbours identifies the ball on the bad frames."""
    candidates = flight(conf=0.35)
    for frame in (10, 11, 12):
        candidates[frame].append(cand(candidates[frame][0].x + 140.0, 300.0, conf=0.9))

    chosen = BallTrajectoryDP().solve(candidates)

    for frame in (10, 11, 12):
        assert chosen[frame].x == 100.0 + 20.0 * frame


def test_gap_is_left_empty_rather_than_guessed():
    candidates = flight(n=10)
    candidates.update(flight(n=10, x0=1000.0, y0=100.0, start=40))

    chosen = BallTrajectoryDP().solve(candidates)

    assert all(f not in chosen for f in range(10, 40))
    assert len(chosen) == 20


def test_separate_flights_are_both_tracked():
    """A restart must be affordable: two rallies are not one trajectory."""
    candidates = flight(n=12)
    candidates.update(flight(n=12, x0=800.0, y0=200.0, vx=-25.0, start=20))

    chosen = BallTrajectoryDP().solve(candidates)

    assert len([f for f in chosen if f < 12]) == 12
    assert len([f for f in chosen if f >= 20]) == 12


def test_interpolation_fills_short_gaps_only():
    chosen = {0: cand(0.0, 0.0), 3: cand(30.0, 0.0), 20: cand(500.0, 0.0)}

    filled = interpolate_gaps(chosen, max_gap=4)

    assert filled[1].x == 10.0 and filled[2].x == 20.0
    assert filled[1].confidence == 0.0, "interpolated points must be identifiable"
    assert 10 not in filled, "a 17-frame hole is not interpolated"


def test_empty_input():
    assert BallTrajectoryDP().solve({}) == {}
    assert BallTrajectoryDP().solve({5: []}) == {}


def test_long_video_is_split_into_segments():
    """Segmenting keeps the edge count linear; the result must stay coherent."""
    candidates = {}
    for rally in range(6):
        candidates.update(flight(n=25, start=rally * 200, y0=300.0 + rally))

    solver = BallTrajectoryDP()
    chosen = solver.solve(candidates)

    assert solver.stats["segments"] == 6
    assert len(chosen) == 150
