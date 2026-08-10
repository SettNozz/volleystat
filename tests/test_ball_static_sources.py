"""Rules for telling a fixed object from a ball in play.

Both errors are expensive, so both directions are tested: furniture must be
removed even when the detector is completely confident about it, and a ball
that merely pauses — in the server's hands, at the apex of a lob — must
survive, because those frames are real data the FSM reads.
"""

from __future__ import annotations

import pytest

from src.tracking.ball_static_sources import (
    clean_ball_annotations,
    find_static_sources,
    find_static_sources_in_candidates,
    suppress_jump_islands,
    suppress_static_annotations,
    suppress_static_candidates,
)
from src.tracking.ball_track_gate import BallCandidate
from src.tracking.ball_trajectory_dp import BallTrajectoryDP

FPS = 30.0


def _parked(frames, x=500.0, y=345.0, jitter=0.0):
    return [(f, x + (jitter if f % 2 else -jitter), y) for f in frames]


def _flight(frames, x0=200.0, y0=300.0):
    return [
        (f, x0 + 18.0 * i, y0 - 12.0 * i + 0.6 * i * i)
        for i, f in enumerate(frames)
    ]


def _annotations(points, total=None):
    total = total or (points[-1][0] + 1)
    by_frame = {f: (x, y) for f, x, y in points}
    return [
        {"f": f, "t": f / FPS, "ball": list(by_frame[f]) if f in by_frame else None}
        for f in range(total)
    ]


# ----------------------------------------------------------------- detection


def test_object_that_never_moves_for_seconds_is_a_fixed_source():
    sources = find_static_sources(_parked(range(120), jitter=0.7), fps=FPS)
    assert len(sources) == 1
    assert sources[0].seconds == pytest.approx(119 / FPS, abs=0.05)
    assert sources[0].spread_px < 3


def test_a_short_pause_is_not_a_fixed_source():
    """A server holds the ball; that is still the ball."""
    assert find_static_sources(_parked(range(24)), fps=FPS) == []


def test_a_ball_passing_through_the_same_spot_is_not_a_fixed_source():
    twice = _flight(range(0, 20)) + _flight(range(200, 220))
    assert find_static_sources(twice, fps=FPS) == []


def test_threshold_scales_with_frame_width():
    """A 4K frame shows the same physical wobble as three times the pixels."""
    wobbly = [(f, 500.0 + 9.0 * (f % 2), 345.0) for f in range(120)]
    assert find_static_sources(wobbly, fps=FPS, px_scale=1.0) == []
    assert len(find_static_sources(wobbly, fps=FPS, px_scale=3.0)) == 1


# --------------------------------------------------------------- suppression


def test_static_points_are_cleared_and_preserved_for_audit():
    annotations = _annotations(_parked(range(120)))
    summary = suppress_static_annotations(annotations, fps=FPS)

    assert summary.removed_points == 120
    assert all(a["ball"] is None for a in annotations)
    assert all(a["ball_static"] for a in annotations)
    assert annotations[0]["ball_static_point"] == [500.0, 345.0]


def test_real_flight_is_untouched():
    annotations = _annotations(_flight(range(60)))
    summary = suppress_static_annotations(annotations, fps=FPS)
    assert summary.removed_points == 0
    assert all(a["ball"] for a in annotations)


def test_sparse_timeline_from_skip_frames_still_finds_the_object():
    annotations = _annotations(_parked(range(0, 120, 2)))
    assert suppress_static_annotations(annotations, fps=FPS).removed_points > 0


# -------------------------------------------------------------- jump islands


def test_island_reached_by_an_impossible_jump_is_removed():
    """The 3:07 case of video 3: teleport, sit still, teleport back."""
    points = (
        _flight(range(0, 12))
        + _parked(range(12, 26), x=549.0, y=348.0)
        + _flight(range(26, 40), x0=560.0, y0=40.0)
    )
    annotations = _annotations(points)
    summary = suppress_jump_islands(annotations, fps=FPS)

    assert summary.removed_points == 14
    assert all(annotations[f]["ball"] is None for f in range(12, 26))
    assert all(annotations[f]["ball"] for f in range(0, 12))
    assert all(annotations[f]["ball"] for f in range(26, 40))


def test_a_short_fragment_that_actually_flies_is_kept():
    """Not every jump is a lie: the ball reappears far away after an occlusion."""
    points = _flight(range(0, 12)) + _flight(range(12, 24), x0=900.0, y0=650.0)
    annotations = _annotations(points)
    assert suppress_jump_islands(annotations, fps=FPS).removed_points == 0


def test_frames_without_a_ball_are_never_touched():
    annotations = [{"f": f, "ball": None} for f in range(50)]
    report = clean_ball_annotations(annotations, fps=FPS)
    assert report["removed_points"] == 0


# --------------------------------------------------- candidates and solver


def _cloud():
    """A confident parked ball all the time; the real ball only mid-clip."""
    cands = {}
    for f in range(120):
        frame = [BallCandidate(x=500.0, y=345.0, confidence=1.0, size=12)]
        if 40 <= f <= 70:
            i = f - 40
            frame.append(
                BallCandidate(x=200.0 + 18.0 * i, y=300.0 - 12.0 * i + 0.6 * i * i,
                              confidence=0.55, size=12)
            )
        cands[f] = frame
    return cands


def test_candidate_suppression_removes_only_the_fixed_object():
    cands = _cloud()
    sources = find_static_sources_in_candidates(cands, fps=FPS)
    assert len(sources) == 1

    removed = suppress_static_candidates(cands, sources)
    assert removed == 120
    survivors = [c for frame in cands.values() for c in frame]
    assert len(survivors) == 31
    assert all(c.confidence == 0.55 for c in survivors)


def test_solver_follows_the_real_ball_once_the_bait_is_gone():
    """Why this matters: a confident static run is *cheap* for the solver."""
    baited = BallTrajectoryDP().solve(_cloud())
    on_parked = sum(1 for c in baited.values() if abs(c.x - 500) < 5)
    # The failure this module exists to prevent: the path spends most of its
    # length on the parked object, and jumps to and from it around the flight —
    # that alternation is what the viewer draws as a scribble.
    assert on_parked > 60

    cands = _cloud()
    suppress_static_candidates(cands, find_static_sources_in_candidates(cands, fps=FPS))
    cleaned = BallTrajectoryDP().solve(cands)
    assert all(abs(c.x - 500) > 5 for c in cleaned.values())
    assert len(cleaned) >= 25
