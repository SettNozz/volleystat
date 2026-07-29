"""Ball selection must follow the moving ball, not the brightest distractor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking.ball_track_gate import BallCandidate, BallTrackGate


def cand(x, y, conf=0.5, size=12.0):
    return BallCandidate(x=x, y=y, confidence=conf, size=size)


def acquire(gate, start_frame=0, x=100.0, y=100.0, step=10.0):
    """Feed a coherent moving run so the gate locks on. Returns next frame idx."""
    frame = start_frame
    for i in range(4):
        gate.update(frame, [cand(x + i * step, y)])
        frame += 1
    return frame


def test_static_distractor_never_acquires_track():
    """A high-confidence blob that never moves is not a ball."""
    gate = BallTrackGate()
    for frame in range(60):
        assert gate.update(frame, [cand(566.0, 340.0, conf=0.9)]) is None


def test_moving_run_acquires_track():
    gate = BallTrackGate()
    got = None
    for i in range(4):
        got = gate.update(i, [cand(100.0 + i * 12, 100.0, conf=0.4)])
    assert got is not None
    assert gate.last_position is not None


def test_confident_distractor_does_not_steal_moving_track():
    """The screenshot bug: a 300 px jump onto brighter kit, then back."""
    gate = BallTrackGate()
    frame = acquire(gate, x=900.0, y=200.0, step=-10.0)

    # Ball continues its arc while a much more confident blob sits far away.
    for i in range(6):
        ball = cand(860.0 - i * 10, 200.0, conf=0.3)
        blob = cand(566.0, 340.0, conf=0.95)
        chosen = gate.update(frame + i, [blob, ball])
        assert chosen is not None
        assert chosen.x == ball.x, "gate followed the distractor"


def test_no_detection_rather_than_teleport():
    """With only an out-of-gate candidate, report nothing instead of jumping."""
    gate = BallTrackGate()
    frame = acquire(gate)
    assert gate.update(frame, [cand(900.0, 700.0, conf=0.99)]) is None


def test_reacquires_after_long_gap():
    """A ball reappearing elsewhere is picked up once it shows coherent motion."""
    gate = BallTrackGate()
    frame = acquire(gate)
    for i in range(20):
        gate.update(frame + i, [])
    frame += 20

    got = None
    for i in range(4):
        got = gate.update(frame + i, [cand(400.0 + i * 15, 500.0, conf=0.6)])
    assert got is not None
    assert gate.last_position[0] > 400.0


def test_tracked_ball_may_slow_to_a_stop():
    """Motion is required to acquire a track, not to keep one."""
    gate = BallTrackGate()
    frame = acquire(gate, x=100.0, y=100.0, step=10.0)
    last = gate.last_position
    for i in range(30):
        got = gate.update(frame + i, [cand(last[0], last[1], conf=0.5)])
        assert got is not None


def test_dormant_track_yields_to_a_moving_ball():
    """A distractor the gate locked onto must not shut out the live ball."""
    gate = BallTrackGate()
    frame = acquire(gate)
    # Park the track: same spot for well over still_timeout_frames.
    parked = gate.last_position
    for i in range(40):
        gate.update(frame + i, [cand(parked[0], parked[1], conf=0.9)])
    frame += 40

    # A ball now travels coherently elsewhere; it should take the track over.
    got = None
    for i in range(4):
        got = gate.update(frame + i, [
            cand(parked[0], parked[1], conf=0.9),          # the parked blob
            cand(300.0 + i * 20, 150.0, conf=0.4),         # the moving ball
        ])
    assert got is not None
    assert got.x > 300.0, "gate stayed on the parked distractor"


def test_reset_clears_track():
    gate = BallTrackGate()
    acquire(gate)
    assert gate.last_position is not None
    gate.reset()
    assert gate.last_position is None
