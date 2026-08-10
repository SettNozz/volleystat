"""Motion-gated selection of the ball among competing YOLO detections.

The detector fires on more than the ball: bright kit, line markings, the score
board, a ball resting in a corner. The old selection rule took the highest
confidence detection within 400 px of the previous accepted one, which is a
third of the frame — so on any frame where a distractor outscored the ball the
track teleported, and the drawn trajectory zig-zagged between the two.

Two physical facts do the filtering here:

* **A ball in flight moves smoothly.** Its next position is close to a constant
  velocity prediction, so candidates are scored by agreement with that
  prediction and only accepted inside a gate that grows with the gap since the
  last sighting — not by confidence alone.
* **A ball in play is never still.** A track is therefore only *acquired* from
  a run of detections that actually travel. A static distractor can hold a
  position for hundreds of frames and never qualify. Motion is required to pick
  the ball up, not to keep it: once tracked, a ball may slow to a stop at the
  apex of a lob or roll to rest without being dropped.

When nothing plausible is visible the gate reports no detection rather than
guessing. Short gaps are filled later by interpolation, which draws a straight
line — far closer to the truth than a 300 px jump onto a distractor.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class BallCandidate:
    """One raw detection considered for this frame."""
    x: float
    y: float
    confidence: float
    size: float


class BallTrackGate:
    """Picks at most one candidate per frame, keeping the trajectory coherent."""

    def __init__(
        self,
        # How far the ball can physically travel in one frame, in px. This is
        # the hard gate and it is measured from the last accepted POSITION, not
        # from the velocity prediction: a contact reverses the ball's direction,
        # so right at a hit — the frames that matter most — the prediction is
        # wrong by twice the speed while the true position is still close by.
        # 75 px/frame at 30 fps ≈ 32 m/s at 720p match framing, above any real
        # spike. Measured on video 3, raising this to 100 buys ~1% more kept
        # frames while noticeably widening the step distribution, so the tighter
        # physical bound is close to free.
        max_travel_px_per_frame: float = 75.0,
        # Detector centre jitter allowed on top, independent of the gap.
        gate_jitter_px: float = 25.0,
        gate_max_px: float = 320.0,
        # After this many frames without a sighting the velocity estimate is
        # worthless; stop extrapolating and require a fresh acquisition.
        coast_frames: int = 10,
        # Detections needed to acquire a new track, and how far they must
        # travel in total for the run to count as a moving object.
        acquire_frames: int = 3,
        acquire_min_travel_px: float = 18.0,
        # Consecutive-frame step limit while assembling an acquisition run.
        acquire_max_step_px: float = 110.0,
        # Confidence pull when several candidates sit inside the gate. Small:
        # geometry is the more trustworthy signal here.
        confidence_weight: float = 0.35,
        # A track that stops moving goes dormant: still reported (a ball really
        # can come to rest) but no longer allowed to block re-acquisition.
        # Without this a distractor that the gate locked onto keeps winning the
        # gate every frame, and the live ball elsewhere is never even considered.
        still_eps_px: float = 3.0,
        still_timeout_frames: int = 20,
    ):
        self.max_travel_px_per_frame = max_travel_px_per_frame
        self.gate_jitter_px = gate_jitter_px
        self.gate_max_px = gate_max_px
        self.coast_frames = coast_frames
        self.acquire_frames = acquire_frames
        self.acquire_min_travel_px = acquire_min_travel_px
        self.acquire_max_step_px = acquire_max_step_px
        self.confidence_weight = confidence_weight
        self.still_eps_px = still_eps_px
        self.still_timeout_frames = still_timeout_frames

        self._last: Optional[Tuple[int, float, float]] = None   # frame, x, y
        self._prev: Optional[Tuple[int, float, float]] = None
        self._pending: List[Tuple[int, float, float, float]] = []  # frame,x,y,size
        self._last_size: Optional[float] = None
        self._moved_at: int = -1   # last frame the track actually travelled

        # True when this frame's result started a new track rather than
        # continuing one. The step across it is a change of hypothesis, not ball
        # movement, so consumers must break the trajectory there instead of
        # reading it as a very fast contact.
        self.acquired_this_frame: bool = False

        self.stats = {"accepted": 0, "gated_out": 0, "acquired": 0, "no_candidate": 0}

    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._last = None
        self._prev = None
        self._pending.clear()
        self._last_size = None
        self._moved_at = -1

    @property
    def last_position(self) -> Optional[Tuple[float, float]]:
        return (self._last[1], self._last[2]) if self._last else None

    def _predict(self, frame_idx: int) -> Optional[Tuple[float, float, int]]:
        """(x, y, dt) predicted for this frame, or None when not tracking."""
        if self._last is None:
            return None
        lf, lx, ly = self._last
        dt = frame_idx - lf
        if dt <= 0 or dt > self.coast_frames:
            return None
        if self._prev is not None:
            pf, px, py = self._prev
            span = lf - pf
            if 0 < span <= self.coast_frames:
                vx, vy = (lx - px) / span, (ly - py) / span
                return lx + vx * dt, ly + vy * dt, dt
        return lx, ly, dt

    def _gate_radius(self, dt: int) -> float:
        return min(self.max_travel_px_per_frame * dt + self.gate_jitter_px,
                   self.gate_max_px)

    def update(
        self, frame_idx: int, candidates: Sequence[BallCandidate],
    ) -> Optional[BallCandidate]:
        """Return the accepted candidate for this frame, or None."""
        self.acquired_this_frame = False
        if not candidates:
            self.stats["no_candidate"] += 1
            self._expire(frame_idx)
            return None

        predicted = self._predict(frame_idx)
        if predicted is not None:
            px, py, dt = predicted
            lf, lx, ly = self._last
            gate = self._gate_radius(dt)
            best, best_score = None, None
            for cand in candidates:
                # Hard gate: could the ball have got here at all?
                travel = ((cand.x - lx) ** 2 + (cand.y - ly) ** 2) ** 0.5
                if travel > gate:
                    continue
                # Soft score: agreement with the predicted flight, eased by
                # confidence. A candidate that continues the trajectory beats
                # one that merely sits within reach.
                miss = ((cand.x - px) ** 2 + (cand.y - py) ** 2) ** 0.5
                score = miss / gate - self.confidence_weight * cand.confidence
                if best_score is None or score < best_score:
                    best, best_score = cand, score
            # A dormant track must not shut out a ball that is visibly moving
            # somewhere else, so try the out-of-gate candidates first.
            if self._is_dormant(frame_idx):
                outside = [
                    c for c in candidates
                    if ((c.x - lx) ** 2 + (c.y - ly) ** 2) ** 0.5 > gate
                ]
                if outside:
                    promoted = self._try_acquire(frame_idx, outside)
                    if promoted is not None:
                        return promoted

            if best is not None:
                self._accept(frame_idx, best)
                self.stats["accepted"] += 1
                return best
            self.stats["gated_out"] += 1

        # Not tracking, or everything fell outside the gate: try to acquire.
        return self._try_acquire(frame_idx, candidates)

    # ------------------------------------------------------------------

    def _accept(self, frame_idx: int, cand: BallCandidate) -> None:
        travelling = True
        if self._last is not None:
            moved = ((cand.x - self._last[1]) ** 2 + (cand.y - self._last[2]) ** 2) ** 0.5
            travelling = moved > self.still_eps_px
        if travelling:
            self._moved_at = frame_idx
            # A live track outranks any rival hypothesis, so drop it. A parked
            # one must not, or the moving ball's run can never accumulate.
            self._pending.clear()
        self._prev = self._last
        self._last = (frame_idx, cand.x, cand.y)
        self._last_size = cand.size

    def _is_dormant(self, frame_idx: int) -> bool:
        """Has the track sat essentially still for longer than a ball in play would?"""
        return (self._moved_at >= 0
                and frame_idx - self._moved_at > self.still_timeout_frames)

    def _expire(self, frame_idx: int) -> None:
        """Drop a track whose sightings have run out, and stale pending points."""
        if self._last is not None and frame_idx - self._last[0] > self.coast_frames:
            self._last = None
            self._prev = None
        cutoff = frame_idx - self.acquire_frames * 3
        self._pending = [p for p in self._pending if p[0] >= cutoff]

    def _try_acquire(
        self, frame_idx: int, candidates: Sequence[BallCandidate],
    ) -> Optional[BallCandidate]:
        """Start a new track from a run of detections that actually travels."""
        self._expire(frame_idx)

        # Highest-confidence candidate is the seed; a genuine ball reappearing
        # is usually the only thing moving, so ambiguity resolves next frame.
        cand = max(candidates, key=lambda c: c.confidence)

        if self._pending:
            lf, lx, ly, _ = self._pending[-1]
            span = max(frame_idx - lf, 1)
            step = ((cand.x - lx) ** 2 + (cand.y - ly) ** 2) ** 0.5 / span
            if step > self.acquire_max_step_px:
                # Incoherent with the run so far — restart from this detection.
                self._pending = []
        self._pending.append((frame_idx, cand.x, cand.y, cand.size))

        if len(self._pending) < self.acquire_frames:
            return None

        run = self._pending[-self.acquire_frames:]
        travel = sum(
            ((run[i][1] - run[i - 1][1]) ** 2 + (run[i][2] - run[i - 1][2]) ** 2) ** 0.5
            for i in range(1, len(run))
        )
        if travel < self.acquire_min_travel_px:
            # A distractor sitting still. Keep the newest points so the run can
            # still qualify the moment it starts moving.
            self._pending = self._pending[-self.acquire_frames:]
            return None

        # Adopt the run: seed velocity from its last two points.
        self._prev = (run[-2][0], run[-2][1], run[-2][2])
        self._last = (run[-1][0], run[-1][1], run[-1][2])
        self._last_size = run[-1][3]
        self._moved_at = frame_idx   # the run qualified by travelling
        self._pending.clear()
        self.stats["acquired"] += 1
        self.acquired_this_frame = True
        return cand
