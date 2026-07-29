"""Offline selection of the ball trajectory from the full candidate cloud.

:class:`~src.tracking.ball_track_gate.BallTrackGate` decides one frame at a
time using only the past, because the annotated video needs a position while
the frame is still in hand. Once the video has been read, that constraint is
self-imposed: the whole candidate cloud is available, so the trajectory can be
chosen globally.

That matters because the detector's mistakes are not evenly spread. On a frame
where a bright shirt outscores the ball, no causal rule can know it was wrong —
but the *path* through that frame gives it away, because the ball's true
position on the frames before and after leaves a smooth curve that the impostor
does not lie on. Measured on 232 human-labeled frames of video 3, the gate
recovered 0.64 of the balls its detector had found (214 FP/min at best sweep);
this solver recovers **0.85 at 78 FP/min** from the same candidates.

Cost model — a state is an *edge* (previous candidate → current candidate),
because the smoothness term needs a velocity and a velocity needs two points:

* **deviation** from the constant-velocity prediction, per frame of gap;
* **confidence reward**, small: geometry is the more trustworthy signal;
* **stillness penalty** — this one is not optional. Deviation from constant
  velocity is *zero* for an object that never moves, so without it a scoreboard
  or a ball resting in a corner is the cheapest path in the graph. A ball in
  play always travels; charging for slowness is the same physical rule the gate
  encodes as "a ball in play is never still". It is charged only when *two
  consecutive* legs are near-motionless, which is what separates a static
  object from a ball hanging at the apex of a lob: the apex is slow for a frame
  or two with a fast leg on either side, a scoreboard is slow forever. Because
  the term no longer fires on real balls it can be steep enough to outweigh any
  confidence advantage a distractor might have.
* **restart penalty** to open a fresh track, so a rally break or a spell of
  out-of-play frames does not force one continuous path through the whole video.

The timeline is cut at gaps no edge could span, and each piece is solved on its
own — exact, and it keeps the edge count linear in video length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ball_track_gate import BallCandidate

_INF = float("inf")


@dataclass
class _Cand:
    frame: int
    x: float
    y: float
    confidence: float
    size: float


class BallTrajectoryDP:
    """Min-cost path through the per-frame candidate cloud."""

    def __init__(
        self,
        # Longest gap an edge may span. Beyond this the velocity estimate is
        # worthless and the timeline is cut instead.
        max_gap_frames: int = 10,
        # Hard limit on candidate-to-candidate speed, px per frame. 90 px/frame
        # at 30 fps is above any real spike at 720p match framing.
        max_step_px: float = 90.0,
        # Pull of detector confidence. Deliberately the same order as a few tens
        # of pixels of deviation: it breaks ties, it does not decide paths.
        # Measured on the video-3 GT window, 100.0 buys 0.832 recall at 101
        # FP/min and 50.0 gives 0.802 at 44. The lower value is the default
        # because a wrong point invents a contact for the FSM, while a missing
        # one is a case the FSM already handles.
        conf_weight: float = 50.0,
        # Cost of skipping frames within one track. Zero measured best — the
        # deviation term already divides by the gap.
        gap_penalty: float = 0.0,
        # Cost of abandoning the current track and starting a new one.
        restart_penalty: float = 400.0,
        # Charge for standing still, applied below ``still_px`` of travel per
        # frame on two consecutive legs. Without it a static distractor is the
        # global optimum. Measured on the video-3 GT window: 5.0 gives 0.789
        # recall at 41 FP/min, 12.0 gives 0.716 at 13, 30.0 gives 0.655 at 26.
        # 12.0 is the default because false positives are what the trajectory
        # and the FSM actually suffer from, and 12 x 3 px still outweighs the
        # confidence advantage any single distractor can hold.
        still_penalty: float = 12.0,
        still_px: float = 3.0,
        # Safety valve: a single segment with more edges than this is split into
        # overlapping chunks so pathological videos cannot blow up memory.
        max_segment_frames: int = 6000,
        chunk_overlap_frames: int = 120,
    ):
        self.max_gap_frames = max_gap_frames
        self.max_step_px = max_step_px
        self.conf_weight = conf_weight
        self.gap_penalty = gap_penalty
        self.restart_penalty = restart_penalty
        self.still_penalty = still_penalty
        self.still_px = still_px
        self.max_segment_frames = max_segment_frames
        self.chunk_overlap_frames = chunk_overlap_frames

        self.stats = {"frames_with_candidates": 0, "frames_chosen": 0, "segments": 0}

    # ------------------------------------------------------------------

    def solve(
        self, candidates: Mapping[int, Sequence[BallCandidate]]
    ) -> Dict[int, BallCandidate]:
        """Pick at most one candidate per frame. Returns {frame: candidate}."""
        frames = sorted(f for f, c in candidates.items() if c)
        self.stats = {"frames_with_candidates": len(frames), "frames_chosen": 0, "segments": 0}
        if not frames:
            return {}

        chosen: Dict[int, BallCandidate] = {}
        for segment in self._segments(frames):
            self.stats["segments"] += 1
            cands = [
                _Cand(f, c.x, c.y, c.confidence, c.size)
                for f in segment
                for c in candidates[f]
            ]
            for cand in self._solve_segment(cands):
                chosen[cand.frame] = BallCandidate(
                    x=cand.x, y=cand.y, confidence=cand.confidence, size=cand.size
                )
        self.stats["frames_chosen"] = len(chosen)
        return chosen

    # ------------------------------------------------------------------

    def _segments(self, frames: List[int]) -> List[List[int]]:
        """Split where no edge could reach across, then cap segment length."""
        pieces: List[List[int]] = [[frames[0]]]
        for prev, cur in zip(frames, frames[1:]):
            if cur - prev > self.max_gap_frames:
                pieces.append([cur])
            else:
                pieces[-1].append(cur)

        out: List[List[int]] = []
        for piece in pieces:
            if piece[-1] - piece[0] <= self.max_segment_frames:
                out.append(piece)
                continue
            # Overlapping chunks: the overlap is longer than any edge, so a
            # track crossing the seam is still seen whole by one of the chunks.
            start = piece[0]
            while start <= piece[-1]:
                end = start + self.max_segment_frames
                chunk = [f for f in piece if start <= f <= end]
                if chunk:
                    out.append(chunk)
                start = end - self.chunk_overlap_frames + 1
        return out

    def _solve_segment(self, cands: List[_Cand]) -> List[_Cand]:
        if len(cands) < 2:
            return list(cands)

        by_frame: Dict[int, List[int]] = {}
        for i, c in enumerate(cands):
            by_frame.setdefault(c.frame, []).append(i)

        # Edges, in order of the frame they end on.
        edges: List[Tuple[int, int]] = []
        incoming: Dict[int, List[int]] = {}
        for ci, c in enumerate(cands):
            for dt in range(1, self.max_gap_frames + 1):
                for pj in by_frame.get(c.frame - dt, ()):
                    p = cands[pj]
                    if math.dist((c.x, c.y), (p.x, p.y)) <= self.max_step_px * dt:
                        incoming.setdefault(ci, []).append(len(edges))
                        edges.append((pj, ci))
        if not edges:
            best = max(cands, key=lambda c: c.confidence)
            return [best]

        order = sorted(range(len(edges)), key=lambda e: cands[edges[e][1]].frame)
        cost = [_INF] * len(edges)
        parent: List[Optional[int]] = [None] * len(edges)
        # Best cost reachable before the frame currently being expanded, and the
        # edge that achieved it — a restart continues from there.
        prefix_best: Dict[int, Tuple[float, Optional[int]]] = {}
        running: Tuple[float, Optional[int]] = (_INF, None)
        last_frame: Optional[int] = None

        for e in order:
            a, b = edges[e]
            ca, cb = cands[a], cands[b]
            if cb.frame != last_frame:
                prefix_best[cb.frame] = running
                last_frame = cb.frame

            prior_cost, prior_edge = prefix_best.get(cb.frame, (_INF, None))
            best = self.restart_penalty + (prior_cost if prior_cost < _INF else 0.0)
            best_parent = prior_edge

            dt = cb.frame - ca.frame
            step = math.dist((cb.x, cb.y), (ca.x, ca.y)) / dt
            base = (
                self.gap_penalty * (dt - 1)
                - self.conf_weight * (ca.confidence + cb.confidence) / 2
            )
            for pe in incoming.get(a, ()):
                if cost[pe] == _INF:
                    continue
                cp = cands[edges[pe][0]]
                span = ca.frame - cp.frame
                vx, vy = (ca.x - cp.x) / span, (ca.y - cp.y) / span
                deviation = math.dist((cb.x, cb.y), (ca.x + vx * dt, ca.y + vy * dt)) / dt
                prev_step = math.dist((ca.x, ca.y), (cp.x, cp.y)) / span
                still = self.still_penalty * max(0.0, self.still_px - min(step, prev_step))
                c = cost[pe] + deviation + base + still
                if c < best:
                    best, best_parent = c, pe

            cost[e] = best
            parent[e] = best_parent
            if best < running[0]:
                running = (best, e)

        # Backtrack; restarts chain segment to segment, so one walk covers all.
        picked: Dict[int, _Cand] = {}
        e = running[1]
        seen = set()
        while e is not None and e not in seen:
            seen.add(e)
            for ci in edges[e]:
                c = cands[ci]
                picked.setdefault(c.frame, c)
            e = parent[e]
        return [picked[f] for f in sorted(picked)]


def interpolate_gaps(
    chosen: Mapping[int, BallCandidate], max_gap: int = 4
) -> Dict[int, BallCandidate]:
    """Fill short gaps on a straight line between chosen points.

    A straight line across four frames is far closer to the truth than the
    nearest distractor, and it is what the viewer needs to draw an unbroken
    trail. Interpolated points carry confidence 0.0 so callers can tell them
    apart from detections.
    """
    out = dict(chosen)
    frames = sorted(chosen)
    for a, b in zip(frames, frames[1:]):
        span = b - a
        if not 1 < span <= max_gap + 1:
            continue
        ca, cb = chosen[a], chosen[b]
        for step in range(1, span):
            t = step / span
            out[a + step] = BallCandidate(
                x=ca.x + (cb.x - ca.x) * t,
                y=ca.y + (cb.y - ca.y) * t,
                confidence=0.0,
                size=ca.size + (cb.size - ca.size) * t,
            )
    return out
