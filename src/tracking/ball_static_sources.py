"""Suppression of fixed objects the ball detector keeps firing on.

A gym is full of things that look like a volleyball to a small detector: a
second ball parked on the floor or in a bag, a ball in a spectator's hands on
the bench, a bright lamp, a round logo on the wall. They are not a problem
because they are *detected* — they are a problem because they are detected
**with high confidence on every single frame**, and the trajectory solver's
cost per edge is negative for a confident candidate. A long, cheap, stationary
run therefore beats a short, correct flight on total cost, so the chosen path
walks onto the fixed object, sits there, jumps back to the real ball, and jumps
away again. That alternation is the scribble the viewer draws.

The physical fact that separates the two is not appearance and not confidence,
it is **duration of stillness**. A ball in play is never still for long: even a
server holding it before the toss moves it, and the longest genuinely-still
stretch measured across four analysed videos on a clean recording
(``6_tiPcRYGcBtc``) is 0.8 s. On the two videos that show the scribble, single
stretches of 6.3 s and 12.2 s sit within a couple of pixels — 12-14% of every
"ball" position in the whole file.

So: a candidate stream that stays inside a few pixels for longer than
``min_seconds`` is a fixed object, and every candidate belonging to it is
removed *before* the solver runs, so the solver is never offered the bait. The
same test is applied to an already-chosen trajectory, which is what cleans up
legacy results and the live-gate path.

Nothing is invented in place of the removed points: the frame becomes "ball not
seen", which is what it truly was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# A fixed object holds position to within a pixel or two; a slowly drifting
# ball does not. Measured at 1280-wide framing and scaled with the frame.
DEFAULT_RADIUS_PX = 6.0
# Longest still stretch seen on a clean recording is 0.8 s; the shortest
# offending stretch on a bad one is 2.3 s. 1.5 s sits between them.
DEFAULT_MIN_SECONDS = 1.5
# A fixed object is detected almost every frame. Requiring most of the span to
# be occupied keeps a real ball that merely passes through the same spot twice
# from being mistaken for one.
DEFAULT_MIN_OCCUPANCY = 0.5
# Frames the source may be missing before the stretch is considered over.
DEFAULT_MAX_ABSENT_SEC = 0.4


@dataclass
class StaticSource:
    """A position that produced ball detections while never moving."""

    x: float
    y: float
    first_frame: int
    last_frame: int
    frames: int
    seconds: float
    spread_px: float

    def contains(self, x: float, y: float, radius: float) -> bool:
        return math.dist((self.x, self.y), (x, y)) <= radius

    def covers(self, frame: int, margin: int = 0) -> bool:
        return self.first_frame - margin <= frame <= self.last_frame + margin

    def as_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames": self.frames,
            "seconds": round(self.seconds, 2),
            "spread_px": round(self.spread_px, 2),
        }


@dataclass
class StaticSuppressionSummary:
    sources: List[Dict[str, Any]] = field(default_factory=list)
    removed_points: int = 0
    removed_candidates: int = 0
    longest_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sources": self.sources,
            "removed_points": self.removed_points,
            "removed_candidates": self.removed_candidates,
            "longest_seconds": round(self.longest_seconds, 2),
        }


# ----------------------------------------------------------------------


def _stretches(
    points: Sequence[Tuple[int, float, float]],
    fps: float,
    radius: float,
    min_seconds: float,
    min_occupancy: float,
    max_absent_frames: int,
) -> List[StaticSource]:
    """Find maximal stretches whose points all sit inside ``radius``.

    Greedy and single pass: a stretch grows while every new point stays within
    the radius of the stretch's running centre, which is what "did not move"
    means for a fixed object seen through a noisy detector.
    """
    sources: List[StaticSource] = []
    if not points:
        return sources

    start = 0
    cx, cy = points[0][1], points[0][2]
    count = 1

    def close(index: int) -> None:
        run = points[start:index]
        if len(run) < 2:
            return
        span_frames = run[-1][0] - run[0][0]
        seconds = span_frames / max(fps, 1e-6)
        if seconds < min_seconds:
            return
        occupancy = len(run) / max(1, span_frames + 1)
        # Sparse timelines (``--skip-frames``) can never reach a per-frame
        # occupancy, so measure against the stride actually present.
        stride = min((b[0] - a[0]) for a, b in zip(run, run[1:]))
        if occupancy * max(1, stride) < min_occupancy:
            return
        mx = sum(p[1] for p in run) / len(run)
        my = sum(p[2] for p in run) / len(run)
        spread = max(math.dist((p[1], p[2]), (mx, my)) for p in run)
        sources.append(
            StaticSource(
                x=mx, y=my, first_frame=run[0][0], last_frame=run[-1][0],
                frames=len(run), seconds=seconds, spread_px=spread,
            )
        )

    for i in range(1, len(points)):
        f, x, y = points[i]
        gap = f - points[i - 1][0]
        if gap > max_absent_frames or math.dist((x, y), (cx, cy)) > radius:
            close(i)
            start, cx, cy, count = i, x, y, 1
            continue
        # Running mean keeps the centre from drifting with a single outlier.
        count += 1
        cx += (x - cx) / count
        cy += (y - cy) / count
    close(len(points))
    return sources


def find_static_sources(
    points: Sequence[Tuple[int, float, float]],
    fps: float = 30.0,
    px_scale: float = 1.0,
    radius_px: float = DEFAULT_RADIUS_PX,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
    max_absent_sec: float = DEFAULT_MAX_ABSENT_SEC,
) -> List[StaticSource]:
    """Positions that held still long enough that no ball in play could have."""
    ordered = sorted(points, key=lambda p: p[0])
    return _stretches(
        ordered,
        fps=fps,
        radius=radius_px * px_scale,
        min_seconds=min_seconds,
        min_occupancy=min_occupancy,
        max_absent_frames=max(1, int(round(max_absent_sec * fps))),
    )


def _densest_stream(
    candidates: Mapping[int, Sequence[Any]], radius: float
) -> List[Tuple[int, float, float]]:
    """One representative point per frame per spatial cell.

    A fixed object may not be the *chosen* candidate on a given frame, so the
    search cannot run on the trajectory alone; it runs on the whole cloud, one
    stream per cell.
    """
    out: List[Tuple[int, float, float]] = []
    for frame in sorted(candidates):
        for cand in candidates[frame]:
            out.append((frame, float(cand.x), float(cand.y)))
    return out


def find_static_sources_in_candidates(
    candidates: Mapping[int, Sequence[Any]],
    fps: float = 30.0,
    px_scale: float = 1.0,
    radius_px: float = DEFAULT_RADIUS_PX,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
    max_absent_sec: float = DEFAULT_MAX_ABSENT_SEC,
) -> List[StaticSource]:
    """Same test, run per spatial cell over the full candidate cloud."""
    radius = radius_px * px_scale
    cell = radius * 2.0
    buckets: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = {}
    for frame, cands in candidates.items():
        for cand in cands:
            x, y = float(cand.x), float(cand.y)
            # Offset copies so a source sitting on a cell border is not split.
            for dx in (0.0, cell / 2):
                for dy in (0.0, cell / 2):
                    key = (int((x + dx) // cell), int((y + dy) // cell), dx > 0, dy > 0)
                    buckets.setdefault(key, []).append((frame, x, y))

    found: List[StaticSource] = []
    for stream in buckets.values():
        stream.sort(key=lambda p: p[0])
        # One point per frame: duplicates inside a cell are the same object.
        deduped: List[Tuple[int, float, float]] = []
        for point in stream:
            if not deduped or point[0] != deduped[-1][0]:
                deduped.append(point)
        found.extend(
            find_static_sources(
                deduped, fps=fps, px_scale=px_scale, radius_px=radius_px,
                min_seconds=min_seconds, min_occupancy=min_occupancy,
                max_absent_sec=max_absent_sec,
            )
        )

    # The offset copies mean the same object is reported up to four times.
    merged: List[StaticSource] = []
    for source in sorted(found, key=lambda s: (-s.frames, s.first_frame)):
        if any(
            other.contains(source.x, source.y, radius * 2)
            and not (source.last_frame < other.first_frame
                     or source.first_frame > other.last_frame)
            for other in merged
        ):
            continue
        merged.append(source)
    return merged


def suppress_static_candidates(
    candidates: Dict[int, List[Any]],
    sources: Sequence[StaticSource],
    px_scale: float = 1.0,
    radius_px: float = DEFAULT_RADIUS_PX,
    margin_frames: int = 30,
) -> int:
    """Drop candidates that belong to a fixed object. Returns how many."""
    if not sources:
        return 0
    radius = radius_px * px_scale * 1.5
    removed = 0
    for frame, cands in candidates.items():
        keep = []
        for cand in cands:
            if any(
                s.covers(frame, margin_frames) and s.contains(cand.x, cand.y, radius)
                for s in sources
            ):
                removed += 1
                continue
            keep.append(cand)
        candidates[frame] = keep
    return removed


def suppress_static_annotations(
    annotations: Sequence[Dict[str, Any]],
    fps: float = 30.0,
    px_scale: float = 1.0,
    radius_px: float = DEFAULT_RADIUS_PX,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    min_occupancy: float = DEFAULT_MIN_OCCUPANCY,
    max_absent_sec: float = DEFAULT_MAX_ABSENT_SEC,
) -> StaticSuppressionSummary:
    """Clear trajectory points that belong to a fixed object.

    The frame keeps ``ball_static`` so the removal is auditable, and the point
    itself is preserved in ``ball_static_point`` rather than silently lost.
    """
    points = [
        (int(a.get("f", i)), float(a["ball"][0]), float(a["ball"][1]))
        for i, a in enumerate(annotations)
        if a.get("ball")
    ]
    summary = StaticSuppressionSummary()
    sources = find_static_sources(
        points, fps=fps, px_scale=px_scale, radius_px=radius_px,
        min_seconds=min_seconds, min_occupancy=min_occupancy,
        max_absent_sec=max_absent_sec,
    )
    if not sources:
        return summary

    radius = radius_px * px_scale * 1.5
    summary.sources = [s.as_dict() for s in sources]
    summary.longest_seconds = max(s.seconds for s in sources)
    for i, ann in enumerate(annotations):
        ball = ann.get("ball")
        if not ball:
            continue
        frame = int(ann.get("f", i))
        if not any(
            s.covers(frame) and s.contains(float(ball[0]), float(ball[1]), radius)
            for s in sources
        ):
            continue
        ann["ball_static_point"] = [round(float(ball[0]), 1), round(float(ball[1]), 1)]
        ann["ball_static"] = True
        ann["ball"] = None
        ann["ball_confidence"] = 0.0
        summary.removed_points += 1
    return summary


def suppress_jump_islands(
    annotations: Sequence[Dict[str, Any]],
    fps: float = 30.0,
    px_scale: float = 1.0,
    max_step_px_per_frame: float = 90.0,
    max_island_seconds: float = 0.8,
    max_island_travel_px: float = 25.0,
) -> StaticSuppressionSummary:
    """Remove short motionless islands entered or left by an impossible jump.

    This is the fast version of the same failure. At 3:07 of video 3 the track
    leaves the real ball at the top of the frame, crosses 341 px in one frame
    (four times the physical limit), sits within two pixels for fourteen
    frames, hops 91 px twice, and jumps 338 px back to the ball. Every one of
    those transitions is impossible for a thrown object, and the thing in the
    middle never moves — so the island is furniture, not a ball.

    The test needs all three properties together. A short segment that *flies*
    is a real fragment of a rally and is kept; a long stationary stretch is
    already handled by :func:`suppress_static_annotations`; and a large step
    on its own is often just a detection gap the solver bridged legitimately.
    """
    summary = StaticSuppressionSummary()
    indexed = [
        (i, int(a.get("f", i)), float(a["ball"][0]), float(a["ball"][1]))
        for i, a in enumerate(annotations)
        if a.get("ball")
    ]
    if len(indexed) < 3:
        return summary

    limit = max_step_px_per_frame * px_scale
    travel_limit = max_island_travel_px * px_scale

    # Cut the trajectory wherever a transition is physically impossible.
    breaks = [0]
    impossible: List[bool] = [False]
    for k in range(1, len(indexed)):
        _, f0, x0, y0 = indexed[k - 1]
        _, f1, x1, y1 = indexed[k]
        dt = max(1, f1 - f0)
        jumped = math.dist((x0, y0), (x1, y1)) > limit * dt
        if jumped or annotations[indexed[k][0]].get("ball_new"):
            breaks.append(k)
            impossible.append(jumped)
    breaks.append(len(indexed))
    impossible.append(False)

    for seg_index in range(len(breaks) - 1):
        start, end = breaks[seg_index], breaks[seg_index + 1]
        run = indexed[start:end]
        if not run:
            continue
        entered_by_jump = impossible[seg_index]
        left_by_jump = (
            impossible[seg_index + 1] if seg_index + 1 < len(impossible) else False
        )
        if not (entered_by_jump or left_by_jump):
            continue
        seconds = (run[-1][1] - run[0][1]) / max(fps, 1e-6)
        if seconds > max_island_seconds:
            continue
        travel = max(math.dist((run[0][2], run[0][3]), (p[2], p[3])) for p in run)
        if travel > travel_limit:
            continue  # it flew: a real fragment, however short

        summary.sources.append(
            {
                "kind": "jump_island",
                "x": round(run[0][2], 1),
                "y": round(run[0][3], 1),
                "first_frame": run[0][1],
                "last_frame": run[-1][1],
                "frames": len(run),
                "seconds": round(seconds, 2),
                "travel_px": round(travel, 1),
                "entered_by_jump": entered_by_jump,
                "left_by_jump": left_by_jump,
            }
        )
        summary.longest_seconds = max(summary.longest_seconds, seconds)
        for i, _, x, y in run:
            ann = annotations[i]
            ann["ball_static_point"] = [round(x, 1), round(y, 1)]
            ann["ball_static"] = True
            ann["ball"] = None
            ann["ball_confidence"] = 0.0
            summary.removed_points += 1
    return summary


def clean_ball_annotations(
    annotations: Sequence[Dict[str, Any]],
    fps: float = 30.0,
    px_scale: float = 1.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Both rules, in the order that makes each one's job easier.

    Long fixed objects go first: removing them turns some of the remaining
    mess into plain gaps, which leaves fewer islands to judge.
    """
    static = suppress_static_annotations(annotations, fps=fps, px_scale=px_scale, **kwargs)
    islands = suppress_jump_islands(annotations, fps=fps, px_scale=px_scale)
    return {
        "static_sources": static.as_dict(),
        "jump_islands": islands.as_dict(),
        "removed_points": static.removed_points + islands.removed_points,
    }


def frozen_report(
    points: Sequence[Tuple[int, float, float]],
    fps: float = 30.0,
    px_scale: float = 1.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Diagnostic: how much of a trajectory is a fixed object pretending."""
    sources = find_static_sources(points, fps=fps, px_scale=px_scale, **kwargs)
    covered = sum(s.frames for s in sources)
    return {
        "points": len(points),
        "static_sources": len(sources),
        "static_points": covered,
        "static_share": round(covered / max(1, len(points)), 4),
        "longest_seconds": round(max((s.seconds for s in sources), default=0.0), 2),
        "worst": [s.as_dict() for s in sorted(sources, key=lambda s: -s.seconds)[:5]],
    }
