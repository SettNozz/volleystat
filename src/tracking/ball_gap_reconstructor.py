"""Offline reconstruction of ball positions hidden by players or court objects.

The detector and :class:`BallTrajectoryDP` remain authoritative for visible
balls.  This module only fills gaps bounded by two detector-backed anchors.
Every synthesized point keeps provenance and confidence, and uncertain gaps
stay empty.  Leading/trailing gaps are never extrapolated.

Two policies are exposed for visual comparison:

``conservative``
    Short interpolation, or a longer smooth curve only when the predicted path
    is visibly occluded by a player/net object.

``contact_aware``
    Same safety checks, but a known/high-evidence contact may split the curve
    into two flights with a direction change at the player's body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

Point = Tuple[float, float]
Velocity = Tuple[float, float]


@dataclass(frozen=True)
class ContactHint:
    """Known contact inside a detector gap."""

    frame: int
    label: str = "other"


@dataclass(frozen=True)
class ReconstructedPoint:
    x: float
    y: float
    source: str
    confidence: float
    contact: bool = False

    def json_point(self) -> List[float]:
        return [round(self.x, 1), round(self.y, 1)]


@dataclass(frozen=True)
class ReconstructionConfig:
    short_gap_frames: int = 4
    conservative_max_gap_frames: int = 18
    occluded_max_gap_frames: int = 45
    contact_max_gap_frames: int = 75
    max_speed_px_at_1280: float = 90.0
    max_accel_px_at_1280: float = 40.0
    player_bbox_margin_px_at_1280: float = 12.0
    net_margin_px_at_1280: float = 14.0
    edge_margin_ratio: float = 0.035
    min_occlusion_ratio: float = 0.25
    restart_min_occlusion_ratio: float = 0.70
    restart_min_anchor_points: int = 3
    restart_anchor_support_frames: int = 18
    inferred_contact_turn_deg: float = 28.0


@dataclass
class ReconstructionSummary:
    mode: str
    detected: int = 0
    reconstructed: int = 0
    contacts: int = 0
    out_of_frame: int = 0
    missing: int = 0
    gaps_filled: int = 0
    gaps_rejected: int = 0


def _point(value: object) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _angle_deg(a: Velocity, b: Velocity) -> float:
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosine = _clamp((a[0] * b[0] + a[1] * b[1]) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


class BallGapReconstructor:
    """Build conservative and contact-aware alternatives from saved annotations."""

    MODES = ("conservative", "contact_aware")

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        *,
        net_tape_top: Optional[float] = None,
        net_tape_bottom: Optional[float] = None,
        net_floor_left: Optional[Sequence[float]] = None,
        net_floor_right: Optional[Sequence[float]] = None,
        config: Optional[ReconstructionConfig] = None,
    ):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        self.width = float(frame_width)
        self.height = float(frame_height)
        self.scale = self.width / 1280.0
        self.config = config or ReconstructionConfig()
        self.net_top = float(net_tape_top) if net_tape_top is not None else None
        self.net_bottom = float(net_tape_bottom) if net_tape_bottom is not None else None
        self.net_left = _point(net_floor_left)
        self.net_right = _point(net_floor_right)

    # ------------------------------------------------------------------
    # Public API

    def apply(
        self,
        annotations: List[MutableMapping[str, object]],
        *,
        active_mode: str = "conservative",
        contact_hints: Iterable[ContactHint] = (),
        keep_variants: bool = True,
    ) -> Dict[str, ReconstructionSummary]:
        """Mutate annotations with raw/final provenance and both alternatives."""
        if active_mode not in self.MODES:
            raise ValueError(f"unknown reconstruction mode: {active_mode}")

        self._capture_raw(annotations)
        hints = {int(h.frame): h for h in contact_hints}
        variants: Dict[str, Dict[int, ReconstructedPoint]] = {}
        summaries: Dict[str, ReconstructionSummary] = {}
        out_flags: Dict[str, set[int]] = {}
        for mode in self.MODES:
            points, flags, summary = self._reconstruct(annotations, mode, hints)
            variants[mode] = points
            out_flags[mode] = flags
            summaries[mode] = summary

        for ann in annotations:
            frame = int(ann.get("f", -1))
            if keep_variants:
                for mode in self.MODES:
                    point = variants[mode].get(frame)
                    prefix = f"ball_{mode}"
                    ann[prefix] = point.json_point() if point else None
                    ann[f"{prefix}_source"] = (
                        point.source if point else
                        "out_of_frame" if frame in out_flags[mode] else "missing"
                    )
                    ann[f"{prefix}_confidence"] = round(point.confidence, 3) if point else 0.0

            chosen = variants[active_mode].get(frame)
            ann["ball"] = chosen.json_point() if chosen else None
            ann["ball_source"] = (
                chosen.source if chosen else
                "out_of_frame" if frame in out_flags[active_mode] else "missing"
            )
            ann["ball_confidence"] = round(chosen.confidence, 3) if chosen else 0.0
            ann["ball_out_of_frame"] = True if frame in out_flags[active_mode] else None
            ann["ball_interp"] = (
                True if chosen is not None and chosen.source != "detected" else None
            )
            ann["ball_contact_reconstructed"] = (
                True if chosen is not None and chosen.contact else None
            )

        return summaries

    # ------------------------------------------------------------------

    def _capture_raw(self, annotations: List[MutableMapping[str, object]]) -> None:
        """Freeze detector/DP anchors before any reconstruction changes ``ball``."""
        for ann in annotations:
            if "ball_raw" in ann:
                continue
            visible = _point(ann.get("ball"))
            # Existing linear fills are not detector-backed anchors.
            if ann.get("ball_interp"):
                visible = None
            ann["ball_raw"] = [round(visible[0], 1), round(visible[1], 1)] if visible else None
            raw_conf = ann.get("ball_confidence")
            try:
                confidence = float(raw_conf) if raw_conf is not None else (1.0 if visible else 0.0)
            except (TypeError, ValueError):
                confidence = 1.0 if visible else 0.0
            ann["ball_raw_confidence"] = round(_clamp(confidence, 0.0, 1.0), 3)

    def _reconstruct(
        self,
        annotations: List[MutableMapping[str, object]],
        mode: str,
        hints: Mapping[int, ContactHint],
    ) -> Tuple[Dict[int, ReconstructedPoint], set[int], ReconstructionSummary]:
        raw: Dict[int, Point] = {}
        by_frame: Dict[int, MutableMapping[str, object]] = {}
        ordered_frames: List[int] = []
        for ann in annotations:
            frame = int(ann.get("f", len(ordered_frames)))
            by_frame[frame] = ann
            ordered_frames.append(frame)
            point = _point(ann.get("ball_raw"))
            if point is not None:
                raw[frame] = point

        result: Dict[int, ReconstructedPoint] = {}
        summary = ReconstructionSummary(mode=mode)
        for frame, point in raw.items():
            conf = float(by_frame[frame].get("ball_raw_confidence") or 1.0)
            result[frame] = ReconstructedPoint(*point, "detected", _clamp(conf, 0.0, 1.0))
        summary.detected = len(result)
        out_of_frame: set[int] = set()

        anchors = sorted(raw)
        for left_frame, right_frame in zip(anchors, anchors[1:]):
            gap_frames = [f for f in ordered_frames if left_frame < f < right_frame]
            if not gap_frames:
                continue
            restart = bool(by_frame[right_frame].get("ball_new"))

            left = raw[left_frame]
            right = raw[right_frame]
            v_left = self._anchor_velocity(anchors, raw, left_frame, previous=True, fallback=right)
            v_right = self._anchor_velocity(anchors, raw, right_frame, previous=False, fallback=left)
            smooth = self._hermite_path(left_frame, left, right_frame, right, v_left, v_right)

            evidence = self._occlusion_evidence(gap_frames, smooth, by_frame)
            gap_len = len(gap_frames)
            gap_hints = [h for f, h in sorted(hints.items()) if left_frame < f < right_frame]
            inferred = None if gap_hints else self._infer_contact(
                gap_frames, smooth, by_frame, v_left, v_right,
            )

            # A DP restart normally marks a different hypothesis.  It can also
            # be the same in-play ball re-acquired after a player hid it.  Only
            # bridge that second case: both anchors and most missing frames
            # must be live play, and player/net overlap must be strong.  The
            # contact-aware variant may use a direction break plus weaker
            # overlap; conservative still requires 70% occlusion evidence.
            if restart:
                live_ratio = self._live_play_ratio(gap_frames, by_frame)
                endpoints_live = all(
                    self._is_live_play_state(by_frame[f].get("state"))
                    for f in (left_frame, right_frame)
                )
                minimum_support = 2 if gap_hints else self.config.restart_min_anchor_points
                anchors_supported = all((
                    self._restart_anchor_support(
                        anchors, by_frame, left_frame, forward=False,
                    ) >= minimum_support,
                    self._restart_anchor_support(
                        anchors, by_frame, right_frame, forward=True,
                    ) >= minimum_support,
                ))
                contact_evidence = bool(gap_hints or inferred is not None)
                enough_evidence = (
                    evidence >= self.config.restart_min_occlusion_ratio
                    if mode == "conservative"
                    else evidence >= self.config.restart_min_occlusion_ratio
                    or (
                        contact_evidence
                        and evidence >= self.config.min_occlusion_ratio
                    )
                )
                if (
                    not endpoints_live
                    or live_ratio < 0.8
                    or not anchors_supported
                    or not enough_evidence
                ):
                    summary.gaps_rejected += 1
                    continue

            if self._is_out_of_frame_gap(left, right, v_left, v_right, smooth):
                out_of_frame.update(gap_frames)
                summary.out_of_frame += len(gap_frames)
                continue

            if gap_len <= self.config.short_gap_frames:
                chosen = self._linear_path(left_frame, left, right_frame, right)
                source = "interpolated_short"
                confidence = max(0.62, 0.86 - gap_len * 0.04)
                contacts: set[int] = set()
            elif mode == "conservative":
                if gap_len > self.config.conservative_max_gap_frames:
                    summary.gaps_rejected += 1
                    continue
                if evidence < self.config.min_occlusion_ratio:
                    summary.gaps_rejected += 1
                    continue
                chosen = smooth
                source = "reconstructed_conservative"
                confidence = 0.58 + min(0.18, evidence * 0.22) - gap_len * 0.003
                contacts = set()
            else:
                contact_frames = [h.frame for h in gap_hints]
                if inferred is not None:
                    contact_frames.append(inferred)
                if contact_frames:
                    if gap_len > self.config.contact_max_gap_frames:
                        summary.gaps_rejected += 1
                        continue
                    chosen = self._contact_path(
                        left_frame, left, right_frame, right,
                        v_left, v_right, contact_frames, smooth, by_frame,
                        {h.frame: h.label for h in gap_hints},
                    )
                    if not chosen:
                        summary.gaps_rejected += 1
                        continue
                    source = "reconstructed_contact_aware"
                    confidence = 0.82 if gap_hints else 0.66
                    confidence += min(0.08, evidence * 0.1)
                    confidence -= gap_len * 0.002
                    contacts = set(contact_frames)
                else:
                    if gap_len > self.config.occluded_max_gap_frames:
                        summary.gaps_rejected += 1
                        continue
                    if evidence < self.config.min_occlusion_ratio:
                        summary.gaps_rejected += 1
                        continue
                    chosen = smooth
                    source = "reconstructed_occlusion"
                    confidence = 0.57 + min(0.16, evidence * 0.2) - gap_len * 0.0025
                    contacts = set()

            if not self._path_is_plausible(chosen, allow_contact=bool(contacts)):
                summary.gaps_rejected += 1
                continue

            for frame in gap_frames:
                point = chosen.get(frame)
                if point is None:
                    continue
                result[frame] = ReconstructedPoint(
                    point[0], point[1], source,
                    _clamp(confidence, 0.05, 0.95),
                    contact=frame in contacts,
                )
                summary.reconstructed += 1
                summary.contacts += int(frame in contacts)
            summary.gaps_filled += 1

        covered = set(result) | out_of_frame
        summary.missing = sum(1 for f in ordered_frames if f not in covered)
        return result, out_of_frame, summary

    @staticmethod
    def _is_live_play_state(value: object) -> bool:
        return str(value or "").lower() in {"serving", "rally_active"}

    def _live_play_ratio(
        self,
        frames: Sequence[int],
        annotations: Mapping[int, Mapping[str, object]],
    ) -> float:
        if not frames:
            return 0.0
        return sum(
            self._is_live_play_state(annotations[frame].get("state"))
            for frame in frames
        ) / len(frames)

    def _restart_anchor_support(
        self,
        anchors: Sequence[int],
        annotations: Mapping[int, Mapping[str, object]],
        frame: int,
        *,
        forward: bool,
    ) -> int:
        """Count nearby detector anchors belonging to one DP track."""
        index = anchors.index(frame)
        indices = (
            range(index, len(anchors)) if forward
            else range(index, -1, -1)
        )
        count = 0
        for anchor_index in indices:
            anchor = anchors[anchor_index]
            if abs(anchor - frame) > self.config.restart_anchor_support_frames:
                break
            if forward and anchor_index > index and annotations[anchor].get("ball_new"):
                break
            count += 1
            if not forward and annotations[anchor].get("ball_new"):
                break
        return count

    # ------------------------------------------------------------------
    # Geometry and evidence

    def _anchor_velocity(
        self,
        anchors: Sequence[int],
        raw: Mapping[int, Point],
        frame: int,
        *,
        previous: bool,
        fallback: Point,
    ) -> Velocity:
        index = anchors.index(frame)
        neighbour_index = index - 1 if previous else index + 1
        if 0 <= neighbour_index < len(anchors):
            neighbour_frame = anchors[neighbour_index]
            dt = abs(frame - neighbour_frame)
            if 0 < dt <= 12:
                if previous:
                    return (
                        (raw[frame][0] - raw[neighbour_frame][0]) / dt,
                        (raw[frame][1] - raw[neighbour_frame][1]) / dt,
                    )
                return (
                    (raw[neighbour_frame][0] - raw[frame][0]) / dt,
                    (raw[neighbour_frame][1] - raw[frame][1]) / dt,
                )
        fallback_index = index + 1 if previous else index - 1
        fallback_frame = anchors[fallback_index] if 0 <= fallback_index < len(anchors) else frame
        dt = max(1, abs(frame - fallback_frame))
        if previous:
            return ((fallback[0] - raw[frame][0]) / dt, (fallback[1] - raw[frame][1]) / dt)
        return ((raw[frame][0] - fallback[0]) / dt, (raw[frame][1] - fallback[1]) / dt)

    def _linear_path(self, fa: int, a: Point, fb: int, b: Point) -> Dict[int, Point]:
        span = fb - fa
        return {
            frame: (
                a[0] + (b[0] - a[0]) * ((frame - fa) / span),
                a[1] + (b[1] - a[1]) * ((frame - fa) / span),
            )
            for frame in range(fa + 1, fb)
        }

    def _hermite_path(
        self, fa: int, a: Point, fb: int, b: Point,
        va: Velocity, vb: Velocity,
    ) -> Dict[int, Point]:
        span = fb - fa
        max_speed = self.config.max_speed_px_at_1280 * self.scale
        va = self._limit_velocity(va, max_speed)
        vb = self._limit_velocity(vb, max_speed)
        out: Dict[int, Point] = {}
        for frame in range(fa + 1, fb):
            t = (frame - fa) / span
            t2, t3 = t * t, t * t * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            out[frame] = (
                h00 * a[0] + h10 * span * va[0] + h01 * b[0] + h11 * span * vb[0],
                h00 * a[1] + h10 * span * va[1] + h01 * b[1] + h11 * span * vb[1],
            )
        return out

    @staticmethod
    def _limit_velocity(velocity: Velocity, maximum: float) -> Velocity:
        speed = math.hypot(*velocity)
        if speed <= maximum or speed < 1e-9:
            return velocity
        scale = maximum / speed
        return velocity[0] * scale, velocity[1] * scale

    def _path_is_plausible(self, path: Mapping[int, Point], *, allow_contact: bool) -> bool:
        if not path:
            return False
        ordered = [path[f] for f in sorted(path)]
        margin = 0.02 * max(self.width, self.height)
        if any(not (-margin <= x <= self.width + margin and -margin <= y <= self.height + margin)
               for x, y in ordered):
            return False
        max_speed = self.config.max_speed_px_at_1280 * self.scale
        if allow_contact:
            max_speed *= 1.35
        velocities = [
            (b[0] - a[0], b[1] - a[1]) for a, b in zip(ordered, ordered[1:])
        ]
        if any(math.hypot(*v) > max_speed for v in velocities):
            return False
        if allow_contact:
            return True
        max_accel = self.config.max_accel_px_at_1280 * self.scale
        return not any(
            math.hypot(b[0] - a[0], b[1] - a[1]) > max_accel
            for a, b in zip(velocities, velocities[1:])
        )

    def _occlusion_evidence(
        self,
        frames: Sequence[int],
        path: Mapping[int, Point],
        annotations: Mapping[int, Mapping[str, object]],
    ) -> float:
        if not frames:
            return 0.0
        hits = 0
        for frame in frames:
            point = path.get(frame)
            if point is None:
                continue
            ann = annotations[frame]
            if self._inside_player(point, ann) or self._inside_net_object(point):
                hits += 1
        return hits / len(frames)

    def _inside_player(self, point: Point, ann: Mapping[str, object]) -> bool:
        margin = self.config.player_bbox_margin_px_at_1280 * self.scale
        for pose in ann.get("poses") or []:
            if not isinstance(pose, Mapping):
                continue
            bbox = pose.get("bb")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = map(float, bbox[:4])
            if x1 - margin <= point[0] <= x2 + margin and y1 - margin <= point[1] <= y2 + margin:
                return True
        return False

    def _inside_net_object(self, point: Point) -> bool:
        margin = self.config.net_margin_px_at_1280 * self.scale
        if self.net_top is not None and self.net_bottom is not None:
            x_min, x_max = 0.0, self.width
            if self.net_left and self.net_right:
                x_min, x_max = sorted((self.net_left[0], self.net_right[0]))
            if x_min - margin <= point[0] <= x_max + margin \
                    and self.net_top - margin <= point[1] <= self.net_bottom + margin:
                return True
        for post in (self.net_left, self.net_right):
            if post and abs(point[0] - post[0]) <= margin * 1.5 \
                    and self.net_top is not None \
                    and self.net_top - margin <= point[1] <= post[1] + margin:
                return True
        return False

    def _is_out_of_frame_gap(
        self,
        left: Point,
        right: Point,
        v_left: Velocity,
        v_right: Velocity,
        path: Mapping[int, Point],
    ) -> bool:
        margin_x = self.width * self.config.edge_margin_ratio
        margin_y = self.height * self.config.edge_margin_ratio

        def outward(point: Point, velocity: Velocity) -> Optional[str]:
            if point[0] <= margin_x and velocity[0] < 0:
                return "left"
            if point[0] >= self.width - margin_x and velocity[0] > 0:
                return "right"
            if point[1] <= margin_y and velocity[1] < 0:
                return "top"
            if point[1] >= self.height - margin_y and velocity[1] > 0:
                return "bottom"
            return None

        def inward(point: Point, velocity: Velocity) -> Optional[str]:
            if point[0] <= margin_x and velocity[0] > 0:
                return "left"
            if point[0] >= self.width - margin_x and velocity[0] < 0:
                return "right"
            if point[1] <= margin_y and velocity[1] > 0:
                return "top"
            if point[1] >= self.height - margin_y and velocity[1] < 0:
                return "bottom"
            return None

        edge_out = outward(left, v_left)
        edge_in = inward(right, v_right)
        if edge_out is not None and edge_out == edge_in:
            return True
        return any(x < 0 or x > self.width or y < 0 or y > self.height for x, y in path.values())

    # ------------------------------------------------------------------
    # Contact-aware reconstruction

    def _infer_contact(
        self,
        frames: Sequence[int],
        smooth: Mapping[int, Point],
        annotations: Mapping[int, Mapping[str, object]],
        v_left: Velocity,
        v_right: Velocity,
    ) -> Optional[int]:
        if _angle_deg(v_left, v_right) < self.config.inferred_contact_turn_deg:
            return None
        candidates = [
            frame for frame in frames
            if smooth.get(frame) is not None and self._inside_player(smooth[frame], annotations[frame])
        ]
        if not candidates:
            return None
        middle = (frames[0] + frames[-1]) / 2.0
        return min(candidates, key=lambda frame: abs(frame - middle))

    def _contact_path(
        self,
        left_frame: int,
        left: Point,
        right_frame: int,
        right: Point,
        v_left: Velocity,
        v_right: Velocity,
        contact_frames: Sequence[int],
        smooth: Mapping[int, Point],
        annotations: Mapping[int, Mapping[str, object]],
        labels: Mapping[int, str],
    ) -> Dict[int, Point]:
        contacts = sorted(set(f for f in contact_frames if left_frame < f < right_frame))
        waypoints: List[Tuple[int, Point]] = [(left_frame, left)]
        for frame in contacts:
            predicted = smooth.get(frame)
            if predicted is None:
                continue
            waypoints.append((frame, self._contact_point(predicted, annotations[frame], labels.get(frame, "other"))))
        waypoints.append((right_frame, right))
        if len(waypoints) < 3:
            return {}

        out: Dict[int, Point] = {}
        for index, ((fa, a), (fb, b)) in enumerate(zip(waypoints, waypoints[1:])):
            span = max(1, fb - fa)
            secant = ((b[0] - a[0]) / span, (b[1] - a[1]) / span)
            va = v_left if index == 0 else secant
            vb = v_right if index == len(waypoints) - 2 else secant
            out.update(self._hermite_path(fa, a, fb, b, va, vb))
            if index < len(waypoints) - 2:
                out[fb] = b
        return out

    def _contact_point(self, predicted: Point, ann: Mapping[str, object], label: str) -> Point:
        poses = [pose for pose in (ann.get("poses") or []) if isinstance(pose, Mapping)]
        if not poses:
            return predicted

        margin = self.config.player_bbox_margin_px_at_1280 * self.scale

        def bbox_distance(pose: Mapping[str, object]) -> float:
            bbox = pose.get("bb")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return float("inf")
            x1, y1, x2, y2 = map(float, bbox[:4])
            cx = _clamp(predicted[0], x1 - margin, x2 + margin)
            cy = _clamp(predicted[1], y1 - margin, y2 + margin)
            return _distance(predicted, (cx, cy))

        pose = min(poses, key=bbox_distance)
        bbox = pose.get("bb")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return predicted
        x1, y1, x2, y2 = map(float, bbox[:4])
        clamped = (
            _clamp(predicted[0], x1 - margin, x2 + margin),
            _clamp(predicted[1], y1 - margin, y2 + margin),
        )

        keypoints = pose.get("kp") or []
        visible: Dict[int, Point] = {}
        for index, kp in enumerate(keypoints):
            if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                try:
                    if float(kp[2]) >= 0.25:
                        visible[index] = (float(kp[0]), float(kp[1]))
                except (TypeError, ValueError):
                    pass

        lower = label.lower()
        if any(name in lower for name in ("attack", "serve", "block", "set")):
            preferred = {7, 8, 9, 10, 0}
        elif any(name in lower for name in ("reception", "pass", "dig")):
            preferred = {7, 8, 9, 10, 13, 14, 15, 16}
        else:
            preferred = set(visible)

        candidates: List[Tuple[float, Point]] = [(0.0, clamped)]
        penalty = 14.0 * self.scale
        for index, point in visible.items():
            candidates.append((0.0 if index in preferred else penalty, point))
        return min(candidates, key=lambda item: _distance(predicted, item[1]) + item[0])[1]
