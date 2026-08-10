"""Attack origins and shot directions in court coordinates.

Only confirmed FSM ``attack`` events are counted.  Each attack is anchored to
the attacking player's feet, projected to court metres, and assigned to the
left, centre, or right third of the net plus a front-row / back-row marker at
the 3 m attack line.  This deliberately excludes generic trajectory turns:
passes, digs, detector restarts, and interpolation artefacts are not attack
origins.

The direction of the shot is measured, never assumed.  The target is the next
contact by the defending team — a point on the floor, so the same homography
that places the attacker is valid for it.  When the rally ends on that attack
(an ace-like kill, or the ball going out) there is no defender contact, and the
fallback is where the ball crossed the net, marked as an estimate so the viewer
can draw it differently.

``left`` and ``right`` are named the way the video shows them, not by the sign
of whatever world frame the calibration happened to produce; see
``court_setups.world_x_screen_sign``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

HEATMAP_VERSION = 4

# Contacts that can end an attack. A serve never can, and another attack from
# the same team means the defending contact was missed by the FSM.
_DEFENDING_CONTACTS = ("reception", "dig", "block", "pass", "attack")


class _CourtProjector(Protocol):
    def image_to_world(self, pt: Tuple[float, float]) -> Tuple[float, float]: ...

    def world_to_image(self, pt: Tuple[float, float]) -> Tuple[float, float]: ...


@dataclass(frozen=True)
class AttackHeatmapConfig:
    """Court dimensions, direction rule, and duplicate-event suppression."""

    min_separation_frames: int = 10
    court_half_width_m: float = 4.5
    court_half_length_m: float = 9.0
    attack_line_m: float = 3.0
    origin_width_margin_m: float = 1.5
    # Below this angle away from the net normal the ball travelled parallel to
    # the sideline: an attack down the line.
    line_angle_deg: float = 20.0
    target_max_gap_sec: float = 4.0
    net_cross_search_frames: int = 90
    min_target_distance_m: float = 0.75


class AttackHeatmap:
    """Aggregate confirmed attacks into three net zones with shot directions."""

    ZONES = ("left", "center", "right")
    ROWS = ("front", "back")

    def __init__(self, fps: float, config: Optional[AttackHeatmapConfig] = None):
        self.fps = float(fps or 30.0)
        self.config = config or AttackHeatmapConfig()

    def analyze(
        self,
        annotations: List[Dict[str, Any]],
        projector: Optional[_CourtProjector] = None,
        events: Optional[List[Dict[str, Any]]] = None,
        net_y: Optional[float] = None,
        net_x_left: Optional[float] = None,
        net_x_right: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return JSON-serializable attack origins, directions, and zone counts.

        ``events`` is the cleaned FSM event stream produced after trajectory
        post-processing.  Deriving attacks from ``annotation.ev`` is retained
        for callers that only have the annotation file.
        """
        frame_annotations = {
            int(annotation["f"]): annotation for annotation in annotations
        }
        ordered_events = sorted(
            events or [], key=lambda event: int(event.get("frame", 0)),
        )
        attacks = self._attack_events(annotations, events)
        screen_sign = self._screen_sign(projector)
        candidates: List[Dict[str, Any]] = []

        for event in attacks:
            hit = self._build_hit(
                event,
                frame_annotations,
                ordered_events,
                projector,
                screen_sign,
                net_y,
                net_x_left,
                net_x_right,
            )
            if hit is not None:
                candidates.append(hit)

        # The FSM can emit adjacent corrections for the same physical contact.
        # Keep one origin per short temporal neighbourhood.
        hits: List[Dict[str, Any]] = []
        for hit in sorted(candidates, key=lambda item: item["frame"]):
            if (
                hits
                and hit["frame"] - hits[-1]["frame"]
                < self.config.min_separation_frames
            ):
                if hit["confidence"] > hits[-1]["confidence"]:
                    hits[-1] = hit
                continue
            hits.append(hit)

        return {
            "version": HEATMAP_VERSION,
            "algorithm": "fsm_attack_origin_with_measured_target",
            "zone_layout": "three_net_widths_by_front_back_row",
            "orientation": "screen",
            "hits": hits,
            "zones": self._zone_summary(hits),
            "directions": self._direction_summary(hits),
            # Keep the existing keys for Viewer/backward compatibility.
            "total_hits": len(hits),
            "total_attacks": len(hits),
            "rejected_attacks": len(attacks) - len(hits),
        }

    # ── event assembly ────────────────────────────────────────────────

    def _build_hit(
        self,
        event: Dict[str, Any],
        frame_annotations: Dict[int, Dict[str, Any]],
        ordered_events: List[Dict[str, Any]],
        projector: Optional[_CourtProjector],
        screen_sign: float,
        net_y: Optional[float],
        net_x_left: Optional[float],
        net_x_right: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        frame = int(event["frame"])
        annotation = frame_annotations.get(frame)
        if not annotation or not annotation.get("ball"):
            return None
        # A restart or interpolated point has no trustworthy local contact
        # geometry and must never create an attack origin.
        if annotation.get("ball_interp") or annotation.get("ball_new"):
            return None
        if projector is None:
            return None

        ball = (float(annotation["ball"][0]), float(annotation["ball"][1]))
        team = self._normalise_team(event.get("team"))
        # Projecting an airborne ball onto the ground can land many metres
        # outside the court and does not describe where the attack began.
        origin_image = self._nearest_pose_foot(
            annotation.get("poses") or [], ball, team,
        )
        if origin_image is None:
            return None
        origin_world = self._to_world(projector, origin_image)
        if origin_world is None:
            return None

        own_sign = 1.0 if origin_world[1] >= 0 else -1.0
        origin_court = self._to_court(origin_world, screen_sign, own_sign)

        zone = self._zone_of(origin_court[0])
        if zone is None:
            return None

        target = self._find_target(
            frame,
            team,
            ordered_events,
            frame_annotations,
            projector,
            screen_sign,
            own_sign,
            net_y,
            net_x_left,
            net_x_right,
        )
        return {
            "frame": frame,
            "time": round(float(event.get("time", frame / self.fps)), 3),
            "x": round(ball[0], 1),
            "y": round(ball[1], 1),
            "team": team,
            "confidence": round(float(event.get("confidence", 1.0)), 3),
            "zone": zone,
            "row": "front" if abs(origin_court[1]) < self.config.attack_line_m
                   else "back",
            "zone_source": "attacker_foot",
            "origin_image": self._round_point(origin_image),
            "origin_world": self._round_point(origin_world),
            # Court frame: x is screen-left to screen-right, y is depth with
            # the attacker's own half positive and the net at zero. This is
            # what the mirrored visual-stats court draws.
            "origin_court": self._round_point(origin_court),
            "target": target,
            "direction": self._direction(origin_court, target),
        }

    def _find_target(
        self,
        attack_frame: int,
        team: Optional[str],
        ordered_events: List[Dict[str, Any]],
        frame_annotations: Dict[int, Dict[str, Any]],
        projector: _CourtProjector,
        screen_sign: float,
        own_sign: float,
        net_y: Optional[float],
        net_x_left: Optional[float],
        net_x_right: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Where the attacked ball actually went.

        The ball crossing the net is the reliable landmark.  Team labels on
        contacts come from the same court geometry the user is correcting, so
        a mislabelled defender must not hide a shot that visibly crossed.
        """
        crossing = self._net_crossing_target(
            attack_frame,
            frame_annotations,
            net_y,
            net_x_left,
            net_x_right,
        )
        defender = self._defender_contact(
            attack_frame,
            team,
            ordered_events,
            after_frame=crossing["frame"] if crossing else None,
        )
        if defender is not None:
            frame = int(defender["frame"])
            annotation = frame_annotations.get(frame)
            ball = annotation.get("ball") if annotation else None
            if annotation and ball:
                foot = self._nearest_pose_foot(
                    annotation.get("poses") or [],
                    (float(ball[0]), float(ball[1])),
                    self._normalise_team(defender.get("team")),
                )
                world = self._to_world(projector, foot) if foot else None
                court = (
                    self._to_court(world, screen_sign, own_sign)
                    if world is not None else None
                )
                # A "defender" standing on the attacking half is a mislabelled
                # or mis-projected contact, not where the shot landed.
                if court is not None and court[1] < 0:
                    return {
                        "frame": frame,
                        "time": round(float(defender.get(
                            "time", frame / self.fps)), 3),
                        "source": "defender_contact",
                        "event_type": str(defender.get("type", "")),
                        "image": self._round_point(foot),
                        "world": self._round_point(world),
                        "court": self._round_point(court),
                    }

        return crossing

    def _defender_contact(
        self,
        attack_frame: int,
        team: Optional[str],
        ordered_events: List[Dict[str, Any]],
        after_frame: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """First contact that can only have happened on the far side.

        With ``after_frame`` (the net crossing) the next contact of any kind
        is the defender's, whatever the FSM labelled it.  Without a crossing
        the only remaining evidence is the team label, so a contact by the
        attacking team ends the search: the ball never left their half.
        """
        limit = attack_frame + int(self.config.target_max_gap_sec * self.fps)
        floor = attack_frame if after_frame is None else max(attack_frame, after_frame)
        for event in ordered_events:
            frame = int(event.get("frame", -1))
            if frame <= floor:
                continue
            if frame > limit:
                return None
            if str(event.get("type", "")).lower() not in _DEFENDING_CONTACTS:
                continue
            if after_frame is None:
                if team is not None and self._normalise_team(event.get("team")) == team:
                    return None
            return event
        return None

    def _net_crossing_target(
        self,
        attack_frame: int,
        frame_annotations: Dict[int, Dict[str, Any]],
        net_y: Optional[float],
        net_x_left: Optional[float],
        net_x_right: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Lateral position where the ball passed the net, as an estimate.

        The ball is above the floor, so the floor homography cannot place it.
        Interpolating between the net posts gives the lateral position along
        the net and ignores the ball's height, which is why the result is
        flagged as an estimate rather than a measured floor point.
        """
        if net_y is None or net_x_left is None or net_x_right is None:
            return None
        span = float(net_x_right) - float(net_x_left)
        if abs(span) < 1e-6:
            return None

        previous = frame_annotations.get(attack_frame, {}).get("ball")
        if not previous:
            return None
        previous_y = float(previous[1])
        for offset in range(1, self.config.net_cross_search_frames + 1):
            annotation = frame_annotations.get(attack_frame + offset)
            if not annotation:
                continue
            ball = annotation.get("ball")
            if not ball:
                continue
            current_y = float(ball[1])
            if (previous_y - net_y) * (current_y - net_y) <= 0 and previous_y != current_y:
                ratio = (net_y - previous_y) / (current_y - previous_y)
                cross_x = float(previous[0]) + ratio * (float(ball[0]) - float(previous[0]))
                # The posts are read straight off the image, so this fraction
                # already runs screen-left to screen-right and needs no sign
                # correction from the world frame.
                lateral = (cross_x - float(net_x_left)) / span - 0.5
                half_width = self.config.court_half_width_m
                court_x = lateral * 2.0 * half_width
                # A crossing outside the posts is either a shot past the
                # antenna or a mis-measured net span. Either way the lateral
                # scale stops being meaningful past the antenna, so the point
                # is pinned there and flagged instead of drawn off-court.
                outside = abs(court_x) > half_width
                court_x = max(-half_width, min(half_width, court_x))
                return {
                    "outside_net_span": outside,
                    "frame": attack_frame + offset,
                    "time": round((attack_frame + offset) / self.fps, 3),
                    "source": "net_crossing_estimate",
                    "event_type": "",
                    "image": [round(cross_x, 1), round(float(net_y), 1)],
                    "world": None,
                    "court": [round(court_x, 3), 0.0],
                }
            previous = ball
            previous_y = current_y
        return None

    def _direction(
        self,
        origin_court: Tuple[float, float],
        target: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Classify the measured shot as down the line or cross court."""
        if not target or not target.get("court"):
            return None
        tx, ty = float(target["court"][0]), float(target["court"][1])
        dx = tx - origin_court[0]
        dy = ty - origin_court[1]
        distance = math.hypot(dx, dy)
        if distance < self.config.min_target_distance_m:
            return None
        angle = math.degrees(math.atan2(abs(dx), abs(dy)))
        return {
            "kind": "line" if angle < self.config.line_angle_deg else "diagonal",
            "angle_deg": round(angle, 1),
            "dx_m": round(dx, 2),
            "dy_m": round(dy, 2),
            "distance_m": round(distance, 2),
            "target_source": target.get("source"),
        }

    # ── geometry helpers ──────────────────────────────────────────────

    @staticmethod
    def _screen_sign(projector: Optional[_CourtProjector]) -> float:
        """+1 when a larger world x is further right on screen."""
        world_to_image = getattr(projector, "world_to_image", None)
        if world_to_image is None:
            return 1.0
        try:
            right = world_to_image((4.5, 0.0))[0]
            left = world_to_image((-4.5, 0.0))[0]
        except Exception:
            return 1.0
        return 1.0 if right >= left else -1.0

    @staticmethod
    def _to_world(
        projector: _CourtProjector,
        point: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if point is None:
            return None
        try:
            wx, wy = projector.image_to_world(point)
        except Exception:
            return None
        if not (math.isfinite(wx) and math.isfinite(wy)):
            return None
        return float(wx), float(wy)

    @staticmethod
    def _to_court(
        world: Tuple[float, float],
        screen_sign: float,
        own_sign: Optional[float],
    ) -> Tuple[float, float]:
        """World metres -> screen-oriented court metres.

        ``own_sign`` normalises depth so the attacking half is positive and
        the defending half negative, which lets the mirrored visual-stats
        court draw both teams' attacks in one picture.
        """
        depth = world[1] if own_sign is None else world[1] * own_sign
        return world[0] * screen_sign, depth

    def _zone_of(self, court_x: float) -> Optional[str]:
        half_width = self.config.court_half_width_m
        # Poses have already passed CourtPlayerGate. Keep a small cross-court
        # margin for ankle/homography noise; depth does not affect which third
        # of the net the attacker occupies.
        if abs(court_x) > half_width + self.config.origin_width_margin_m:
            return None
        third = half_width / 3.0
        if court_x < -third:
            return "left"
        if court_x > third:
            return "right"
        return "center"

    def _zone_summary(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary = []
        for zone in self.ZONES:
            in_zone = [hit for hit in hits if hit["zone"] == zone]
            summary.append({
                "id": zone,
                "count": len(in_zone),
                "front": sum(1 for hit in in_zone if hit["row"] == "front"),
                "back": sum(1 for hit in in_zone if hit["row"] == "back"),
            })
        return summary

    @staticmethod
    def _direction_summary(hits: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"line": 0, "diagonal": 0, "unknown": 0}
        for hit in hits:
            direction = hit.get("direction")
            kind = direction.get("kind") if direction else None
            counts[kind if kind in counts else "unknown"] += 1
        return counts

    @staticmethod
    def _attack_events(
        annotations: List[Dict[str, Any]],
        events: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if events is not None:
            return [
                event for event in events
                if str(event.get("type", "")).lower() == "attack"
            ]

        derived: List[Dict[str, Any]] = []
        for annotation in annotations:
            if any(str(event).lower() == "attack"
                   for event in annotation.get("ev") or []):
                derived.append({
                    "frame": int(annotation["f"]),
                    "time": annotation.get("t"),
                    "type": "attack",
                })
        return derived

    @staticmethod
    def _round_point(
        point: Optional[Tuple[float, float]],
    ) -> Optional[List[float]]:
        if point is None:
            return None
        return [round(float(point[0]), 3), round(float(point[1]), 3)]

    @classmethod
    def _nearest_pose_foot(
        cls,
        poses: List[Dict[str, Any]],
        ball_pt: Tuple[float, float],
        team: Optional[str],
    ) -> Optional[Tuple[float, float]]:
        """Find the teammate most likely to have contacted the ball."""
        bx, by = ball_pt
        best: Optional[Tuple[float, float]] = None
        best_score = float("inf")
        for pose in poses:
            if team and cls._normalise_team(pose.get("team")) != team:
                continue
            bbox = pose.get("bb") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            keypoints = pose.get("kp") or []
            ankles = []
            for index in (15, 16):
                if (
                    index < len(keypoints)
                    and len(keypoints[index]) >= 3
                    and float(keypoints[index][2]) >= 0.2
                ):
                    ankles.append((
                        float(keypoints[index][0]),
                        float(keypoints[index][1]),
                    ))
            if ankles:
                foot = (
                    sum(point[0] for point in ankles) / len(ankles),
                    max(point[1] for point in ankles),
                )
            else:
                foot = ((x1 + x2) / 2.0, y2)
            score = abs(foot[0] - bx) + 0.30 * abs(foot[1] - by)
            if score < best_score:
                best_score, best = score, foot
        return best

    @staticmethod
    def _normalise_team(value: Any) -> Optional[str]:
        raw = str(value or "").lower()
        if raw in ("a", "team_a"):
            return "team_a"
        if raw in ("b", "team_b"):
            return "team_b"
        return None
