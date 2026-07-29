"""Three-zone attack-origin heatmap.

Only confirmed FSM ``attack`` events are counted.  Each attack is anchored to
the nearest on-court player from the attacking team, projected to court
coordinates, and assigned to the left, centre, or right third of the net.
This deliberately excludes generic trajectory turns: passes, digs, detector
restarts, and interpolation artefacts are not attack origins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


class _CourtProjector(Protocol):
    def image_to_world(self, pt: Tuple[float, float]) -> Tuple[float, float]: ...


@dataclass(frozen=True)
class AttackHeatmapConfig:
    """Court dimensions and duplicate-event suppression."""

    min_separation_frames: int = 10
    court_half_width_m: float = 4.5
    origin_width_margin_m: float = 1.5


class AttackHeatmap:
    """Aggregate confirmed attacks into three equal net-width zones."""

    ZONES = ("left", "center", "right")

    def __init__(self, fps: float, config: Optional[AttackHeatmapConfig] = None):
        self.fps = float(fps or 30.0)
        self.config = config or AttackHeatmapConfig()

    def analyze(
        self,
        annotations: List[Dict[str, Any]],
        projector: Optional[_CourtProjector] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return JSON-serializable attack origins and three zone counts.

        ``events`` is the cleaned FSM event stream produced after trajectory
        post-processing.  Deriving attacks from ``annotation.ev`` is retained
        for callers that only have the annotation file.
        """
        frame_annotations = {
            int(annotation["f"]): annotation for annotation in annotations
        }
        attacks = self._attack_events(annotations, events)
        candidates: List[Dict[str, Any]] = []

        for event in attacks:
            frame = int(event["frame"])
            annotation = frame_annotations.get(frame)
            if not annotation or not annotation.get("ball"):
                continue
            # A restart or interpolated point has no trustworthy local contact
            # geometry and must never create an attack origin.
            if annotation.get("ball_interp") or annotation.get("ball_new"):
                continue

            x, y = map(float, annotation["ball"][:2])
            team = self._normalise_team(event.get("team"))
            zone, source = self._zone_for(
                (x, y), annotation, projector, team,
            )
            if zone is None:
                continue
            candidates.append({
                "frame": frame,
                "time": round(float(event.get("time", frame / self.fps)), 3),
                "x": round(x, 1),
                "y": round(y, 1),
                "team": team,
                "confidence": round(float(event.get("confidence", 1.0)), 3),
                "zone": zone,
                "zone_source": source,
            })

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

        counts = {zone: 0 for zone in self.ZONES}
        for hit in hits:
            counts[hit["zone"]] += 1

        return {
            "version": 2,
            "algorithm": "fsm_attack_origin",
            "zone_layout": "three_equal_net_widths",
            "hits": hits,
            "zones": [
                {"id": zone, "count": counts[zone]} for zone in self.ZONES
            ],
            # Keep the existing key for Viewer/backward compatibility.
            "total_hits": len(hits),
            "total_attacks": len(hits),
            "rejected_attacks": len(attacks) - len(hits),
        }

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

    def _zone_for(
        self,
        image_pt: Tuple[float, float],
        annotation: Dict[str, Any],
        projector: Optional[_CourtProjector],
        team: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        if projector is None:
            return None, None

        pose_foot = self._nearest_pose_foot(
            annotation.get("poses") or [], image_pt, team,
        )
        if pose_foot is None:
            # Projecting an airborne ball onto the ground can land many metres
            # outside the court and does not describe where the attack began.
            return None, None
        try:
            wx, _ = projector.image_to_world(pose_foot)
        except Exception:
            return None, None

        half_width = self.config.court_half_width_m
        # Poses have already passed CourtPlayerGate. Keep a small cross-court
        # margin for ankle/homography noise; depth does not affect which third
        # of the net the attacker occupies.
        if abs(wx) > half_width + self.config.origin_width_margin_m:
            return None, "attacker_foot"
        third_boundary = half_width / 3.0
        if wx < -third_boundary:
            return "left", "attacker_foot"
        if wx > third_boundary:
            return "right", "attacker_foot"
        return "center", "attacker_foot"

    @classmethod
    def _nearest_pose_foot(
        cls,
        poses: List[Dict[str, Any]],
        ball_pt: Tuple[float, float],
        team: Optional[str],
    ) -> Optional[Tuple[float, float]]:
        """Find the attacking teammate most likely to have contacted the ball."""
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
