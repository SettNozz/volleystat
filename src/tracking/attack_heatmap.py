"""Trajectory-only attack/contact heatmap.

The detector deliberately runs *after* ball-trajectory cleanup.  It does not
depend on rally state or event labels: a candidate hit is a locally sharp turn
of the ball velocity vector.  Court zones are an approximate ground-plane
projection of the hit image coordinate, suitable for an aggregate heatmap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


class _CourtProjector(Protocol):
    def image_to_world(self, pt: Tuple[float, float]) -> Tuple[float, float]: ...
    def team_of_world_y(self, wy: float) -> str: ...


@dataclass(frozen=True)
class AttackHeatmapConfig:
    """Conservative defaults for 30-fps volleyball footage."""

    window_frames: int = 3
    min_speed_px_per_sec: float = 120.0
    min_direction_change_deg: float = 50.0
    min_velocity_consistency: float = 0.72
    min_separation_frames: int = 10
    court_half_width_m: float = 4.5
    court_half_length_m: float = 9.0
    attack_line_m: float = 3.0


class AttackHeatmap:
    """Extract sharp trajectory turns and aggregate them into court zones."""

    def __init__(self, fps: float, config: Optional[AttackHeatmapConfig] = None):
        self.fps = float(fps or 30.0)
        self.config = config or AttackHeatmapConfig()

    def analyze(
        self,
        annotations: List[Dict[str, Any]],
        projector: Optional[_CourtProjector] = None,
    ) -> Dict[str, Any]:
        """Return JSON-serializable hit candidates and 3×2 zone counts.

        Interpolated observations are useful for display but never become hit
        centres or velocity anchors: otherwise an artificial straight line in
        a detector gap could look like a contact.
        """
        samples = [a for a in annotations if a.get("ball")]
        candidates: List[Dict[str, Any]] = []
        w = self.config.window_frames
        for i in range(w, len(samples) - w):
            before, centre, after = samples[i - w], samples[i], samples[i + w]
            segment = samples[i - w:i + w + 1]
            if any(sample.get("ball_interp") for sample in segment):
                continue
            f0, f1, f2 = int(before["f"]), int(centre["f"]), int(after["f"])
            if f1 <= f0 or f2 <= f1:
                continue
            incoming, speed_before, consistency_before = self._segment_velocity(samples[i - w:i + 1])
            outgoing, speed_after, consistency_after = self._segment_velocity(samples[i:i + w + 1])
            if min(consistency_before, consistency_after) < self.config.min_velocity_consistency:
                continue
            if min(speed_before, speed_after) < self.config.min_speed_px_per_sec:
                continue
            cosine = float(np.dot(incoming, outgoing) / (speed_before * speed_after))
            angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
            if angle < self.config.min_direction_change_deg:
                continue
            confidence = self._confidence(
                angle, speed_before, speed_after, min(consistency_before, consistency_after),
            )
            x, y = (float(centre["ball"][0]), float(centre["ball"][1]))
            zone, zone_source = self._zone_for((x, y), centre, projector)
            candidates.append({
                "frame": f1,
                "time": round(float(centre.get("t", f1 / self.fps)), 3),
                "x": round(x, 1),
                "y": round(y, 1),
                "angle_deg": round(angle, 1),
                "speed_before": round(speed_before, 1),
                "speed_after": round(speed_after, 1),
                "trajectory_consistency": round(min(consistency_before, consistency_after), 3),
                "confidence": round(confidence, 3),
                "zone": zone,
                "zone_source": zone_source,
            })

        # Non-maximum suppression in time: a physical hit may produce sharp
        # turns in adjacent frame windows. Keep only the strongest one.
        accepted: List[Dict[str, Any]] = []
        for hit in sorted(candidates, key=lambda h: h["confidence"], reverse=True):
            if all(abs(hit["frame"] - other["frame"]) >= self.config.min_separation_frames
                   for other in accepted):
                accepted.append(hit)
        hits = sorted(accepted, key=lambda h: h["frame"])

        zone_counts: Dict[str, int] = {}
        for hit in hits:
            zone = hit["zone"]
            if zone:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
        return {
            "version": 1,
            "algorithm": "trajectory_turn",
            "window_frames": w,
            "min_direction_change_deg": self.config.min_direction_change_deg,
            "min_speed_px_per_sec": self.config.min_speed_px_per_sec,
            "hits": hits,
            "zones": [{"id": zone, "count": count} for zone, count in sorted(zone_counts.items())],
            "total_hits": len(hits),
        }

    def _segment_velocity(self, samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, float, float]:
        """Mean velocity plus directional coherence for a short segment."""
        velocities = []
        for previous, current in zip(samples, samples[1:]):
            dt_frames = int(current["f"]) - int(previous["f"])
            if dt_frames <= 0:
                continue
            p0 = np.asarray(previous["ball"], dtype=np.float64)
            p1 = np.asarray(current["ball"], dtype=np.float64)
            velocities.append((p1 - p0) * self.fps / dt_frames)
        if not velocities:
            return np.zeros(2), 0.0, 0.0
        vectors = np.asarray(velocities)
        mean = vectors.mean(axis=0)
        magnitudes = np.linalg.norm(vectors, axis=1)
        speed = float(np.linalg.norm(mean))
        # 1.0 = all steps agree perfectly; 0.0 = random walk / jitter.
        consistency = speed / max(float(magnitudes.mean()), 1e-6)
        return mean, speed, consistency

    def _confidence(
        self,
        angle_deg: float,
        speed_before: float,
        speed_after: float,
        consistency: float,
    ) -> float:
        angle_score = (angle_deg - self.config.min_direction_change_deg) / (
            180.0 - self.config.min_direction_change_deg
        )
        speed_score = min(speed_before, speed_after) / (self.config.min_speed_px_per_sec * 3.0)
        return float(np.clip(
            0.50 * angle_score + 0.30 * min(speed_score, 1.0) + 0.20 * consistency,
            0.0, 1.0,
        ))

    def _zone_for(
        self,
        image_pt: Tuple[float, float],
        annotation: Dict[str, Any],
        projector: Optional[_CourtProjector],
    ) -> Tuple[Optional[str], Optional[str]]:
        if projector is None:
            return None, None
        # A ball at contact is above the court, so direct image→ground
        # projection can land outside the court.  The nearest retained player
        # pose gives the hitter's ground location and therefore a meaningful
        # attack zone.  Fall back to the ball only when no pose is available.
        pose_foot = self._nearest_pose_foot(annotation.get("poses") or [], image_pt)
        ground_pt = pose_foot or image_pt
        source = "nearest_player_foot" if pose_foot else "ball_projection"
        try:
            wx, wy = projector.image_to_world(ground_pt)
        except Exception:
            return None, None
        c = self.config
        if abs(wx) > c.court_half_width_m or abs(wy) > c.court_half_length_m:
            return None, source
        team = projector.team_of_world_y(wy)
        depth = "front" if abs(wy) <= c.attack_line_m else "back"
        column = ("left" if wx < -c.court_half_width_m / 3.0
                  else "right" if wx > c.court_half_width_m / 3.0 else "middle")
        return f"{team}_{depth}_{column}", source

    @staticmethod
    def _nearest_pose_foot(
        poses: List[Dict[str, Any]],
        ball_pt: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Best ground anchor among the on-court skeletons in one frame."""
        bx, by = ball_pt
        best: Optional[Tuple[float, float]] = None
        best_score = float("inf")
        for pose in poses:
            bbox = pose.get("bb") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            keypoints = pose.get("kp") or []
            ankles = []
            for idx in (15, 16):
                if idx < len(keypoints) and len(keypoints[idx]) >= 3 and float(keypoints[idx][2]) >= 0.2:
                    ankles.append((float(keypoints[idx][0]), float(keypoints[idx][1])))
            if ankles:
                foot = (sum(p[0] for p in ankles) / len(ankles), max(p[1] for p in ankles))
            else:
                foot = ((x1 + x2) / 2.0, y2)
            # Ball-to-foot vertical distance is naturally larger than the
            # horizontal one, hence the lower y weight.  Retained poses have
            # already passed CourtPlayerGate, so spectators cannot win here.
            score = abs(foot[0] - bx) + 0.30 * abs(foot[1] - by)
            if score < best_score:
                best_score, best = score, foot
        return best
