"""
Player pose extraction and skeleton drawing.

Turns YOLO-pose (17 COCO keypoints) results into ``PlayerPose`` objects with
team assignment (by net_y), optional ByteTrack IDs, and court-bounds filtering.
Also provides skeleton drawing for the annotated video overlay.

COCO keypoint indices:
    0 nose        1 l_eye       2 r_eye       3 l_ear      4 r_ear
    5 l_shoulder  6 r_shoulder  7 l_elbow     8 r_elbow
    9 l_wrist    10 r_wrist    11 l_hip      12 r_hip
   13 l_knee    14 r_knee     15 l_ankle    16 r_ankle
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Keypoint index constants (COCO order used by YOLO-pose)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Bones as pairs of keypoint indices
SKELETON: List[Tuple[int, int]] = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
]

KP_CONF_MIN = 0.30  # keypoints below this confidence are treated as missing

# Broad margins are only for an already-confirmed player that briefly chases a
# ball outside the court.  They must never be used to admit a new track: bench
# players and spectators often stand 1–1.5 m past a sideline.
COURT_MARGIN_X = 1.5        # broad grace margin beyond sidelines
COURT_MARGIN_SERVE = 4.0    # broad grace margin behind end lines (serve zone)
COURT_ENTRY_MARGIN_X = 0.35 # new member: small calibration/ankle tolerance only
COURT_ENTRY_MARGIN_SERVE = 4.0  # a server may legally begin behind the end line


class CourtProjector:
    """
    Maps image points to court coordinates (meters) using the calibration
    homography (world → image, world frame: x ∈ [-4.5, 4.5] across the court,
    y ∈ [-9, 9] along it, y = 0 is the line under the net).

    The near side (closer to camera = the observed team_b) is detected
    automatically by projecting both end lines and comparing image y, so it
    works regardless of which line was clicked as "front" during calibration.
    """

    def __init__(
        self,
        world_to_image,
        half_width: float = 4.5,
        half_length: float = 9.0,
        net_floor_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        net_tape_band: Optional[Tuple[float, float, float, float]] = None,
    ):
        H = np.asarray(world_to_image, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"Expected 3x3 homography, got {H.shape}")
        self._H = H
        self._H_inv = np.linalg.inv(H)  # raises LinAlgError if singular
        self.half_width = half_width
        self.half_length = half_length
        # Which world-y sign is the near side? Larger image y = closer to camera.
        y_pos = self.world_to_image((0.0, half_length))[1]
        y_neg = self.world_to_image((0.0, -half_length))[1]
        self._near_sign = 1.0 if y_pos > y_neg else -1.0
        self._net_floor_line = net_floor_line
        self._net_tape_band = net_tape_band

    @property
    def has_net_floor_line(self) -> bool:
        """True when a post-base line is available for pixel-accurate sides."""
        return self._net_floor_line is not None

    def net_floor_y(self, x: float) -> Optional[float]:
        """Y of the floor directly under the net at image coordinate ``x``."""
        if self._net_floor_line is None:
            return None
        (x1, y1), (x2, y2) = self._net_floor_line
        if abs(x2 - x1) < 1e-6:
            return (y1 + y2) / 2.0
        t = (x - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    def net_floor_points(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return self._net_floor_line

    def bbox_occludes_net(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Whether a player silhouette overlaps the visible net band."""
        if self._net_tape_band is None:
            return False
        net_x1, net_x2, net_top, net_bottom = self._net_tape_band
        x1, y1, x2, y2 = bbox
        return (
            x2 >= net_x1 and x1 <= net_x2
            and y2 >= net_top and y1 <= net_bottom
        )

    def world_to_image(self, wpt: Tuple[float, float]) -> Tuple[float, float]:
        v = self._H @ np.array([wpt[0], wpt[1], 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def image_to_world(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        v = self._H_inv @ np.array([pt[0], pt[1], 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def team_of_world_y(self, wy: float) -> str:
        """team_b = near half (the observed team), team_a = far half."""
        return "team_b" if wy * self._near_sign > 0 else "team_a"

    def team_of(
        self,
        foot_image_pt: Tuple[float, float],
        near_tolerance_px: float = 0.0,
    ) -> str:
        """team_b = near half (the observed team), team_a = far half.

        A net post-base line is the primary divider.  It lives on the ground
        plane, unlike the tape detected by SAM, and therefore avoids mixing
        players in front of the net with players visible through the mesh.
        """
        floor_y = self.net_floor_y(foot_image_pt[0])
        if floor_y is not None:
            # Pose ankles often land slightly above the actual shoe sole,
            # particularly for front-row players partly hidden by the net.
            # Give only the near side a small, caller-controlled cushion; a
            # genuine far-side player remains many pixels above the floor line.
            return "team_b" if foot_image_pt[1] >= floor_y - near_tolerance_px else "team_a"
        _, wy = self.image_to_world(foot_image_pt)
        return self.team_of_world_y(wy)

    def inside_score(
        self,
        foot_image_pt: Tuple[float, float],
        margin_x: float = COURT_MARGIN_X,
        margin_y: float = COURT_MARGIN_SERVE,
    ) -> float:
        """
        How far inside the (margin-expanded) play area the point is, in meters.
        Positive = inside, negative = outside. Used both as an on-court test
        (score >= 0) and to rank candidates when capping players per team.
        """
        wx, wy = self.image_to_world(foot_image_pt)
        dx = (self.half_width + margin_x) - abs(wx)
        dy = (self.half_length + margin_y) - abs(wy)
        return min(dx, dy)

    def entry_score(self, foot_image_pt: Tuple[float, float]) -> float:
        """Strict score used only to admit a new on-court team member."""
        return self.inside_score(
            foot_image_pt,
            margin_x=COURT_ENTRY_MARGIN_X,
            margin_y=COURT_ENTRY_MARGIN_SERVE,
        )


@dataclass(eq=False)  # eq off: numpy keypoints field breaks generated __eq__
class PlayerPose:
    """One detected player with keypoints."""
    team: str                       # "team_a" (far side) / "team_b" (near side)
    center: Tuple[float, float]     # bbox center (cx, cy) — same as legacy player point
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: np.ndarray           # (17, 3) float32: x, y, conf
    track_id: Optional[int] = None  # ByteTrack ID (None if tracking unavailable)
    back_facing_score: Optional[float] = None  # 0..1 face-hidden/back evidence
    team_prior: Optional[str] = None
    source: str = "yolo_pose"
    mask_instance_id: Optional[int] = None

    @property
    def bbox_h(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)

    def kp(self, idx: int) -> Optional[Tuple[float, float]]:
        """Keypoint (x, y) if confident enough, else None."""
        x, y, c = self.keypoints[idx]
        if c < KP_CONF_MIN:
            return None
        return float(x), float(y)

    def wrists(self) -> List[Tuple[float, float]]:
        return [p for p in (self.kp(L_WRIST), self.kp(R_WRIST)) if p is not None]

    def ankles(self) -> List[Tuple[float, float]]:
        return [p for p in (self.kp(L_ANKLE), self.kp(R_ANKLE)) if p is not None]

    def foot_point(self) -> Tuple[float, float]:
        """Ground contact estimate: mean of visible ankles, else bbox bottom-center."""
        ankles = self.ankles()
        if ankles:
            return (sum(a[0] for a in ankles) / len(ankles),
                    max(a[1] for a in ankles))
        return ((self.bbox[0] + self.bbox[2]) / 2.0, self.bbox[3])


def estimate_back_facing_score(pose: PlayerPose) -> Optional[float]:
    """Estimate back-facing evidence from face-vs-torso keypoint visibility.

    YOLO11's anatomical left/right order is not stable enough to distinguish
    front from back in real match footage.  A hidden face is a safer signal:
    compare nose/eye/ear confidence with shoulders and hips.  This remains a
    soft prior and is only consumed together with net occlusion and geometry.
    """
    face_indices = (0, 1, 2, 3, 4)
    torso_indices = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    face_conf = float(np.mean([pose.keypoints[i, 2] for i in face_indices]))
    torso_conf = float(np.mean([pose.keypoints[i, 2] for i in torso_indices]))
    if torso_conf < KP_CONF_MIN:
        return None
    face_ratio = face_conf / max(torso_conf, 1e-6)
    nose_ratio = float(pose.keypoints[NOSE, 2]) / max(torso_conf, 1e-6)
    return float(np.clip(1.0 - 0.7 * face_ratio - 0.3 * nose_ratio, 0.0, 1.0))


def extract_player_poses(
    result,
    net_y: float,
    court_x_range: Optional[Tuple[float, float]] = None,
    min_bbox_h: float = 0.0,
    projector: Optional[CourtProjector] = None,
) -> List[PlayerPose]:
    """
    Convert one ultralytics Results object into a list of PlayerPose.

    Team assignment: by court half via ``projector`` (foot world-y vs the line
    under the net — correct even when the far team's feet are below the net
    tape in pixels), or legacy bbox-center y vs ``net_y`` without one.

    On-court filtering is NOT done here — pass the result through
    ``CourtPlayerGate.filter()`` (stateful, with track hysteresis).
    Without a projector, the legacy ``court_x_range`` filter applies.
    """
    poses: List[PlayerPose] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return poses

    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id
    ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(boxes)
    if result.keypoints is not None:
        kps = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
    else:
        # Non-pose model fallback: keep centers working, keypoints all-zero
        kps = np.zeros((len(boxes), 17, 3), dtype=np.float32)

    for (x1, y1, x2, y2), tid, kp in zip(boxes, ids, kps):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if (y2 - y1) < min_bbox_h:
            continue

        pose = PlayerPose(
            team="",
            center=(float(cx), float(cy)),
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            keypoints=kp.astype(np.float32),
            track_id=int(tid) if tid is not None else None,
        )
        pose.back_facing_score = estimate_back_facing_score(pose)

        if projector is not None:
            pose.team = projector.team_of(pose.foot_point())
        else:
            if court_x_range is not None and not (court_x_range[0] <= cx <= court_x_range[1]):
                continue
            pose.team = "team_a" if cy < net_y else "team_b"

        poses.append(pose)

    return poses


class CourtPlayerGate:
    """
    Stateful on-court filter with track hysteresis.

    Hard cut-offs lose real players: someone chasing the ball 2–3 m outside
    the sideline, or whose feet are briefly mis-estimated (occluded ankles),
    would vanish from the overlay and the features. The gate instead works
    like a membership list:

      - a track becomes a MEMBER only after its feet enter the strict court
        area (small sideline tolerance, legal serve-zone depth) — at ANY
        frame, not just the first one: players occluded at video start join
        as soon as visible
      - a member stays visible while tracked, even outside the court, as long
        as it stays within EXTRA_MARGIN of the play area and was on court
        within the last GRACE_FRAMES
      - OCCLUSION pass: a player emerging from behind a teammate has no
        visible ankles, so the foot falls back to bbox-bottom (= the
        occluder's body) and projects off court. Such detections (no ankles
        AND bbox overlapping an already-kept player AND within EXTRA_MARGIN)
        are kept VISIBLE immediately; full membership is granted once the
        ankles confirm the player is standing on the court
      - spectators/benches/referees never step on court → never members
        (and they are fully visible, so the occlusion pass doesn't admit them)
      - team is sticky per track: assigned where the player clearly stands in
        one half, so it doesn't flicker for players acting at the net
      - at most ``max_per_team`` players kept (ranked by how far inside the
        court they stand)

    Detections without a track_id fall back to the plain on-court test.
    """

    GRACE_FRAMES = 150      # ~5 s at 30 fps: membership survives off-court this long
    EXTRA_MARGIN = 3.0      # meters beyond base margins members may roam
    # A track first seen at the net is frequently assigned to the wrong half:
    # YOLO can put an ankle on the occluding player or the net line.  Do not
    # make that first sample permanent.  Lock only after several consistent,
    # ankle-confirmed observations well away from the centre line.
    TEAM_LOCK_DEPTH = 0.75  # |world y| (smoothed) needed to cast a side vote
    TEAM_LOCK_VOTES = 2     # net signed votes needed before team becomes sticky
    WY_EMA_ALPHA = 0.50     # react quickly when an occluded player reappears

    def __init__(self, projector: CourtProjector, max_per_team: Optional[int] = 6):
        self.projector = projector
        self.max_per_team = max_per_team
        self._last_inside: Dict[int, int] = {}   # track_id -> frame of last on-court foot
        self._team_of: Dict[int, str] = {}       # track_id -> sticky team
        self._wy_ema: Dict[int, float] = {}      # track_id -> smoothed foot world-y
        self._team_votes: Dict[int, int] = {}    # track_id -> signed, trusted half votes
        self._frame = -1

    def reset(self) -> None:
        self._last_inside.clear()
        self._team_of.clear()
        self._wy_ema.clear()
        self._team_votes.clear()
        self._frame = -1

    OCCLUSION_OVERLAP = 0.25  # bbox intersection / smaller-bbox area to count as occluded-by

    def filter(
        self,
        poses: List[PlayerPose],
        serve_setup: bool = False,
    ) -> List[PlayerPose]:
        self._frame += 1
        proj = self.projector

        # Precompute foot/scores once per pose. ``entry_score`` is strict and
        # prevents sideline spectators from ever earning membership; the broad
        # score is retained for grace after a real player has been confirmed.
        entries = []
        for pose in poses:
            foot = pose.foot_point()
            entries.append((pose, foot, proj.entry_score(foot), proj.inside_score(foot)))

        kept: List[PlayerPose] = []
        scores: List[float] = []
        deferred = []  # candidates for the occlusion pass

        # --- Pass 1: strict on-court test + member grace ---
        for pose, foot, entry_score, score in entries:
            tid = pose.track_id
            if entry_score >= 0:
                # Membership only when the feet are trusted (visible ankles);
                # bbox-bottom feet still show the player but don't grant grace
                if tid is not None and pose.ankles():
                    self._last_inside[tid] = self._frame
                keep = True
            else:
                # Member grace: recently-on-court tracks may roam a bit outside
                keep = (
                    tid is not None
                    and tid in self._last_inside
                    and self._frame - self._last_inside[tid] <= self.GRACE_FRAMES
                    and score >= -self.EXTRA_MARGIN
                )
            if keep:
                kept.append(pose)
                scores.append(score)
            else:
                deferred.append((pose, foot, score))

        # --- Pass 2: occluded players emerging behind kept players ---
        # No visible ankles → foot estimate is the occluder's body, so the
        # strict test wrongly rejects. Keep them visible if they overlap an
        # already-kept player and aren't far outside the play area.
        for pose, foot, score in deferred:
            if pose.ankles():
                continue  # feet were visible — the strict rejection stands
            if score < -self.EXTRA_MARGIN:
                continue
            if any(_bbox_overlap(pose.bbox, k.bbox) >= self.OCCLUSION_OVERLAP for k in kept):
                kept.append(pose)
                scores.append(score)

        # A detected ground-level net divider is more reliable than the legacy
        # homography here.  Assign every visible foot directly to its side.
        # This fixes the concrete failure where players in front of the net
        # were rendered gray because their old world_y calibration was offset.
        if proj.has_net_floor_line:
            for pose in kept:
                # Cap the cushion so a large foreground bbox cannot absorb a
                # player actually standing behind the net.  Small/far poses
                # still receive enough tolerance for ankle-keypoint noise.
                tolerance = float(np.clip(pose.bbox_h * 0.06, 5.0, 18.0))
                pose.team = proj.team_of(pose.foot_point(), near_tolerance_px=tolerance)

                # Soft observed-team prior.  The camera-side team commonly
                # faces away from the camera and physically occludes the net.
                # Use this only inside a narrow geometric ambiguity band; it
                # can recover a near player whose ankles were pulled upward by
                # net/teammate occlusion, but never move a clear far-side pose.
                back_score = pose.back_facing_score
                if (
                    back_score is not None
                    and back_score >= 0.45
                    and proj.bbox_occludes_net(pose.bbox)
                ):
                    pose.team_prior = "back_facing_net_occlusion"
                    foot = pose.foot_point()
                    floor_y = proj.net_floor_y(foot[0])
                    ambiguity = float(np.clip(
                        pose.bbox_h * (0.14 if serve_setup else 0.10),
                        10.0,
                        40.0 if serve_setup else 30.0,
                    ))
                    if (
                        floor_y is not None
                        and pose.team == "team_a"
                        and foot[1] >= floor_y - ambiguity
                    ):
                        pose.team = "team_b"
                        pose.team_prior = (
                            "serve_back_facing_net_override"
                            if serve_setup else "back_facing_net_override"
                        )

        # Sticky team: require repeated, trusted ankle measurements before
        # locking.  A player may enter the video while occluded; their first
        # recovered ankle position is not enough evidence to decide a team.
        for pose in kept:
            if proj.has_net_floor_line:
                continue
            tid = pose.track_id
            if tid is None:
                continue
            foot, entry_score, score = next(
                (foot, entry_score, score)
                for p, foot, entry_score, score in entries if p is pose
            )

            if pose.ankles() and entry_score >= 0.0:
                _, wy = proj.image_to_world(foot)
                old_ema = self._wy_ema.get(tid, wy)
                wy_ema = self.WY_EMA_ALPHA * wy + (1.0 - self.WY_EMA_ALPHA) * old_ema
                self._wy_ema[tid] = wy_ema

                # Votes are based on the smoothed point.  Opposite evidence
                # cancels a bad early sample instead of preserving it forever.
                if abs(wy_ema) >= self.TEAM_LOCK_DEPTH and tid not in self._team_of:
                    vote = 1 if proj.team_of_world_y(wy_ema) == "team_b" else -1
                    votes = int(np.clip(self._team_votes.get(tid, 0) + vote,
                                        -self.TEAM_LOCK_VOTES, self.TEAM_LOCK_VOTES))
                    self._team_votes[tid] = votes
                    if abs(votes) >= self.TEAM_LOCK_VOTES:
                        self._team_of[tid] = "team_b" if votes > 0 else "team_a"

            # Until a team is locked, use the latest trusted smoothed point.
            # This remains responsive while the player emerges from occlusion.
            if tid in self._team_of:
                pose.team = self._team_of[tid]
            elif tid in self._wy_ema:
                pose.team = proj.team_of_world_y(self._wy_ema[tid])

        # Purge tracks unseen on court for too long
        stale = [t for t, f in self._last_inside.items()
                 if self._frame - f > self.GRACE_FRAMES]
        for t in stale:
            self._last_inside.pop(t, None)
            self._team_of.pop(t, None)
            self._team_votes.pop(t, None)
            self._wy_ema.pop(t, None)

        # Cap per team, keeping the most-inside players
        if self.max_per_team is not None:
            kept_idx: List[int] = []
            for team in ("team_a", "team_b"):
                members = [(scores[i], i) for i, p in enumerate(kept) if p.team == team]
                members.sort(key=lambda t: -t[0])
                kept_idx.extend(i for _, i in members[:self.max_per_team])
            kept = [kept[i] for i in sorted(kept_idx)]

        return kept


def _bbox_overlap(a: Tuple[float, float, float, float],
                  b: Tuple[float, float, float, float]) -> float:
    """Intersection area as a fraction of the smaller bbox (0..1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1.0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
    return inter / min(area_a, area_b)


def poses_to_teams(poses: List[PlayerPose]) -> Dict[str, List[Tuple[float, float]]]:
    """Legacy format expected by FSM and StreamingEventPredictor."""
    teams: Dict[str, List[Tuple[float, float]]] = {"team_a": [], "team_b": []}
    for p in poses:
        teams[p.team].append(p.center)
    return teams


def draw_skeleton(
    frame: np.ndarray,
    pose: PlayerPose,
    color: Tuple[int, int, int],
    bone_thickness: int = 2,
    kp_radius: int = 3,
    draw_id: bool = False,
) -> None:
    """Draw one player's skeleton in place."""
    kp = pose.keypoints
    for i, j in SKELETON:
        if kp[i, 2] >= KP_CONF_MIN and kp[j, 2] >= KP_CONF_MIN:
            cv2.line(frame,
                     (int(kp[i, 0]), int(kp[i, 1])),
                     (int(kp[j, 0]), int(kp[j, 1])),
                     color, bone_thickness, cv2.LINE_AA)
    for i in range(kp.shape[0]):
        if kp[i, 2] >= KP_CONF_MIN:
            cv2.circle(frame, (int(kp[i, 0]), int(kp[i, 1])), kp_radius, color, -1, cv2.LINE_AA)
    if draw_id and pose.track_id is not None:
        x1, y1 = int(pose.bbox[0]), int(pose.bbox[1])
        cv2.putText(frame, f"#{pose.track_id}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
