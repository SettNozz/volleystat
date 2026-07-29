"""
Volleyball game analyzer pipeline.

Combines YOLO ball detection + YOLO-pose skeleton + FSM event tracking.
Works entirely in pixel coordinates. No world coordinate transforms needed.
"""

import cv2
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project root
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.detection.player_pose import (
    CourtPlayerGate, CourtProjector, PlayerPose, draw_skeleton,
    extract_player_poses, poses_to_teams,
)
from src.tracking.integrated_volleyball_fsm import CourtBounds, IntegratedVolleyballFSM
from src.tracking.attack_heatmap import AttackHeatmap
from src.tracking.ball_track_gate import BallCandidate, BallTrackGate
from src.tracking.ball_trajectory_dp import BallTrajectoryDP, interpolate_gaps
from src.tracking.pose_feature_extractor import POSE_FEATURE_NAMES, PoseFeatureExtractor
from src.tracking.serve_reception import annotate_rallies
from src.utils.logger import get_logger

log = get_logger("Analyzer")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Measured on 232 human-labeled frames of video 3 (results/ball_eval): this
# model finds the ball on 0.72 of them at imgsz 1280, against 0.59 for the
# curated8 weights that used to be the default. The old "curated8 55% vs
# fullframe 38%" comparison counted frames where the detector fired, not frames
# where it was right.
_BALL_MODEL = "models/best_fullframe/weights/best.pt"
_BALL_IMGSZ = 1280
# Second opinion for the candidate cloud. curated8 sliced into tiles finds balls
# the primary model misses (union recall 0.93 vs 0.72); the trajectory solver
# sorts out which of the extra candidates are real. Costs a second forward pass
# per tile, so it is opt-in.
_BALL_MODEL_2 = "models/yolov8_curated_training/yolov8n_volleyball_curated8/weights/best.pt"
_BALL_TILE_GRID_2 = (3, 2)
_POSE_MODEL = "models/pretrained/yolo11m-pose.pt"
_POSE_FALLBACK = "yolo11n-pose.pt"  # will be auto-downloaded


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationData:
    """Key pixel-space references extracted from court_calibration.json."""
    net_y: float        # Net line Y position (pixels)
    net_x_left: float  # Net left X (pixels)
    net_x_right: float # Net right X (pixels)
    court_top: float   # Approx top of play area (pixels)
    court_bottom: float
    court_left: float
    court_right: float
    frame_width: int
    frame_height: int
    # SAM-detected net tape edges (optional, in same coordinate system as net_y)
    net_tape_top: Optional[float] = None
    net_tape_bottom: Optional[float] = None
    # Ground-level line between the two net-post bases. Unlike the tape, this
    # line lies on the court and can reliably divide the teams.
    net_floor_left: Optional[Tuple[float, float]] = None
    net_floor_right: Optional[Tuple[float, float]] = None
    # ``sam3_post_mask`` is trusted; older Hough values are automatically
    # refreshed once by SAM3 before being used as the team divider.
    net_floor_method: Optional[str] = None
    # Homography world (court meters) → image pixels, scaled to current frame.
    # Enables the on-court player filter and court-half team split.
    homography: Optional[List[List[float]]] = None


@dataclass
class RallySummary:
    start_frame: int
    end_frame: int
    duration_frames: int
    events: List[Dict]
    contacts_team_a: int = 0
    contacts_team_b: int = 0


@dataclass
class GameStats:
    video_name: str = ""
    total_frames: int = 0
    processed_frames: int = 0
    ball_detected_frames: int = 0
    fps: float = 30.0
    rallies: List[Dict] = field(default_factory=list)
    all_events: List[Dict] = field(default_factory=list)
    # Per-frame data for HTML viewer overlay (ball pos, players, FSM state)
    frame_annotations: List[Dict] = field(default_factory=list)
    # Trajectory-only contact/attack candidates aggregated into court zones.
    attack_heatmap: Dict[str, Any] = field(default_factory=dict)
    processing_time_sec: float = 0.0

    @property
    def ball_detection_rate(self) -> float:
        return self.ball_detected_frames / max(self.processed_frames, 1)

    def to_dict(self) -> Dict:
        return {
            "video": self.video_name,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "ball_detection_rate": round(self.ball_detection_rate, 3),
            "total_rallies": len(self.rallies),
            "rallies": self.rallies,
            "all_events": self.all_events,
            "attack_heatmap": self.attack_heatmap,
            "processing_time_sec": round(self.processing_time_sec, 1),
        }


# ---------------------------------------------------------------------------
# GameAnalyzer
# ---------------------------------------------------------------------------

class GameAnalyzer:
    """
    Main pipeline:
        video / frames dir → ball + pose detection → FSM → GameStats

    Usage:
        analyzer = GameAnalyzer()
        stats = analyzer.process_video("game.mp4", output_dir="results/")
        # or
        stats = analyzer.process_frames("data/frames/", output_dir="results/")
    """

    # Default event model for overlay (LSTM v4 — best macro_f1=0.377, 1211 clips)
    _DEFAULT_EVENT_CHECKPOINT = "action_recognition/models/checkpoints_lstm_v4/best_event_sequence.pt"

    def __init__(
        self,
        calibration_path: Optional[str] = "court_calibration.json",
        ball_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        output_video: bool = True,
        skip_frames: int = 0,
        net_y_override: Optional[float] = None,
        event_checkpoint_path: Optional[str] = None,
        max_players_per_team: int = 6,
        pose_imgsz: Optional[int] = None,
        refresh_net_floor: bool = False,
        ball_ensemble: bool = False,
        ball_trajectory_dp: bool = True,
    ):
        """
        Args:
            calibration_path: Path to court_calibration.json (relative to project root or absolute)
            ball_model_path:  Override ball detection model
            pose_model_path:  Override pose model
            output_video:     Produce annotated video / frame outputs
            skip_frames:      Process every (skip_frames+1)-th frame for speed
            event_checkpoint_path: Optional path to event sequence model (Transformer v3) for overlay.
            ball_ensemble: Also collect candidates from a second, tiled detector.
            ball_trajectory_dp: Re-pick the ball trajectory globally once the whole
                       video has been read (see BallTrajectoryDP). The live gate
                       still runs, because the annotated video needs a position
                       while the frame is in hand.
        """
        self.output_video = output_video
        self.skip_frames = skip_frames
        self.net_y_override = net_y_override
        self.ball_ensemble = ball_ensemble
        self.ball_trajectory_dp = ball_trajectory_dp
        # Raw per-frame candidates, kept for the global trajectory pass.
        self._ball_candidates: Dict[int, List[BallCandidate]] = {}

        self._pose_track_warned = False
        self.max_players_per_team = max_players_per_team
        self.pose_imgsz = pose_imgsz  # e.g. 1280 improves recall on small/far players
        self.refresh_net_floor = refresh_net_floor
        self._projector: Optional[CourtProjector] = None
        self._projector_calib: Optional[CalibrationData] = None
        self._court_gate: Optional[CourtPlayerGate] = None

        # Resolve and store calibration path for SAM net write-back
        self._calibration_path: Optional[Path] = None
        if calibration_path:
            p = Path(calibration_path)
            self._calibration_path = p if p.is_absolute() else _ROOT / p

        self.calibration = self._load_calibration(calibration_path)
        self._init_ball_detector(ball_model_path or str(_ROOT / _BALL_MODEL))
        self._init_pose_model(pose_model_path or str(_ROOT / _POSE_MODEL))
        self._event_predictor = self._init_event_predictor(event_checkpoint_path)
        self._warmup()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_calibration(self, path: Optional[str]) -> Optional[CalibrationData]:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        if not p.exists():
            log.warning(f"Calibration not found: {p} — using defaults")
            return None

        with open(p) as f:
            data = json.load(f)

        net_pts   = data.get("net_line_points", [])
        back_pts  = data.get("back_line_points", [])
        left_pts  = data.get("left_sideline_points", [])
        right_pts = data.get("right_sideline_points", [])

        net_y      = float(np.mean([p[1] for p in net_pts]))   if net_pts   else 800.0
        net_xs     = [p[0] for p in net_pts]
        court_bot  = float(np.mean([p[1] for p in back_pts]))  if back_pts  else 1500.0
        court_left = float(min(p[0] for p in left_pts))        if left_pts  else 0.0
        court_right= float(max(p[0] for p in right_pts))       if right_pts else float(data.get("frame_width", 3840))

        # Load SAM-detected net tape positions if available.
        # net_*_y_720p are stored in 720p space — convert immediately to the
        # native calibration resolution so all CalibrationData fields share
        # one coordinate system (then _scale_calibration scales uniformly).
        sam = data.get("net_detection_sam", {})
        native_h = float(data.get("frame_height", 2160))
        sam_sy = native_h / 720.0
        net_tape_top    = float(sam["net_top_y_720p"])    * sam_sy if sam.get("net_top_y_720p")    else None
        net_tape_bottom = float(sam["net_bottom_y_720p"]) * sam_sy if sam.get("net_bottom_y_720p") else None
        floor_left_720 = sam.get("net_floor_left_720p")
        floor_right_720 = sam.get("net_floor_right_720p")
        native_w = float(data.get("frame_width", 3840))
        net_floor_left = (
            (float(floor_left_720[0]) * native_w / 1280.0,
             float(floor_left_720[1]) * sam_sy)
            if floor_left_720 else None
        )
        net_floor_right = (
            (float(floor_right_720[0]) * native_w / 1280.0,
             float(floor_right_720[1]) * sam_sy)
            if floor_right_720 else None
        )
        net_floor_method = str(sam.get("net_floor_method")) if floor_left_720 and floor_right_720 else None
        if net_floor_left and net_floor_right:
            # Calibrations persisted before the floor line was levelled still
            # carry a noise tilt, so re-level on load instead of requiring a
            # fresh detection run.
            from src.detection.net_detector import level_floor_line
            net_floor_left, net_floor_right = level_floor_line(net_floor_left, net_floor_right)

        return CalibrationData(
            net_y=net_y,
            net_x_left=float(min(net_xs)) if net_xs else court_left,
            net_x_right=float(max(net_xs)) if net_xs else court_right,
            # court_top is set very high in the frame (low y) so that balls in the air
            # (above the net) are NOT flagged as "out of bounds".
            court_top=50.0,
            court_bottom=court_bot,
            court_left=court_left,
            court_right=court_right,
            frame_width=data.get("frame_width", 3840),
            frame_height=data.get("frame_height", 2160),
            net_tape_top=net_tape_top,
            net_tape_bottom=net_tape_bottom,
            net_floor_left=net_floor_left,
            net_floor_right=net_floor_right,
            net_floor_method=net_floor_method,
            homography=data.get("perspective_matrix"),
        )

    def _scale_calibration(self, actual_w: int, actual_h: int) -> "CalibrationData":
        """Return calibration scaled to actual frame size, with optional net_y override."""
        c = self.calibration
        if c is None:
            # No calibration: build defaults with override if provided
            net_y = self.net_y_override if self.net_y_override else actual_h * 0.5
            return CalibrationData(
                net_y=net_y, net_x_left=0.0, net_x_right=float(actual_w),
                court_top=10.0, court_bottom=float(actual_h),
                court_left=0.0, court_right=float(actual_w),
                frame_width=actual_w, frame_height=actual_h,
            )
        if c.frame_width != actual_w or c.frame_height != actual_h:
            sx = actual_w / c.frame_width
            sy = actual_h / c.frame_height
            log.info(f"Scaling calibration {c.frame_width}x{c.frame_height} "
                     f"→ {actual_w}x{actual_h}  (sx={sx:.3f} sy={sy:.3f})")
            # Homography maps world → calibration pixels; compose with the
            # pixel scaling so it maps world → actual-resolution pixels.
            homography_scaled = None
            if c.homography is not None:
                S = np.diag([sx, sy, 1.0])
                homography_scaled = (S @ np.asarray(c.homography, dtype=np.float64)).tolist()
            c = CalibrationData(
                net_y=c.net_y * sy,
                net_x_left=c.net_x_left * sx,
                net_x_right=c.net_x_right * sx,
                court_top=c.court_top * sy,
                court_bottom=c.court_bottom * sy,
                court_left=c.court_left * sx,
                court_right=c.court_right * sx,
                frame_width=actual_w,
                frame_height=actual_h,
                net_tape_top=c.net_tape_top * sy if c.net_tape_top else None,
                net_tape_bottom=c.net_tape_bottom * sy if c.net_tape_bottom else None,
                net_floor_left=(c.net_floor_left[0] * sx, c.net_floor_left[1] * sy)
                    if c.net_floor_left else None,
                net_floor_right=(c.net_floor_right[0] * sx, c.net_floor_right[1] * sy)
                    if c.net_floor_right else None,
                net_floor_method=c.net_floor_method,
                homography=homography_scaled,
            )
        if self.net_y_override is not None:
            log.info(f"net_y override: {c.net_y:.1f} -> {self.net_y_override}")
            c = CalibrationData(
                net_y=self.net_y_override,
                net_x_left=c.net_x_left, net_x_right=c.net_x_right,
                court_top=c.court_top, court_bottom=c.court_bottom,
                court_left=c.court_left, court_right=c.court_right,
                frame_width=c.frame_width, frame_height=c.frame_height,
                net_tape_top=c.net_tape_top, net_tape_bottom=c.net_tape_bottom,
                net_floor_left=c.net_floor_left, net_floor_right=c.net_floor_right,
                net_floor_method=c.net_floor_method,
                homography=c.homography,
            )
        return c

    def _init_ball_detector(self, model_path: str):
        from detection.detect_ball_on_video import BallDetectorVideo
        self.ball_detector = BallDetectorVideo(model_path, imgsz=_BALL_IMGSZ)
        self.ball_detector_2 = None
        if self.ball_ensemble:
            self.ball_detector_2 = BallDetectorVideo(
                str(_ROOT / _BALL_MODEL_2), imgsz=640, tile_grid=_BALL_TILE_GRID_2,
            )
            log.info(f"Ball ensemble: second opinion from {Path(_BALL_MODEL_2).parent.parent.name} "
                     f"tiled {_BALL_TILE_GRID_2[0]}x{_BALL_TILE_GRID_2[1]}")
        self.ball_gate = BallTrackGate()

    def _init_pose_model(self, model_path: str):
        from ultralytics import YOLO
        if Path(model_path).exists():
            self.pose_model = YOLO(model_path)
            log.info(f"Pose model loaded: {Path(model_path).name}")
        else:
            log.warning(f"Pose model not found at {model_path} — downloading {_POSE_FALLBACK}")
            self.pose_model = YOLO(_POSE_FALLBACK)

    def _init_event_predictor(self, event_checkpoint_path: Optional[str]):
        """Load sliding-window event predictor (Transformer v3) for overlay; None if disabled or file missing."""
        if event_checkpoint_path == "":
            return None
        path = event_checkpoint_path if event_checkpoint_path else self._DEFAULT_EVENT_CHECKPOINT
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        if not p.exists():
            log.info(f"Event overlay disabled: checkpoint not found at {p}")
            return None
        try:
            from action_recognition.inference.streaming_event_predictor import StreamingEventPredictor
            return StreamingEventPredictor(p)
        except Exception as e:
            log.warning(f"Event overlay disabled: {e}")
            return None

    def _warmup(self):
        """
        Run one dummy inference on both models to trigger CUDA JIT / cudnn benchmark.
        torch.backends.cudnn.benchmark=True (set by BallDetectorVideo) profiles all CUDA
        kernels on the first call — this takes 5+ minutes without warmup.
        Running warmup upfront means the main loop starts fast.
        """
        import torch
        if not torch.cuda.is_available():
            return
        log.info("Warming up CUDA kernels (first-time only, may take a few minutes)...")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        t0 = time.time()
        self.ball_detector.detect_ball(dummy)
        log.info(f"  Ball model warmed up in {time.time()-t0:.1f}s")
        t0 = time.time()
        self.pose_model(dummy, verbose=False, device="cuda", half=True)
        log.info(f"  Pose model warmed up in {time.time()-t0:.1f}s")

    # Frames without ball detection before forcing rally end.
    # At 30fps, 150 frames = 5 seconds. Ball can be temporarily occluded by players
    # or miss detection for several seconds during an active rally — don't end too soon.
    NO_BALL_RALLY_TIMEOUT = 150

    # Lower than BallDetectorVideo's own 0.2, deliberately. Once BallTrackGate
    # decides by trajectory agreement rather than by confidence, a faint but
    # geometrically correct detection is worth more than a confident distractor,
    # and admitting them raises kept frames from ~70% to ~78-86% without
    # reintroducing jumps.
    BALL_MIN_CONFIDENCE = 0.1

    # Two detectors reporting centres this close are seeing the same object.
    BALL_MERGE_RADIUS = 12.0

    def _make_fsm(self, fps: float = 30.0) -> IntegratedVolleyballFSM:
        if self.calibration:
            c = self.calibration
            bounds = CourtBounds(
                left=c.court_left,
                right=c.court_right,
                top=c.court_top,
                bottom=c.court_bottom,
                net_y=c.net_y,
            )
        else:
            bounds = CourtBounds(left=0, right=3840, top=50, bottom=1700, net_y=830)

        fsm = IntegratedVolleyballFSM(court_bounds=bounds, fps=fps)

        # 4K GoPro: player bbox center is ~300px from hands/feet.
        # Default 40px radius misses every contact — increase substantially.
        fsm.event_tracker.contact_radius = 300.0

        return fsm

    # ------------------------------------------------------------------
    # Per-frame detection
    # ------------------------------------------------------------------

    def _detect_ball(
        self, frame: np.ndarray, frame_idx: int,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Returns (center_xy, confidence) or (None, 0.0).

        Selection is delegated to BallTrackGate. Picking the highest-confidence
        detection within a fixed radius let bright kit and other static
        look-alikes steal the track for a few frames at a time, which is what
        made the drawn trajectory jump hundreds of pixels and back.
        """
        candidates = self._ball_candidates_for(frame)
        # The global trajectory pass runs after the video is read, so every
        # candidate has to be kept, not just the one the gate picks now.
        if self.ball_trajectory_dp and candidates:
            self._ball_candidates[frame_idx] = candidates

        chosen = self.ball_gate.update(frame_idx, candidates)
        if chosen is None:
            return None, 0.0
        return (chosen.x, chosen.y), chosen.confidence

    def _ball_candidates_for(self, frame: np.ndarray) -> List[BallCandidate]:
        """Every plausible ball in this frame, from one or both detectors."""
        def to_candidates(detections):
            return [
                BallCandidate(
                    x=(x1 + x2) / 2,
                    y=(y1 + y2) / 2,
                    confidence=conf,
                    size=max(x2 - x1, y2 - y1),
                )
                for x1, y1, x2, y2, conf in detections
                if conf >= self.BALL_MIN_CONFIDENCE
            ]

        candidates = to_candidates(self.ball_detector.detect_ball(frame))
        if self.ball_detector_2 is None:
            return candidates

        # Merge the second opinion; the two models often report the same ball.
        for extra in to_candidates(self.ball_detector_2.detect_ball(frame)):
            if not any(
                math.hypot(extra.x - c.x, extra.y - c.y) < self.BALL_MERGE_RADIUS
                for c in candidates
            ):
                candidates.append(extra)
        return candidates

    def _detect_players(
        self, frame: np.ndarray
    ) -> Tuple[Dict[str, List[Tuple[float, float]]], List[PlayerPose]]:
        """
        YOLO-pose + ByteTrack. Returns:
            teams: {"team_a": [(cx,cy), ...], "team_b": [...]} — legacy format
                   for FSM / StreamingEventPredictor (team_a = far side, y < net_y)
            poses: List[PlayerPose] — full 17-keypoint skeletons with track IDs
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs = dict(verbose=False, device=device, half=(device == "cuda"))
        if self.pose_imgsz:
            kwargs["imgsz"] = self.pose_imgsz

        try:
            results = self.pose_model.track(
                frame, persist=True, tracker="bytetrack.yaml", **kwargs,
            )
        except Exception as e:
            if not self._pose_track_warned:
                log.warning(f"Pose tracking unavailable ({e}) — using per-frame inference (no IDs)")
                self._pose_track_warned = True
            results = self.pose_model(frame, **kwargs)

        net_y = self.calibration.net_y if self.calibration else 830.0

        # Preferred filter: court homography — foot point must land on the
        # court (± serve-zone margins). Kills spectators/benches/referees and
        # splits teams by court half instead of pixel y vs the net tape.
        projector = self._get_projector()

        # Legacy fallback (no homography in calibration): x-range heuristic
        court_x_range = None
        min_bbox_h = 0.0
        if self.calibration:
            c = self.calibration
            margin = 0.06 * c.frame_width
            court_x_range = (c.court_left - margin, c.court_right + margin)
            min_bbox_h = 0.025 * c.frame_height  # tiny far-away false detections

        poses = extract_player_poses(
            results[0] if results else None,
            net_y=net_y,
            court_x_range=court_x_range,
            min_bbox_h=min_bbox_h,
            projector=projector,
        )

        # On-court filter with track hysteresis: spectators/benches never get
        # in; players chasing a ball outside the lines don't drop out.
        if projector is not None:
            if self._court_gate is None or self._court_gate.projector is not projector:
                self._court_gate = CourtPlayerGate(projector, self.max_players_per_team)
            poses = self._court_gate.filter(poses)

        return poses_to_teams(poses), poses

    def _get_projector(self) -> Optional[CourtProjector]:
        """CourtProjector for the current (scaled) calibration, cached."""
        c = self.calibration
        if c is None or c.homography is None:
            return None
        if self._projector_calib is not c:  # calibration object changed (rescaled)
            try:
                floor_line = None
                if c.net_floor_left and c.net_floor_right:
                    floor_line = (c.net_floor_left, c.net_floor_right)
                self._projector = CourtProjector(c.homography, net_floor_line=floor_line)
            except Exception as e:
                log.warning(f"Court projector unavailable ({e}) — using x-range player filter")
                self._projector = None
            self._projector_calib = c
        return self._projector

    def _maybe_run_sam_net_detection(
        self,
        video_path: str,
        frame_w: float,
        frame_h: float,
        refresh_floor: bool = False,
    ):
        """Run SAM net-tape segmentation if not already in calibration.

        Results (720p space) are persisted into the calibration JSON so this
        runs once per camera setup; in-memory calibration gets the values in
        native calibration resolution.
        """
        c = self.calibration
        if (c is not None and c.net_tape_top is not None
                and c.net_floor_left is not None and c.net_floor_right is not None
                and c.net_floor_method == "sam3_post_mask"
                and not refresh_floor):
            log.info(f"Net tape already calibrated: top={c.net_tape_top:.0f}px "
                     f"bottom={c.net_tape_bottom:.0f}px (calibration px)")
            return

        log.info("Net tape not in calibration — running SAM net segmentation on first 5 s …")
        try:
            from src.detection.net_detector import NetDetector, save_sam_net_to_calibration

            net_y = c.net_y if c else frame_h * 0.5
            net_x_left = c.net_x_left if c else 0.0
            net_x_right = c.net_x_right if c else frame_w

            detector = NetDetector()
            if c is not None and c.net_tape_top is not None and c.net_tape_bottom is not None:
                top = c.net_tape_top * 720.0 / c.frame_height
                bot = c.net_tape_bottom * 720.0 / c.frame_height
                log.info("Net tape already calibrated; detecting only post bases")
            else:
                top, bot = detector.detect_from_video(
                    video_path=video_path,
                    net_y_hint=net_y,
                    net_x_left=net_x_left,
                    net_x_right=net_x_right,
                    frame_h=c.frame_height if c else frame_h,
                    frame_w=c.frame_width if c else frame_w,
                )
            if top is None:
                log.warning("SAM net detection returned no result — net tape bounds unavailable")
                return

            # In-memory: convert 720p → native calibration resolution
            floor_720 = detector.detect_floor_line_sam3(
                video_path=video_path,
                calibration_path=str(self._calibration_path) if self._calibration_path else "",
            )
            floor_method = "sam3_post_mask" if floor_720 else None
            if floor_720 is None:
                floor_720 = detector.detect_floor_line_from_video(
                    video_path=video_path,
                    net_x_left=net_x_left,
                    net_x_right=net_x_right,
                    frame_h=c.frame_height if c else frame_h,
                    frame_w=c.frame_width if c else frame_w,
                    net_tape_top=top * (c.frame_height if c else frame_h) / 720.0,
                )
                floor_method = "hough_post_base" if floor_720 else None

            if c is not None:
                sam_sy = c.frame_height / 720.0
                self.calibration = CalibrationData(
                    net_y=c.net_y, net_x_left=c.net_x_left, net_x_right=c.net_x_right,
                    court_top=c.court_top, court_bottom=c.court_bottom,
                    court_left=c.court_left, court_right=c.court_right,
                    frame_width=c.frame_width, frame_height=c.frame_height,
                    net_tape_top=top * sam_sy, net_tape_bottom=bot * sam_sy,
                    net_floor_left=(floor_720[0][0] * c.frame_width / 1280.0,
                                    floor_720[0][1] * sam_sy) if floor_720 else c.net_floor_left,
                    net_floor_right=(floor_720[1][0] * c.frame_width / 1280.0,
                                     floor_720[1][1] * sam_sy) if floor_720 else c.net_floor_right,
                    net_floor_method=floor_method or c.net_floor_method,
                    homography=c.homography,
                )
            # Persist original 720p values for future runs
            if self._calibration_path and self._calibration_path.exists():
                save_sam_net_to_calibration(
                    str(self._calibration_path), top, bot, floor_720, floor_method,
                )

        except Exception as exc:
            log.warning(f"SAM net detection failed ({exc}) — continuing without net tape bounds")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    # Skeleton overlay styling: team_b (near side — the team we track) is
    # highlighted; team_a (far side) is dimmed for context.
    _SKELETON_STYLE = {
        "team_b": {"color": (0, 220, 255), "bone": 3, "kp": 5, "draw_id": True},   # bright yellow
        "team_a": {"color": (180, 120, 80), "bone": 1, "kp": 2, "draw_id": False}, # muted blue-gray
    }

    def _draw_frame(
        self,
        frame: np.ndarray,
        ball_pos: Optional[Tuple[float, float]],
        player_teams: Dict[str, List[Tuple[float, float]]],
        fsm_status: Dict,
        event_display: Optional[str] = None,
        player_poses: Optional[List[PlayerPose]] = None,
    ) -> np.ndarray:
        out = frame.copy()

        # Court outline from homography (thin) — shows the on-court filter area
        projector = self._get_projector()
        if projector is not None:
            hw, hl = projector.half_width, projector.half_length
            quad = np.array([
                projector.world_to_image(w) for w in
                [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
            ], dtype=np.int32)
            cv2.polylines(out, [quad], isClosed=True, color=(200, 200, 200), thickness=2)
            # Line under the net (world y=0) — the actual team split reference
            floor_line = projector.net_floor_points()
            if floor_line is not None:
                p0, p1 = floor_line
                cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                         (255, 0, 255), 3)
                for label, point in (("A", p0), ("B", p1)):
                    px, py = int(point[0]), int(point[1])
                    cv2.circle(out, (px, py), 12, (255, 0, 255), -1)
                    cv2.putText(out, label, (px + 14, py - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            else:
                p0 = projector.world_to_image((-hw, 0.0))
                p1 = projector.world_to_image((hw, 0.0))
                cv2.line(out, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])),
                         (200, 200, 200), 2)

        # Net: SAM-detected tape band if available, else calibration net_y line
        if self.calibration:
            c = self.calibration
            x0, x1 = int(c.net_x_left), int(c.net_x_right)
            # The physical post bases are the authoritative ends of the net.
            # Legacy calibration ``net_x_*`` is only a tape/court estimate and
            # may stop short of a post on a new camera angle.
            if c.net_floor_left and c.net_floor_right:
                x0, x1 = sorted((int(c.net_floor_left[0]), int(c.net_floor_right[0])))
            if c.net_tape_top is not None and c.net_tape_bottom is not None:
                t, b = int(c.net_tape_top), int(c.net_tape_bottom)
                band = out.copy()
                cv2.rectangle(band, (x0, t), (x1, b), (0, 180, 255), -1)
                cv2.addWeighted(band, 0.25, out, 0.75, 0, out)
                cv2.line(out, (x0, t), (x1, t), (0, 140, 255), 2)
                cv2.line(out, (x0, b), (x1, b), (0, 90, 220), 2)
            else:
                ny = int(c.net_y)
                cv2.line(out, (x0, ny), (x1, ny), (255, 220, 0), 4)

        # Ball
        if ball_pos:
            cx, cy = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(out, (cx, cy), 18, (0, 255, 0), 3)
            cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)

        # Players — skeletons when keypoints are available, dots as fallback
        if player_poses:
            for pose in player_poses:
                st = self._SKELETON_STYLE[pose.team]
                draw_skeleton(out, pose, st["color"],
                              bone_thickness=st["bone"], kp_radius=st["kp"],
                              draw_id=st["draw_id"])
        else:
            colors = {"team_a": (255, 80, 0), "team_b": (0, 80, 255)}
            for team, pts in player_teams.items():
                for px, py in pts:
                    cv2.circle(out, (int(px), int(py)), 10, colors[team], -1)

        # FSM state overlay (line 1)
        state = fsm_status.get("game_state", "?")
        phase = fsm_status.get("rally_phase") or ""
        team  = fsm_status.get("current_team", "?")
        contacts = fsm_status.get("team_contacts", 0)
        text = f"{state} | {phase} | {team} [{contacts}]"
        cv2.putText(out, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

        # Event model overlay (line 2): sliding-window Transformer v3 prediction
        if event_display is not None and event_display != "-":
            ev_text = f"Event: {event_display}"
            cv2.putText(out, ev_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 200, 0), 4)

        return out

    # ------------------------------------------------------------------
    # Core processing loop
    # ------------------------------------------------------------------

    def _process_frames_iter(
        self,
        frame_iter,           # iterator yielding (frame_idx, frame_np)
        fps: float,
        total: int,
        stats: GameStats,
        out_writer=None,
        out_frames_dir: Optional[Path] = None,
    ) -> GameStats:
        """Shared processing loop used by both process_video and process_frames."""
        fsm = self._make_fsm(fps)
        pose_fx = PoseFeatureExtractor()  # fresh per video — track IDs restart
        if self._court_gate is not None:
            self._court_gate.reset()      # membership must not leak across videos
        self.ball_gate.reset()           # nor may a ball track

        # Accumulated rallies across FSM resets
        completed_rallies: List[Dict] = []
        current_rally_events: List[Dict] = []
        current_rally_start: Optional[int] = None
        prev_state = None

        # No-ball timeout: if ball not detected for this many frames during RALLY_ACTIVE,
        # force the rally to end (ball went out / rally over).
        frames_since_ball = 0

        # Serve cooldown: suppress repeated "serve" predictions — a real serve can happen
        # at most once per rally. After showing "serve", block it for at least this many
        # frames so the model can't flicker back to "serve" mid-rally.
        _SERVE_COOLDOWN_SEC  = 8.0          # minimum seconds between two serve displays
        _serve_cooldown_frames = int(fps * _SERVE_COOLDOWN_SEC)
        _last_serve_shown_frame: int = -_serve_cooldown_frames  # allow first serve immediately

        # Sticky overlay: after rally ends, keep showing the last in-rally prediction for
        # a short window so the screen doesn't go blank between rallies.
        _STICKY_FRAMES = int(fps * 2.0)     # show last prediction for 2 sec after rally ends
        _last_rally_display: str = ""
        _last_rally_display_frame: int = -_STICKY_FRAMES

        t0 = time.time()

        for frame_idx, frame in frame_iter:
            stats.total_frames += 1

            # Frame skipping
            if self.skip_frames > 0 and frame_idx % (self.skip_frames + 1) != 0:
                if out_writer:
                    out_writer.write(frame)
                continue

            stats.processed_frames += 1
            timestamp = frame_idx / fps

            # --- Detect ---
            ball_pos, ball_conf = self._detect_ball(frame, frame_idx)
            if ball_pos:
                stats.ball_detected_frames += 1
                frames_since_ball = 0
            else:
                frames_since_ball += 1

            # No-ball timeout: end rally/serve if ball gone too long
            fsm_status_pre = fsm.get_current_status()
            cur_state_pre = fsm_status_pre.get("game_state", "")
            if frames_since_ball >= self.NO_BALL_RALLY_TIMEOUT and cur_state_pre in ("rally_active", "serving"):
                log.warning(f"f{frame_idx}: [{cur_state_pre}] ended — no ball for {frames_since_ball} frames (timeout)")
                fsm._end_rally(frame_idx, "no_ball_timeout")
                frames_since_ball = 0

            player_teams, player_poses = self._detect_players(frame)

            # Pose-derived features (keypoints + ball) — saved per frame for the
            # event model dataset and viewer. Reference for "wrist above net":
            # SAM net tape top (physical top of the net) when available.
            if self.calibration:
                net_y_now = self.calibration.net_tape_top or self.calibration.net_y
            else:
                net_y_now = 830.0
            pose_features = pose_fx.update(frame_idx, player_poses, ball_pos, net_y_now)

            # --- FSM update ---
            fsm_events = fsm.update(
                frame_idx=frame_idx,
                ball_pos=ball_pos,
                player_positions=player_teams,
                timestamp=timestamp,
            )
            fsm_status = fsm.get_current_status()
            cur_state = fsm_status.get("game_state")

            # Track rally boundaries
            if cur_state == "rally_active" and current_rally_start is None:
                current_rally_start = frame_idx

            for ev in fsm_events:
                ev_dict = {
                    "frame": ev.frame_idx,
                    "time": round(ev.timestamp, 2),
                    "type": ev.event_type,
                    "game_state": str(ev.game_state.value),
                    "rally_phase": str(ev.rally_phase.value) if ev.rally_phase else None,
                    "team": ev.team_controlling,
                    "contact_seq": ev.contacts_count,
                }
                stats.all_events.append(ev_dict)
                current_rally_events.append(ev_dict)

            # Rally ended → save summary before FSM resets it
            # Provisional rallies from the raw trajectory. They are replaced
            # wholesale by the clean-trajectory pass after outlier rejection;
            # kept here only so a run that ends early still has something.
            if cur_state == "rally_end" and prev_state != "rally_end":
                if current_rally_events:
                    current_rally_events.sort(key=lambda e: e["frame"])
                    r_start = min(current_rally_events[0]["frame"],
                                  current_rally_start
                                  if current_rally_start is not None
                                  else current_rally_events[0]["frame"])
                    rally_dict = {
                        "start_frame": r_start,
                        "end_frame": frame_idx,
                        "duration_frames": frame_idx - r_start,
                        "duration_sec": round((frame_idx - r_start) / fps, 2),
                        "events": list(current_rally_events),
                        "contacts_team_a": sum(1 for e in current_rally_events if e["team"] == "team_a"),
                        "contacts_team_b": sum(1 for e in current_rally_events if e["team"] == "team_b"),
                    }
                    completed_rallies.append(rally_dict)
                current_rally_start = None
                current_rally_events = []

            prev_state = cur_state

            # --- Event model (Transformer) prediction for overlay and viewer comparison ---
            # FSM gate: only show LSTM prediction when ball is actually in play.
            # During serve_prep / serve_ready players stand in serving position, causing
            # the LSTM to continuously output "serve" between rallies — suppress that.
            _LSTM_ACTIVE_STATES = {"serving", "rally_active"}
            event_display = None
            if self._event_predictor is not None and self.calibration is not None:
                raw_display = self._event_predictor.add_frame(
                    ball_pos, player_teams, self.calibration.net_y, fps,
                    fsm_state=cur_state or "",   # let predictor exclude impossible classes
                    pose_features=pose_features,  # 8 pose channels (new checkpoints)
                )
                # Gate: only pass prediction through when FSM says ball is in play
                if raw_display is not None and cur_state in _LSTM_ACTIVE_STATES:
                    if raw_display == "serve":
                        # Serve cooldown: enforce minimum gap between serve displays.
                        # During rally_active the predictor already suppresses serve, but
                        # "serving" state can still briefly overlap with a false serve
                        # prediction lingering in the smoothing window.
                        frames_since_serve = frame_idx - _last_serve_shown_frame
                        if frames_since_serve >= _serve_cooldown_frames:
                            _last_serve_shown_frame = frame_idx
                            _last_rally_display = raw_display
                            _last_rally_display_frame = frame_idx
                            event_display = raw_display
                        # else: too soon — suppress this serve prediction silently
                    else:
                        _last_rally_display = raw_display
                        _last_rally_display_frame = frame_idx
                        event_display = raw_display
                elif (raw_display is not None
                      and cur_state not in _LSTM_ACTIVE_STATES
                      and _last_rally_display
                      and _last_rally_display != "serve"
                      and frame_idx - _last_rally_display_frame < _STICKY_FRAMES):
                    # Sticky: rally just ended — keep last non-serve prediction briefly visible
                    event_display = _last_rally_display

            # --- Collect per-frame annotation for viewer (FSM + Transformer for comparison) ---
            frame_events_this = [e for e in stats.all_events if e["frame"] == frame_idx]
            ann = {
                "f": frame_idx,
                "t": round(timestamp, 3),
                "ball": [round(ball_pos[0], 1), round(ball_pos[1], 1)] if ball_pos else None,
                # The gate started a new track here: the step from the previous
                # position is a change of hypothesis, not the ball moving.
                "ball_new": True if (ball_pos and self.ball_gate.acquired_this_frame) else None,
                "p": [[round(x, 1), round(y, 1)]
                      for x, y in player_teams.get("team_a", []) + player_teams.get("team_b", [])],
                "state": cur_state,
                "phase": fsm_status.get("rally_phase"),
                "ev": [e["type"] for e in frame_events_this],
                # Full skeletons: id, team (a/b), bbox, 17×[x,y,conf]
                "poses": [
                    {
                        "id": p.track_id,
                        "team": p.team[-1],  # "a" / "b"
                        "bb": [round(v, 1) for v in p.bbox],
                        "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)]
                               for x, y, c in p.keypoints],
                    }
                    for p in player_poses
                ],
                # Pose-derived features in POSE_FEATURE_NAMES order
                "pose_f": pose_features,
            }
            if event_display is not None:
                ann["event_model"] = event_display
            stats.frame_annotations.append(ann)

            # --- Annotate output ---
            if self.output_video and (out_writer or out_frames_dir):
                annotated = self._draw_frame(
                    frame, ball_pos, player_teams, fsm_status,
                    event_display=event_display, player_poses=player_poses,
                )
                if out_writer:
                    out_writer.write(annotated)
                if out_frames_dir:
                    cv2.imwrite(str(out_frames_dir / f"frame_{frame_idx:05d}.jpg"), annotated)

            # Progress log every 5 seconds of video
            log_interval = max(1, int(fps * 5))
            if frame_idx % log_interval == 0 and frame_idx > 0:
                elapsed = time.time() - t0
                pct = frame_idx / max(total, 1) * 100
                fps_proc = stats.processed_frames / max(elapsed, 0.01)
                ball_pct = stats.ball_detected_frames / max(stats.processed_frames, 1) * 100
                log.info(f"[{pct:5.1f}%] f{frame_idx}/{total} | {fps_proc:.1f} fps | "
                         f"state={cur_state} | ball={ball_pct:.0f}%")

        stats.rallies = completed_rallies
        stats.processing_time_sec = time.time() - t0

        # Post-process: re-pick the whole trajectory from the candidate cloud.
        # The gate above had to choose with only the past in hand; now the
        # frames after each decision are available too, and they are what
        # identifies the ball on frames where a distractor scored higher.
        if self.ball_trajectory_dp:
            self._apply_trajectory_dp(stats)

        # Post-process: remove trajectory jumps (false positives inconsistent with neighbours)
        n_removed = self._filter_ball_outliers(stats.frame_annotations)
        if n_removed:
            log.info(f"Outlier filter: removed {n_removed} false-positive frames")

        # Post-process: fill short gaps in ball detection by linear interpolation
        n_filled = self._interpolate_ball_positions(stats.frame_annotations)
        if n_filled:
            log.info(f"Ball interpolation: filled {n_filled} missing frames")

        # The inline FSM above has to run on raw per-frame detections, because
        # the annotated video needs a state to draw while the frame is in hand.
        # Outlier rejection only becomes possible once the whole trajectory is
        # known, and every false detection the filter removes had already been
        # fed to the FSM as a contact. Re-run it over the cleaned trajectory so
        # the saved events and rallies come from the same data the heatmap uses.
        stats.all_events, stats.rallies = self._recompute_events_on_clean_trajectory(
            stats.frame_annotations, fps,
        )
        completed_rallies = stats.rallies

        # Serve-centric derivation: serve → reception (+outcome) from rally
        # structure. Adds "serve_reception" to each rally dict.
        sr_counts = annotate_rallies(completed_rallies, fps)
        if sr_counts["serves"]:
            log.info(f"Serve/reception: {sr_counts['serves']} serves, "
                     f"{sr_counts['success']} received OK, "
                     f"{sr_counts['failed']} failed, "
                     f"{sr_counts['no_reception']} no reception (ace/fault)")

        # Heatmap intentionally uses the cleaned trajectory and never the FSM:
        # any sharp, high-speed trajectory turn is a hit candidate even if a
        # rally was not recognized.
        stats.attack_heatmap = AttackHeatmap(fps).analyze(
            stats.frame_annotations, projector=self._get_projector(),
        )
        log.info(f"Attack heatmap: {stats.attack_heatmap['total_hits']} trajectory-turn hits")

        return stats

    def _apply_trajectory_dp(self, stats: "GameStats") -> None:
        """Replace the gate's per-frame picks with a globally chosen path.

        Rewrites ``ball`` on every annotation, so everything downstream — the
        outlier filter, the FSM re-run, the heatmap, the viewer — reads the same
        trajectory. ``ball_new`` is dropped: it marked a gate re-acquisition,
        and the solver does not re-acquire, it chooses once.
        """
        if not self._ball_candidates:
            return

        solver = BallTrajectoryDP()
        chosen = interpolate_gaps(solver.solve(self._ball_candidates), max_gap=4)

        before = sum(1 for a in stats.frame_annotations if a.get("ball"))
        for ann in stats.frame_annotations:
            cand = chosen.get(ann["f"])
            ann["ball"] = [round(cand.x, 1), round(cand.y, 1)] if cand else None
            ann["ball_new"] = None
            if cand is not None and cand.confidence == 0.0:
                ann["ball_interp"] = True
        after = sum(1 for a in stats.frame_annotations if a.get("ball"))

        stats.ball_detected_frames = after
        log.info(
            f"Trajectory solver: {after} frames with a ball (gate had {before}), "
            f"from {solver.stats['frames_with_candidates']} frames with candidates "
            f"in {solver.stats['segments']} segments"
        )
        self._ball_candidates.clear()

    def _recompute_events_on_clean_trajectory(
        self, annotations: List[Dict], fps: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Re-run the FSM over the cleaned trajectory; return (events, rallies).

        Per-frame ``state``/``phase``/``ev`` fields are rewritten in place so the
        viewer overlay agrees with the saved rally list.
        """
        fsm = self._make_fsm(fps)
        events: List[Dict] = []
        rallies: List[Dict] = []
        cur_events: List[Dict] = []
        cur_start: Optional[int] = None
        prev_state: Optional[str] = None
        frames_since_ball = 0

        for ann in annotations:
            frame_idx = int(ann["f"])
            ball = ann.get("ball")
            ball_pos = (float(ball[0]), float(ball[1])) if ball else None
            frames_since_ball = 0 if ball_pos else frames_since_ball + 1

            teams: Dict[str, List[Tuple[float, float]]] = {"team_a": [], "team_b": []}
            for pose in ann.get("poses") or []:
                key = "team_a" if pose.get("team") in ("a", "team_a") else "team_b"
                bb = pose.get("bb")
                if bb:
                    teams[key].append(((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0))

            pre_state = fsm.get_current_status().get("game_state", "")
            if (frames_since_ball >= self.NO_BALL_RALLY_TIMEOUT
                    and pre_state in ("rally_active", "serving")):
                fsm._end_rally(frame_idx, "no_ball_timeout")
                frames_since_ball = 0

            fsm_events = fsm.update(
                frame_idx=frame_idx,
                ball_pos=ball_pos,
                player_positions=teams,
                timestamp=frame_idx / fps,
                # Both an interpolated fill and a track restart make the local
                # trajectory unusable for measuring a turn.
                ball_interp=bool(ann.get("ball_interp") or ann.get("ball_new")),
            )
            status = fsm.get_current_status()
            state = status.get("game_state")

            ann["state"] = state
            ann["phase"] = status.get("rally_phase")
            ann["ev"] = [e.event_type for e in fsm_events]

            if state == "rally_active" and cur_start is None:
                cur_start = frame_idx

            for ev in fsm_events:
                d = {
                    "frame": ev.frame_idx,
                    "time": round(ev.timestamp, 2),
                    "type": ev.event_type,
                    "game_state": str(ev.game_state.value),
                    "rally_phase": str(ev.rally_phase.value) if ev.rally_phase else None,
                    "team": ev.team_controlling,
                    "contact_seq": ev.contacts_count,
                }
                events.append(d)
                cur_events.append(d)

            if state == "rally_end" and prev_state != "rally_end":
                if cur_events:
                    # The serve is published on the net crossing but carries the
                    # earlier frame of its own contact, so emission order is not
                    # frame order. serve_reception reads events positionally.
                    cur_events.sort(key=lambda e: e["frame"])
                    # A rally spans serve → ball dead; cur_start (first
                    # RALLY_ACTIVE frame) would clip the serve off the front.
                    r_start = min(cur_events[0]["frame"],
                                  cur_start if cur_start is not None else cur_events[0]["frame"])
                    rallies.append({
                        "start_frame": r_start,
                        "end_frame": frame_idx,
                        "duration_frames": frame_idx - r_start,
                        "duration_sec": round((frame_idx - r_start) / fps, 2),
                        "events": list(cur_events),
                        "contacts_team_a": sum(1 for e in cur_events if e["team"] == "team_a"),
                        "contacts_team_b": sum(1 for e in cur_events if e["team"] == "team_b"),
                    })
                cur_start, cur_events = None, []
            prev_state = state

        events.sort(key=lambda e: e["frame"])
        log.info(f"Clean-trajectory pass: {len(rallies)} rallies, {len(events)} events")
        return events, rallies

    def _filter_ball_outliers(
        self,
        annotations: List[Dict],
        max_gap: int = 4,
        detour_ratio: float = 2.5,
        min_jump_px: float = 30.0,
    ) -> int:
        """
        Remove ball detections that are inconsistent with neighbouring frames.

        A detection at frame N is a false positive if:
        - It is within `max_gap` frames of both a previous and a next detection
        - The detour distance (prev→curr + curr→next) exceeds `detour_ratio` × the
          direct distance (prev→next) — i.e. the ball "jumps out and back"
        - The jump size is at least `min_jump_px` (ignore tiny wiggles)

        Removed frames are cleared (ball=None) so interpolation can fill them later.
        Returns the number of frames removed.
        """
        detected = [i for i, ann in enumerate(annotations)
                    if ann.get("ball") and not ann.get("ball_interp")]
        removed = 0

        for k in range(1, len(detected) - 1):
            i_prev, i_curr, i_next = detected[k - 1], detected[k], detected[k + 1]

            # Only consider closely-spaced detections
            if i_curr - i_prev > max_gap or i_next - i_curr > max_gap:
                continue

            # Skip if any neighbour was already filtered this pass
            bp = annotations[i_prev]["ball"]
            bc = annotations[i_curr]["ball"]
            bn = annotations[i_next]["ball"]
            if bp is None or bc is None or bn is None:
                continue

            xp, yp = bp
            xc, yc = bc
            xn, yn = bn

            d_in   = np.hypot(xc - xp, yc - yp)   # prev → curr
            d_out  = np.hypot(xn - xc, yn - yc)   # curr → next
            d_skip = np.hypot(xn - xp, yn - yp)   # prev → next (direct)

            if d_skip < 1.0:
                d_skip = 1.0

            jump = max(d_in, d_out)
            if jump < min_jump_px:
                continue  # tiny movement, not an outlier

            if (d_in + d_out) > detour_ratio * d_skip:
                annotations[i_curr]["ball"] = None
                annotations[i_curr]["ball_filtered"] = True
                removed += 1

        return removed

    def _interpolate_ball_positions(self, annotations: List[Dict], max_gap: int = 10) -> int:
        """
        Linearly interpolate ball positions for short detection gaps.

        Only fills gaps of at most `max_gap` consecutive missing frames — longer gaps
        mean the ball is genuinely out of sight.  Interpolated frames are tagged with
        ``"ball_interp": True`` so the viewer can style them differently.

        Returns the number of frames filled.
        """
        if not annotations:
            return 0

        # Indices of annotation entries that have a confirmed ball detection
        detected = [i for i, ann in enumerate(annotations) if ann.get("ball") and not ann.get("ball_interp")]
        filled = 0

        for k in range(len(detected) - 1):
            i0, i1 = detected[k], detected[k + 1]
            gap = i1 - i0 - 1
            if gap <= 0 or gap > max_gap:
                continue
            x0, y0 = annotations[i0]["ball"]
            x1, y1 = annotations[i1]["ball"]
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                annotations[i0 + step]["ball"] = [
                    round(x0 + t * (x1 - x0), 1),
                    round(y0 + t * (y1 - y0), 1),
                ]
                annotations[i0 + step]["ball_interp"] = True
                filled += 1

        return filled

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: str = "results",
        max_duration_sec: Optional[float] = None,
    ) -> GameStats:
        """Process a video file."""
        vp = Path(video_path)
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vp}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(max_duration_sec * fps) if max_duration_sec else total

        log.section(f"Video: {vp.name}")
        log.info(f"Resolution={w}x{h}  fps={fps:.1f}  frames={total}  skip={self.skip_frames}")

        # Auto-detect net tape with SAM if not already in calibration
        self._maybe_run_sam_net_detection(
            str(vp), float(w), float(h), refresh_floor=self.refresh_net_floor,
        )

        # Scale calibration to actual frame resolution (e.g., 4K calib on 720p video)
        saved_calib = self.calibration
        self.calibration = self._scale_calibration(w, h)

        out_writer = None
        if self.output_video:
            out_path = od / f"{vp.stem}_analyzed.mp4"
            # Prefer mp4v first to avoid OpenH264 DLL warning on Windows (avc1/H264 need it)
            for codec in ("mp4v", "avc1", "H264"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                if out_writer.isOpened():
                    log.info(f"VideoWriter codec: {codec}")
                    break
                out_writer = None
            if out_writer is None:
                log.warning("VideoWriter failed — annotated video will not be saved")

        def frame_iter():
            idx = 0
            while idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1

        stats = GameStats(video_name=vp.name, fps=fps)
        scaled_calib = self.calibration  # already scaled — needed for _save_and_print
        try:
            self._process_frames_iter(
                frame_iter(), fps, max_frames, stats,
                out_writer=out_writer,
            )
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            self.calibration = saved_calib  # restore original

        self._save_and_print(stats, od, vp.stem,
                             original_video_path=str(vp.resolve()),
                             calibration_override=scaled_calib)
        return stats

    def process_frames(
        self,
        frames_dir: str,
        output_dir: str = "results",
        fps: float = 30.0,
        max_frames: Optional[int] = None,
    ) -> GameStats:
        """Process a directory of JPEG/PNG frames (e.g. for testing)."""
        fd = Path(frames_dir)
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png"}
        frame_files = sorted(p for p in fd.iterdir() if p.suffix.lower() in exts)
        if max_frames:
            frame_files = frame_files[:max_frames]

        total = len(frame_files)
        log.section(f"Frames dir: {fd.name}")
        log.info(f"frames={total}  fps={fps:.1f}")

        out_frames_dir = None
        if self.output_video:
            out_frames_dir = od / "annotated_frames"
            out_frames_dir.mkdir(exist_ok=True)

        def frame_iter():
            for idx, fp in enumerate(frame_files):
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                yield idx, frame

        stats = GameStats(video_name=fd.name)
        self._process_frames_iter(
            frame_iter(), fps, total, stats,
            out_frames_dir=out_frames_dir,
        )

        self._save_and_print(stats, od, fd.name)
        return stats

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_and_print(self, stats: GameStats, output_dir: Path, stem: str,
                        original_video_path: str = "",
                        calibration_override: Optional["CalibrationData"] = None):
        out_path = output_dir / f"{stem}_stats.json"
        with open(out_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2, default=str)

        # Use the scaled calibration that was actually used during processing
        calib = calibration_override or self.calibration

        # Save compact per-frame annotations separately (for HTML viewer overlay)
        ann_path = output_dir / f"{stem}_annotations.json"
        ann_data = {
            "fps": stats.fps,
            "video": stats.video_name,
            "original_video_path": original_video_path,
            "frame_width": calib.frame_width if calib else 1280,
            "frame_height": calib.frame_height if calib else 720,
            "net_y": calib.net_y if calib else 370.0,
            "net_x_left": calib.net_x_left if calib else 0.0,
            "net_x_right": calib.net_x_right if calib else 1280.0,
            "net_tape_top": calib.net_tape_top if calib else None,
            "net_tape_bottom": calib.net_tape_bottom if calib else None,
            "net_floor_left": list(calib.net_floor_left) if calib and calib.net_floor_left else None,
            "net_floor_right": list(calib.net_floor_right) if calib and calib.net_floor_right else None,
            "net_floor_method": calib.net_floor_method if calib else None,
            "pose_feature_names": POSE_FEATURE_NAMES,
            "attack_heatmap": stats.attack_heatmap,
            "frames": stats.frame_annotations,
        }
        with open(ann_path, "w") as f:
            json.dump(ann_data, f, separators=(",", ":"),
                      default=lambda o: float(o) if hasattr(o, "__float__") else str(o))

        log.section(f"Done: {stem}")
        log.info(f"Frames processed : {stats.processed_frames}/{stats.total_frames}")
        log.info(f"Ball detected    : {stats.ball_detection_rate*100:.1f}%")
        log.info(f"Rallies detected : {len(stats.rallies)}")
        log.info(f"Total events     : {len(stats.all_events)}")
        log.info(f"Processing time  : {stats.processing_time_sec:.1f}s")
        log.info(f"Stats saved to   : {out_path}")
