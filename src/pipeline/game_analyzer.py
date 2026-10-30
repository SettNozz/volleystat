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
from dataclasses import asdict, dataclass, field, replace
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from src.models.model_registry import ResolvedModel, resolve_model
from src.tracking.ball_static_sources import (
    clean_ball_annotations,
    find_static_sources_in_candidates,
    suppress_static_candidates,
)
from src.tracking.ball_trajectory_smoother import (
    DEFAULT_WINDOW as BALL_SMOOTH_DEFAULT_WINDOW,
    jitter_metric,
    resolve_window as resolve_ball_smooth_window,
    smooth_ball_annotations,
)
from src.tracking.ball_gap_reconstructor import (
    BallGapReconstructor,
    ContactHint,
)
from src.tracking.pose_feature_extractor import POSE_FEATURE_NAMES, PoseFeatureExtractor
from src.tracking.serve_reception import annotate_rallies
from src.tracking.match_analytics import (
    annotate_match_phases,
    annotate_score_timeline,
    enrich_match_analytics,
    filter_non_game_records,
)
from src.utils.logger import get_logger
from src.utils.path_utils import local_path_from_uri
from src.models.inference_cache import (
    DEFAULT_CACHE, CachedForward, CacheMiss, LazyModel, ModelOutputCache,
    array_payload, digest, file_digest, model_spec,
    pack_pose_result, payload_array, unpack_pose_result, PoseTracker, FrozenOSNet, osnet_feature_store,
)

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
class _EventOverlayState:
    """Per-pass display policy shared by provisional and final predictions."""
    fps: float
    last_serve_frame: Optional[int] = None
    last_rally_display: str = ""
    last_rally_frame: int = 0

    def update(self, raw_display: Optional[str], state: str, frame: int) -> Optional[str]:
        if raw_display is not None and state in {"serving", "rally_active"}:
            if raw_display == "serve":
                if (self.last_serve_frame is not None
                        and frame - self.last_serve_frame < int(self.fps * 8.0)):
                    return None
                self.last_serve_frame = frame
            self.last_rally_display = raw_display
            self.last_rally_frame = frame
            return raw_display
        if (raw_display is not None
                and state not in {"serving", "rally_active"}
                and self.last_rally_display
                and self.last_rally_display != "serve"
                and frame - self.last_rally_frame < int(self.fps * 2.0)):
            return self.last_rally_display
        return None


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
    # SAM3 post X plus the configured physical net-height geometry is
    # trusted; older Hough values are refreshed before becoming the divider.
    net_floor_method: Optional[str] = None
    net_height_m: float = 2.34
    net_post_spacing_m: Optional[float] = None
    # Homography world (court meters) → image pixels, scaled to current frame.
    # Enables the on-court player filter and court-half team split.
    homography: Optional[List[List[float]]] = None
    # Full court outline in image pixels:
    # front-left, front-right, back-right, back-left.
    court_corners: Optional[List[Tuple[float, float]]] = None


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
    # Confirmed playable rallies only. Service attempts that end before a
    # controlled follow-up are kept separately instead of inflating this list.
    rallies: List[Dict] = field(default_factory=list)
    service_attempts: List[Dict] = field(default_factory=list)
    rally_lifecycle: Dict[str, int] = field(default_factory=dict)
    match_analytics: Dict[str, Any] = field(default_factory=dict)
    all_events: List[Dict] = field(default_factory=list)
    # With the dense contact spotter enabled it owns ``all_events``, and the
    # FSM's own contacts are kept here so nothing is lost by the swap.
    all_events_fsm: List[Dict] = field(default_factory=list)
    # Likewise the spotter's serve-anchored rallies replace the FSM's, which
    # stay here. Empty when the spotter did not run.
    rallies_fsm: List[Dict] = field(default_factory=list)
    service_attempts_fsm: List[Dict] = field(default_factory=list)
    rally_lifecycle_fsm: Dict[str, int] = field(default_factory=dict)
    # Spotter diagnostics, provenance and decoded rally phases.
    contact_spotter: Dict[str, Any] = field(default_factory=dict)
    player_identity: Dict[str, Any] = field(default_factory=dict)
    player_contacts: Dict[str, Any] = field(default_factory=dict)
    model_cache: Dict[str, Any] = field(default_factory=dict)
    # Per-frame data for HTML viewer overlay (ball pos, players, FSM state)
    frame_annotations: List[Dict] = field(default_factory=list)
    # Trajectory-only contact/attack candidates aggregated into court zones.
    attack_heatmap: Dict[str, Any] = field(default_factory=dict)
    ball_reconstruction: Dict[str, Any] = field(default_factory=dict)
    # Window used, points moved, and the jitter measured before/after.
    ball_smoothing: Dict[str, Any] = field(default_factory=dict)
    # Fixed objects the detector fired on, and what was dropped because of it.
    ball_static_suppression: Dict[str, Any] = field(default_factory=dict)
    # Which weights produced this result: registry name, version, source.
    models: Dict[str, Any] = field(default_factory=dict)
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
            "service_attempts": self.service_attempts,
            "rally_lifecycle": self.rally_lifecycle,
            "match_analytics": self.match_analytics,
            "all_events": self.all_events,
            "all_events_fsm": self.all_events_fsm,
            "rallies_fsm": self.rallies_fsm,
            "service_attempts_fsm": self.service_attempts_fsm,
            "rally_lifecycle_fsm": self.rally_lifecycle_fsm,
            "contact_spotter": self.contact_spotter,
            "player_identity": self.player_identity,
            "player_contacts": self.player_contacts,
            "model_cache": self.model_cache,
            "attack_heatmap": self.attack_heatmap,
            "ball_reconstruction": self.ball_reconstruction,
            "ball_smoothing": self.ball_smoothing,
            "ball_static_suppression": self.ball_static_suppression,
            "models": self.models,
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

    # Four-class causal production fit over both full-video gold datasets.
    # Loading stays fail-safe: if this local weight is absent, overlay disables.
    _DEFAULT_EVENT_CHECKPOINT: Optional[str] = (
        "action_recognition/models/experiments_three_full_gold_multiscale_v2/"
        "multiscale_final_three_videos_seed42/final_event_sequence.pt"
    )

    def __init__(
        self,
        calibration_path: Optional[str] = "court_calibration.json",
        ball_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        output_video: bool = False,
        skip_frames: int = 0,
        net_y_override: Optional[float] = None,
        event_checkpoint_path: Optional[str] = None,
        max_players_per_team: int = 6,
        pose_imgsz: Optional[int] = 2560,
        refresh_net_floor: bool = False,
        ball_ensemble: bool = False,
        ball_trajectory_dp: bool = True,
        ball_batch_size: int = 1,
        net_height_m: float = 2.34,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        enable_pose: bool = True,
        enable_net_detection: bool = True,
        match_segments: Optional[List[Dict[str, Any]]] = None,
        team_identity_map: Optional[Dict[str, str]] = None,
        ball_reconstruction_mode: str = "off",
        reconstruction_affects_fsm: bool = False,
        reconstruction_contact_hints: Optional[List[Dict[str, Any]]] = None,
        ball_smooth_window: Any = BALL_SMOOTH_DEFAULT_WINDOW,
        suppress_static_ball_sources: bool = True,
        use_model_registry: bool = True,
        person_masks_path: Optional[str] = None,
        contact_spotter_path: Optional[str] = None,
        enable_player_identity: bool = True,
        player_identity_side: str = "b",
        player_identity_seed: Optional[str] = None,
        player_reid_assets: Optional[str] = None,
        player_reid_device: str = "auto",
        model_cache_dir: Optional[str] = None,
        require_model_cache: bool = False,
    ):
        """
        Args:
            calibration_path: Path to court_calibration.json (relative to project root or absolute)
            ball_model_path:  Override ball detection model
            pose_model_path:  Override pose model
            output_video:     Produce annotated video / frame outputs
            skip_frames:      Process every (skip_frames+1)-th frame for speed
            event_checkpoint_path: Optional event-sequence checkpoint override.
            ball_ensemble: Also collect candidates from a second, tiled detector.
            ball_trajectory_dp: Re-pick the ball trajectory globally once the whole
                       video has been read (see BallTrajectoryDP). The live gate
                       still runs, because the annotated video needs a position
                       while the frame is in hand.
            ball_batch_size: Batch this many frames for the ball YOLO. Pose
                       tracking remains sequential so ByteTrack IDs stay valid.
            net_height_m: Physical net-top height. Mixed games default to
                       2.34 m; men's games use 2.43 m.
            progress_callback: Optional observer receiving phase, real frame
                       counts, throughput and ETA during analysis.
            enable_pose: Disable pose inference for ball-only dataset refreshes.
            enable_net_detection: Disable SAM net calibration for ball-only
                       pre-label jobs where court geometry is intentionally absent.
            ball_reconstruction_mode: ``off``, ``conservative``,
                       ``contact_aware`` or ``compare``. Compare stores both
                       alternatives and uses conservative as active output.
            reconstruction_affects_fsm: Let only high-confidence reconstructed
                       positions enter the clean FSM pass. Disabled while the
                       two policies are under visual evaluation.
            ball_smooth_window: Odd frame count (or a preset name: light /
                       moderate / strong / off) for the local quadratic fit
                       that removes per-frame detector wobble from the chosen
                       trajectory. 0 disables it. Contacts are detected first
                       and the fit never crosses one, so hits stay angular.
            suppress_static_ball_sources: Drop detections that belong to a
                       fixed object (a spare ball on the floor, a bag, a lamp).
                       They are confident on every frame, and a confident
                       candidate is *cheap* for the solver, so a stationary run
                       can outscore a real flight. See ball_static_sources.
        """
        self.progress_callback = progress_callback
        self.enable_pose = bool(enable_pose)
        self.enable_player_identity = bool(enable_player_identity and enable_pose)
        if player_identity_side not in {"a", "b"}:
            raise ValueError("player_identity_side must be a or b")
        if player_reid_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("player_reid_device must be auto, cpu or cuda")
        self.player_identity_side = player_identity_side
        self.player_identity_seed = player_identity_seed
        self.player_reid_assets = player_reid_assets
        self.player_reid_device = player_reid_device
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else DEFAULT_CACHE
        self.require_model_cache = require_model_cache
        self._model_cache = None
        self._ball_specs = {}
        self._pose_spec = None
        self.enable_net_detection = bool(enable_net_detection)
        self.match_segments = list(match_segments or [])
        self.team_identity_map = dict(team_identity_map or {})
        mode = str(ball_reconstruction_mode).replace("-", "_")
        if mode not in {"off", "conservative", "contact_aware", "compare"}:
            raise ValueError(f"unknown ball_reconstruction_mode: {mode}")
        self.ball_reconstruction_mode = mode
        self.reconstruction_affects_fsm = bool(reconstruction_affects_fsm)
        self.ball_smooth_window = resolve_ball_smooth_window(ball_smooth_window)
        self.suppress_static_ball_sources = bool(suppress_static_ball_sources)
        self.use_model_registry = bool(use_model_registry)
        # name -> {source, version, path, ...}; written into the stats file.
        self.model_provenance: Dict[str, Dict[str, Any]] = {}
        self._static_source_report: Dict[str, Any] = {}
        self._current_fps = 30.0
        self.reconstruction_contact_hints = [
            ContactHint(int(item["frame"]), str(item.get("label") or item.get("type") or "other"))
            for item in (reconstruction_contact_hints or [])
            if item.get("frame") is not None
        ]
        self._emit_progress(
            phase="initializing",
            progress=0.0,
            message="Loading detection models...",
        )
        self.output_video = output_video
        self.skip_frames = skip_frames
        # Loaded on first use: a checkpoint is useless until there are final
        # trajectories and poses to run it over.
        self.contact_spotter_path = contact_spotter_path
        self._contact_spotter = None
        self.net_y_override = net_y_override
        self.ball_ensemble = ball_ensemble
        self.ball_trajectory_dp = ball_trajectory_dp
        self.ball_batch_size = max(1, int(ball_batch_size))
        self.net_height_m = float(net_height_m)
        if not 2.0 <= self.net_height_m <= 2.6:
            raise ValueError("net_height_m must be between 2.0 and 2.6")
        # Raw per-frame candidates, kept for the global trajectory pass.
        self._ball_candidates: Dict[int, List[BallCandidate]] = {}

        self._pose_track_warned = False
        self.max_players_per_team = max_players_per_team
        self.pose_imgsz = pose_imgsz or 2560  # Preserve small/far players in native 4K footage.
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
        self._init_ball_detector(ball_model_path)
        self.pose_model = None
        if self.enable_pose:
            self._init_pose_model(pose_model_path)
        self._person_pose_recovery = None
        if person_masks_path:
            if not self.enable_pose or self.pose_model is None:
                raise ValueError("SAM-guided recovery requires YOLO-pose")
            from src.detection.sam_pose_recovery import SamGuidedPoseRecovery
            self._person_pose_recovery = SamGuidedPoseRecovery(
                person_masks_path, self.model_provenance["pose_estimator"]["path"],
            )
        self._event_predictor = self._init_event_predictor(event_checkpoint_path)
        self._warmup()
        self._emit_progress(
            phase="initializing",
            progress=0.0,
            message="Models ready; opening video...",
        )

    def _emit_progress(self, **payload: Any) -> None:
        """Report progress without letting a UI observer break analysis."""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(dict(payload))
        except Exception as exc:
            log.warning(f"Progress callback failed: {exc}")

    def _prepare_model_cache(self, source, fps, width, height, limit, source_identity=None):
        # __new__-based subsystem tests deliberately do not construct production models.
        if not hasattr(self, 'model_cache_dir'):
            return None
        self._model_cache = ModelOutputCache(source, self.model_cache_dir, self.require_model_cache,
                                             source_identity=source_identity)
        cache = self._model_cache
        self._ball_stages = {name: cache.stage('ball_'+name, spec) for name,spec in self._ball_specs.items()}
        scope = {'limit': limit, 'skip_frames': self.skip_frames, 'fps': fps, 'size': [width,height]}
        self._pose_sequence = None
        self._pose_cache_read = False
        if self.enable_pose:
            self._pose_sequence, self._pose_cache_read = cache.sequence({
                'model': self._pose_spec, 'scope': scope,
                'adapter': hashlib.sha256(inspect.getsource(PoseTracker).encode()).hexdigest()})
            self._raw_pose_stage = cache.stage('pose_raw', {k:v for k,v in self._pose_spec.items() if k!='tracker'})
            self._cached_pose_tracker = PoseTracker(self._pose_tracker_path)
        self._cache_pass = digest({'ball': self._ball_stages, 'pose': self._pose_spec if self.enable_pose else None,
                                   'scope': scope})
        previous = cache.get('detection_passes', self._cache_pass)
        if previous is None:
            length = cache.get('source_video', 'decoded_total') if source_identity is None else None
            count = min(limit,length['count']) if length else limit
            frames = list(range(0,count,self.skip_frames+1))
            if (all(cache.covers(stage,frames) for stage in self._ball_stages.values())
                    and (not self.enable_pose or cache.covers(self._raw_pose_stage,frames))):
                previous = {'total_frames':count,'frame_indices':list(range(count)),
                            'processed_frames':len(frames)}
        if self.require_model_cache and previous is None:
            raise CacheMiss('No completed model-output pass for this video/range; inference is disabled')
        self._cached_pass = previous
        self._cache_scope_limit = limit
        self._cache_visited_frames = []
        self._cache_frame_size = (width,height)
        recovery = getattr(self, '_person_pose_recovery', None)
        if recovery is not None:
            self._recovery_model_sha = file_digest(recovery.checkpoint)
            self._recovery_backend_sha = file_digest(_ROOT/'src/detection/sam_pose_recovery.py')
            self._recovery_pose_code_sha = file_digest(_ROOT/'src/detection/player_pose.py')
            original = getattr(recovery, '_uncached_predict', recovery._predict)
            recovery._uncached_predict = original
            stage = cache.stage('sam_crop_pose_raw', model_spec(recovery.checkpoint,
                backend=hashlib.sha256(inspect.getsource(type(recovery)._predict).encode()).hexdigest(),
                imgsz=640, conf=.15))
            def predict_crop(crop, device):
                key = digest({'shape': list(crop.shape), 'sha256': hashlib.sha256(crop.tobytes()).hexdigest()})
                return unpack_pose_result(cache.compute(stage,key,lambda:pack_pose_result(original(crop,device))))
            recovery._predict = predict_crop
        predictor = self._event_predictor
        if predictor is not None:
            model = predictor.model
            if isinstance(model, CachedForward):
                model = model.model
            predictor.model = CachedForward(model, cache, model_spec(
                self.model_provenance['event_sequence']['path'],
                backend=file_digest(_ROOT/'action_recognition/models/sequence_event_model.py')))
        log.info(f'Model cache: {cache.path}; completed detection pass={previous is not None}')
        return previous

    def _finish_detection_cache(self, stats):
        cache = getattr(self, '_model_cache', None)
        if cache is not None:
            if (getattr(self, '_cache_full_source_limit', None)==self._cache_scope_limit
                    and stats.total_frames < self._cache_scope_limit-1):
                raise RuntimeError('Video ended early; raw cache retained but detection pass is not complete')
            if self._pose_sequence and not self._pose_cache_read:
                cache.seal_sequence(self._pose_sequence)
            cache.put('detection_passes', self._cache_pass,
                      {'total_frames': stats.total_frames, 'processed_frames': stats.processed_frames,
                       'frame_indices': self._cache_visited_frames})
            if getattr(self, '_cache_full_source_limit', None)==self._cache_scope_limit:
                cache.put('source_video','decoded_total',{'count':stats.total_frames})

    def _cached_frame_iterator(self):
        # Pixel-free replay: detectors only read cached values, downstream uses
        # poses/candidates. Image-dependent recovery/rendering use real frames.
        empty = np.empty((0,0,3), dtype=np.uint8)
        for frame in self._cached_pass['frame_indices']:
            yield frame, empty

    def _cache_dense_probabilities(self, spotter, checkpoint):
        cache = getattr(self, '_model_cache', None)
        if cache is None:
            return
        module = sys.modules[type(spotter).__module__]
        backend = hashlib.sha256((inspect.getsource(type(spotter).probabilities)
                                  +inspect.getsource(module.predict_dense)).encode()).hexdigest()
        stage = cache.stage('contact_probabilities', model_spec(checkpoint, backend=backend,
            architecture=file_digest(_ROOT/'action_recognition/rally_spotting/model.py'),
            window=spotter.window, stride=spotter.stride,
            mean=spotter.mean, deviation=spotter.deviation))
        original = getattr(spotter, '_uncached_probabilities', spotter.probabilities)
        spotter._uncached_probabilities = original
        def probabilities(features):
            key = digest({'shape': list(features.shape), 'dtype': features.dtype.str,
                          'sha256': hashlib.sha256(features.tobytes()).hexdigest()})
            return payload_array(cache.compute(stage,key,lambda:array_payload(original(features))))
        spotter.probabilities = probabilities

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
        stored_net_height = sam.get("net_height_m")
        if stored_net_height is None:
            stored_net_height = (
                data.get("metadata", {}).get("court_geometry", {}).get("net_height", 2.43)
            )
        net_height_m = float(stored_net_height)
        net_post_spacing_m = (
            float(sam["net_post_spacing_m"])
            if sam.get("net_post_spacing_m") is not None else None
        )
        raw_corners = data.get("court_corners")
        court_corners = (
            [(float(point[0]), float(point[1])) for point in raw_corners]
            if raw_corners and len(raw_corners) == 4 else None
        )
        if net_floor_left and net_floor_right:
            # Calibrations persisted before the floor line was levelled still
            # carry a noise tilt, so re-level on load instead of requiring a
            # fresh detection run.
            from src.detection.net_detector import (
                anchor_net_top_to_calibration,
                infer_net_bottom_y_from_floor,
                level_floor_line,
            )
            net_floor_left, net_floor_right = level_floor_line(net_floor_left, net_floor_right)
            if net_tape_top is not None:
                anchored_top = anchor_net_top_to_calibration(
                    net_tape_top,
                    net_y,
                    net_height_hint=60.0 * sam_sy,
                )
                if anchored_top != net_tape_top:
                    log.warning(
                        f"Stored SAM net top {net_tape_top:.0f}px is inconsistent "
                        f"with the calibrated tape {net_y:.0f}px; repairing it"
                    )
                net_tape_top = anchored_top
                floor_y = (net_floor_left[1] + net_floor_right[1]) / 2.0
                net_tape_bottom = infer_net_bottom_y_from_floor(
                    net_tape_top,
                    floor_y,
                    net_height_m,
                )

        return self._validated_net_floor(CalibrationData(
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
            net_height_m=net_height_m,
            net_post_spacing_m=net_post_spacing_m,
            homography=data.get("perspective_matrix"),
            court_corners=court_corners,
        ))

    @staticmethod
    def _validated_net_floor(calibration: Optional[CalibrationData]) -> Optional[CalibrationData]:
        """Reject a geometrically impossible detected divider, in memory only.

        An invalid Hough/SAM floor line can put both teams on one side; the
        six-player cap then drops real players. Use the calibrated court's
        centre line at the detected post X coordinates, preserving its slope.
        """
        c = calibration
        if c is None or c.homography is None or not (c.net_floor_left and c.net_floor_right):
            return c
        if c.net_floor_method == "calibrated_court_midline_fallback":
            return c
        from src.detection.net_detector import NetDetector, infer_net_bottom_y_from_floor
        spacing = c.net_post_spacing_m
        if spacing is None:
            spacing = NetDetector.estimate_post_spacing_m(
                (c.net_floor_left[0] * 1280 / c.frame_width,
                 c.net_floor_left[1] * 720 / c.frame_height),
                (c.net_floor_right[0] * 1280 / c.frame_width,
                 c.net_floor_right[1] * 720 / c.frame_height),
                c.homography, c.frame_width, c.frame_height,
            )
        if NetDetector.post_spacing_is_plausible(spacing):
            return c
        try:
            projector = CourtProjector(c.homography)
            a = projector.world_to_image((-projector.half_width, 0.0))
            b = projector.world_to_image((projector.half_width, 0.0))
            if not np.isfinite([*a, *b]).all() or abs(b[0] - a[0]) < 1e-6:
                return c
            def on_midline(x):
                return (float(x), float(a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0])))
            left, right = on_midline(c.net_floor_left[0]), on_midline(c.net_floor_right[0])
            bottom = c.net_tape_bottom
            if c.net_tape_top is not None:
                bottom = infer_net_bottom_y_from_floor(c.net_tape_top, (left[1] + right[1]) / 2, c.net_height_m)
        except (ValueError, TypeError, ZeroDivisionError, np.linalg.LinAlgError):
            return c
        log.warning(
            f"Rejecting {c.net_floor_method} team divider (post spacing {spacing:.2f} m); "
            "using calibrated court midline. Source calibration is unchanged."
        )
        return replace(c, net_floor_left=left, net_floor_right=right,
                       net_floor_method="calibrated_court_midline_fallback",
                       net_post_spacing_m=None, net_tape_bottom=bottom)

    def _scale_calibration(self, actual_w: int, actual_h: int) -> "CalibrationData":
        """Return calibration scaled to actual frame size, with optional net_y override."""
        c = self._validated_net_floor(self.calibration)
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
                net_height_m=c.net_height_m,
                net_post_spacing_m=c.net_post_spacing_m,
                homography=homography_scaled,
                court_corners=[
                    (point[0] * sx, point[1] * sy)
                    for point in c.court_corners
                ] if c.court_corners else None,
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
                net_height_m=c.net_height_m,
                net_post_spacing_m=c.net_post_spacing_m,
                homography=c.homography,
                court_corners=c.court_corners,
            )
        return c

    def _resolve(self, name: str, override: Optional[str] = None) -> ResolvedModel:
        """Ask the registry which weights this run should use, and record it.

        Provenance is kept per analyzer and copied into ``*_stats.json``: a
        result that cannot say which model produced it is the problem the
        registry exists to solve.
        """
        resolved = resolve_model(name, override=override,
                                 use_registry=self.use_model_registry)
        self.model_provenance[name] = resolved.as_dict()
        return resolved

    def _init_ball_detector(self, model_path: Optional[str]):
        primary = self._resolve("ball_detector", model_path)
        imgsz = int(primary.extra.get("imgsz") or _BALL_IMGSZ)
        def detector(path, size, grid=None):
            from detection.detect_ball_on_video import BallDetectorVideo
            return BallDetectorVideo(str(path), imgsz=size, tile_grid=grid)
        backend = file_digest(_ROOT/'scripts/detection/detect_ball_on_video.py')
        self._ball_specs['primary'] = model_spec(primary.path, backend=backend, imgsz=imgsz,
                                                  batch_size=self.ball_batch_size)
        self.ball_detector = LazyModel(lambda: detector(primary.path, imgsz))
        self.ball_detector_2 = None
        if self.ball_ensemble:
            second = self._resolve("ball_detector_tiled")
            grid = tuple(second.extra.get("tile_grid") or _BALL_TILE_GRID_2)
            size = int(second.extra.get("imgsz") or 640)
            self._ball_specs['tiled'] = model_spec(second.path, backend=backend, imgsz=size, tile_grid=grid)
            self.ball_detector_2 = LazyModel(lambda: detector(second.path, size, grid))
            log.info(f"Ball ensemble: second opinion from {second} "
                     f"tiled {grid[0]}x{grid[1]}")
        self.ball_gate = BallTrackGate()
        self._ball_px_scale = 1.0

    # Ball selection is tuned in pixels on 1280-wide footage. Everything that
    # bounds how far the ball may travel between frames has to grow with the
    # frame, or 4K input — where the same shot covers three times the pixels —
    # reads as physically impossible motion and the track is never picked up.
    _BALL_TUNING_WIDTH = 1280.0

    def _configure_ball_selection_for(self, frame_width: int) -> None:
        scale = float(frame_width) / self._BALL_TUNING_WIDTH
        self._ball_px_scale = scale
        if abs(scale - 1.0) < 1e-6:
            self.ball_gate = BallTrackGate()
            return
        self.ball_gate = BallTrackGate(
            max_travel_px_per_frame=75.0 * scale,
            gate_jitter_px=25.0 * scale,
            gate_max_px=320.0 * scale,
            acquire_min_travel_px=18.0 * scale,
            acquire_max_step_px=110.0 * scale,
            still_eps_px=3.0 * scale,
        )
        log.info(f"Ball selection scaled x{scale:.2f} for {frame_width}px-wide frames")

    def _init_pose_model(self, model_path: Optional[str]):
        resolved = self._resolve("pose_estimator", model_path)
        if resolved.path.exists():
            from importlib import metadata
            tracker = metadata.distribution('ultralytics').locate_file('ultralytics/cfg/trackers/bytetrack.yaml')
            self._pose_tracker_path = tracker
            backend = hashlib.sha256(inspect.getsource(type(self)._infer_pose_model).encode()).hexdigest()
            self._pose_spec = model_spec(resolved.path, backend=backend,
                                         imgsz=self.pose_imgsz, conf=.1, batch=1,
                                         precision='fp16_cuda_fp32_cpu')
            self._pose_spec['tracker'] = file_digest(tracker)
            def load():
                from ultralytics import YOLO
                return YOLO(str(resolved.path))
            self.pose_model = LazyModel(load)
            log.info(f"Pose model resolved (loads only on cache miss): {resolved}")
        else:
            raise FileNotFoundError(f'Pose checkpoint unavailable: {resolved.path}; install it before inference')

    def _init_event_predictor(self, event_checkpoint_path: Optional[str]):
        """Load an explicitly selected event predictor, or disable the overlay."""
        if not event_checkpoint_path and not self._DEFAULT_EVENT_CHECKPOINT:
            log.info(
                "Event overlay disabled by default; pass --event-checkpoint "
                "with a validated checkpoint to enable it"
            )
            return None
        # The registry decides which of the twenty-odd checkpoint directories is
        # the production event model; an explicit --event-checkpoint still wins.
        resolved = self._resolve("event_sequence", event_checkpoint_path)
        p = resolved.path
        if not p.exists():
            log.info(f"Event overlay disabled: checkpoint not found at {p}")
            return None
        try:
            from action_recognition.inference.streaming_event_predictor import StreamingEventPredictor
            log.info(f"Event model: {resolved}")
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
        if isinstance(self.ball_detector, LazyModel):
            return  # A cache hit must never execute dummy inference.
        import torch
        if not torch.cuda.is_available():
            return
        log.info("Warming up CUDA kernels (first-time only, may take a few minutes)...")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        t0 = time.time()
        self.ball_detector.detect_ball(dummy)
        log.info(f"  Ball model warmed up in {time.time()-t0:.1f}s")
        if self.pose_model is not None:
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
        # Scale legacy pixel thresholds from their 1280px tuning space. The
        # physical motion/contact geometry must not change when input switches
        # between 720p, 1080p and 4K.
        frame_width = float(self.calibration.frame_width) if self.calibration else 3840.0
        px_scale = frame_width / self._BALL_TUNING_WIDTH
        fsm.event_tracker.contact_radius = 100.0 * px_scale
        fsm.event_tracker.min_speed_px_per_sec = 120.0 * px_scale
        # Keep candidate recall high: net crossing + reception/follow-up now
        # provide the precision, so the initial velocity gate can be permissive.
        fsm.min_serve_speed = 200.0 * px_scale
        fsm.ROLLING_SPEED_PX_PER_FRAME = 6.0 * px_scale
        fsm.CARRIED_SPEED_PX_PER_FRAME = 20.0 * px_scale
        fsm.FLOOR_MARGIN_PX = 70.0 * px_scale
        fsm.SERVE_OUT_MARGIN_PX = 30.0 * px_scale
        fsm.RALLY_OUT_MARGIN_PX = 50.0 * px_scale
        fsm.BLOCK_NET_MARGIN = 80.0 * px_scale

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
        candidates = self._ball_candidates_for(frame, frame_idx)
        return self._select_ball_candidate(candidates, frame_idx)

    def _select_ball_candidate(
        self, candidates: List[BallCandidate], frame_idx: int,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Feed precomputed candidates through the stateful live gate."""
        # The global trajectory pass runs after the video is read, so every
        # candidate has to be kept, not just the one the gate picks now.
        if self.ball_trajectory_dp and candidates:
            self._ball_candidates[frame_idx] = candidates

        chosen = self.ball_gate.update(frame_idx, candidates)
        if chosen is None:
            return None, 0.0
        return (chosen.x, chosen.y), chosen.confidence

    def _candidates_from_detections(self, detections) -> List[BallCandidate]:
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

    def _ball_candidates_for(self, frame: np.ndarray, frame_idx=None, primary=None) -> List[BallCandidate]:
        """Every plausible ball in this frame, from one or both detectors."""
        def raw(name, detector):
            cache = getattr(self, '_model_cache', None)
            def infer():
                return [[float(v) for v in row] for row in detector.detect_ball(frame)]
            return cache.compute(self._ball_stages[name], frame_idx, infer) if cache else infer()
        candidates = self._candidates_from_detections(primary if primary is not None else raw('primary', self.ball_detector))
        if self.ball_detector_2 is None:
            return candidates

        # Merge the second opinion; the two models often report the same ball.
        for extra in self._candidates_from_detections(
            raw('tiled', self.ball_detector_2)
        ):
            if not any(
                math.hypot(extra.x - c.x, extra.y - c.y) < self.BALL_MERGE_RADIUS
                for c in candidates
            ):
                candidates.append(extra)
        return candidates

    def _infer_pose_model(self, frame, kwargs):
        settings = self._pose_spec['settings']
        result = self.pose_model.predict(frame, conf=settings.get('conf',.1),
                                         batch=settings.get('batch',1), mode='track', **kwargs)
        return pack_pose_result(result[0] if result else None)

    def _detect_players(
        self,
        frame: np.ndarray,
        fsm_state: str = "",
    ) -> Tuple[Dict[str, List[Tuple[float, float]]], List[PlayerPose]]:
        """
        YOLO-pose + ByteTrack. Returns:
            teams: {"team_a": [(cx,cy), ...], "team_b": [...]} — legacy format
                   for FSM / StreamingEventPredictor (team_a = far side, y < net_y)
            poses: List[PlayerPose] — full 17-keypoint skeletons with track IDs
        """
        if self.pose_model is None:
            return {"team_a": [], "team_b": []}, []

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs = dict(verbose=False, device=device, half=(device == "cuda"))
        if self.pose_imgsz:
            kwargs["imgsz"] = self.pose_imgsz

        cache = getattr(self, '_model_cache', None)
        frame_index = getattr(self, '_current_pose_frame', None)
        saved = cache.get(self._pose_sequence, frame_index) if cache and self._pose_cache_read else None
        if saved is not None:
            results = [unpack_pose_result(saved)]
        elif cache and not self._pose_cache_read:
            def infer_pose():
                return self._infer_pose_model(frame, kwargs)
            raw = cache.compute(self._raw_pose_stage, frame_index, infer_pose)
            saved = self._cached_pose_tracker.update(raw, self._cache_frame_size)
            cache.put(self._pose_sequence, frame_index, saved)
            results = [unpack_pose_result(saved)]
        else:
            if cache and self._pose_cache_read:
                raise CacheMiss(f'Incomplete/corrupt sealed pose sequence at {frame_index}')
            try:
                results = self.pose_model.track(
                    frame, persist=True, tracker="bytetrack.yaml", **kwargs,
                )
            except Exception as e:
                if not self._pose_track_warned:
                    log.warning(f"Pose tracking unavailable ({e}) — using per-frame inference (no IDs)")
                    self._pose_track_warned = True
                results = self.pose_model(frame, **kwargs)
            if cache:
                cache.put(self._pose_sequence, frame_index, pack_pose_result(results[0] if results else None))

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

        recovery = getattr(self, "_person_pose_recovery", None)
        if recovery is not None:
            if cache:
                def pack_pose(pose):
                    value = asdict(pose)
                    value['keypoints'] = pose.keypoints.tolist()
                    return value
                spec = {'sidecar': recovery.sidecar_hash, 'weights': self._recovery_model_sha,
                        'backend': self._recovery_backend_sha, 'pose_code': self._recovery_pose_code_sha,
                        'geometry': asdict(self.calibration)}
                stage = cache.stage('sam_guided_pose', spec)
                key = digest({'frame': frame_index, 'poses': [pack_pose(p) for p in poses]})
                counters = ('covered_frames', 'crop_inferences', 'proposals_recovered')
                def recover():
                    before = {name: getattr(recovery, name) for name in counters}
                    output = recovery.recover(frame, frame_index, poses, projector, device)
                    delta = {name: getattr(recovery, name)-before[name] for name in counters}
                    for name in counters:
                        setattr(recovery, name, before[name])
                    return {'poses': [pack_pose(p) for p in output], 'counters': delta}
                value = cache.compute(stage, key, recover)
                poses = [PlayerPose(**{**p, 'keypoints': np.asarray(p['keypoints'], dtype=np.float32)})
                         for p in value['poses']]
                for name, delta in value['counters'].items():
                    setattr(recovery, name, getattr(recovery, name)+delta)
            else:
                poses = recovery.recover(frame, frame_index, poses, projector, device)

        # On-court filter with track hysteresis: spectators/benches never get
        # in; players chasing a ball outside the lines don't drop out.
        if projector is not None:
            if self._court_gate is None or self._court_gate.projector is not projector:
                self._court_gate = CourtPlayerGate(projector, self.max_players_per_team)
            poses = self._court_gate.filter(
                poses,
                serve_setup=fsm_state in {"serve_prep", "serve_ready"},
            )

        if recovery is not None:
            recovery.accepted_person_frames += sum(p.source == "sam_guided_yolo_pose" for p in poses)
        return poses_to_teams(poses), poses

    def _get_projector(self) -> Optional[CourtProjector]:
        """CourtProjector for the current (scaled) calibration, cached."""
        c = self.calibration
        if c is None or c.homography is None:
            return None
        if self._projector_calib is not c:  # calibration object changed (rescaled)
            try:
                floor_line = None
                net_tape_band = None
                if c.net_floor_left and c.net_floor_right:
                    floor_line = (c.net_floor_left, c.net_floor_right)
                if (
                    c.net_floor_left and c.net_floor_right
                    and c.net_tape_top is not None and c.net_tape_bottom is not None
                ):
                    x1, x2 = sorted((c.net_floor_left[0], c.net_floor_right[0]))
                    net_tape_band = (x1, x2, c.net_tape_top, c.net_tape_bottom)
                self._projector = CourtProjector(
                    c.homography,
                    net_floor_line=floor_line,
                    net_tape_band=net_tape_band,
                )
            except Exception as e:
                log.warning(f"Court projector unavailable ({e}) — using x-range player filter")
                self._projector = None
            self._projector_calib = c
        return self._projector

    def _maybe_run_sam_net_detection(self, video_path, frame_w, frame_h, refresh_floor=False):
        cache = getattr(self, '_model_cache', None)
        if cache is None or not self.enable_net_detection:
            return self._run_net_detection_uncached(video_path, frame_w, frame_h, refresh_floor)
        import os
        resolved = self._resolve('net_sam')
        worker = _ROOT/'scripts/detection/detect_net_posts_sam3.py'
        python = Path(os.environ.get('SAM3_PYTHON') or (_ROOT/'.sam3_env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')))
        sam3 = {'python':str(python),'available':python.is_file(),
                'worker_sha256':file_digest(worker) if worker.is_file() else None}
        if python.is_file():
            try:
                from huggingface_hub import try_to_load_from_cache
                for name in ('sam3.pt','config.json'):
                    path = try_to_load_from_cache('facebook/sam3',name)
                    sam3[name] = file_digest(path) if isinstance(path,str) and Path(path).is_file() else None
            except ImportError:
                sam3['checkpoint_resolution']='unavailable'
        base = {'weights': file_digest(resolved.path) if resolved.path.is_file() else None,
                'backend': file_digest(_ROOT/'src/detection/net_detector.py'),
                'size': [frame_w,frame_h], 'net_height_m': self.net_height_m,
                'refresh_floor': refresh_floor,
                'sam3_worker':sam3}
        def stage():
            return cache.stage('net_geometry', {**base, 'calibration': asdict(self.calibration) if self.calibration else None})
        key = stage()
        def infer():
            self._run_net_detection_uncached(video_path, frame_w, frame_h, refresh_floor)
            return {'calibration': asdict(self.calibration) if self.calibration else None,
                    'models': {k:v for k,v in self.model_provenance.items() if k.startswith('net_')}}
        result = cache.compute(key, 'result', infer)
        self.calibration = CalibrationData(**result['calibration']) if result['calibration'] else None
        self.model_provenance.update(result['models'])
        cache.put(stage(), 'result', result)  # Calibration write-back must not force another SAM pass.

    def _run_net_detection_uncached(
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
        if not self.enable_net_detection:
            log.info("Net detection disabled for this analysis")
            return

        c = self.calibration
        if (c is not None and c.net_tape_top is not None
                and c.net_floor_left is not None and c.net_floor_right is not None
                and c.net_floor_method == "sam3_post_x_physical_height"
                and abs(c.net_height_m - self.net_height_m) < 1e-6
                and not refresh_floor):
            log.info(f"Net tape already calibrated: top={c.net_tape_top:.0f}px "
                     f"bottom={c.net_tape_bottom:.0f}px (calibration px)")
            return

        log.info("Net tape not in calibration — running SAM net segmentation on first 5 s …")
        try:
            from src.detection.net_detector import (
                NetDetector,
                infer_net_bottom_y_from_floor,
                save_sam_net_to_calibration,
            )

            net_y = c.net_y if c else frame_h * 0.5
            net_x_left = c.net_x_left if c else 0.0
            net_x_right = c.net_x_right if c else frame_w

            detector = NetDetector(net_height_m=self.net_height_m,
                                   use_model_registry=self.use_model_registry)
            if detector.model_provenance:
                self.model_provenance["net_sam"] = detector.model_provenance
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
                net_tape_top=top,
                net_tape_bottom=bot,
            )
            floor_method = "sam3_post_x_physical_height" if floor_720 else None
            floor_spacing_m = None
            if floor_720 is not None and c is not None:
                floor_spacing_m = detector.estimate_post_spacing_m(
                    floor_720[0], floor_720[1], c.homography,
                    c.frame_width, c.frame_height,
                )
                if not detector.post_spacing_is_plausible(floor_spacing_m):
                    log.warning(
                        f"SAM3 post spacing {floor_spacing_m:.2f} m is outside "
                        "the expected 10-11 m; trying Hough fallback"
                    )
                    floor_720 = None
                    floor_method = None
            if floor_720 is None:
                floor_x_priors = detector.project_net_floor_x_priors(
                    c.homography if c else None,
                    c.frame_width if c else frame_w,
                    c.frame_height if c else frame_h,
                )
                floor_720 = detector.detect_floor_line_from_video(
                    video_path=video_path,
                    net_x_left=net_x_left,
                    net_x_right=net_x_right,
                    frame_h=c.frame_height if c else frame_h,
                    frame_w=c.frame_width if c else frame_w,
                    net_tape_top=top * (c.frame_height if c else frame_h) / 720.0,
                    net_tape_bottom=bot * (c.frame_height if c else frame_h) / 720.0,
                    floor_x_priors=floor_x_priors,
                )
                floor_method = "hough_post_x_physical_height" if floor_720 else None
                if floor_720 is not None and c is not None:
                    floor_spacing_m = detector.estimate_post_spacing_m(
                        floor_720[0], floor_720[1], c.homography,
                        c.frame_width, c.frame_height,
                    )
                    if not detector.post_spacing_is_plausible(floor_spacing_m):
                        log.warning(
                            f"Hough post spacing {floor_spacing_m:.2f} m is outside "
                            "the expected 10-11 m; keep it only as a low-confidence fallback"
                        )
            if floor_spacing_m is not None:
                log.info(f"Net post spacing: {floor_spacing_m:.2f} m (expected 10-11 m)")

            if floor_720 is not None:
                floor_y_720 = (floor_720[0][1] + floor_720[1][1]) / 2.0
                physics_bot = infer_net_bottom_y_from_floor(
                    top,
                    floor_y_720,
                    self.net_height_m,
                )
                log.info(
                    f"Net mesh bottom from physical proportions: {bot:.1f}px -> "
                    f"{physics_bot:.1f}px (1.00m / {self.net_height_m:.2f}m)"
                )
                bot = physics_bot

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
                    net_height_m=self.net_height_m,
                    net_post_spacing_m=floor_spacing_m,
                    homography=c.homography,
                    court_corners=c.court_corners,
                )
            # Persist original 720p values for future runs
            if self._calibration_path and self._calibration_path.exists():
                save_sam_net_to_calibration(
                    str(self._calibration_path), top, bot, floor_720, floor_method,
                    net_height_m=self.net_height_m,
                    net_post_spacing_m=floor_spacing_m,
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
        ball_source: str = "detected",
        ball_confidence: Optional[float] = None,
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
            ball_colors = {
                "detected": (0, 255, 0),
                "interpolated_short": (0, 200, 255),
                "reconstructed_conservative": (255, 212, 0),
                "reconstructed_contact_aware": (216, 79, 255),
                "reconstructed_occlusion": (250, 139, 167),
            }
            color = ball_colors.get(ball_source, (0, 255, 0))
            cv2.circle(out, (cx, cy), 18, color, 3)
            cv2.circle(out, (cx, cy), 5, color, -1)
            if ball_source != "detected":
                label = ball_source.replace("reconstructed_", "")
                if ball_confidence is not None:
                    label += f" {float(ball_confidence):.2f}"
                cv2.putText(
                    out, label, (cx + 22, max(24, cy - 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )

        # Players — skeletons when keypoints are available, dots as fallback
        if player_poses:
            for pose in player_poses:
                st = self._SKELETON_STYLE[pose.team]
                draw_skeleton(out, pose, st["color"],
                              bone_thickness=st["bone"], kp_radius=st["kp"],
                              draw_id=st["draw_id"] or pose.player_identity_processed)
                if pose.team_prior:
                    x1, y1 = int(pose.bbox[0]), int(pose.bbox[1])
                    cv2.putText(
                        out, "BACK/NET prior", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
                    )
        else:
            colors = {"team_a": (255, 80, 0), "team_b": (0, 80, 255)}
            for team, pts in player_teams.items():
                for px, py in pts:
                    cv2.circle(out, (int(px), int(py)), 10, colors[team], -1)

        # Confirmed lifecycle overlay (line 1). The old raw FSM tuple often
        # displayed internally valid but user-meaningless combinations such as
        # ``serving | | team_b [0]``.
        lifecycle = fsm_status.get("lifecycle") or {}
        lifecycle_status = lifecycle.get("status", "unconfirmed")
        lifecycle_labels = {
            "unconfirmed": "Waiting for serve",
            "serve_candidate": "Serve candidate",
            "serve_confirmed": "Serve confirmed",
            "reception_pending": "Reception confirmed",
            "playable_rally": "Playable rally",
            "ace_no_reception": "Ace / no reception",
            "reception_failed": "Reception failed",
            "serve_fault": "Serve fault",
        }
        team  = fsm_status.get("current_team", "?")
        contacts = fsm_status.get("team_contacts", 0)
        text = lifecycle_labels.get(lifecycle_status, lifecycle_status)
        if lifecycle_status in {"reception_pending", "playable_rally"}:
            text += f" | {team} [{contacts}]"
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

        overlay = _EventOverlayState(fps)
        if self._event_predictor is not None:
            self._event_predictor.reset()

        t0 = time.time()
        progress_interval = max(1, int(fps / 2))

        def report_frame_progress(frame_idx: int, force: bool = False) -> None:
            if not force and frame_idx % progress_interval:
                return
            elapsed = time.time() - t0
            frame_current = min(frame_idx + 1, total)
            frame_progress = frame_current / max(total, 1)
            source_fps = frame_current / max(elapsed, 0.01)
            eta_sec = (
                (total - frame_current) / source_fps
                if source_fps > 0 and frame_current < total
                else 0.0
            )
            self._emit_progress(
                phase="analyzing",
                progress=round(frame_progress * 97.0, 2),
                frame_current=frame_current,
                frame_total=total,
                video_percent=round(frame_progress * 100.0, 2),
                processing_fps=round(source_fps, 2),
                eta_sec=round(eta_sec, 1),
                elapsed_sec=round(elapsed, 1),
                message=(
                    f"Analyzing frame {frame_current:,} of {total:,} "
                    f"({frame_progress * 100.0:.1f}%)"
                ),
            )

        batch_enabled = self.ball_batch_size > 1 and self.ball_detector_2 is None
        if batch_enabled:
            log.info(f"Ball YOLO batch size: {self.ball_batch_size}")

        def frame_chunks():
            chunk = []
            for item in frame_iter:
                chunk.append(item)
                if len(chunk) >= self.ball_batch_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        def frames_with_ball_candidates():
            nonlocal batch_enabled
            for frame_chunk in frame_chunks():
                cache = getattr(self, '_model_cache', None)
                if cache is not None:
                    raw = {}
                    missing = []
                    for idx,image in frame_chunk:
                        if self.skip_frames > 0 and idx % (self.skip_frames+1):
                            continue
                        value = cache.get(self._ball_stages['primary'], idx)
                        if value is None:
                            missing.append((idx,image))
                        else:
                            raw[idx] = value
                    if missing and cache.require:
                        raise CacheMiss(f'Missing ball model outputs at frame {missing[0][0]}')
                    if missing:
                        if batch_enabled and len(missing) > 1:
                            try:
                                values = self.ball_detector.detect_ball_batch([image for _,image in missing])
                            except Exception as error:
                                log.warning(f'Ball batch unavailable ({error}); using single-frame model outputs')
                                batch_enabled = False
                                values = [self.ball_detector.detect_ball(image) for _,image in missing]
                            if len(values) != len(missing):
                                raise ValueError('Ball batch output count differs from input')
                        else:
                            values = [self.ball_detector.detect_ball(image) for _,image in missing]
                        cache.computed['ball_primary'] += len(missing)
                        for (idx,_),value in zip(missing,values):
                            raw[idx] = [[float(v) for v in row] for row in value]
                            cache.put(self._ball_stages['primary'], idx, raw[idx])
                    for idx,image in frame_chunk:
                        yield idx,image,(self._ball_candidates_for(image,idx,raw[idx]) if idx in raw else None)
                    continue
                batched_candidates: Dict[int, List[BallCandidate]] = {}
                if batch_enabled:
                    processable = [
                        (idx, frame)
                        for idx, frame in frame_chunk
                        if self.skip_frames <= 0
                        or idx % (self.skip_frames + 1) == 0
                    ]
                    if len(processable) > 1:
                        try:
                            detections = self.ball_detector.detect_ball_batch(
                                [frame for _, frame in processable]
                            )
                            batched_candidates = {
                                idx: self._candidates_from_detections(frame_detections)
                                for (idx, _), frame_detections
                                in zip(processable, detections)
                            }
                        except Exception as exc:
                            log.warning(
                                f"Ball batch inference unavailable ({exc}); "
                                "falling back to per-frame inference"
                            )
                            batch_enabled = False
                for frame_idx, frame in frame_chunk:
                    yield frame_idx, frame, batched_candidates.get(frame_idx)

        for frame_idx, frame, candidates in frames_with_ball_candidates():
            if getattr(self, '_model_cache', None) is not None:
                self._cache_visited_frames.append(frame_idx)
            stats.total_frames += 1

            # Frame skipping
            if self.skip_frames > 0 and frame_idx % (self.skip_frames + 1) != 0:
                if out_writer:
                    out_writer.write(frame)
                report_frame_progress(frame_idx)
                continue

            stats.processed_frames += 1
            timestamp = frame_idx / fps

            # --- Detect ---
            if candidates is None:
                ball_pos, ball_conf = self._detect_ball(frame, frame_idx)
            else:
                ball_pos, ball_conf = self._select_ball_candidate(
                    candidates, frame_idx,
                )
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

            self._current_pose_frame = frame_idx
            player_teams, player_poses = self._detect_players(
                frame,
                fsm_state=cur_state_pre,
            )

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
                    "lifecycle": dict(ev.lifecycle) if ev.lifecycle else None,
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
            event_display = None
            if self._event_predictor is not None and self.calibration is not None:
                raw_display = self._event_predictor.add_frame(
                    ball_pos, player_teams, self.calibration.net_y, fps,
                    fsm_state=cur_state or "",   # let predictor exclude impossible classes
                    video_time=frame_idx / fps,
                    pose_features=pose_features,  # 8 pose channels (new checkpoints)
                    frame_size=(
                        self.calibration.frame_width,
                        self.calibration.frame_height,
                    ),
                    ball_new=bool(ball_pos and self.ball_gate.acquired_this_frame),
                )
                event_display = overlay.update(raw_display, cur_state or "", frame_idx)
                guard = getattr(self._event_predictor, 'serve_guard', None)
                if guard is not None and guard.reason(event_display):
                    event_display = None

            # --- Collect per-frame annotation for viewer (FSM + Transformer for comparison) ---
            frame_events_this = [e for e in stats.all_events if e["frame"] == frame_idx]
            ann = {
                "f": frame_idx,
                "t": round(timestamp, 3),
                "ball": [round(ball_pos[0], 1), round(ball_pos[1], 1)] if ball_pos else None,
                "ball_confidence": round(float(ball_conf), 4) if ball_pos else 0.0,
                # The gate started a new track here: the step from the previous
                # position is a change of hypothesis, not the ball moving.
                "ball_new": True if (ball_pos and self.ball_gate.acquired_this_frame) else None,
                "p": [[round(x, 1), round(y, 1)]
                      for x, y in player_teams.get("team_a", []) + player_teams.get("team_b", [])],
                "state": cur_state,
                "phase": fsm_status.get("rally_phase"),
                "lifecycle": fsm_status.get("lifecycle"),
                "current_team": fsm_status.get("current_team"),
                "contacts": fsm_status.get("team_contacts", 0),
                "ev": [e["type"] for e in frame_events_this],
                # Full skeletons: id, team (a/b), bbox, 17×[x,y,conf]
                "poses": [
                    {
                        "id": p.track_id,
                        "team": p.team[-1],  # "a" / "b"
                        "bb": [round(v, 1) for v in p.bbox],
                        "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 2)]
                               for x, y, c in p.keypoints],
                        "back": (
                            round(float(p.back_facing_score), 2)
                            if p.back_facing_score is not None else None
                        ),
                        "team_prior": p.team_prior,
                        "source": p.source,
                        "mask_instance_id": p.mask_instance_id,
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
            report_frame_progress(frame_idx, force=(frame_idx + 1 >= total))

        self._finish_detection_cache(stats)
        stats.rallies = completed_rallies
        stats.processing_time_sec = time.time() - t0
        # Written last: SAM net detection may add its own entry mid-run.
        stats.models = dict(self.model_provenance)
        self._emit_progress(
            phase="trajectory",
            progress=98.0,
            frame_current=min(stats.total_frames, total),
            frame_total=total,
            video_percent=100.0,
            processing_fps=round(
                stats.total_frames / max(stats.processing_time_sec, 0.01), 2,
            ),
            eta_sec=None,
            elapsed_sec=round(stats.processing_time_sec, 1),
            message="Optimizing the complete ball trajectory...",
        )

        # Post-process: re-pick the whole trajectory from the candidate cloud.
        # The gate above had to choose with only the past in hand; now the
        # frames after each decision are available too, and they are what
        # identifies the ball on frames where a distractor scored higher.
        self._current_fps = float(fps or 30.0)
        solved = self._apply_trajectory_dp(stats) if self.ball_trajectory_dp else False

        # These two clean up after the greedy gate and must not run after the
        # solver, which supersedes both: its output is smooth by construction,
        # and it leaves a frame empty *because* no candidate there was
        # plausible. Filling those holes anyway — the old pass bridged gaps of
        # up to 10 frames — replaces a considered "unknown" with a guess, and
        # measured on the GT window it added 46 wrong positions for 15 right.
        # They are skipped on the solver actually having run, not on the flag:
        # a run with no candidates still has gate output that needs them.
        if not solved:
            # Post-process: remove trajectory jumps (false positives inconsistent with neighbours)
            n_removed = self._filter_ball_outliers(stats.frame_annotations)
            if n_removed:
                log.info(f"Outlier filter: removed {n_removed} false-positive frames")

            # Post-process: fill short gaps in ball detection by linear interpolation
            n_filled = self._interpolate_ball_positions(stats.frame_annotations)
            if n_filled:
                log.info(f"Ball interpolation: filled {n_filled} missing frames")

        # Safety net over the final trajectory, after the legacy gate clean-up so
        # nothing can re-fill a removed island. The candidate-level pass above
        # only runs when the solver does; the live gate, `--no-ball-dp`, and any
        # result loaded from an older run still need the same two physical
        # rules applied to the points that were actually chosen.
        if self.suppress_static_ball_sources:
            report = clean_ball_annotations(
                stats.frame_annotations,
                fps=self._current_fps,
                px_scale=self._ball_px_scale,
            )
            report.update(getattr(self, "_static_source_report", {}))
            stats.ball_static_suppression = report
            if report["removed_points"]:
                log.info(
                    f"Static/jump suppression: {report['removed_points']} trajectory "
                    f"points cleared "
                    f"({report['static_sources']['removed_points']} fixed object, "
                    f"{report['jump_islands']['removed_points']} jump islands)"
                )

        # Which candidate is the ball is settled above; where the ball is is
        # not. The chosen point is a detector box centre and it breathes a few
        # pixels every frame, which is the zig-zag visible in the drawn trail.
        # Fit it out *before* the FSM re-run so events are measured on the same
        # trajectory the viewer shows — and so contact detection sees a clean
        # incoming leg instead of noise it has to be tolerant of.
        self._smooth_ball_trajectory(stats)

        # The inline FSM above has to run on raw per-frame detections, because
        # the annotated video needs a state to draw while the frame is in hand.
        # Outlier rejection only becomes possible once the whole trajectory is
        # known, and every false detection the filter removes had already been
        # fed to the FSM as a contact. Re-run it over the cleaned trajectory so
        # the saved events and rallies come from the same data the heatmap uses.
        (
            stats.all_events,
            stats.rallies,
            stats.service_attempts,
        ) = self._recompute_events_on_clean_trajectory(
            stats.frame_annotations, fps,
        )
        if self.match_segments:
            stats.all_events = filter_non_game_records(
                stats.all_events, self.match_segments
            )
            stats.service_attempts = filter_non_game_records(
                stats.service_attempts,
                self.match_segments,
                frame_field="start_frame",
            )
            stats.rallies = [
                attempt for attempt in stats.rallies
                if not attempt.get("excluded_from_match_stats")
                and attempt in stats.service_attempts
            ]
        # Serve-centric derivation adds exact lifecycle analytics to every
        # service attempt and returns the canonical counters used by the viewer.
        stats.rally_lifecycle = annotate_rallies(stats.service_attempts, fps)
        # Context enrichment is deliberately a post-processing pass.  A point
        # is confirmed by the *next* serve, and stable team IDs / possession
        # numbers also require the complete ordered attempt list.
        stats.match_analytics = enrich_match_analytics(
            stats.service_attempts,
            team_identity_map=self.team_identity_map,
            match_segments=self.match_segments,
        )
        annotate_match_phases(stats.frame_annotations, self.match_segments)
        annotate_score_timeline(stats.frame_annotations, stats.match_analytics)
        self._recompute_features_and_predictions(stats.frame_annotations, fps)
        # Strictly after the recompute: the spotter reads the same final ball
        # coordinates and ``pose_f`` the exported annotations carry.
        self._spot_contacts(stats, fps)
        # The spotter, when it ran, logged its own rallies; these are the FSM's.
        fsm_lifecycle = stats.rally_lifecycle_fsm or stats.rally_lifecycle
        if fsm_lifecycle.get("service_attempts"):
            lc = fsm_lifecycle
            log.info(
                "FSM rally lifecycle: "
                f"{lc['service_attempts']} attempts, "
                f"{lc['confirmed_serves']} crossed net, "
                f"{lc['playable_rallies']} playable, "
                f"{lc['aces_no_reception']} ace/no reception, "
                f"{lc['reception_failed']} failed receptions, "
                f"{lc['serve_faults']} serve faults"
            )

        # Count only confirmed attacks. Generic trajectory turns also include
        # passes, digs and detector restarts, which create a noisy "heatmap"
        # unrelated to the requested attack-origin statistic.
        # Already scaled to the actual frame size by ``process_video``; the
        # projector on the next line reads the same object.
        calib = self.calibration
        stats.attack_heatmap = AttackHeatmap(fps).analyze(
            stats.frame_annotations,
            projector=self._get_projector(),
            events=stats.all_events,
            # Needed only when a rally ends on the attack and no defender
            # ever touches the ball; then the net crossing is the target.
            net_y=calib.net_y if calib else None,
            # Post bases mark the physical ends of the net; the calibration
            # net_x clicks can sit well inside them.
            net_x_left=(
                calib.net_floor_left[0] if calib and calib.net_floor_left
                else (calib.net_x_left if calib else None)
            ),
            net_x_right=(
                calib.net_floor_right[0] if calib and calib.net_floor_right
                else (calib.net_x_right if calib else None)
            ),
        )
        log.info(
            f"Attack zones: {stats.attack_heatmap['total_attacks']} "
            "confirmed attacks across left/center/right"
        )

        return stats

    def _recompute_features_and_predictions(self, annotations: List[Dict], fps: float) -> None:
        """Replay saved poses against final ball coordinates and clean FSM states.

        Features use the exported ``ball`` trajectory, including accepted
        reconstruction. The separate reconstruction-affects-FSM policy remains
        enforced by the clean FSM replay. Raw observations/provenance stay intact.
        """
        pose_fx = PoseFeatureExtractor()
        predictor = self._event_predictor
        if predictor is not None:
            predictor.reset()
        overlay = _EventOverlayState(fps)
        calibration = self.calibration
        pose_net_y = (
            calibration.net_tape_top
            if calibration is not None and calibration.net_tape_top is not None
            else (calibration.net_y if calibration is not None else 830.0)
        )
        for ann in annotations:
            frame_idx = int(ann["f"])
            poses = self._poses_from_annotation(ann)
            ball = ann.get("ball")
            ball_pos = tuple(float(value) for value in ball) if ball is not None else None
            if ann.get("ball_out_of_frame"):
                ball_pos = None
            ann["pose_f"] = pose_fx.update(frame_idx, poses, ball_pos, pose_net_y)
            ann.pop("event_model", None)
            ann.pop('event_model_decision', None)
            if predictor is None or calibration is None:
                continue
            teams = {"team_a": [], "team_b": []}
            for pose in poses:
                teams[pose.team].append(pose.center)
            # ``p`` stores the exact rounded centers used by offline feature
            # builders, grouped team_a then team_b. Avoid bbox-rounding drift.
            points = ann.get("p")
            if points is not None and len(points) == len(poses):
                split = len(teams["team_a"])
                teams = {"team_a": points[:split], "team_b": points[split:]}
            state = ann.get("state") or ""
            raw_display = predictor.add_frame(
                ball_pos, teams, calibration.net_y, fps,
                fsm_state=state,
                video_time=frame_idx / fps,
                pose_features=ann["pose_f"],
                frame_size=(calibration.frame_width, calibration.frame_height),
                ball_new=bool(ann.get("ball_new")),
            )
            display = overlay.update(raw_display, state, frame_idx)
            guard = getattr(predictor, 'serve_guard', None)
            if guard is not None and guard.reason(display):
                display = None
            if getattr(predictor, 'last_decision', None) is not None:
                ann['event_model_decision'] = dict(predictor.last_decision)
            if display is not None:
                ann["event_model"] = display

    def _spot_contacts(self, stats: "GameStats", fps: float) -> None:
        """Replace the FSM's contact events with the dense spotter's.

        Measured leave-one-video-out on the three gold videos, the spotter puts
        a typed contact on the right frame far more often than the FSM
        (0.79 against 0.156 four-class macro F1 at +/-5 frames). The contact
        events are replaced, and so are rallies and service attempts: they are
        rebuilt from the spotter's contacts, each rally opening on its serve
        (see ``rally_segmentation``). Score and match analytics stay
        FSM-derived; they were computed before this point and need team sides
        the spotter does not produce.

        The FSM's events, rallies, attempts and counters are kept in their
        ``*_fsm`` fields so enabling this never loses what the previous path
        produced.
        """
        # Read defensively: tests build an analyzer with ``__new__`` and set
        # only the attributes the path under test needs, as elsewhere here.
        checkpoint = getattr(self, "contact_spotter_path", None)
        if not checkpoint:
            return
        from action_recognition.rally_spotting.contact_spotter import (
            ContactSpotter, phase_segments,
        )
        from action_recognition.rally_spotting import dense_contacts as dc
        from action_recognition.rally_spotting.rally_segmentation import (
            segment_rallies,
        )

        if getattr(self, "_contact_spotter", None) is None:
            self._contact_spotter = ContactSpotter.load(checkpoint)
            log.info(
                f"Contact spotter: {checkpoint} "
                f"({len(self._contact_spotter.members)} member(s))"
            )
        self._cache_dense_probabilities(self._contact_spotter, checkpoint)
        calib = self.calibration
        payload = {
            "fps": fps,
            "frame_width": calib.frame_width if calib else 1280,
            "frame_height": calib.frame_height if calib else 720,
            "net_y": calib.net_y if calib else 370.0,
            "frames": stats.frame_annotations,
        }
        try:
            spotted = self._contact_spotter.run(payload)
        except Exception as error:
            if isinstance(error, CacheMiss):
                raise
            # A failed spotter must not cost the whole analysis; the FSM
            # events already in ``all_events`` stay as they are.
            log.warning(f"Contact spotter failed, keeping FSM events: {error}")
            stats.contact_spotter = {"enabled": True, "error": str(error)}
            return

        stats.all_events_fsm = list(stats.all_events)
        events = []
        for event in spotted.events:
            events.append({
                "frame": event["frame"],
                "time": round(event["frame"] / fps, 2),
                "type": event["type"],
                "core_type": event["core_type"],
                "confidence": event["confidence"],
                "phase_before": event["phase_before"],
                "phase_after": event["phase_after"],
                "source": event["source"],
                # The spotter does not attribute a contact to a side; these
                # stay empty rather than carrying a guess.
                "team": None,
                "game_state": None,
                "rally_phase": None,
            })
        serve_probability = (
            spotted.probabilities[:, dc.LABEL_TYPES.index("serve")]
            if spotted.probabilities is not None else None
        )
        segmented = segment_rallies(
            events, fps,
            serve_probability=serve_probability,
            phase_labels=spotted.phase_labels,
            frame_numbers=spotted.frame_numbers,
        )
        # A missed serve recovered from the serve probability is still the
        # model's evidence, so it joins the event list, marked ``inferred``
        # with its own source; the review queue reads only decoded contacts.
        events = sorted(
            events + segmented["inferred_serves"],
            key=lambda event: int(event["frame"]),
        )
        stats.all_events = events
        stats.rallies_fsm = list(stats.rallies)
        stats.service_attempts_fsm = list(stats.service_attempts)
        stats.rally_lifecycle_fsm = dict(stats.rally_lifecycle)
        attempts = segmented["service_attempts"]
        match_segments = getattr(self, "match_segments", None)
        if match_segments:
            attempts = filter_non_game_records(
                attempts, match_segments, frame_field="start_frame",
            )
        kept_ids = {attempt["service_attempt_id"] for attempt in attempts}
        stats.service_attempts = attempts
        stats.rallies = [
            rally for rally in segmented["rallies"]
            if rally["service_attempt_id"] in kept_ids
        ]
        lifecycle = dict(segmented["rally_lifecycle"])
        if match_segments:
            lifecycle["service_attempts"] = len(attempts)
            lifecycle["playable_rallies"] = len(stats.rallies)
            lifecycle["serves_not_played"] = len(attempts) - len(stats.rallies)
            lifecycle["excluded_as_non_game"] = (
                len(segmented["service_attempts"]) - len(attempts)
            )
        stats.rally_lifecycle = lifecycle
        by_frame: Dict[int, List[str]] = {}
        for event in events:
            by_frame.setdefault(int(event["frame"]), []).append(event["type"])
        for ann in stats.frame_annotations:
            frame_index = int(ann["f"])
            ann["ev_fsm"] = list(ann.get("ev") or [])
            ann["ev"] = by_frame.get(frame_index, [])
        phases = phase_segments(spotted.phase_labels, spotted.frame_numbers)
        stats.contact_spotter = {
            "enabled": True,
            "checkpoint": str(checkpoint),
            "diagnostics": spotted.diagnostics,
            "phase_segments": phases,
            "rally_segmentation": {
                "rules": segmented["rules"],
                "inferred_serves": len(segmented["inferred_serves"]),
                "groups_without_serve": segmented["unanchored_groups"],
            },
            "replaced": {
                "all_events": "dense_contact_spotter",
                "fsm_events_kept_in": "all_events_fsm",
                "per_frame_fsm_events_kept_in": "ev_fsm",
                "rallies": "spotter_rallies_v1",
                "fsm_rallies_kept_in": [
                    "rallies_fsm", "service_attempts_fsm", "rally_lifecycle_fsm",
                ],
            },
            "not_replaced": ["match_analytics", "attack_heatmap"],
        }
        log.info(
            f"Contact spotter: {len(events)} events "
            f"({spotted.diagnostics.get('per_type')}, "
            f"{len(segmented['inferred_serves'])} inferred serves), "
            f"{len(stats.rallies)} rallies from {len(attempts)} serves, "
            f"{len(phases)} phase segments, FSM had "
            f"{len(stats.all_events_fsm)} events / {len(stats.rallies_fsm)} rallies"
        )

    @staticmethod
    def _poses_from_annotation(ann: Dict) -> List[PlayerPose]:
        poses = []
        for saved in ann.get("poses") or []:
            bbox = tuple(float(value) for value in saved.get("bb", ()))
            keypoints = np.asarray(saved.get("kp") or [], dtype=np.float32)
            if len(bbox) != 4 or keypoints.shape != (17, 3):
                continue
            poses.append(PlayerPose(
                team="team_a" if saved.get("team") in ("a", "team_a") else "team_b",
                center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                bbox=bbox,
                keypoints=keypoints,
                track_id=saved.get("id"),
                back_facing_score=saved.get("back"),
                source=saved.get("source", "yolo_pose"),
                mask_instance_id=saved.get("mask_instance_id"),
                team_prior=saved.get("team_prior"),
                player_id=saved.get("player_id"),
                player_identity_processed="player_id" in saved,
            ))
        return poses

    def _apply_trajectory_dp(self, stats: "GameStats") -> bool:
        """Replace the gate's per-frame picks with a globally chosen path.

        Rewrites ``ball`` on every annotation, so everything downstream — the
        FSM re-run, the heatmap, the viewer — reads the same trajectory.
        ``ball_new`` is rebuilt from physically disconnected solver segments.
        A restart is a change of hypothesis, not a ball flight, so the FSM,
        heatmap and Viewer must not measure or draw across it.

        Returns whether the trajectory was actually replaced. A run with no
        candidates at all (an empty clip, or every frame below the confidence
        floor) leaves the gate's output in place, and the caller must then keep
        the legacy clean-up passes that output still needs.
        """
        if not self._ball_candidates:
            return False

        # A fixed object in the gym — a spare ball on the floor, a bag, a lamp —
        # is detected confidently on every frame, and a confident candidate has
        # *negative* cost in the solver. A long stationary run therefore beats a
        # short correct flight on total cost, which is how the chosen path ends
        # up alternating between the real ball and a piece of furniture. Take
        # the bait off the table before the solver ever sees it.
        if self.suppress_static_ball_sources:
            sources = find_static_sources_in_candidates(
                self._ball_candidates,
                fps=float(getattr(self, "_current_fps", 30.0) or 30.0),
                px_scale=self._ball_px_scale,
            )
            removed = suppress_static_candidates(
                self._ball_candidates, sources, px_scale=self._ball_px_scale,
            )
            if sources:
                log.info(
                    f"Static ball sources: {len(sources)} fixed object(s), "
                    f"{removed} candidates dropped "
                    f"(longest {max(s.seconds for s in sources):.1f} s)"
                )
            self._static_source_report = {
                "candidate_sources": [s.as_dict() for s in sources],
                "candidates_removed": removed,
            }

        solver = BallTrajectoryDP(px_scale=self._ball_px_scale)
        solved = solver.solve(self._ball_candidates)
        chosen = (
            interpolate_gaps(solved, max_gap=4, max_step_px=solver.max_step_px)
            if self.ball_reconstruction_mode == "off"
            else solved
        )

        before = sum(1 for a in stats.frame_annotations if a.get("ball"))
        for ann in stats.frame_annotations:
            cand = chosen.get(ann["f"])
            ann["ball"] = [round(cand.x, 1), round(cand.y, 1)] if cand else None
            ann["ball_confidence"] = round(float(cand.confidence), 4) if cand else 0.0
            ann["ball_new"] = (
                True if cand is not None and ann["f"] in solver.track_start_frames
                else None
            )
            ann["ball_interp"] = (
                True if cand is not None and cand.confidence == 0.0 else None
            )

        if self.ball_reconstruction_mode != "off":
            active_mode = (
                "conservative"
                if self.ball_reconstruction_mode == "compare"
                else self.ball_reconstruction_mode
            )
            calibration = self.calibration
            reconstructor = BallGapReconstructor(
                int(calibration.frame_width if calibration else 1280),
                int(calibration.frame_height if calibration else 720),
                net_tape_top=calibration.net_tape_top if calibration else None,
                net_tape_bottom=calibration.net_tape_bottom if calibration else None,
                net_floor_left=calibration.net_floor_left if calibration else None,
                net_floor_right=calibration.net_floor_right if calibration else None,
            )
            summaries = reconstructor.apply(
                stats.frame_annotations,
                active_mode=active_mode,
                contact_hints=self.reconstruction_contact_hints,
                keep_variants=True,
            )
            stats.ball_reconstruction = {
                "requested_mode": self.ball_reconstruction_mode,
                "active_mode": active_mode,
                "affects_fsm": self.reconstruction_affects_fsm,
                "variants": {
                    name: vars(summary) for name, summary in summaries.items()
                },
            }
        after = sum(1 for a in stats.frame_annotations if a.get("ball"))

        stats.ball_detected_frames = after
        log.info(
            f"Trajectory solver: {after} frames with a ball (gate had {before}), "
            f"from {solver.stats['frames_with_candidates']} frames with candidates "
            f"in {solver.stats['segments']} segments / "
            f"{len(solver.track_start_frames)} physical tracks"
        )
        self._ball_candidates.clear()
        return True

    def _smooth_ball_trajectory(self, stats: "GameStats") -> None:
        """Fit per-frame detector wobble out of the chosen trajectory.

        Records the jitter (mean absolute second difference of the path) either
        side of the pass, because "looks smoother" is not a measurement and the
        window is a tuning knob that has to be judged on numbers as well as on
        the video.
        """
        window = getattr(self, "ball_smooth_window", 0)
        if not window:
            stats.ball_smoothing = {"window": 0}
            return

        before = jitter_metric(stats.frame_annotations)
        summary = smooth_ball_annotations(
            stats.frame_annotations,
            window=window,
            px_scale=getattr(self, "_ball_px_scale", 1.0),
        )
        summary["jitter_before"] = before
        summary["jitter_after"] = jitter_metric(stats.frame_annotations)
        stats.ball_smoothing = summary
        log.info(
            f"Ball smoothing (window {summary['window']}): "
            f"{summary['moved']}/{summary['points']} points moved, "
            f"mean {summary['mean_shift_px']} px, "
            f"{summary['contacts_preserved']} contacts kept sharp, "
            f"jitter {before['mean']} -> {summary['jitter_after']['mean']} px/frame"
        )

    def _recompute_events_on_clean_trajectory(
        self, annotations: List[Dict], fps: float,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Re-run the FSM over the cleaned trajectory.

        Per-frame ``state``/``phase``/``ev`` fields are rewritten in place so the
        viewer overlay agrees with the saved rally list. Returns all events,
        confirmed playable rallies, and every terminal service attempt.
        """
        fsm = self._make_fsm(fps)
        events: List[Dict] = []
        rallies: List[Dict] = []
        service_attempts: List[Dict] = []
        cur_events: List[Dict] = []
        prev_state: Optional[str] = None
        frames_since_ball = 0
        last_frame_idx = 0
        reconstruction_mode = getattr(self, "ball_reconstruction_mode", "off")
        reconstruction_affects_fsm = bool(
            getattr(self, "reconstruction_affects_fsm", False)
        )

        def event_dict(ev) -> Dict:
            return {
                "frame": ev.frame_idx,
                "time": round(ev.timestamp, 2),
                "type": ev.event_type,
                "game_state": str(ev.game_state.value),
                "rally_phase": (
                    str(ev.rally_phase.value) if ev.rally_phase else None
                ),
                "team": ev.team_controlling,
                "contact_seq": ev.contacts_count,
                "lifecycle": dict(ev.lifecycle) if ev.lifecycle else None,
            }

        def finish_attempt(end_frame: int, lifecycle: Dict) -> None:
            nonlocal cur_events
            if lifecycle.get("status") not in {
                "playable_rally",
                "ace_no_reception",
                "reception_failed",
                "serve_fault",
            }:
                cur_events = []
                return

            cur_events.sort(key=lambda e: e["frame"])
            start_frame = lifecycle.get("start_frame")
            if start_frame is None:
                start_frame = lifecycle.get("serve_frame")
            if start_frame is None and cur_events:
                start_frame = cur_events[0]["frame"]
            if start_frame is None:
                cur_events = []
                return

            duration = max(0, int(end_frame) - int(start_frame))
            attempt = {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "duration_frames": duration,
                "duration_sec": round(duration / fps, 2),
                "events": list(cur_events),
                "contacts_team_a": sum(
                    1 for e in cur_events if e["team"] == "team_a"
                ),
                "contacts_team_b": sum(
                    1 for e in cur_events if e["team"] == "team_b"
                ),
                "lifecycle": dict(lifecycle),
            }
            service_attempts.append(attempt)
            if lifecycle.get("is_playable"):
                rallies.append(attempt)
            cur_events = []

        for ann in annotations:
            frame_idx = int(ann["f"])
            last_frame_idx = frame_idx
            ball = ann.get("ball")
            if reconstruction_mode != "off":
                if not reconstruction_affects_fsm:
                    ball = ann.get("ball_raw")
                elif ann.get("ball_source") != "detected" \
                        and float(ann.get("ball_confidence") or 0.0) < 0.8:
                    ball = None
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
                ball_interp=bool(
                    ann.get("ball_new")
                    or (
                        ann.get("ball_interp")
                        and not (
                            reconstruction_affects_fsm
                            and ann.get("ball_contact_reconstructed")
                            and float(ann.get("ball_confidence") or 0.0) >= 0.8
                        )
                    )
                ),
            )
            status = fsm.get_current_status()
            state = status.get("game_state")

            ann["state"] = state
            ann["phase"] = status.get("rally_phase")
            ann["ev"] = [e.event_type for e in fsm_events]
            ann["lifecycle"] = status.get("lifecycle")
            ann["current_team"] = status.get("current_team")
            ann["contacts"] = status.get("team_contacts", 0)

            for ev in fsm_events:
                d = event_dict(ev)
                events.append(d)
                cur_events.append(d)

            if state == "rally_end" and prev_state != "rally_end":
                finish_attempt(frame_idx, status.get("lifecycle") or {})
            prev_state = state

        # Preserve an attempt cut off by the end of the source video.
        tail_status = fsm.get_current_status()
        tail_lifecycle = tail_status.get("lifecycle") or {}
        if (
            tail_status.get("game_state") in ("serving", "rally_active")
            and tail_lifecycle.get("serve_frame") is not None
        ):
            fsm._end_rally(last_frame_idx, "video_end")
            for ev in fsm.update(
                frame_idx=last_frame_idx + 1,
                ball_pos=None,
                player_positions={"team_a": [], "team_b": []},
                timestamp=(last_frame_idx + 1) / fps,
            ):
                d = event_dict(ev)
                events.append(d)
                cur_events.append(d)
            finish_attempt(
                last_frame_idx,
                fsm.get_current_status().get("lifecycle") or {},
            )

        events.sort(key=lambda e: e["frame"])
        log.info(
            "Clean-trajectory pass: "
            f"{len(service_attempts)} service attempts, "
            f"{len(rallies)} playable rallies, {len(events)} events"
        )
        return events, rallies, service_attempts

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

    def _apply_player_identity(self, stats: "GameStats", images, fps: float,
                               video_path: Optional[Path] = None) -> None:
        """Separate offline pass: stable IDs never replace tracker/FSM inputs."""
        if not getattr(self, "enable_player_identity", False):
            stats.player_identity = {"enabled": False, "status": "disabled"}
            return
        from src.models.player_reid_encoder import DEFAULT_ASSETS, OSNetEncoder
        from src.tracking.player_identity_workflow import METHOD, run_identity_pass
        started = time.time()
        side = self.player_identity_side
        assets = Path(self.player_reid_assets) if self.player_reid_assets else DEFAULT_ASSETS
        if not assets.is_absolute():
            assets = _ROOT / assets
        override = str(assets / "osnet_msmt17.pth") if self.player_reid_assets else None
        resolved = self._resolve("player_reid", override)
        cache = getattr(self, '_model_cache', None)
        if cache is None and video_path is not None and hasattr(self, 'model_cache_dir'):
            cache = self._model_cache = ModelOutputCache(video_path, self.model_cache_dir, self.require_model_cache)
        feature_cache = None
        try:
            if cache is None:
                encoder = OSNetEncoder(assets, resolved.path, self.player_reid_device)
            else:
                encoder = FrozenOSNet(assets, resolved.path, self.player_reid_device)
                size = getattr(self, '_cache_frame_size', None)
                if size is None:
                    size = (self.calibration.frame_width,self.calibration.frame_height)
                width,height = size
                feature_cache = osnet_feature_store(cache, assets, resolved.path, width,height)
        except FileNotFoundError as exc:
            if self.player_reid_assets or self.player_identity_seed:
                raise
            log.warning(f"Player identity unavailable: {exc}. Run scripts/models/install_player_reid.py")
            stats.player_identity = {"enabled": True, "status": "model_unavailable", "method": METHOD,
                                     "side": side, "error": str(exc)}
            for ann in stats.frame_annotations:
                for pose in ann.get("poses") or []:
                    if pose.get("team") in (side, f"team_{side}"):
                        pose["player_id"] = None
                        pose["identity"] = {"method": METHOD, "reason": "model_unavailable",
                                            "label_source": "model", "verified": False}
        else:
            def progress(done, total):
                if done % max(1, int(fps * 2)) == 0 or done == total:
                    elapsed = max(time.time() - started, .001)
                    self._emit_progress(phase="player_identity", progress=97.0 + done/max(total, 1),
                                        frame_current=done, frame_total=total,
                                        processing_fps=round(done/elapsed, 2),
                                        eta_sec=round(elapsed/max(done, 1)*(total-done), 1),
                                        message=f"Assigning stable player IDs {done:,}/{total:,}")
            seed = Path(self.player_identity_seed) if self.player_identity_seed else None
            if seed is not None and not seed.is_absolute():
                seed = _ROOT / seed
            stats.player_identity = run_identity_pass(
                stats.frame_annotations, images, fps, encoder, side=side,
                roster_size=self.max_players_per_team or 6, seed_path=seed,
                video_path=video_path, progress=progress,
                feature_cache=feature_cache,
            )
            stats.player_identity["model_resolution"] = resolved.as_dict()
            self.model_provenance["player_reid"].update(encoder.provenance)
            log.info(f"Player identity: {stats.player_identity['status']}, "
                     f"{stats.player_identity.get('assigned', 0)} assigned observations")
        stats.models = dict(self.model_provenance)
        elapsed = time.time() - started
        stats.player_identity["processing_time_sec"] = round(elapsed, 3)
        stats.processing_time_sec += elapsed

    def _draw_saved_annotation(self, frame: np.ndarray, ann: Dict) -> np.ndarray:
        teams = {"team_a": [], "team_b": []}
        for saved in ann.get("poses") or []:
            box = saved.get("bb") or []
            if len(box) == 4:
                team = "team_a" if saved.get("team") in ("a", "team_a") else "team_b"
                teams[team].append(((box[0]+box[2])/2., (box[1]+box[3])/2.))
        ball = ann.get("ball")
        drawn = self._draw_frame(
            frame, (float(ball[0]), float(ball[1])) if ball else None, teams,
            {"game_state": ann.get("state"), "rally_phase": ann.get("phase"),
             "lifecycle": ann.get("lifecycle"), "current_team": ann.get("current_team"),
             "team_contacts": ann.get("contacts", 0)},
            event_display=ann.get("event_model"), player_poses=self._poses_from_annotation(ann),
            ball_source=str(ann.get("ball_source") or "detected"),
            ball_confidence=float(ann.get("ball_confidence") or 0.),
        )
        if not ball and ann.get("ball_out_of_frame"):
            cv2.putText(drawn, "BALL OUT OF FRAME", (20, frame.shape[0]-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
        return drawn

    def _render_final_video(
        self,
        video_path: Path,
        output_path: Path,
        annotations: List[Dict],
        fps: float,
        frame_width: int,
        frame_height: int,
        max_frames: int,
    ) -> bool:
        """Render final trajectories and roster IDs from the exported annotations."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.warning(f"Final render cannot open source video: {video_path}")
            return False

        writer = None
        for codec in ("mp4v", "avc1", "H264"):
            candidate = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*codec), fps,
                (frame_width, frame_height),
            )
            if candidate.isOpened():
                writer = candidate
                log.info(f"Final VideoWriter codec: {codec}")
                break
            candidate.release()
        if writer is None:
            cap.release()
            log.warning("Final VideoWriter failed — annotated video not saved")
            return False

        by_frame = {int(ann["f"]): ann for ann in annotations}
        rendered = 0
        try:
            while rendered < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                ann = by_frame.get(rendered)
                if ann is None:
                    writer.write(frame)
                    rendered += 1
                    continue

                drawn = self._draw_saved_annotation(frame, ann)
                writer.write(drawn)
                rendered += 1
                if rendered % max(1, int(fps * 2)) == 0:
                    self._emit_progress(
                        phase="rendering",
                        progress=98.0 + rendered / max(max_frames, 1),
                        frame_current=rendered,
                        frame_total=max_frames,
                        video_percent=round(100.0 * rendered / max(max_frames, 1), 2),
                        message=f"Rendering final annotated video {rendered:,}/{max_frames:,}",
                    )
        finally:
            cap.release()
            writer.release()
        log.info(f"Final annotated video: {output_path} ({rendered:,} frames)")
        return rendered > 0

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
        vp = local_path_from_uri(video_path)
        od = Path(output_dir)
        if any((od/f"{vp.stem}{suffix}").exists()
               for suffix in ("_stats.json", "_annotations.json", "_analyzed.mp4")):
            raise FileExistsError(f"Analysis result already exists for {vp.stem}; choose a fresh output directory")
        od.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vp}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recovery = getattr(self, "_person_pose_recovery", None)
        if recovery is not None:
            try:
                recovery.validate_video(vp, fps, w, h)
            except Exception:
                cap.release()
                raise
        max_frames = int(max_duration_sec * fps) if max_duration_sec else total
        max_frames = min(max_frames, total) if total > 0 else max_frames
        self._cache_full_source_limit = total
        try:
            cached_pass = self._prepare_model_cache(vp, fps, w, h, max_frames)
        except Exception:
            cap.release()
            if getattr(self, '_model_cache', None) is not None:
                self._model_cache.db.close()
            raise

        log.section(f"Video: {vp.name}")
        log.info(f"Resolution={w}x{h}  fps={fps:.1f}  frames={total}  skip={self.skip_frames}")
        self._emit_progress(
            phase="calibration",
            progress=0.0,
            frame_current=0,
            frame_total=max_frames,
            video_percent=0.0,
            message=f"Preparing {w}x{h} video ({max_frames:,} frames)...",
        )

        # Auto-detect net tape with SAM if not already in calibration
        try:
            self._maybe_run_sam_net_detection(
                str(vp), float(w), float(h), refresh_floor=self.refresh_net_floor,
            )
        except Exception:
            cap.release()
            if getattr(self, '_model_cache', None) is not None:
                self._model_cache.db.close()
            raise

        # Scale calibration to actual frame resolution (e.g., 4K calib on 720p video)
        saved_calib = self.calibration
        self.calibration = self._scale_calibration(w, h)
        self._configure_ball_selection_for(w)

        out_writer = None
        out_path = od / f"{vp.stem}_analyzed.mp4"
        defer_final_render = (
            self.output_video and (self.ball_reconstruction_mode != "off"
                                   or getattr(self, "enable_player_identity", False))
        )
        if self.output_video and not defer_final_render:
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
                self._cached_frame_iterator() if cached_pass and not recovery and out_writer is None else frame_iter(), fps, max_frames, stats,
                out_writer=out_writer,
            )
            if getattr(self, "enable_player_identity", False):
                def identity_images():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    yield from frame_iter()
                self._apply_player_identity(stats, identity_images(), fps, vp)
            else:
                stats.player_identity = {"enabled": False, "status": "disabled"}
            if defer_final_render:
                self._render_final_video(
                    vp, out_path, stats.frame_annotations, fps, w, h, max_frames,
                )
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            self.calibration = saved_calib  # restore original
            if getattr(self, '_model_cache', None) is not None:
                stats.model_cache = self._model_cache.summary()
                self._model_cache.db.close()

        self._emit_progress(
            phase="saving",
            progress=99.0,
            frame_current=max_frames,
            frame_total=max_frames,
            video_percent=100.0,
            eta_sec=0.0,
            message="Saving statistics and frame annotations...",
        )
        self._save_and_print(stats, od, vp.stem,
                             original_video_path=str(vp.resolve()),
                             calibration_override=scaled_calib)
        self._emit_progress(
            phase="done",
            progress=100.0,
            frame_current=max_frames,
            frame_total=max_frames,
            video_percent=100.0,
            eta_sec=0.0,
            message="Analysis complete",
        )
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
        if (any((od/f"{fd.name}{suffix}").exists() for suffix in ("_stats.json", "_annotations.json"))
                or (od/"annotated_frames").exists()):
            raise FileExistsError(f"Frame analysis already exists for {fd.name}; choose a fresh output directory")
        od.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png"}
        frame_files = sorted(p for p in fd.iterdir() if p.suffix.lower() in exts)
        all_frame_files = frame_files
        if max_frames:
            frame_files = frame_files[:max_frames]

        total = len(frame_files)
        self._cache_full_source_limit = None
        cached_pass = None
        if frame_files and hasattr(self, 'model_cache_dir'):
            first = cv2.imread(str(frame_files[0]))
            if first is None:
                raise ValueError('First source frame cannot be read')
            h,w = first.shape[:2]
            identity = {'kind':'frames','files': [[p.name,file_digest(p)] for p in all_frame_files]}
            cached_pass = self._prepare_model_cache(fd, fps, w, h, total, source_identity=identity)
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

        stats = GameStats(video_name=fd.name, fps=fps)
        stats.models = dict(self.model_provenance)
        self._process_frames_iter(
            self._cached_frame_iterator() if cached_pass and not self.output_video else frame_iter(), fps, total, stats,
            out_frames_dir=None if getattr(self, "enable_player_identity", False) else out_frames_dir,
        )

        self._apply_player_identity(stats, frame_iter(), fps)
        if out_frames_dir is not None and getattr(self, "enable_player_identity", False):
            by_frame = {ann["f"]: ann for ann in stats.frame_annotations}
            for index, image in frame_iter():
                if index in by_frame:
                    cv2.imwrite(str(out_frames_dir/f"frame_{index:05d}.jpg"),
                                self._draw_saved_annotation(image, by_frame[index]))

        self._save_and_print(stats, od, fd.name)
        if getattr(self, '_model_cache', None) is not None:
            self._model_cache.db.close()
        return stats

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_and_print(self, stats: GameStats, output_dir: Path, stem: str,
                        original_video_path: str = "",
                        calibration_override: Optional["CalibrationData"] = None):
        from src.tracking.player_contacts import build_player_contacts
        stats.player_contacts = build_player_contacts(
            stats.all_events, stats.frame_annotations, stats.fps, stats.player_identity,
        )
        if getattr(self, '_model_cache', None) is not None:
            stats.model_cache = self._model_cache.summary()
        out_path = output_dir / f"{stem}_stats.json"
        with open(out_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2, default=str)

        # Use the scaled calibration that was actually used during processing
        calib = calibration_override or self.calibration

        # Save compact per-frame annotations separately (for HTML viewer overlay)
        ann_path = output_dir / f"{stem}_annotations.json"
        team_prior_counts: Dict[str, int] = {}
        for frame_ann in stats.frame_annotations:
            for pose in frame_ann.get("poses") or []:
                prior = pose.get("team_prior")
                if prior:
                    team_prior_counts[prior] = team_prior_counts.get(prior, 0) + 1
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
            "net_height_m": calib.net_height_m if calib else self.net_height_m,
            "net_post_spacing_m": calib.net_post_spacing_m if calib else None,
            "court_corners": [
                list(point) for point in calib.court_corners
            ] if calib and calib.court_corners else None,
            "team_divider_prior_counts": team_prior_counts,
            "player_detection": {
                "pose_imgsz": getattr(self, "pose_imgsz", None),
                "sam_guided_recovery": self._person_pose_recovery.summary()
                    if getattr(self, "_person_pose_recovery", None) is not None else {"enabled": False},
            },
            "pose_feature_names": POSE_FEATURE_NAMES,
            "attack_heatmap": stats.attack_heatmap,
            "ball_reconstruction": stats.ball_reconstruction,
            "contact_spotter": stats.contact_spotter,
            "player_identity": stats.player_identity,
            "player_contacts": stats.player_contacts,
            "model_cache": stats.model_cache,
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
