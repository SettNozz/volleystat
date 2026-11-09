"""
Volleyball game analyzer pipeline.

Combines YOLO ball detection + YOLO-pose skeleton + FSM event tracking.
Works entirely in pixel coordinates. No world coordinate transforms needed.
"""

import cv2
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# Project root
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.detection.player_pose import CourtPlayerGate, CourtProjector
from src.tracking.attack_heatmap import AttackHeatmap
from src.tracking.ball_track_gate import BallCandidate
from src.tracking.ball_size import drop_unobserved_ball_sizes, observed_ball_size
from src.tracking.ball_static_sources import clean_ball_annotations
from src.tracking.ball_trajectory_smoother import (
    DEFAULT_WINDOW as BALL_SMOOTH_DEFAULT_WINDOW, resolve_window as resolve_ball_smooth_window,
)
from src.tracking.ball_gap_reconstructor import ContactHint
from src.tracking.court_pose_consistency import reject_unsupported_sideline_poses
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
from src.pipeline.model_prefetch import prefetch_model_outputs, threaded_iter
from src.models.inference_cache import DEFAULT_CACHE, CacheMiss, file_digest

# Re-exported: callers import these names from this module.
from src.pipeline.analyzer.defaults import (
    _BALL_MODEL, _BALL_IMGSZ, _BALL_MODEL_2, _BALL_TILE_GRID_2, _POSE_MODEL, _POSE_FALLBACK,
)
from src.pipeline.analyzer.types import _EventOverlayState, CalibrationData, RallySummary, GameStats
from src.pipeline.analyzer.cache import ModelCacheMixin
from src.pipeline.analyzer.calibration import CalibrationMixin
from src.pipeline.analyzer.models import ModelLoadingMixin
from src.pipeline.analyzer.detection import DetectionMixin
from src.pipeline.analyzer.ball_trajectory import BallTrajectoryMixin
from src.pipeline.analyzer.events import EventsMixin
from src.pipeline.analyzer.identity import IdentityMixin
from src.pipeline.analyzer.rendering import RenderingMixin

log = get_logger("Analyzer")


# ---------------------------------------------------------------------------
# GameAnalyzer
# ---------------------------------------------------------------------------

class GameAnalyzer(
    ModelCacheMixin,
    CalibrationMixin,
    ModelLoadingMixin,
    DetectionMixin,
    BallTrajectoryMixin,
    EventsMixin,
    IdentityMixin,
    RenderingMixin,
):
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
        ball_batch_size: Optional[int] = 1,
        ball_imgsz: Optional[int] = None,
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
        model_prefetch: bool = True,
        enable_contact_author: bool = True,
        contact_author_model: Optional[str] = None,
        contact_author_roster: Optional[str] = None,
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
            ball_batch_size: Ball YOLO batch size. None chooses automatically.
            ball_imgsz: Ball YOLO input size. None takes the registry entry's
                       imgsz; set it with an explicit ball model trained at
                       another size.
                       Pose tracking remains sequential so ByteTrack IDs stay valid.
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
            model_prefetch: Decode frames and run stateless ball/pose model
                       inference ahead of the ordered analysis loop on
                       background threads (see pipeline/model_prefetch.py).
                       Identical model calls and cache keys; only overlap.
        """
        self.model_prefetch = bool(model_prefetch)
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
        self.enable_contact_author = enable_contact_author
        self.contact_author_model = contact_author_model
        self.contact_author_roster = contact_author_roster
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
        self.ball_imgsz = ball_imgsz
        self._auto_ball_batch = ball_batch_size is None
        self.ball_batch_size = 1 if self._auto_ball_batch else max(1, int(ball_batch_size))
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
        self.pose_runner = None
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

    # Ball selection is tuned in pixels on 1280-wide footage. Everything that
    # bounds how far the ball may travel between frames has to grow with the
    # frame, or 4K input — where the same shot covers three times the pixels —
    # reads as physically impossible motion and the track is never picked up.
    _BALL_TUNING_WIDTH = 1280.0

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

    # Chunks (of ball_batch_size frames) decoded and inferred beyond the one the
    # loop is consuming. Two keep the GPU fed without holding many 4K frames.
    PREFETCH_CHUNKS_AHEAD = 2
    IDENTITY_DECODE_DEPTH = 16
    # Raw pose is the heaviest GPU stage. Each thread has its own GpuPoseModel
    # (CUDA graph + stream), so one frame's upload/NMS overlaps another's
    # network. Measured 2026-11-07 (RTX 5070 Ti, i9-10900F, 4K VP9), full cold
    # pipeline: before CUDA graphs two workers beat three (17.6 vs 16.9 fps);
    # with graphs the loop (24 fps) is bound by GPU time (pose ~20 ms + OSNet
    # ~10 ms per frame) and decoding (35.6 fps without identity), so two.
    POSE_PREFETCH_WORKERS = 2

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    # Skeleton overlay styling: team_b (near side — the team we track) is
    # highlighted; team_a (far side) is dimmed for context.
    _SKELETON_STYLE = {
        "team_b": {"color": (0, 220, 255), "bone": 3, "kp": 5, "draw_id": True},   # bright yellow
        "team_a": {"color": (180, 120, 80), "bone": 1, "kp": 2, "draw_id": False}, # muted blue-gray
    }

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
        batch_limit = self.ball_batch_size
        if batch_enabled:
            log.info(f"Ball YOLO batch size: {self.ball_batch_size}")

        prefetch_jobs = self._model_prefetch_jobs(total)
        if prefetch_jobs:
            log.info(f"Model prefetch: {', '.join(prefetch_jobs)} computed ahead on background threads")
            source = prefetch_model_outputs(frame_iter, prefetch_jobs, self.ball_batch_size,
                                            chunks_ahead=self.PREFETCH_CHUNKS_AHEAD)
            # Closed explicitly by process_video before the capture is released:
            # after an exception the traceback would keep this generator alive.
            self._prefetch_source = source
        else:
            source = ((idx, image, {}) for idx, image in frame_iter)

        def frame_chunks():
            chunk = []
            for item in source:
                chunk.append(item)
                if len(chunk) >= batch_limit:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

        def detect_cached_batch(images):
            nonlocal batch_limit, batch_enabled
            if not batch_enabled or len(images) == 1:
                return [self.ball_detector.detect_ball(image) for image in images]
            if len(images) > batch_limit:
                return [value for start in range(0, len(images), batch_limit)
                        for value in detect_cached_batch(images[start:start + batch_limit])]
            try:
                return self.ball_detector.detect_ball_batch(images)
            except Exception as error:
                if getattr(self, '_auto_ball_batch', False) and 'out of memory' in str(error).lower():
                    batch_limit = max(1, len(images) // 2)
                    log.warning(f'GPU ran out of memory; retrying ball detection with batch {batch_limit}')
                else:
                    log.warning(f'Ball batch unavailable ({error}); using single-frame model outputs')
                    batch_enabled = False
            if batch_enabled:
                import torch
                torch.cuda.empty_cache()
            return detect_cached_batch(images)

        def frames_with_ball_candidates():
            nonlocal batch_enabled
            for frame_chunk in frame_chunks():
                cache = getattr(self, '_model_cache', None)
                if cache is not None:
                    raw = {}
                    missing = []
                    ahead_values = []
                    for idx,image,ahead in frame_chunk:
                        if self.skip_frames > 0 and idx % (self.skip_frames+1):
                            continue
                        value = cache.get(self._ball_stages['primary'], idx)
                        if value is None and ahead.get('ball_primary') is not None:
                            ahead_values.append((idx, ahead['ball_primary']))
                        elif value is None:
                            missing.append((idx,image))
                        else:
                            raw[idx] = value
                    if ahead_values:
                        cache.computed['ball_primary'] += len(ahead_values)
                        for idx,value in ahead_values:
                            raw[idx] = [[float(v) for v in row] for row in value]
                            cache.put(self._ball_stages['primary'], idx, raw[idx])
                    if missing and cache.require:
                        raise CacheMiss(f'Missing ball model outputs at frame {missing[0][0]}')
                    if missing:
                        values = detect_cached_batch([image for _,image in missing])
                        if len(values) != len(missing):
                            raise ValueError('Ball batch output count differs from input')
                        cache.computed['ball_primary'] += len(missing)
                        for (idx,_),value in zip(missing,values):
                            raw[idx] = [[float(v) for v in row] for row in value]
                            cache.put(self._ball_stages['primary'], idx, raw[idx])
                    for idx,image,ahead in frame_chunk:
                        yield idx,image,(self._ball_candidates_for(image,idx,raw[idx],prefetched=ahead)
                                         if idx in raw else None),ahead
                    continue
                batched_candidates: Dict[int, List[BallCandidate]] = {}
                if batch_enabled:
                    processable = [
                        (idx, frame)
                        for idx, frame, _ in frame_chunk
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
                for frame_idx, frame, ahead in frame_chunk:
                    yield frame_idx, frame, batched_candidates.get(frame_idx), ahead

        for frame_idx, frame, candidates, prefetched in frames_with_ball_candidates():
            self._prefetched_outputs = (frame_idx, prefetched)
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
                # Longer side of the observed detector box, px. Never set on
                # interpolated or reconstructed points: those have no box.
                "ball_size": observed_ball_size(
                    getattr(self, "_last_ball_size", None) if ball_pos else None
                ),
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
            self._prefetch_embeddings(ann, frame)

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

        return self._run_offline_passes(stats, fps)

    def _run_offline_passes(self, stats: GameStats, fps: float) -> GameStats:
        """Whole-video passes after the frame loop.

        Order matters: sideline pose validation, trajectory solve/clean-up,
        smoothing, clean FSM replay, match analytics, final features/predictions,
        contact spotter, then the attack heatmap.
        """
        self._current_fps = float(fps or 30.0)
        projector = self._get_projector()
        if projector is not None:
            rejected_poses = reject_unsupported_sideline_poses(
                stats.frame_annotations, projector, self._current_fps,
                sample_step=getattr(self, 'skip_frames', 0) + 1,
            )
            if rejected_poses:
                log.info(f"Sideline pose check: rejected {rejected_poses} unsupported observations")

        # Post-process: re-pick the whole trajectory from the candidate cloud.
        # The gate above had to choose with only the past in hand; now the
        # frames after each decision are available too, and they are what
        # identifies the ball on frames where a distractor scored higher.
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
        # Every pass that removes, bridges or reconstructs a ball point has
        # run. A box size stays only on the observations it was measured on.
        drop_unobserved_ball_sizes(stats.frame_annotations)

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
            projector=projector,
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
        self._cache_video_path = vp
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

        def frame_iter(pinned=False):
            from src.detection.pose_gpu import pinned_frame
            idx = 0
            while idx < max_frames:
                ret, frame = cap.read(pinned_frame((h, w, 3))) if pinned else cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1

        stats = GameStats(video_name=vp.name, fps=fps)
        scaled_calib = self.calibration  # already scaled — needed for _save_and_print
        try:
            self._start_embedding_prefetch()
            self._process_frames_iter(
                self._cached_frame_iterator() if cached_pass and not recovery and out_writer is None
                else frame_iter(pinned=self._decode_into_pinned_memory()), fps, max_frames, stats,
                out_writer=out_writer,
            )
            self._close_prefetch()
            self._finish_embedding_prefetch()
            if getattr(self, "enable_player_identity", False):
                def identity_images():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    yield from frame_iter()
                # Decode ahead while OSNet encodes; closed before cap is reused.
                images = (threaded_iter(identity_images(), self.IDENTITY_DECODE_DEPTH, name='identity-decode')
                          if getattr(self, 'model_prefetch', False) else identity_images())
                try:
                    self._apply_player_identity(stats, images, fps, vp)
                finally:
                    images.close()
            else:
                stats.player_identity = {"enabled": False, "status": "disabled"}
            self._apply_contact_authors(stats, str(vp.resolve()), w, h)
            if defer_final_render:
                self._render_final_video(
                    vp, out_path, stats.frame_annotations, fps, w, h, max_frames,
                )
        finally:
            self._close_prefetch()
            self._finish_embedding_prefetch(store_results=False)
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
        w, h = 1280, 720
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
        self._apply_contact_authors(stats, "", w if frame_files else 1280, h if frame_files else 720,
            images=lambda wanted: ((f, image) for f, image in frame_iter() if f in wanted))
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

    def _save_and_print(self, stats: GameStats, output_dir: Path, stem: str,
                        original_video_path: str = "",
                        calibration_override: Optional["CalibrationData"] = None):
        from src.tracking.player_contacts import build_player_contacts
        if not stats.player_contacts:
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
            "net_tape_endpoint_y": calib.net_tape_endpoint_y if calib else None,
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
