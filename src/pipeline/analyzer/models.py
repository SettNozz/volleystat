"""Model resolution, loading, warm-up and FSM construction.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

import time
import hashlib
import inspect
from typing import Optional

import numpy as np

from src.tracking.integrated_volleyball_fsm import CourtBounds, IntegratedVolleyballFSM
from src.tracking.ball_track_gate import BallTrackGate
from src.models.model_registry import ResolvedModel, resolve_model
from src.utils.logger import get_logger
from src.models.inference_cache import LazyModel, file_digest, model_spec
from src.pipeline.analyzer.defaults import _ROOT, _BALL_IMGSZ, _BALL_TILE_GRID_2

log = get_logger("Analyzer")


class ModelLoadingMixin:

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
        imgsz = int(getattr(self, "ball_imgsz", None) or primary.extra.get("imgsz") or _BALL_IMGSZ)
        self.model_provenance["ball_detector"]["imgsz"] = imgsz
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
            from src.detection import pose_gpu
            backend = hashlib.sha256((inspect.getsource(type(self)._infer_pose_model)
                                      + inspect.getsource(pose_gpu)).encode()).hexdigest()
            self._pose_spec = model_spec(resolved.path, backend=backend,
                                         imgsz=self.pose_imgsz, conf=.1, batch=1,
                                         precision='fp16_cuda_fp32_cpu',
                                         preprocess=pose_gpu.PREPROCESS)
            self._pose_spec['tracker'] = file_digest(tracker)
            def load():
                from ultralytics import YOLO
                return YOLO(str(resolved.path))
            self.pose_model = LazyModel(load)
            # Raw cache-backed detections: letterbox, network and NMS on the GPU.
            imgsz = self.pose_imgsz or 640
            self.pose_runner = LazyModel(lambda: pose_gpu.GpuPoseModel(resolved.path, imgsz, conf=.1))
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
