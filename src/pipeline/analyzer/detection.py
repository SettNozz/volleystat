"""Per-frame ball and player detection, including background prefetch.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

import math
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.detection.player_pose import (
    CourtPlayerGate, PlayerPose, extract_player_poses, poses_to_teams,
)
from src.tracking.ball_track_gate import BallCandidate
from src.utils.logger import get_logger
from src.pipeline.model_prefetch import PrefetchJob
from src.utils.cuda_streams import disable_ultralytics_device_sync, own_stream
from src.models.inference_cache import (
    CacheMiss, LazyModel, digest, pack_pose_result, unpack_pose_result,
)

log = get_logger("Analyzer")


class DetectionMixin:

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

    def _ball_candidates_for(self, frame: np.ndarray, frame_idx=None, primary=None,
                             prefetched=None) -> List[BallCandidate]:
        """Every plausible ball in this frame, from one or both detectors.

        ``prefetched`` holds raw detector outputs already computed for this
        frame by the background model prefetch (keyed ``ball_<name>``).
        """
        def raw(name, detector):
            cache = getattr(self, '_model_cache', None)
            def infer():
                ahead = (prefetched or {}).get('ball_' + name)
                value = ahead if ahead is not None else detector.detect_ball(frame)
                return [[float(v) for v in row] for row in value]
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
        runner = getattr(self, 'pose_runner', None)
        if runner is not None:
            # src/detection/pose_gpu.py: same weights/conf/imgsz, GPU letterbox.
            return runner([frame])[0]
        settings = self._pose_spec['settings']
        result = self.pose_model.predict(frame, conf=settings.get('conf',.1),
                                         batch=settings.get('batch',1), mode='track', **kwargs)
        return pack_pose_result(result[0] if result else None)

    def _close_prefetch(self) -> None:
        source = getattr(self, '_prefetch_source', None)
        self._prefetch_source = None
        if source is not None:
            source.close()

    def _decode_into_pinned_memory(self) -> bool:
        """Pinned frames only pay off when the GPU pose runner still has to run."""
        if getattr(self, 'pose_runner', None) is None or getattr(self, '_pose_cache_read', False):
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _pose_predict_kwargs(self) -> Dict[str, Any]:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = dict(verbose=False, device=device, half=(device == "cuda"))
        if self.pose_imgsz:
            kwargs["imgsz"] = self.pose_imgsz
        return kwargs

    def _model_prefetch_jobs(self, total: int) -> Dict[str, PrefetchJob]:
        """Stateless model outputs this pass still has to compute, as prefetch jobs.

        Only cache-backed passes qualify: the cache is what turns pose into a
        stateless raw stage (ByteTrack is replayed in order on this thread).
        Each job runs the exact call the serial path would make, so values and
        cache keys do not depend on whether prefetch ran.
        """
        cache = getattr(self, '_model_cache', None)
        if not getattr(self, 'model_prefetch', False) or cache is None or cache.require:
            return {}
        step = self.skip_frames + 1
        frames = range(0, max(0, int(total)), step)

        def missing(stage):
            stored = cache.items(stage)
            absent = {index for index in frames if str(index) not in stored}
            return lambda index, absent=absent: index in absent

        jobs: Dict[str, PrefetchJob] = {}
        primary = missing(self._ball_stages['primary'])
        if any(primary(index) for index in frames):
            batch_state = {'enabled': self.ball_batch_size > 1 and self.ball_detector_2 is None}

            # The ball thread gets its own CUDA stream and Ultralytics' stage
            # timer stops being a device-wide barrier (src/utils/cuda_streams.py).
            disable_ultralytics_device_sync()

            def ball_batch(images):
                with own_stream():
                    if batch_state['enabled']:
                        try:
                            return self.ball_detector.detect_ball_batch(images)
                        except Exception as error:
                            log.warning(f'Ball batch unavailable ({error}); using single-frame model outputs')
                            batch_state['enabled'] = False
                    return [self.ball_detector.detect_ball(image) for image in images]

            def ball_single(image):
                with own_stream():
                    return self.ball_detector.detect_ball(image)

            jobs['ball_primary'] = PrefetchJob(primary, ball_single, ball_batch)
        if self.ball_detector_2 is not None and 'tiled' in self._ball_stages:
            tiled = missing(self._ball_stages['tiled'])
            if any(tiled(index) for index in frames):
                jobs['ball_tiled'] = PrefetchJob(tiled, lambda image: self.ball_detector_2.detect_ball(image))
        if self.enable_pose and self.pose_model is not None and not self._pose_cache_read:
            pose = missing(self._raw_pose_stage)
            if any(pose(index) for index in frames):
                infer, workers = self._pose_worker_inference()
                jobs['pose_raw'] = PrefetchJob(pose, infer, workers=workers)
        return jobs

    def _pose_worker_inference(self) -> Tuple[Callable[[np.ndarray], Any], int]:
        """Thread-safe raw pose inference: one YOLO instance per worker thread.

        An Ultralytics model holds its predictor state, so concurrent calls need
        separate instances. Every thread still runs the unchanged
        ``_infer_pose_model`` (its source is part of the pose cache key) with
        the same weights and settings; only the instance differs.
        """
        import queue
        import threading
        from types import SimpleNamespace
        kwargs = self._pose_predict_kwargs()
        pool: "queue.Queue" = queue.Queue()
        pool.put(self)
        # Only a lazily built model can be instantiated again; anything else
        # (an injected model) stays on a single thread.
        runner = getattr(self, 'pose_runner', None)
        engine = runner if runner is not None else self.pose_model
        factory = engine.factory if isinstance(engine, LazyModel) else None
        workers = max(1, int(self.POSE_PREFETCH_WORKERS)) if factory else 1
        hosts = [self]
        for _ in range(workers - 1):
            extra = LazyModel(factory)
            hosts.append(SimpleNamespace(pose_model=extra if runner is None else self.pose_model,
                                         pose_runner=extra if runner is not None else None,
                                         _pose_spec=self._pose_spec))
        size = getattr(self, '_cache_frame_size', None)
        for host in hosts:
            # GPU runners capture their CUDA graph now, before any worker thread
            # (ball, decode, OSNet) is using the GPU.
            if host is not self:
                pool.put(host)
            if runner is not None and size is not None:
                host.pose_runner.prepare((size[1], size[0], 3))
        local = threading.local()

        def infer(image):
            host = getattr(local, 'host', None)
            if host is None:
                host = local.host = pool.get_nowait()
            return type(self)._infer_pose_model(host, image, kwargs)
        return infer, workers

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

        kwargs = self._pose_predict_kwargs()

        cache = getattr(self, '_model_cache', None)
        frame_index = getattr(self, '_current_pose_frame', None)
        saved = cache.get(self._pose_sequence, frame_index) if cache and self._pose_cache_read else None
        if saved is not None:
            results = [unpack_pose_result(saved)]
        elif cache and not self._pose_cache_read:
            ahead = getattr(self, '_prefetched_outputs', None)
            def infer_pose():
                # Raw detections computed ahead by the same _infer_pose_model call.
                if ahead and ahead[0] == frame_index and 'pose_raw' in ahead[1]:
                    return ahead[1]['pose_raw']
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
