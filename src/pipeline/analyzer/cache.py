"""Model-output cache wiring: capture, replay and verification.

Mixin of :class:`src.pipeline.game_analyzer.GameAnalyzer`; methods moved verbatim.
"""

import json
import sys
import hashlib
import inspect
import subprocess

import numpy as np

from src.utils.logger import get_logger
from src.models.inference_cache import (
    CachedForward, CacheMiss, ModelOutputCache, array_payload, digest, file_digest, model_spec, pack_pose_result, payload_array, unpack_pose_result, PoseTracker,
)
from src.pipeline.analyzer.defaults import _ROOT

log = get_logger("Analyzer")


class ModelCacheMixin:

    def _prepare_model_cache(self, source, fps, width, height, limit, source_identity=None):
        # __new__-based subsystem tests deliberately do not construct production models.
        if not hasattr(self, 'model_cache_dir'):
            return None
        self._model_cache = ModelOutputCache(source, self.model_cache_dir, self.require_model_cache,
                                             source_identity=source_identity)
        cache = self._model_cache
        if getattr(self, '_auto_ball_batch', False):
            self._select_ball_batch(cache, limit)
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
            stages = list(self._ball_stages.values())
            if self.enable_pose:
                stages.append(self._raw_pose_stage)

            def covers(count):
                return count > 0 and all(
                    cache.covers(stage, range(0, count, self.skip_frames + 1)) for stage in stages
                )

            complete = covers(count)
            if (not complete and length is None and source_identity is None
                    and getattr(self, '_cache_full_source_limit', None) == limit and stages
                    and all(cache.covers(stage, [0]) for stage in stages)):
                # A prior run may have cached every real frame but failed EOF
                # validation before it could publish the decoded source length.
                verified_count = self._verified_decoded_frame_count()
                if verified_count is not None and 0 < verified_count < limit and covers(verified_count):
                    count = verified_count
                    complete = True
                    cache.put('source_video', 'decoded_total', {'count': count})
            if complete:
                frames = range(0,count,self.skip_frames+1)
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

    def _select_ball_batch(self, cache, limit):
        import torch

        primary = self._ball_specs['primary']
        settings = {key: value for key, value in primary['settings'].items() if key != 'batch_size'}
        best_coverage = 0
        batch_size = 1
        stride = self.skip_frames + 1
        rows = cache.db.execute(
            "SELECT item FROM outputs WHERE stage='stage_manifests' AND item LIKE 'ball_primary:%'"
        ).fetchall()
        for (stage,) in rows:
            spec = cache.get('stage_manifests', stage)['spec']
            cached_batch = spec['settings'].get('batch_size')
            if not isinstance(cached_batch, int) or cached_batch < 1:
                continue
            if spec != {**primary, 'settings': {**settings, 'batch_size': cached_batch}}:
                continue
            coverage = cache.db.execute(
                'SELECT COUNT(*) FROM outputs WHERE stage=? AND CAST(item AS INTEGER) < ? '
                'AND CAST(item AS INTEGER) % ? = 0', (stage, limit, stride),
            ).fetchone()[0]
            if coverage > best_coverage:
                best_coverage, batch_size = coverage, cached_batch

        if best_coverage == 0 and torch.cuda.is_available() and self.ball_detector_2 is None:
            free_bytes, _ = torch.cuda.mem_get_info()
            # Leave room for pose inference and CUDA workspaces. If actual ball
            # inference needs more memory, the batch is reduced on demand.
            free_gib = free_bytes / (1024 ** 3)
            batch_size = max(1, min(16, int((free_gib - 2) / 0.5)))
        self.ball_batch_size = batch_size
        self._ball_specs['primary'] = {**primary, 'settings': {**settings, 'batch_size': batch_size}}
        reason = 'cached detections' if best_coverage else 'available device memory'
        log.info(f'Automatic ball batch: {batch_size} ({reason})')

    def _finish_detection_cache(self, stats):
        cache = getattr(self, '_model_cache', None)
        if cache is not None:
            if (getattr(self, '_cache_full_source_limit', None)==self._cache_scope_limit
                    and stats.total_frames < self._cache_scope_limit-1):
                # Container frame counts can include samples hidden by MP4 edit
                # lists. Verify a clean independent decode before sealing a
                # shorter pass; never treat an arbitrary read failure as EOF.
                length = cache.get('source_video', 'decoded_total')
                verified_count = length['count'] if length else self._verified_decoded_frame_count()
                if not stats.total_frames or verified_count != stats.total_frames:
                    raise RuntimeError('Video ended early; raw cache retained but detection pass is not complete')
                log.warning(f'Container reports {self._cache_scope_limit} frames; '
                            f'clean decode verified {verified_count} frames')
            if self._pose_sequence and not self._pose_cache_read:
                cache.seal_sequence(self._pose_sequence)
            cache.put('detection_passes', self._cache_pass,
                      {'total_frames': stats.total_frames, 'processed_frames': stats.processed_frames,
                       'frame_indices': self._cache_visited_frames})
            if getattr(self, '_cache_full_source_limit', None)==self._cache_scope_limit:
                cache.put('source_video','decoded_total',{'count':stats.total_frames})

    def _verified_decoded_frame_count(self):
        source = getattr(self, '_cache_video_path', None)
        if source is None:
            return None
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-threads', '0', '-select_streams', 'v:0',
                 '-count_frames', '-show_entries', 'stream=nb_read_frames',
                 '-of', 'json', str(source)],
                capture_output=True, text=True,
            )
            if result.returncode or result.stderr.strip():
                return None
            return int(json.loads(result.stdout)['streams'][0]['nb_read_frames'])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return None

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
