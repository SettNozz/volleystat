"""Persistent, content-addressed model outputs. Never stores pickle or human labels."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import uuid
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT/'data/inference_cache'
SCHEMA = 'model_outputs_v1'


class CacheMiss(RuntimeError):
    """A replay requested model output that was never captured."""


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'Unsupported cache value: {type(item).__name__}')
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False, default=convert).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@lru_cache
def runtime_versions():
    result = {}
    for package in ('ultralytics', 'torch', 'torchvision', 'numpy', 'Pillow'):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def model_spec(path, *, backend, **settings):
    path = Path(path)
    return {'weights_sha256': file_digest(path), 'backend': backend,
            'runtime': runtime_versions(), 'settings': settings}


def array_payload(value):
    array = np.asarray(value)
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'values': array.tolist()}


def payload_array(value):
    result = np.asarray(value['values'], dtype=value['dtype'])
    return result.reshape(value['shape'])


class ModelOutputCache:
    """One source's immutable values, plus explicitly sealed tracker sequences.

    Stateful ByteTrack outputs are reused only as a complete ordered sequence.
    An interrupted writer is retained, but cannot masquerade as a warm cache.
    Stateless model outputs survive interruptions individually.
    """
    def __init__(self, source, root=DEFAULT_CACHE, require=False, *, source_identity=None):
        self.source = source_identity or {'kind': 'video', 'sha256': file_digest(source)}
        self.path = Path(root)/digest(self.source)
        self.path.mkdir(parents=True, exist_ok=True)
        self.require = require
        self.counts = Counter()
        self.computed = Counter()
        self.db = sqlite3.connect(self.path/'outputs.sqlite3', timeout=60)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS outputs (
                stage TEXT, item TEXT, body BLOB NOT NULL, sha TEXT NOT NULL,
                PRIMARY KEY(stage,item));
            CREATE TABLE IF NOT EXISTS sequences (
                stage TEXT, generation TEXT PRIMARY KEY, complete INTEGER NOT NULL DEFAULT 0);
        ''')
        manifest = self.path/'source.json'
        if not manifest.exists():
            try:
                with manifest.open('x', encoding='utf-8') as stream:
                    json.dump({'schema': SCHEMA, 'source': self.source}, stream, indent=2)
            except FileExistsError:
                pass
        if json.loads(manifest.read_text(encoding='utf-8')) != {'schema': SCHEMA, 'source': self.source}:
            raise ValueError('Cache source/schema mismatch')

    def stage(self, name, spec):
        key = name + ':' + digest({'schema': SCHEMA, 'spec': spec})
        # Specs remain inspectable without loading any model.
        self.put('stage_manifests', key, {'name': name, 'spec': spec})
        return key

    def get(self, stage, item):
        row = self.db.execute('SELECT body,sha FROM outputs WHERE stage=? AND item=?',
                              (stage, str(item))).fetchone()
        if row is None:
            self.counts['misses'] += 1
            return None
        raw = zlib.decompress(row[0])
        if hashlib.sha256(raw).hexdigest() != row[1]:
            raise ValueError(f'Corrupt model cache: {stage}/{item}')
        self.counts['hits'] += 1
        return json.loads(raw)

    def put(self, stage, item, value):
        raw = canonical(value)
        checksum = hashlib.sha256(raw).hexdigest()
        with self.db:
            existing = self.db.execute('SELECT sha FROM outputs WHERE stage=? AND item=?',
                                       (stage, str(item))).fetchone()
            if existing is not None:
                if existing[0] != checksum:
                    raise ValueError('Refusing to replace an existing frozen model output')
                return
            self.db.execute('INSERT INTO outputs VALUES (?,?,?,?)',
                            (stage, str(item), zlib.compress(raw, 3), checksum))
        self.counts['written'] += 1

    def compute(self, stage, item, operation):
        value = self.get(stage, item)
        if value is not None:
            return value
        if self.require:
            raise CacheMiss(f'Missing {stage}/{item}; replay cannot run inference')
        self.computed[stage.split(':')[0]] += 1
        value = operation()
        self.put(stage, item, value)
        return value

    def sequence(self, spec):
        stage = self.stage('pose_bytetrack', spec)
        row = self.db.execute('SELECT generation FROM sequences WHERE stage=? AND complete=1 LIMIT 1',
                              (stage,)).fetchone()
        if row:
            return stage+'/'+row[0], True
        generation = uuid.uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO sequences(stage,generation) VALUES (?,?)', (stage,generation))
        return stage+'/'+generation, False

    def covers(self, stage, frames):
        keys = {row[0] for row in self.db.execute('SELECT item FROM outputs WHERE stage=?', (stage,))}
        return all(str(frame) in keys for frame in frames)

    def seal_sequence(self, sequence):
        generation = sequence.rsplit('/', 1)[1]
        with self.db:
            self.db.execute('UPDATE sequences SET complete=1 WHERE generation=?', (generation,))

    def summary(self):
        return {'schema': SCHEMA, 'directory': str(self.path.resolve()), 'source': self.source,
                'require_cached': self.require, **dict(self.counts),
                'computed_outputs_by_stage': dict(self.computed),
                'labels': 'model-derived, not ground truth'}


class LazyModel:
    """No weights, warmup or GPU allocation until an actual cache miss."""
    def __init__(self, factory):
        self.factory = factory
        self.loaded = None

    def __getattr__(self, name):
        if self.loaded is None:
            self.loaded = self.factory()
        return getattr(self.loaded, name)

    def __call__(self, *args, **kwargs):
        if self.loaded is None:
            self.loaded = self.factory()
        return self.loaded(*args, **kwargs)

    def reset_tracker(self):
        if self.loaded is not None:
            for tracker in getattr(getattr(self.loaded, 'predictor', None), 'trackers', []):
                tracker.reset()


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self, value):
        self.data = _Array(payload_array(value))
        values = self.data.value
        self.xyxy = _Array(values[:, :4])
        self.id = _Array(values[:, 4]) if values.shape[1] == 7 else None
        self.conf = _Array(values[:, -2])
        self.cls = _Array(values[:, -1])

    def __len__(self):
        return len(self.data.value)


def pack_pose_result(result):
    if result is None:
        return {'boxes': None, 'keypoints': None}
    return {'boxes': array_payload(result.boxes.data.cpu().numpy()) if result.boxes is not None else None,
            'keypoints': array_payload(result.keypoints.data.cpu().numpy()) if result.keypoints is not None else None}


def unpack_pose_result(value):
    return SimpleNamespace(boxes=_Boxes(value['boxes']) if value['boxes'] is not None else None,
                           keypoints=SimpleNamespace(data=_Array(payload_array(value['keypoints'])))
                           if value['keypoints'] is not None else None)


class PoseTracker:
    """The same Ultralytics ByteTrack callback, replayed on saved YOLO outputs.

    No neural model is loaded. Empty detections and unconfirmed tracks follow
    on_predict_postprocess_end's behaviour exactly, including keypoint indices.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.tracker = None

    def update(self, raw, size):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import YAML
        if raw['boxes'] is None:
            return raw
        if self.tracker is None:
            config = YAML.load(str(self.config_path))
            if config['tracker_type']!='bytetrack':
                raise ValueError('Cached pose replay currently supports ByteTrack only')
            self.tracker = BYTETracker(SimpleNamespace(**config), frame_rate=30)
        boxes = Boxes(payload_array(raw['boxes']), orig_shape=(size[1],size[0])).cpu().numpy()
        tracks = self.tracker.update(boxes, img=None)
        if not len(tracks):
            return raw
        indices = tracks[:, -1].astype(int)
        tracked = tracks[:, :-1].copy()
        # Results.update clips the callback's boxes to native image bounds.
        tracked[:, [0,2]] = np.clip(tracked[:, [0,2]], 0, size[0])
        tracked[:, [1,3]] = np.clip(tracked[:, [1,3]], 0, size[1])
        return {'boxes': array_payload(tracked),
                'keypoints': array_payload(payload_array(raw['keypoints'])[indices])
                if raw['keypoints'] is not None else None}


class ObservationFeatures:
    """Crop keys exclude roster IDs, pose indices, seed and association policy."""
    def __init__(self, cache, spec, width, height):
        self.cache = cache
        self.stage = cache.stage('osnet_embeddings', spec)
        self.width, self.height = width, height

    def key(self, frame, box):
        x1,y1,x2,y2 = map(round, box)
        box = [max(0,min(self.width,x1)),max(0,min(self.height,y1)),
               max(0,min(self.width,x2)),max(0,min(self.height,y2))]
        return digest({'frame': int(frame), 'crop_xyxy': box})

    def get(self, frame, box):
        value = self.cache.get(self.stage, self.key(frame, box))
        return payload_array(value) if value is not None else None

    def put(self, frame, box, value):
        self.cache.put(self.stage, self.key(frame, box), array_payload(value))


class FrozenOSNet:
    """Pinned encoder with no neural instance until encode is really needed."""
    def __init__(self, assets, weights=None, device='auto', *, threads=None, cuda_memory_fraction=None):
        from src.models.player_reid_encoder import OSNetEncoder, SOURCE_SHA256, WEIGHTS_SHA256
        self.assets=Path(assets)
        self.weights=Path(weights) if weights else self.assets/'osnet_msmt17.pth'
        if file_digest(self.assets/'osnet.py')!=SOURCE_SHA256 or file_digest(self.weights)!=WEIGHTS_SHA256:
            raise ValueError('OSNet code/weights differ from the supported pinned model')
        def load():
            if threads is not None or cuda_memory_fraction is not None:
                import torch
                if threads is not None:
                    torch.set_num_threads(threads)
                if cuda_memory_fraction is not None and device=='cuda':
                    torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction,0)
            return OSNetEncoder(self.assets,self.weights,device)
        self.lazy=LazyModel(load)
        self.crop=OSNetEncoder.crop

    def encode(self,crops):
        return self.lazy.encode(crops)

    @property
    def provenance(self):
        from src.models.player_reid_encoder import SOURCE_SHA256, WEIGHTS_SHA256
        if self.lazy.loaded is not None:
            return {**self.lazy.loaded.provenance,'model_loaded_this_run':True}
        return {'architecture':'osnet_x1_0','weights_path':str(self.weights.resolve()),
                'weights_sha256':WEIGHTS_SHA256,'code_sha256':SOURCE_SHA256,
                'input_hw':[256,128],'normalization':'ImageNet mean/std','flip_average':True,
                'strict_weight_loading':True,'training':False,'model_loaded_this_run':False,
                'device':'frozen_cache','inference':'reused saved model outputs'}


def osnet_feature_store(cache, assets, weights, width, height):
    from src.models.player_reid_encoder import SOURCE_SHA256
    return ObservationFeatures(cache,model_spec(weights,
        backend=file_digest(ROOT/'src/models/player_reid_encoder.py'),code_sha256=SOURCE_SHA256,
        crop='clamped_rounded_full_bbox',flip_average=True),width,height)


def video_frames(video, needed, size=None):
    """Decode only on a missing feature; retrieve only requested source frames."""
    import cv2
    cap=cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError('Source video unavailable')
        if size and (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))!=tuple(size):
            raise ValueError('Source dimensions differ from the feature plan')
        current=-1
        for frame in sorted(needed):
            while current<frame:
                if not cap.grab():
                    raise ValueError(f'Source ended before frame {frame}')
                current+=1
            ok,image=cap.retrieve()
            if not ok:
                raise ValueError(f'Cannot decode source frame {frame}')
            yield frame,image
    finally:
        cap.release()


def collect_embeddings(frames, images, encoder, feature_cache=None, batch_size=8, progress=None):
    observations={p['key']:(f['f'],p) for f in frames for p in f['players']}
    features={}
    if feature_cache is not None:
        for key,(frame,obs) in observations.items():
            value=feature_cache.get(frame,obs['bb'])
            if value is not None:
                features[key]=value
        if len(features)!=len(observations) and feature_cache.cache.require:
            raise CacheMiss('Missing OSNet crop embeddings; replay cannot run inference')
    if len(features)==len(observations):
        return features
    needed={frame for key,(frame,_) in observations.items() if key not in features}
    by_frame={f['f']:f['players'] for f in frames}
    pending,keys=[],[]
    def flush():
        if pending:
            values=encoder.encode(pending)
            if len(values)!=len(keys):
                raise ValueError('Encoder output count differs from saved detections')
            features.update(zip(keys,values))
            if feature_cache is not None:
                feature_cache.cache.counts['osnet_batches']+=1
                feature_cache.cache.computed['osnet_embeddings']+=len(values)
                for key,value in zip(keys,values):
                    frame,obs=observations[key]
                    feature_cache.put(frame,obs['bb'],value)
            pending.clear(); keys.clear()
    visited=0
    for frame,image in (images(needed) if callable(images) else images):
        if frame not in needed:
            continue
        visited+=1
        for obs in by_frame[frame]:
            if obs['key'] not in features:
                pending.append(encoder.crop(image,obs['bb']).copy())
                keys.append(obs['key'])
                if len(pending)>=batch_size:
                    flush()
        if progress:
            progress(visited,len(needed))
        if len(features)+len(keys)==len(observations):
            break
    flush()
    if set(features)!=set(observations):
        raise ValueError('Source images do not cover all saved detections')
    return features


class CachedForward:
    """Cache raw neural logits; phase gates and smoothing are replayed normally."""
    def __init__(self, model, cache, spec):
        self.model, self.cache = model, cache
        self.stage = cache.stage('neural_logits', spec)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, tensor):
        import torch
        inputs = tensor.detach().cpu().numpy()
        key = digest({'shape': list(inputs.shape), 'dtype': inputs.dtype.str,
                      'input_sha256': hashlib.sha256(inputs.tobytes()).hexdigest()})
        value = self.cache.compute(self.stage, key,
                                  lambda: array_payload(self.model(tensor).detach().cpu().numpy()))
        return torch.from_numpy(payload_array(value)).to(tensor.device)
