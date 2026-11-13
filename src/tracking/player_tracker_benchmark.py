"""Controlled tracker association over fixed saved poses, never raw detector gold.

Only IDs change. Original boxes, pose indices, side decisions and cached crop
features stay fixed, making movement labels usable without a geometric rejoin.
This experiment does not reproduce pre-filter production tracking.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from src.tracking.player_identity_scoring import score

BOXMOT_VERSION = '26.0.0'
TRACKERS = ('bytetrack', 'botsort', 'deepocsort')


def align_ids(keys, track_ids, detection_indices):
    """Bind IDs by backend detection indices; unmatched predictions add no pose."""
    if len(track_ids) != len(detection_indices):
        raise ValueError('Track IDs and detection indices differ in length')
    result = dict.fromkeys(keys)
    used_ids, used_indices = set(), set()
    for raw_id, index in zip(track_ids, detection_indices):
        if int(index) != index or int(raw_id) != raw_id or raw_id < 0:
            raise ValueError('Backend returned invalid track ID or detection index')
        index, raw_id = int(index), int(raw_id)
        if index == -1:
            continue
        if index < 0 or index >= len(keys) or index in used_indices or raw_id in used_ids:
            raise ValueError('Backend detection mapping is out of range or non-unique')
        result[keys[index]] = str(raw_id)
        used_indices.add(index)
        used_ids.add(raw_id)
    return result


class FixedPoseTracker:
    """No model construction or video decode; appearance is supplied per row."""

    def __init__(self, name, size):
        self.name, self.size = name, size
        if name == 'bytetrack':
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import ROOT, YAML
            self.config = YAML.load(str(ROOT/'cfg/trackers/bytetrack.yaml'))
            self.backend = BYTETracker(SimpleNamespace(**self.config), frame_rate=30)
        elif name in ('botsort', 'deepocsort'):
            import boxmot
            if boxmot.__version__ != BOXMOT_VERSION:
                raise ValueError(f'Benchmark requires boxmot=={BOXMOT_VERSION}')
            from boxmot import BotSort, BotSortConfig, DeepOcSort, DeepOcSortConfig
            cls, config = ((BotSort, BotSortConfig) if name == 'botsort'
                           else (DeepOcSort, DeepOcSortConfig))
            settings = config(use_cmc=False, use_embeddings=True)
            self.config = asdict(settings)
            self.backend = cls(settings)
        else:
            raise ValueError(f'Unknown tracker: {name}')

    def update(self, players, features):
        keys = [p['key'] for p in players]
        boxes = np.asarray([p['bb'] for p in players], dtype=np.float32).reshape(-1, 4)
        if self.name == 'bytetrack':
            from ultralytics.engine.results import Boxes
            detections = np.column_stack((boxes, np.ones(len(boxes)), np.zeros(len(boxes)))).astype(np.float32)
            tracks = self.backend.update(Boxes(detections, (self.size[1], self.size[0])).cpu().numpy(), img=None)
            return align_ids(keys, tracks[:, 4] if len(tracks) else [],
                             tracks[:, -1] if len(tracks) else [])
        import torch
        from boxmot.structures import Boxes, Detections
        embeddings = np.stack([features[key] for key in keys]) if keys else np.empty((0, 512))
        if not np.isfinite(embeddings).all():
            raise ValueError('Non-finite cached embeddings')
        detections = Detections(
            geometry=Boxes(torch.from_numpy(boxes)), scores=torch.ones(len(keys)),
            class_ids=torch.zeros(len(keys), dtype=torch.int64),
            embeddings=torch.as_tensor(embeddings, dtype=torch.float32))
        tracks = self.backend.update(detections)
        return align_ids(keys, tracks.track_ids.tolist(), tracks.detection_indices.tolist())


def replay_tracks(frames, features, tracker, progress=None):
    """Advance every source frame, including omitted frames and empty detections."""
    predictions, previous = {}, None
    for frame in frames:
        current = frame['f']
        if previous is not None and current <= previous:
            raise ValueError('Tracker replay requires unique ordered frames')
        if previous is not None:
            for _ in range(previous+1, current):
                tracker.update([], features)
        predictions.update(tracker.update(frame['players'], features))
        previous = current
        if progress and current % 2000 == 0:
            progress(current)
    return predictions


def replace_raw_ids(frames, predictions):
    """Copy observations without changing their boxes or human join keys."""
    return [{**frame, 'players': [{**p, 'raw_id': predictions[p['key']]}
                                 for p in frame['players']]} for frame in frames]


def tracking_metrics(predictions, labels, fps):
    """Conditional IDF1 on reviewed observations; no detector recall claim."""
    result = score(predictions, labels, fps)
    assigned = result['correct'] + result['wrong']
    total = result['observations']
    timeline = defaultdict(list)
    for key, person in labels['players'].items():
        timeline[person].append((int(key.split(':')[0]), predictions[key]))
    changes = Counter()
    for items in timeline.values():
        items.sort()
        known = [(f, pid) for f, pid in items if pid is not None]
        for (f0, a), (f1, b) in zip(known, known[1:]):
            if a != b:
                changes['id_changes_across_reviewed_observations'] += 1
                changes['id_changes_after_gap_1s'] += (f1-f0)/fps >= 1.
    return {**result, 'observation_idf1': 2*result['correct']/max(total+assigned, 1),
            'idtp': result['correct'], 'idfp': result['wrong'],
            'idfn': total-result['correct'], 'distinct_track_ids': len({
                predictions[k] for k in labels['players'] if predictions[k] is not None}),
            **dict(changes)}


def roster_predictions(frames, features, fps):
    """Run the current automatic roster stage, with its short-clip fallback."""
    from src.tracking.player_identity import default_seed
    from src.tracking.player_identity_workflow import (
        DISCOVERY_PARAMETERS, PRODUCTION_PARAMETERS, discovery_possible)
    from src.tracking.player_reid import associate
    from src.tracking.player_roster_discovery import (
        RosterNotFound, assign_roster, discover_roster, name_left_to_right)
    fallback = None
    if discovery_possible(frames, fps):
        try:
            found = discover_roster(frames, features, fps, 6, DISCOVERY_PARAMETERS)
            name_left_to_right(frames, found['roster'])
            predictions = assign_roster(frames, features, fps, found['roster'], DISCOVERY_PARAMETERS,
                                        cohort=found['cohort'])
            return {k: p['player_id'] for k, p in predictions.items()}, {
                'method': 'osnet_tracklet_roster_v5', 'tracklets': found['tracklets'],
                'people_found': found['people'], 'parameters': DISCOVERY_PARAMETERS.to_dict()}
        except RosterNotFound as exc:
            fallback = str(exc)
    try:
        seed = default_seed(frames, 6)
    except ValueError:
        return {p['key']: None for f in frames for p in f['players']}, {
            'method': 'no_complete_roster', 'fallback': fallback}
    result = associate(frames, seed, features, fps, True, PRODUCTION_PARAMETERS)
    return {k: p['player_id'] for k, p in result['predictions'].items()}, {
        'method': 'osnet_frozen_gallery_kalman_roster_v4', 'fallback': fallback,
        'parameters': PRODUCTION_PARAMETERS.to_dict()}
