"""Compare player-side policies using fixed saved poses and cached OSNet embeddings."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.inference_cache import ModelOutputCache, ObservationFeatures, file_digest
from src.tracking.player_court_side import frame_sides, measure_poses, preserve_frame_sides, temporal_sides
from src.tracking.player_identity_workflow import PRODUCTION_PARAMETERS, observations_from_annotations
from src.tracking.player_reid import associate, gallery_keys
from src.viewer_app.court_correction import _projector_for


class FrozenOSNetStage:
    """Read one existing stage with the production cache key and checksum checks."""

    get = ModelOutputCache.get

    def __init__(self, db, stage):
        self.db, self.selected = db, stage
        self.counts = Counter()

    def stage(self, name, _spec):
        if not self.selected.startswith(name + ':') or self.get('stage_manifests', self.selected) is None:
            raise ValueError(f'No frozen stage {self.selected}')
        return self.selected


def cached_embeddings(payload, frames, requested_stage):
    db_path = Path(payload['model_cache']['directory']) / 'outputs.sqlite3'
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        stages = [row[0] for row in db.execute(
            "SELECT item FROM outputs WHERE stage='stage_manifests' AND item LIKE 'osnet_embeddings:%'"
        )]
        if requested_stage is None and len(stages) != 1:
            raise ValueError(f'Select one --osnet-stage from {stages}')
        stage = requested_stage or stages[0]
        reader = FrozenOSNetStage(db, stage)
        store = ObservationFeatures(reader, None, payload['frame_width'], payload['frame_height'])
        features = {obs['key']: store.get(frame['f'], obs['bb'])
                    for frame in frames for obs in frame['players']}
        missing = [key for key, value in features.items() if value is None]
        if missing:
            raise ValueError(f'{len(missing)} missing OSNet crops, first keys {missing[:5]}')
        return features, stage


def replay(pool, evidence, seed, features, gallery, fps, allow_tentative):
    side = 'b'
    frames = []
    for frame in pool:
        players = []
        for obs in frame['players']:
            label = evidence[(frame['f'], obs['pose_index'])]
            if obs['key'] in seed['assignments']:
                players.append(obs)
            elif label.tentative and allow_tentative:
                players.append({**obs, 'tentative_side': True})
            elif label.side == side and not label.tentative:
                players.append(obs)
        frames.append({'f': frame['f'], 'players': players})
    used_features = {obs['key']: features[obs['key']] for frame in frames for obs in frame['players']}
    used_features.update((key, features[key]) for keys in gallery.values() for key in keys)
    predictions = associate(frames, seed, used_features, fps, recover=True,
                            parameters=PRODUCTION_PARAMETERS, frozen_gallery=gallery)['predictions']
    rows = {}
    for frame in pool:
        for obs in frame['players']:
            evidence_row = evidence[(frame['f'], obs['pose_index'])]
            prediction = predictions.get(obs['key'], {})
            player_id = prediction.get('player_id')
            rows[obs['key']] = (side if player_id else evidence_row.side, player_id,
                                prediction.get('reason'), evidence_row.reason)
    return rows


def scores(rows, reference, checks, keys):
    stats = Counter()
    for key in keys:
        side, pid, reason, _ = rows[key]
        old_side, old_pid, _, _ = reference[key]
        stats.update({'observations': 1, 'side_known': side is not None, 'assigned_ids': pid is not None,
                      'opposite_side_vs_zero': side is not None and old_side is not None and side != old_side,
                      'lost_side_vs_zero': side is None and old_side is not None,
                      'changed_id_vs_zero': pid is not None and old_pid is not None and pid != old_pid,
                      'lost_id_vs_zero': pid is None and old_pid is not None,
                      'tentative_ids': reason == 'tentative_side_corroborated'})
    for key, expected in checks.items():
        side, pid, _, _ = rows[key]
        if 'side' in expected:
            stats.update({'reviewed': 1, 'reviewed_correct_side': side == expected['side'],
                          'reviewed_unknown_side': side is None,
                          'reviewed_far_with_roster_id': expected['side'] == 'a' and pid is not None})
        if expected.get('no_roster_id'):
            stats.update({'reviewed_non_roster': 1, 'reviewed_non_roster_with_id': pid is not None})
        if 'player_id' in expected:
            stats.update({'reviewed_ids': 1, 'reviewed_correct_id': pid == expected['player_id'],
                          'reviewed_missing_id': pid is None,
                          'reviewed_wrong_id': pid is not None and pid != expected['player_id']})
    return dict(stats)


def frame_span(value):
    start, end = map(int, value.split(':'))
    if not 0 <= start <= end:
        raise ValueError('Window must be START:END with 0 <= START <= END')
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New report JSON')
    parser.add_argument('--window', action='append', type=frame_span, required=True, help='Inclusive START:END')
    parser.add_argument('--osnet-stage')
    parser.add_argument('--seed', type=Path, help='Frozen diagnostic roster tied to these annotations')
    parser.add_argument('--preserve-frame-near', action='store_true', help='Keep the frame policy’s near-side candidates')
    parser.add_argument('--checks', type=Path, help='Visual side checks keyed by frame:pose_index')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.annotations.resolve()
    payload = json.loads(path.read_bytes())
    fps = payload['fps']
    annotations_sha = file_digest(path)
    seed = json.loads(args.seed.read_bytes()) if args.seed else payload['player_identity']['seed']
    if args.seed and seed['annotations_sha256'] != annotations_sha:
        raise ValueError('Diagnostic roster belongs to another annotations file')
    if seed.get('introductions') or payload['player_identity']['side'] != 'b':
        raise ValueError('Experiment requires a fixed near-side roster')
    projector, _ = _projector_for(payload, payload['court_corners'])
    if not projector.has_net_floor_line:
        raise ValueError('Saved net floor required')
    all_frames = payload['frames']
    context = [ann for ann in all_frames if any(lo - fps - 1 <= ann['f'] <= hi + fps for lo, hi in args.window)
               or abs(ann['f'] - seed['frame']) <= PRODUCTION_PARAMETERS.gallery_half_window_frames + fps]
    measurements = measure_poses(context, projector, payload['frame_height'])
    selected = [ann for ann in context if any(lo <= ann['f'] <= hi for lo, hi in args.window)
                or abs(ann['f'] - seed['frame']) <= PRODUCTION_PARAMETERS.gallery_half_window_frames]
    pool = observations_from_annotations(selected, side=None)
    anchor_context = observations_from_annotations(selected, side=None)
    for frame, ann in zip(anchor_context, selected):
        frame['players'] = [obs for obs in frame['players'] if obs['key'] in seed['assignments']
                            or ann['poses'][obs['pose_index']]['team'] in ('b', 'team_b')]
    gallery = gallery_keys(anchor_context, seed, PRODUCTION_PARAMETERS)
    features, stage = cached_embeddings(payload, pool, args.osnet_stage)
    scored_keys = {obs['key'] for frame in pool for obs in frame['players']
                   if abs(frame['f'] - seed['frame']) > PRODUCTION_PARAMETERS.gallery_half_window_frames}
    checks = {}
    if args.checks:
        check_payload = json.loads(args.checks.read_bytes())
        if check_payload['annotations_sha256'] != annotations_sha:
            raise ValueError('Diagnostic checks belong to another annotations file')
        checks = check_payload['checks']
    if not checks.keys() <= scored_keys or any(check.get('side') not in ('a', 'b', None) for check in checks.values()):
        raise ValueError('Side checks must select scored observations and use a/b labels')
    offsets = (0, -10, -5, 5, 10)
    report = {'source': str(path), 'annotations_sha256': annotations_sha, 'windows': args.window,
              'seed': seed, 'gallery': gallery, 'osnet_stage': stage, 'cached_crops': len(features),
              'inference_calls': 0, 'scope': 'fixed post-cap detections, one source, diagnostic checks',
              'preserve_frame_near': args.preserve_frame_near, 'checks': checks,
              'back_scores': {key: next(ann for ann in selected if ann['f'] == int(key.split(':')[0]))
                              ['poses'][int(key.split(':')[1])].get('back') for key in checks}, 'variants': {}}
    for name, temporal, open_gate, back in (
        ('frame_floor', False, False, False), ('history', True, False, False),
        ('history_osnet', True, True, False), ('history_osnet_back', True, True, True),
    ):
        variants = report['variants'][name] = {}
        reference = None
        for offset in offsets:
            if temporal:
                evidence = temporal_sides(context, measurements, fps, offset, use_back=back)
                if args.preserve_frame_near:
                    evidence = preserve_frame_sides(
                        evidence, frame_sides(context, measurements, offset, use_back=back),
                    )
            else:
                evidence = frame_sides(context, measurements, offset)
            rows = replay(pool, evidence, seed, features, gallery, fps, open_gate)
            if reference is None:
                reference = rows
            variants[str(offset)] = {'metrics': scores(rows, reference, checks, scored_keys),
                                     'checks': {key: rows[key] for key in checks}}
            print(name, offset, variants[str(offset)]['metrics'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
