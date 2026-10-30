"""Compare confirmed roster expansion and rejection on one frozen feature cache."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.player_reid_encoder import SOURCE_SHA256, WEIGHTS_SHA256, file_sha256
from src.tracking.player_identity_workflow import (
    PRODUCTION_PARAMETERS, apply_identity_result, load_seed, observations_from_annotations,
)
from src.tracking.player_reid import associate


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def nonidentity_digest(data):
    """Verify every original value except the explicitly replaced identity fields."""
    value = deepcopy(data)
    value.pop('player_identity', None)
    for frame in value['frames']:
        for pose in frame.get('poses') or []:
            pose.pop('player_id', None)
            pose.pop('identity', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--roster', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--appearance-regret-max', type=float,
                        help='Evaluate only a candidate with this maximum local appearance regret (0..1)')
    args = parser.parse_args()
    if args.appearance_regret_max is not None and not 0 <= args.appearance_regret_max <= 1:
        parser.error('Appearance regret must be in 0..1')
    if args.output.exists():
        parser.error('Output exists; choose a fresh versioned directory')
    source = args.annotations.resolve()
    data = json.loads(source.read_text(encoding='utf-8'))
    stem = source.name.removesuffix('_annotations.json')
    stats_path = source.with_name(stem + '_stats.json')
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    plan = json.loads((args.cache/'plan.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    encoder = manifest.get('encoder') or {}
    if (encoder.get('weights_sha256') != WEIGHTS_SHA256
            or encoder.get('code_sha256') != SOURCE_SHA256
            or encoder.get('strict_weight_loading') is not True
            or encoder.get('training') is not False):
        raise ValueError('Cache encoder differs from the frozen supported model')
    video = Path(data['original_video_path'])
    if (plan['schema'] != 'player_reid_embedding_cache_v1'
            or file_sha256(source) != plan['annotations_sha256']
            or manifest['annotations_sha256'] != plan['annotations_sha256']
            or file_sha256(video) != plan['video_sha256']
            or file_sha256(args.cache/'observations.json') != manifest['observations_sha256']
            or file_sha256(args.cache/'embeddings.npz') != manifest['embedding_sha256']):
        raise ValueError('Cache/source fingerprints differ')
    frames = observations_from_annotations(data['frames'], plan['side'])
    cached = json.loads((args.cache/'observations.json').read_text(encoding='utf-8'))
    if frames != cached['frames'] or data['fps'] != cached['fps']:
        raise ValueError('Cached observations differ from source detections')
    if (len(frames), data['frame_width'], data['frame_height']) != (
            plan['source_frames'], plan['width'], plan['height']):
        raise ValueError('Source frame count or resolution differs')
    with np.load(args.cache/'embeddings.npz', allow_pickle=False) as archive:
        keys = archive['keys'].tolist()
        vectors = archive['embeddings']
    expected = [p['key'] for frame in frames for p in frame['players']]
    if keys != expected or vectors.shape != (len(expected), 512) or not np.isfinite(vectors).all():
        raise ValueError('Embedding keys, dimensions or values differ')
    features = dict(zip(keys, vectors))
    seed = load_seed(args.roster, frames, video, plan['side'], 6)
    if not seed.get('introductions'):
        raise ValueError('Experiment needs confirmed roster introductions')
    fixed_seed = {k: v for k, v in seed.items() if k != 'introductions'}
    design = {
        'fixed6': (fixed_seed, False),
        'fixed6_guard': (fixed_seed, True),
        'confirmed_roster': (seed, False),
        'confirmed_roster_guard': (seed, True),
    }
    parameters = PRODUCTION_PARAMETERS
    if args.appearance_regret_max is not None:
        parameters = replace(parameters, local_appearance_regret_max=args.appearance_regret_max)
        design = {'confirmed_roster_guard': (seed, True)}
    args.output.mkdir(parents=True)
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    sources = {}
    for relative in ('src/tracking/player_reid.py', 'src/tracking/player_motion.py',
                     'src/tracking/player_identity_consistency.py',
                     'src/tracking/player_identity_workflow.py',
                     'scripts/evaluation/run_player_roster_experiment.py'):
        sources[relative] = file_sha256(ROOT/relative)
        shutil.copy2(ROOT/relative, snapshot/Path(relative).name)
    write_json(args.output/'experiment_plan.json', {
        'schema': 'player_roster_comparison_v1', 'annotations_sha256': plan['annotations_sha256'],
        'stats_sha256': file_sha256(stats_path), 'video_sha256': plan['video_sha256'],
        'roster_sha256': file_sha256(args.roster), 'cache_manifest_sha256': file_sha256(args.cache/'manifest.json'),
        'source_sha256': sources, 'parameters': parameters.to_dict(),
        'design': {name: {'known_people': len(s['assignments']) + len(s.get('introductions', [])),
                          'guard': guard} for name, (s, guard) in design.items()},
        'selected': 'confirmed_roster_guard', 'training': False,
        'evaluation': 'No held-out human identity checks on target video. Coverage is not accuracy.',
        'human_input': 'New-person confirmation only; appearance anchor and first-frame boundary remain derived.'})
    before = nonidentity_digest(data)
    source_ids = {f'{frame["f"]}:{i}': pose.get('player_id')
                  for frame in data['frames'] for i, pose in enumerate(frame.get('poses') or [])
                  if f'{frame["f"]}:{i}' in features}
    comparison = {}
    selected = None
    for name, (case_seed, guard) in design.items():
        started = time.monotonic()
        result = associate(frames, case_seed, features, data['fps'], True,
                           replace(parameters, tracklet_consistency=guard))
        counts = Counter(p['player_id'] for p in result['predictions'].values())
        comparison[name] = {
            'assigned': len(expected) - counts[None], 'observations': len(expected),
            'coverage': 1 - counts[None]/len(expected),
            'person_observations': {pid: count for pid, count in counts.items() if pid is not None},
            'reasons': dict(Counter(p['reason'] for p in result['predictions'].values())),
            'tracklet_consistency': result['tracklet_consistency'],
            'elapsed_seconds': time.monotonic() - started,
            'agreement_with_source_fixed6': sum(prediction['player_id'] == source_ids[key]
                                                for key, prediction in result['predictions'].items()),
            'source_agreement_is_accuracy': False,
        }
        write_json(args.output/(name + '_predictions.json'), result)
        print(name, json.dumps(comparison[name]), flush=True)
        if name == 'confirmed_roster_guard':
            selected = result
    report = apply_identity_result(data['frames'], seed, selected, plan['side'])
    report.update({'encoder': manifest['encoder'], 'input_annotations_sha256': plan['annotations_sha256'],
                   'source_video': str(video.resolve()), 'cache': str(args.cache.resolve()),
                   'cache_manifest_sha256': file_sha256(args.cache/'manifest.json'),
                   'processing_time_sec': manifest['elapsed_seconds'] + comparison['confirmed_roster_guard']['elapsed_seconds']})
    data['player_identity'] = report
    if before != nonidentity_digest(data):
        raise AssertionError('Nonidentity source values changed')
    stats['player_identity'] = report
    comparison['preservation'] = {'nonidentity_sha256': before, 'unchanged': True,
                                  'includes': 'all raw poses, ball, geometry, FSM and event values'}
    # Keep the original full-analysis timing; derived pass timing is reported separately.
    write_json(args.output/'comparison.json', comparison)
    write_json(args.output/(stem + '_annotations.json'), data)
    write_json(args.output/(stem + '_stats.json'), stats)
    print('Fresh result:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
