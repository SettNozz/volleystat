"""Compare the PR #20 roster policy with and without jersey-number constraints.

Replays fixed saved poses, cached OSNet embeddings and a saved jersey read
cache (``read_jersey_numbers.py``). No neural inference runs here. The base
variant is the final side policy of ``compare_player_court_side.py``
(history + OSNet corroboration + back prior, frame-policy near candidates kept,
zero floor offset).
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.inference_cache import file_digest
from src.tracking.jersey_numbers import JerseyParameters, jersey_constraints
from src.tracking.player_court_side import frame_sides, measure_poses, preserve_frame_sides, temporal_sides
from src.tracking.player_identity_workflow import PRODUCTION_PARAMETERS, observations_from_annotations
from src.tracking.player_reid import associate, gallery_keys
from src.viewer_app.court_correction import _projector_for

_spec = importlib.util.spec_from_file_location('court_side_eval', Path(__file__).with_name('compare_player_court_side.py'))
court_side_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(court_side_eval)


def build_frames(pool, evidence, seed):
    frames = []
    for frame in pool:
        players = []
        for obs in frame['players']:
            label = evidence[(frame['f'], obs['pose_index'])]
            if obs['key'] in seed['assignments']:
                players.append(obs)
            elif label.tentative:
                players.append({**obs, 'tentative_side': True})
            elif label.side == 'b' and not label.tentative:
                players.append(obs)
        frames.append({'f': frame['f'], 'players': players})
    return frames


def rows_from(pool, evidence, predictions):
    rows = {}
    for frame in pool:
        for obs in frame['players']:
            label = evidence[(frame['f'], obs['pose_index'])]
            prediction = predictions.get(obs['key'], {})
            pid = prediction.get('player_id')
            rows[obs['key']] = ('b' if pid else label.side, pid, prediction.get('reason'), label.reason)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--checks', type=Path, required=True)
    parser.add_argument('--reads', type=Path, required=True, help='jersey_number_reads_v1 JSON')
    parser.add_argument('--window', action='append', type=court_side_eval.frame_span, required=True)
    parser.add_argument('--osnet-stage')
    parser.add_argument('--min-prob', type=float, default=JerseyParameters.min_prob)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    path = args.annotations.resolve()
    payload = json.loads(path.read_bytes())
    fps = payload['fps']
    sha = file_digest(path)
    seed = json.loads(args.seed.read_bytes())
    checks = json.loads(args.checks.read_bytes())
    reads = json.loads(args.reads.read_bytes())
    if seed['annotations_sha256'] != sha or checks['annotations_sha256'] != sha or reads['annotations_sha256'] != sha:
        raise ValueError('Seed, checks and reads must belong to these annotations')
    checks = checks['checks']
    projector, _ = _projector_for(payload, payload['court_corners'])
    context = [ann for ann in payload['frames'] if any(lo - fps - 1 <= ann['f'] <= hi + fps for lo, hi in args.window)
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
    features, stage = court_side_eval.cached_embeddings(payload, pool, args.osnet_stage)
    scored = {obs['key'] for frame in pool for obs in frame['players']
              if abs(frame['f'] - seed['frame']) > PRODUCTION_PARAMETERS.gallery_half_window_frames}
    evidence = preserve_frame_sides(temporal_sides(context, measurements, fps, 0, use_back=True),
                                    frame_sides(context, measurements, 0, use_back=True))
    frames = build_frames(pool, evidence, seed)
    used = {obs['key']: features[obs['key']] for frame in frames for obs in frame['players']}
    used.update((key, features[key]) for keys in gallery.values() for key in keys)
    parameters = JerseyParameters(min_prob=args.min_prob)

    def run(jersey):
        return associate(frames, seed, used, fps, recover=True, parameters=PRODUCTION_PARAMETERS,
                         frozen_gallery=gallery, jersey=jersey)['predictions']

    base = run(None)
    reference = rows_from(pool, evidence, base)
    anchor = jersey_constraints(frames, reads['reads'], fps, seed, None, parameters)
    first = jersey_constraints(frames, reads['reads'], fps, seed, base, parameters)
    report = {'source': str(path), 'annotations_sha256': sha, 'osnet_stage': stage, 'windows': args.window,
              'reader': reads['reader'], 'jersey_parameters': parameters.to_dict(), 'inference_calls': 0,
              'scope': 'one source, fixed post-cap detections, diagnostic checks', 'variants': {}}
    for name, jersey in (('pr20_final', None), ('jersey_anchor_roster', anchor), ('jersey_first_pass_roster', first)):
        predictions = base if jersey is None else run(jersey)
        rows = rows_from(pool, evidence, predictions)
        metrics = court_side_eval.scores(rows, reference, checks, scored)
        reasons = Counter(predictions[k]['reason'] for k in scored if k in predictions)
        report['variants'][name] = {
            'metrics': metrics, 'reasons': dict(reasons),
            'roster_numbers': (jersey or {}).get('roster'), 'roster_source': (jersey or {}).get('roster_source'),
            'observations_with_number': len((jersey or {}).get('observations') or {}),
            'checks': {key: rows[key] for key in checks},
        }
        print(name, (jersey or {}).get('roster'), {k: metrics.get(k) for k in (
            'assigned_ids', 'reviewed_ids', 'reviewed_correct_id', 'reviewed_wrong_id', 'reviewed_missing_id',
            'reviewed_correct_side', 'changed_id_vs_zero', 'lost_id_vs_zero')}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
