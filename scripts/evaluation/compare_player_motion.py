"""Matched EMA/Kalman comparison on immutable saved detections/embeddings."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluation.run_player_identity_pilot import digest, write_json
from scripts.evaluation.run_player_reid_experiment import project_inputs
from src.tracking.player_identity import score
from src.tracking.player_reid import ReIDParameters, associate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tracklet-consistency', action='store_true',
                        help='Also score the production local contradiction rejection')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh directory')
    observations, labels, fingerprint = project_inputs(args.project)
    plan = json.loads((args.cache/'plan.json').read_text())
    manifest = json.loads((args.cache/'embedding_manifest.json').read_text())
    if (plan['observations_sha256'] != fingerprint
            or manifest['observations_sha256'] != fingerprint
            or manifest['embedding_sha256'] != digest(args.cache/'embeddings.npz')
            or plan['seed']['assignments'] != labels['seed']['assignments']
            or plan['seed']['frame'] != labels['seed']['frame']):
        raise ValueError('Cache/source/seed fingerprint mismatch')
    args.output.mkdir(parents=True)
    baseline = ReIDParameters(**plan['parameters'])
    design = {'ema': replace(baseline, motion_model='ema'),
              'kalman': replace(baseline, motion_model='kalman')}
    if args.tracklet_consistency:
        design['kalman_roster_guard'] = replace(baseline, motion_model='kalman', tracklet_consistency=True)
    write_json(args.output/'plan.json', {
        'schema':'player_motion_comparison_v1', 'observations_sha256':fingerprint,
        'labels_sha256':digest(args.project/'labels.json'),
        'embedding_sha256':manifest['embedding_sha256'],
        'association_source_sha256':digest(Path(__file__).resolve().parents[2]/'src/tracking/player_reid.py'),
        'motion_source_sha256':digest(Path(__file__).resolve().parents[2]/'src/tracking/player_motion.py'),
        'parameters':{name:p.to_dict() for name,p in design.items()},
        'evaluation_use':'scoring only; existing checks are development estimates'})
    data = np.load(args.cache/'embeddings.npz', allow_pickle=False)
    features = dict(zip(data['keys'].tolist(), data['embeddings']))
    output = {}
    for name, parameters in design.items():
        run = associate(observations['frames'], plan['seed'], features,
                        observations['fps'], True, parameters)
        run['human_checks'] = score(run['predictions'], labels['checks'], set(plan['seed']['assignments']))
        run['reasons'] = dict(Counter(p['reason'] for p in run['predictions'].values()))
        output[name] = run
        print(name, json.dumps(run['human_checks']), flush=True)
    write_json(args.output/'comparison.json', output)


if __name__ == '__main__':
    main()
