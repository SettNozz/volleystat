"""Frozen OSNet/HSV galleries crossed with local-only/lost-recovery association."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluation.run_player_identity_pilot import digest, write_json
from src.tracking.player_identity import compare, score
from src.tracking.player_reid import ReIDParameters, associate, gallery_keys
from src.models.player_reid_encoder import (
    SOURCE_SHA256, WEIGHTS_SHA256, OSNetEncoder, install_assets,
)
from src.models.inference_cache import FrozenOSNet,ModelOutputCache,collect_embeddings,osnet_feature_store,video_frames


def fetch_assets(args):
    """Explicit download only; embedding/evaluation have no network fallback."""
    install_assets(args.output)


def project_inputs(project: Path):
    observations = json.loads((project/'observations.json').read_text(encoding='utf-8'))
    labels = json.loads((project/'labels.json').read_text(encoding='utf-8'))
    fingerprint = digest(project/'observations.json')
    if labels['observations_sha256'] != fingerprint:
        raise ValueError('Human labels do not match frozen observations')
    if not labels.get('seed') or labels['seed'].get('label_source') != 'human' or not labels['seed'].get('verified'):
        raise ValueError('A verified human seed is required')
    return observations, labels, fingerprint


def embed(args):
    import torch

    observations, labels, fingerprint = project_inputs(args.project)
    if args.output.exists():
        raise ValueError('Embedding output exists; use a fresh directory')
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('Batch size and thread count must be positive')
    assets = args.assets
    if digest(assets/'osnet.py') != SOURCE_SHA256 or digest(assets/'osnet_msmt17.pth') != WEIGHTS_SHA256:
        raise ValueError('OSNet code/weights differ from pinned official artifacts')
    source = observations['source']
    video = Path(source['video'])
    if digest(Path(source['annotations'])) != source['annotations_sha256']:
        raise ValueError('Source annotations changed')
    if video.stat().st_size != source['video_size'] or video.stat().st_mtime_ns != source['video_mtime_ns']:
        raise ValueError('Source video changed')
    parameters = ReIDParameters()
    # Freeze design before producing any new evaluation scores.
    args.output.mkdir(parents=True)
    plan = {'schema': 'player_reid_factorial_plan_v1', 'observations_sha256': fingerprint,
            'labels_sha256_at_start': digest(args.project/'labels.json'),
            'seed': labels['seed'], 'parameters': parameters.to_dict(),
            'factors': {'descriptor': ['hsv', 'osnet'], 'recovery': [False, True]},
            'gallery_keys': gallery_keys(observations['frames'], labels['seed'], parameters),
            'evaluation_use': 'checks only score predictions, never galleries/association; development estimate',
            'training': False}
    check_keys = {c['key'] for c in labels['checks']}
    if check_keys.intersection(k for keys in plan['gallery_keys'].values() for k in keys):
        raise ValueError('Seed gallery overlaps evaluation checks; choose a separate seed interval')
    write_json(args.output/'plan.json', plan)
    cache=ModelOutputCache(video)
    encoder=FrozenOSNet(assets,device=args.device,threads=args.threads,
                        cuda_memory_fraction=.1 if args.device=='cuda' else None)
    store=osnet_feature_store(cache,assets,assets/'osnet_msmt17.pth',observations['width'],observations['height'])
    features=collect_embeddings(observations['frames'],
        lambda needed:video_frames(video,needed,(observations['width'],observations['height'])),
        encoder,store,args.batch_size)
    keys=[p['key'] for f in observations['frames'] for p in f['players']]
    vectors=np.stack([features[key] for key in keys])
    if not np.isfinite(vectors).all():
        raise ValueError('Non-finite embeddings')
    np.savez_compressed(args.output/'embeddings.npz', keys=np.asarray(keys), embeddings=vectors)
    provenance = {'schema': 'osnet_msmt17_full_bbox_flip_v1', 'observations_sha256': fingerprint,
                  'assets': json.loads((assets/'manifest.json').read_text()), 'code_sha256': SOURCE_SHA256,
                  'weights_sha256': WEIGHTS_SHA256, 'strict_weight_loading': True,
                  'input_hw': [256,128], 'normalization': 'ImageNet mean/std',
                  'flip_average': 'L2 each view, average, L2 again', 'device': encoder.provenance['device'],
                  'torch_version': torch.__version__, 'cpu_threads': args.threads,
                  'batch_size': args.batch_size, 'inference_dtype': 'float32',
                  'cuda_memory_fraction': .10 if args.device == 'cuda' else None,
                  'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' and encoder.provenance['model_loaded_this_run'] else None,
                  'model_cache':cache.summary(),'encoder':encoder.provenance,
                  'crops': len(keys), 'dimensions': int(vectors.shape[1]),
                  'embedding_sha256': digest(args.output/'embeddings.npz'), 'training': False}
    write_json(args.output/'embedding_manifest.json', provenance)
    cache.db.close()
    print(f'[reid] cached {vectors.shape} embeddings in {args.output.resolve()}', flush=True)


def evaluate(args):
    observations, labels, fingerprint = project_inputs(args.project)
    plan = json.loads((args.cache/'plan.json').read_text())
    metadata = json.loads((args.cache/'embedding_manifest.json').read_text())
    if plan['observations_sha256'] != fingerprint or metadata['observations_sha256'] != fingerprint:
        raise ValueError('Embedding cache does not match observations')
    if plan['seed']['assignments'] != labels['seed']['assignments'] or plan['seed']['frame'] != labels['seed']['frame']:
        raise ValueError('Seed changed after experiment plan was fixed')
    if metadata['embedding_sha256'] != digest(args.cache/'embeddings.npz'):
        raise ValueError('Embedding checksum changed')
    if args.output.exists():
        raise ValueError('Evaluation output exists; use a new file')
    data = np.load(args.cache/'embeddings.npz', allow_pickle=False)
    learned = dict(zip(data['keys'].tolist(), data['embeddings']))
    hsv = {p['key']: p['appearance'] for f in observations['frames'] for p in f['players']}
    expected_keys = set(hsv)
    if set(learned) != expected_keys or len(data['keys']) != len(expected_keys):
        raise ValueError('Learned embeddings do not cover exactly the same saved detections')
    parameters = ReIDParameters(**plan['parameters'])
    gallery = gallery_keys(observations['frames'], labels['seed'], parameters)
    if {c['key'] for c in labels['checks']}.intersection(k for keys in gallery.values() for k in keys):
        raise ValueError('Evaluation labels overlap seed galleries')
    comparison = compare(observations, labels['seed'], labels['checks'])
    comparison.update({'schema': 'player_reid_factorial_result_v1', 'observations_sha256': fingerprint,
                       'labels_sha256': digest(args.project/'labels.json'), 'embedding_manifest': metadata,
                       'experiment_plan': plan, 'development_estimate': True})
    seed_keys = set(labels['seed']['assignments'])
    for descriptor, features in [('hsv', hsv), ('osnet', learned)]:
        for recovery in (False, True):
            name = f'{descriptor}_fixed_{"recovery" if recovery else "local"}'
            run = associate(observations['frames'], labels['seed'], features, observations['fps'], recovery, parameters)
            predictions = run['predictions']
            for frame in observations['frames']:
                assigned = [predictions[p['key']]['player_id'] for p in frame['players']
                            if predictions[p['key']]['player_id'] is not None]
                if len(assigned) != len(set(assigned)):
                    raise AssertionError(f'Duplicate player ID in frame {frame["f"]}')
            checked = score(predictions, labels['checks'], seed_keys)
            per_player = {}
            for pid in sorted(labels['seed']['assignments'].values()):
                counters = Counter('correct' if predictions[c['key']]['player_id'] == pid else
                                   'unknown' if predictions[c['key']]['player_id'] is None else 'wrong'
                                   for c in labels['checks'] if c['player_id'] == pid)
                per_player[pid] = dict(counters)
            assigned = sum(p['player_id'] is not None for p in predictions.values())
            run.update({'human_checks': checked, 'per_player': per_player,
                        'diagnostics': {'observations': len(predictions), 'assigned': assigned,
                                        'coverage': assigned/max(len(predictions),1),
                                        'reasons': dict(Counter(p['reason'] for p in predictions.values()))}})
            comparison['methods'][name] = run
            print(name, json.dumps(checked), per_player, flush=True)
    write_json(args.output, comparison)
    print(f'[reid] saved {args.output.resolve()}', flush=True)


def stress_gap(args):
    """Diagnostic detection blackout; original observations are never changed."""
    observations, labels, fingerprint = project_inputs(args.project)
    if args.output.exists():
        raise ValueError('Stress output exists; use a fresh file')
    plan = json.loads((args.cache/'plan.json').read_text())
    metadata = json.loads((args.cache/'embedding_manifest.json').read_text())
    if plan['observations_sha256'] != fingerprint or metadata['embedding_sha256'] != digest(args.cache/'embeddings.npz'):
        raise ValueError('Stress cache fingerprint mismatch')
    if plan['seed']['assignments'] != labels['seed']['assignments'] or plan['seed']['frame'] != labels['seed']['frame']:
        raise ValueError('Stress seed changed from fixed plan')
    parameters = ReIDParameters(**plan['parameters'])
    fps = observations['fps']
    start = labels['seed']['frame'] + int(np.ceil(.6*fps))
    end = start + int(np.ceil(4.*fps))
    frames = [{'f': f['f'], 'players': [] if start <= f['f'] < end else f['players']}
              for f in observations['frames']]
    if gallery_keys(frames, labels['seed'], parameters) != plan['gallery_keys']:
        raise ValueError('Blackout must not alter initial reference galleries')
    data = np.load(args.cache/'embeddings.npz', allow_pickle=False)
    features = dict(zip(data['keys'].tolist(), data['embeddings']))
    checks = [c for c in labels['checks'] if c['frame'] >= end]
    if not checks:
        raise ValueError('No human checks after the fixed blackout')
    result = {'schema': 'player_reid_synthetic_blackout_v1', 'observations_sha256': fingerprint,
              'labels_sha256': digest(args.project/'labels.json'), 'synthetic': True,
              'blackout_source_frames': [start,end], 'blackout_end_exclusive': True,
              'blackout_seconds': (end-start)/fps,
              'post_blackout_checks': len(checks), 'excluded_checks': len(labels['checks'])-len(checks),
              'selection': 'All existing human checks after blackout; masked checks excluded for both methods',
              'parameters': parameters.to_dict(), 'methods': {}}
    for recover in (False,True):
        name = f'osnet_fixed_{"recovery" if recover else "local"}'
        run = associate(frames, labels['seed'], features, fps, recover, parameters)
        run['human_checks'] = score(run['predictions'], checks, set(labels['seed']['assignments']))
        result['methods'][name] = run
        print('[blackout]',name,json.dumps(run['human_checks']),flush=True)
    write_json(args.output,result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    download = sub.add_parser('fetch-assets', help='Explicitly download pinned official OSNet source, MIT license and weights')
    download.add_argument('--output', type=Path, required=True, help='New assets directory')
    cache = sub.add_parser('embed', help='Cache frozen, pinned OSNet features without training')
    cache.add_argument('--project', required=True, type=Path)
    cache.add_argument('--assets', required=True, type=Path)
    cache.add_argument('--output', required=True, type=Path, help='New cache directory')
    cache.add_argument('--batch-size', type=int, default=32)
    cache.add_argument('--threads', type=int, default=4)
    cache.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    evaluation = sub.add_parser('evaluate', help='Run fixed factorial comparison; human checks score only')
    evaluation.add_argument('--project', required=True, type=Path)
    evaluation.add_argument('--cache', required=True, type=Path)
    evaluation.add_argument('--output', required=True, type=Path, help='New comparison JSON file')
    stress = sub.add_parser('stress-gap', help='Diagnostic four-second detection blackout in memory; score post-gap checks only')
    stress.add_argument('--project',required=True,type=Path)
    stress.add_argument('--cache',required=True,type=Path)
    stress.add_argument('--output',required=True,type=Path,help='New diagnostic JSON file')
    args = parser.parse_args()
    if args.command == 'fetch-assets':
        fetch_assets(args)
    elif args.command == 'embed':
        embed(args)
    elif args.command == 'stress-gap':
        stress_gap(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
