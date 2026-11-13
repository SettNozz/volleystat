"""Compare ByteTrack, BoT-SORT and Deep OC-SORT on fixed saved poses and cached OSNet.

Human movement labels score predictions only. This is an association development
experiment, not a full-pipeline MOT benchmark or permission to promote a tracker.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--movement', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True,
                        help='Exact reviewed analysis; SHA256 binding is mandatory')
    parser.add_argument('--output', type=Path, required=True, help='Fresh experiment directory')
    parser.add_argument('--side', choices=('a', 'b'), default='b')
    parser.add_argument('--model-cache-dir', type=Path, default=ROOT/'data/inference_cache')
    parser.add_argument('--tracker-runtime-dir', type=Path,
                        help='Optional isolated package directory containing boxmot==26.0.0')
    parser.add_argument('--trackers', nargs='+', choices=('bytetrack', 'botsort', 'deepocsort'),
                        default=['bytetrack', 'botsort', 'deepocsort'])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh directory')
    if len(set(args.trackers)) != len(args.trackers):
        parser.error('Tracker names must be unique')
    if args.tracker_runtime_dir:
        sys.path.insert(0, str(args.tracker_runtime_dir.resolve()))

    from src.models.inference_cache import ModelOutputCache, collect_embeddings, osnet_feature_store
    from src.models.player_reid_encoder import DEFAULT_ASSETS, file_sha256
    from src.tracking.player_identity_scoring import human_labels, score
    from src.tracking.player_identity_workflow import observations_from_annotations
    from src.tracking.player_tracker_benchmark import (
        FixedPoseTracker, replay_tracks, replace_raw_ids, roster_predictions, tracking_metrics)

    movement = json.loads(args.movement.read_text(encoding='utf-8'))
    if file_sha256(args.annotations) != movement['annotations_source']['sha256']:
        raise ValueError('Annotations differ from the human-reviewed source')
    annotations = json.loads(args.annotations.read_text(encoding='utf-8'))
    labels = human_labels(movement)
    frames = observations_from_annotations(annotations['frames'], args.side)
    keys = {p['key'] for f in frames for p in f['players']}
    if not set(labels['players']).issubset(keys):
        raise ValueError('Reviewed poses are absent from the selected court side')
    if not labels['players']:
        raise ValueError('No verified human player labels')
    fps = annotations['fps']
    size = (annotations['frame_width'], annotations['frame_height'])
    video = Path(annotations['original_video_path'])
    if file_sha256(video) != movement['video_source']['sha256']:
        raise ValueError('Video differs from the human-reviewed source')
    assets = DEFAULT_ASSETS if DEFAULT_ASSETS.is_absolute() else ROOT/DEFAULT_ASSETS
    cache = ModelOutputCache(video, args.model_cache_dir, require=True)
    try:
        store = osnet_feature_store(cache, assets, assets/'osnet_msmt17.pth', *size)
        features = collect_embeddings(frames, [], None, store)
        cache_report = cache.summary()
        cache_report['embedding_stage'] = store.stage
    finally:
        cache.db.close()
    print(f'Frozen inputs: {len(frames)} frames, {len(keys)} poses, {len(labels["players"])} human labels; '
          f'{len(features)} cached embeddings', flush=True)

    backends = {name: FixedPoseTracker(name, size) for name in args.trackers}
    args.output.mkdir(parents=True)
    plan = {
        'schema': 'fixed_pose_tracker_comparison_v1',
        'annotations': str(args.annotations.resolve()), 'annotations_sha256': file_sha256(args.annotations),
        'movement': str(args.movement.resolve()), 'movement_sha256': file_sha256(args.movement),
        'human_revision': movement.get('revision'), 'video_sha256': movement['video_source']['sha256'],
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in (
            Path(__file__), ROOT/'src/tracking/player_tracker_benchmark.py',
            ROOT/'src/tracking/player_identity_workflow.py', ROOT/'src/tracking/player_roster_discovery.py',
            ROOT/'src/tracking/player_identity_scoring.py')},
        'runtime': {name: metadata.version(name) for name in ('numpy', 'torch', 'ultralytics')},
        'side': args.side, 'fps': fps, 'processed_frames': len(frames),
        'scoring_window': labels['window'], 'scored_poses': len(labels['players']),
        'duplicates_excluded': len(labels['duplicates']), 'cache': cache_report,
        'configs': {name: backend.config for name, backend in backends.items()},
        'protocol': {
            'inputs': 'fixed court-side saved pose boxes, already selected by the original production tracker',
            'confidence': 'constant 1.0 for every input; original detector confidence is not exported',
            'appearance': 'identical existing pinned OSNet crop embeddings; no inference or decoding',
            'geometry': 'retain original boxes/keypoints/pose indices; use only backend detection indices for IDs',
            'camera_motion_compensation': False,
            'labels': 'verified human movement labels used only for scoring, never association or roster seeding',
            'naming': 'global one-to-one ID permutation maximising agreement; no temporal remapping',
            'idf1': 'conditional on reviewed observed positives; ignores unreviewed and excluded observations',
            'roster': 'current automatic roster discovery with production short-clip fallback',
            'limits': ['one development video; no held-out promotion evidence',
                       'does not measure missed detections, HOTA or full MOT IDF1',
                       'does not replay court gating, membership, substitutions, contact authors or events',
                       'fixed side and constant confidence differ from pre-filter production tracking'],
            'promotion': 'keep production unchanged; a candidate needs raw-cache full-pipeline validation'}
    }
    if any(name != 'bytetrack' for name in backends):
        import boxmot
        plan['runtime']['boxmot'] = boxmot.__version__
    write_json(args.output/'plan.json', plan)
    saved_tracks = {p['key']: str(p['raw_id']) if p['raw_id'] is not None else None
                    for f in frames for p in f['players']}
    saved_roster = {f'{f["f"]}:{i}': p.get('player_id') for f in annotations['frames']
                    for i, p in enumerate(f.get('poses') or []) if f'{f["f"]}:{i}' in keys}
    report = {'schema': plan['schema'], 'production_changed': False, 'runs': {},
              'saved_production': {'tracking': tracking_metrics(saved_tracks, labels, fps),
                                   'roster': score(saved_roster, labels, fps)}}

    def record(name, predictions, seconds, config=None):
        started = time.perf_counter()
        roster, provenance = roster_predictions(replace_raw_ids(frames, predictions), features, fps)
        result = {'tracking': tracking_metrics(predictions, labels, fps),
                  'roster': score(roster, labels, fps), 'roster_provenance': provenance,
                  'tracking_seconds': round(seconds, 3),
                  'roster_seconds': round(time.perf_counter()-started, 3), 'config': config}
        write_json(args.output/(name+'_predictions.json'), {'raw_ids': predictions, 'player_ids': roster})
        write_json(args.output/(name+'_metrics.json'), result)
        report['runs'][name] = result
        print(f'{name}: observation IDF1={result["tracking"]["observation_idf1"]:.4f}; '
              f'roster correct={result["roster"]["correct_rate"]:.4f}, '
              f'wrong={result["roster"]["wrong_rate"]:.4f}, '
              f'unknown={result["roster"]["unknown_rate"]:.4f}', flush=True)

    record('saved_tracks_current_roster', saved_tracks, 0.)
    for name, backend in backends.items():
        started = time.perf_counter()
        predictions = replay_tracks(frames, features, backend,
                                    lambda frame: print(f'{name}: frame {frame}', flush=True))
        record(name, predictions, time.perf_counter()-started, backend.config)
    baseline = report['runs']['saved_tracks_current_roster']['roster']
    for name in backends:
        current = report['runs'][name]['roster']
        report['runs'][name]['delta_vs_current_roster'] = {
            k: current[k]-baseline[k] for k in ('correct_rate', 'wrong_rate', 'unknown_rate')}
    write_json(args.output/'metrics.json', report)
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
