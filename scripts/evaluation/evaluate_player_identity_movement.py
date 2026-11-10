"""Score roster identities against human movement review; optional cache-only method replay.

Saved results are scored as they are. ``--replay`` also runs the identity
methods on the saved poses and cached OSNet embeddings (no inference, no video
decode). Human labels are used for scoring only, except in the controls that
are explicitly named ``human_seed``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.player_reid_encoder import file_sha256
from src.tracking.player_identity_scoring import human_labels, score


def saved_predictions(annotations: dict, side: str) -> dict:
    return {f'{row["f"]}:{index}': pose.get('player_id')
            for row in annotations['frames'] for index, pose in enumerate(row.get('poses') or [])
            if pose.get('team') in (side, 'team_'+side)}


def check_binding(path: Path, annotations: dict, movement: dict):
    """Pose indices in the review only mean something for the reviewed analysis."""
    reviewed = movement['annotations_source']['sha256']
    if file_sha256(path) == reviewed:
        return 'reviewed_analysis'
    if (annotations.get('player_identity') or {}).get('input_annotations_sha256') == reviewed:
        return 'identity_replay_of_reviewed_analysis'
    raise ValueError(f'{path} is neither the reviewed analysis nor an identity replay of it')


def replay(annotations: dict, labels: dict, side: str, cache_dir: Path) -> dict:
    from src.models.inference_cache import ModelOutputCache, collect_embeddings, osnet_feature_store
    from src.models.player_reid_encoder import DEFAULT_ASSETS
    from src.tracking.player_identity import default_seed
    from src.tracking.player_identity_workflow import (
        DISCOVERY_PARAMETERS, PRODUCTION_PARAMETERS, observations_from_annotations)
    from src.tracking.player_reid import associate
    from src.tracking.player_roster_discovery import assign_roster, discover_roster, name_left_to_right

    fps = annotations['fps']
    frames = observations_from_annotations(annotations['frames'], side)
    assets = DEFAULT_ASSETS if DEFAULT_ASSETS.is_absolute() else ROOT/DEFAULT_ASSETS
    cache = ModelOutputCache(Path(annotations['original_video_path']), cache_dir, require=True)
    store = osnet_feature_store(cache, assets, assets/'osnet_msmt17.pth',
                                annotations['frame_width'], annotations['frame_height'])
    features = collect_embeddings(frames, [], None, store)
    cache.db.close()
    by_key = {obs['key']: (frame['f'], obs) for frame in frames for obs in frame['players']}
    runs = {}

    def record(name, predictions, started, mapping=None, **extra):
        runs[name] = {**score({k: v['player_id'] for k, v in predictions.items()}, labels, fps, mapping),
                      'seconds': round(time.time()-started, 2), **extra}

    started = time.time()
    seed = default_seed(frames, 6)
    result = associate(frames, seed, features, fps, True, PRODUCTION_PARAMETERS)
    record('seed_frame_automatic', result['predictions'], started, seed_frame=seed['frame'])

    # Control: the same tracker with a correct human roster on one frame.
    labelled = defaultdict(dict)
    for key, pid in labels['players'].items():
        labelled[by_key[key][0]][key] = pid
    people = set(labels['players'].values())
    complete = sorted((sum(by_key[k][1]['occlusion'] for k in keys), f) for f, keys in labelled.items()
                      if set(keys.values()) == people
                      and len(next(fr for fr in frames if fr['f'] == f)['players']) == len(people))
    picks = {}
    if complete:
        ordered = sorted(f for _, f in complete)
        picks = {'earliest': ordered[0], 'cleanest': complete[0][1], 'median': ordered[len(ordered)//2]}
    for name, frame in picks.items():
        started = time.time()
        seed = {'frame': frame, 'assignments': dict(labelled[frame]), 'label_source': 'human', 'verified': True}
        result = associate(frames, seed, features, fps, True, PRODUCTION_PARAMETERS)
        record(f'seed_frame_human_seed_{name}', result['predictions'], started,
               {pid: pid for pid in people}, seed_frame=frame)

    started = time.time()
    found = discover_roster(frames, features, fps, 6, DISCOVERY_PARAMETERS)
    seed = name_left_to_right(frames, found['roster'])
    predictions = assign_roster(frames, features, fps, found['roster'], DISCOVERY_PARAMETERS,
                                cohort=found['cohort'])
    record('roster_discovery', predictions, started, reference_frame=seed['frame'],
           people_found=found['people'], tracklets=found['tracklets'],
           roster_presence_frames={p['player_id']: p['presence_frames'] for p in found['roster']})

    # Control: whole-video roster and gallery, but the sequential tracker.
    started = time.time()
    anchors = {pid: key for key, pid in seed['assignments'].items()}
    gallery = {p['player_id']: [anchors[p['player_id']]]
               + [k for k in p['gallery_keys'][::max(1, len(p['gallery_keys'])//32)]
                  if k != anchors[p['player_id']]] for p in found['roster']}
    gallery = {pid: gallery[pid] for pid in seed['assignments'].values()}
    result = associate(frames, seed, features, fps, True, PRODUCTION_PARAMETERS, frozen_gallery=gallery)
    record('seed_frame_with_discovered_roster_gallery', result['predictions'], started, seed_frame=seed['frame'])
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--movement', type=Path, required=True, help='human_player_movement_v1 export')
    parser.add_argument('--annotations', type=Path, nargs='+', required=True,
                        help='Saved <stem>_annotations.json results to score (the reviewed analysis or identity replays of it)')
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for metrics.json')
    parser.add_argument('--side', choices=('a', 'b'), default='b')
    parser.add_argument('--replay', action='store_true',
                        help='Also run the identity methods on the first annotations file from cached embeddings')
    parser.add_argument('--model-cache-dir', type=Path, default=ROOT/'data'/'inference_cache')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh directory')
    movement = json.loads(args.movement.read_text(encoding='utf-8'))
    labels = human_labels(movement)
    report = {'schema': 'player_identity_movement_score_v1', 'movement': str(args.movement.resolve()),
              'movement_sha256': file_sha256(args.movement), 'revision': movement.get('revision'),
              'video': labels['video'], 'window_frames': labels['window'],
              'scored_poses': len(labels['players']), 'duplicate_poses_left_out': len(labels['duplicates']),
              'protocol': {
                  'join': 'verified human player labels joined to saved poses by frame and pose index',
                  'naming': 'one-to-one model-ID to human-ID naming that maximises agreement, unless a run fixes it',
                  'scope': 'reviewed near-side poses of one development video; missed detections are not scored'},
              'saved': {}, 'replay': {}}
    first = None
    for path in args.annotations:
        annotations = json.loads(path.read_text(encoding='utf-8'))
        binding = check_binding(path, annotations, movement)
        first = first or annotations
        result = score(saved_predictions(annotations, args.side), labels, annotations['fps'])
        result['binding'] = binding
        result['method'] = (annotations.get('player_identity') or {}).get('method')
        report['saved'][str(path)] = result
        print(f'{path.parent.name}: {result["method"]} correct {result["correct_rate"]:.3f} '
              f'wrong {result["wrong_rate"]:.3f} unknown {result["unknown_rate"]:.3f}', flush=True)
    if args.replay:
        report['replay'] = replay(first, labels, args.side, args.model_cache_dir)
        for name, result in report['replay'].items():
            print(f'{name}: correct {result["correct_rate"]:.3f} wrong {result["wrong_rate"]:.3f} '
                  f'unknown {result["unknown_rate"]:.3f} ({result["seconds"]} s)', flush=True)
    args.output.mkdir(parents=True)
    with (args.output/'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
