"""Fresh cached replay must preserve nonidentity analysis and bind exact sources."""
import json
import sys

import numpy as np
import pytest

from scripts.evaluation import run_player_roster_experiment as replay
from src.models.player_reid_encoder import SOURCE_SHA256, WEIGHTS_SHA256, file_sha256
from src.tracking.player_identity_workflow import observations_from_annotations


def test_cached_roster_replay_preserves_ball_events_raw_poses_and_refuses_replacement(tmp_path, monkeypatch):
    video = tmp_path/'source.webm'
    video.write_bytes(b'fingerprinted source, no inference in replay')
    initial = [{'id': i, 'team': 'b', 'bb': [i*60, 0, i*60+30, 100],
                'player_id': f'P{i+1}', 'identity': {'reason': 'old'}, 'kp': [[1, 2, .9]]}
               for i in range(6)]
    newcomer = {'id': 99, 'team': 'b', 'bb': [500, 0, 530, 100],
                'player_id': 'P1', 'identity': {'reason': 'old'}}
    data = {'video': 'source', 'original_video_path': str(video), 'fps': 30.,
            'frame_width': 640, 'frame_height': 360,
            'frames': [{'f': 0, 'poses': initial, 'ball': [12, 20], 'ev': [{'type': 'serve'}]},
                       {'f': 127, 'poses': [newcomer], 'ball': None, 'ball_out_of_frame': True}],
            'player_identity': {'method': 'old'}}
    annotations = tmp_path/'source_annotations.json'
    annotations.write_text(json.dumps(data))
    stats = {'all_events': [{'type': 'serve', 'team': None}], 'rallies': [{'id': 1}],
             'models': {'pose': 'unchanged'}, 'processing_time_sec': 10}
    (tmp_path/'source_stats.json').write_text(json.dumps(stats))
    roster = tmp_path/'roster.json'
    roster.write_text(json.dumps({
        'schema': 'player_identity_roster_v2', 'video_sha256': file_sha256(video),
        'side': 'b', 'max_on_court': 6,
        'initial': {'frame': 0, 'label_source': 'automatic_initialisation', 'verified': False,
                    'players': [{'player_id': p['player_id'], 'bb': p['bb']} for p in initial]},
        'introductions': [{'player_id': 'P7', 'anchor': {'frame': 127, 'bb': newcomer['bb']},
                           'available_from_frame': 120,
                           'identity_confirmation': {'label_source': 'human', 'verified': True,
                                                     'scope': 'new_person', 'note': 'Confirmed substitute'}}]}))
    cache = tmp_path/'cache'
    cache.mkdir()
    frames = observations_from_annotations(data['frames'])
    (cache/'observations.json').write_text(json.dumps({'fps': 30., 'frames': frames}))
    keys = [p['key'] for f in frames for p in f['players']]
    np.savez_compressed(cache/'embeddings.npz', keys=np.asarray(keys), embeddings=np.eye(7, 512, dtype=np.float32))
    fingerprint = file_sha256(annotations)
    (cache/'plan.json').write_text(json.dumps({
        'schema': 'player_reid_embedding_cache_v1', 'side': 'b', 'annotations_sha256': fingerprint,
        'video_sha256': file_sha256(video), 'source_frames': 2, 'width': 640, 'height': 360}))
    (cache/'manifest.json').write_text(json.dumps({
        'annotations_sha256': fingerprint, 'observations_sha256': file_sha256(cache/'observations.json'),
        'embedding_sha256': file_sha256(cache/'embeddings.npz'), 'elapsed_seconds': 1,
        'encoder': {'weights_sha256': WEIGHTS_SHA256, 'code_sha256': SOURCE_SHA256,
                    'strict_weight_loading': True, 'training': False}}))
    output = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['replay', '--annotations', str(annotations), '--cache', str(cache),
                                    '--roster', str(roster), '--output', str(output)])
    replay.main()
    result = json.loads((output/'source_annotations.json').read_text())
    assert result['frames'][1]['poses'][0]['player_id'] == 'P7'
    assert result['player_identity']['roster_size'] == 7
    assert result['player_identity']['max_on_court'] == 6
    assert replay.nonidentity_digest(result) == replay.nonidentity_digest(data)
    result_stats = json.loads((output/'source_stats.json').read_text())
    assert {k: result_stats[k] for k in stats} == stats
    assert file_sha256(annotations) == fingerprint
    with pytest.raises(SystemExit):
        replay.main()
    (cache/'observations.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', [*sys.argv[:-1], str(tmp_path/'another_result')])
    with pytest.raises(ValueError, match='fingerprints'):
        replay.main()
