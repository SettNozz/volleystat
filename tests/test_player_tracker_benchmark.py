"""Association experiment contracts: fixed observations and honest identity scores."""
from copy import deepcopy

import numpy as np
import pytest

from src.tracking.player_tracker_benchmark import (
    BOXMOT_VERSION, FixedPoseTracker, align_ids, replay_tracks, replace_raw_ids, tracking_metrics)


def labels(players):
    return {'players': players, 'excluded': {}, 'duplicates': {}}


def test_backend_indices_preserve_observations_and_ignore_unmatched_predictions():
    assert align_ids(['9:2', '9:7', '9:8'], [12, 30, 80], [1, -1, 0]) == {
        '9:2': '80', '9:7': '12', '9:8': None}


@pytest.mark.parametrize('ids,indices', [([1], [2]), ([1, 2], [0, 0]),
                                      ([1, 1], [0, 1]), ([1], [-2]),
                                      ([1], [0.5]), ([-1], [0]), ([1], [])])
def test_invalid_backend_binding_fails(ids, indices):
    with pytest.raises(ValueError):
        align_ids(['0:0', '0:1'], ids, indices)


def test_frame_gaps_age_tracker_without_creating_detections():
    class Backend:
        def __init__(self):
            self.counts = []

        def update(self, players, features):
            self.counts.append(len(players))
            return {p['key']: '1' for p in players}

    backend = Backend()
    frames = [{'f': 5, 'players': [{'key': '5:0'}]}, {'f': 8, 'players': []},
              {'f': 9, 'players': [{'key': '9:0'}]}]
    assert replay_tracks(frames, {}, backend) == {'5:0': '1', '9:0': '1'}
    assert backend.counts == [1, 0, 0, 0, 1]
    with pytest.raises(ValueError, match='ordered'):
        replay_tracks(list(reversed(frames)), {}, Backend())


def test_tracklet_changes_do_not_mutate_geometry_or_source_labels():
    frames = [{'f': 0, 'players': [{'key': '0:3', 'raw_id': 5, 'bb': [1, 2, 3, 4],
                                   'pose_index': 3, 'occlusion': .2}]}]
    frozen = deepcopy(frames)
    changed = replace_raw_ids(frames, {'0:3': '900'})
    assert frames == frozen
    assert changed[0]['players'][0] == {**frozen[0]['players'][0], 'raw_id': '900'}


def test_conditional_idf1_penalises_fragmentation_with_one_global_mapping():
    gold = labels({'0:0': 'P1', '1:0': 'P1', '2:0': 'P1', '3:0': 'P1'})
    split = tracking_metrics({'0:0': '1', '1:0': '1', '2:0': '2', '3:0': '2'}, gold, 30)
    assert split['observation_idf1'] == .5
    assert split['idtp'] == split['idfp'] == split['idfn'] == 2
    assert split['adjacent_id_switches'] == 1
    assert split['distinct_track_ids'] == 2


def test_unknown_ids_stay_in_denominator_and_missing_keys_fail():
    gold = labels({'0:0': 'P1', '1:0': 'P1'})
    result = tracking_metrics({'0:0': '99', '1:0': None}, gold, 30)
    assert result['observation_idf1'] == pytest.approx(2/3)
    assert result['unknown_rate'] == .5
    with pytest.raises(ValueError, match='absent'):
        tracking_metrics({'0:0': '99'}, gold, 30)


@pytest.mark.parametrize('name', ['bytetrack', 'botsort', 'deepocsort'])
def test_real_backend_keeps_detection_join_and_survives_occlusion(name, monkeypatch):
    if name != 'bytetrack':
        boxmot = pytest.importorskip('boxmot')
        if boxmot.__version__ != BOXMOT_VERSION:
            pytest.skip('Optional benchmark runtime version differs')
    # Embeddings are supplied; no model download or inference is permitted.
    import socket
    def offline(*args, **kwargs):
        raise OSError('Network disabled for cached tracker test')
    monkeypatch.setattr(socket, 'create_connection', offline)
    backend = FixedPoseTracker(name, (640, 480))
    features, frames = {}, []
    for f in range(12):
        players = []
        if f != 6:
            for index, x in enumerate((20+f*2, 300-f*2)):
                key = f'{f}:{index}'
                players.append({'key': key, 'bb': [x, 30, x+60, 220]})
                vector = np.zeros(512, dtype=np.float32)
                vector[index] = 1.
                features[key] = vector
        frames.append({'f': f, 'players': players})
    ids = replay_tracks(frames, features, backend)
    assert set(ids) == set(features)
    for index in (0, 1):
        assert ids[f'0:{index}'] is not None
        assert ids[f'0:{index}'] == ids[f'5:{index}'] == ids[f'11:{index}']
        # Deep OC-SORT can withhold output until the track is reconfirmed.
        assert ids[f'7:{index}'] in (None, ids[f'0:{index}'])
    assert ids['0:0'] != ids['0:1']
