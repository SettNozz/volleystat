"""Whole-video roster discovery: constant players, not whoever stands in one frame."""
from copy import deepcopy

import numpy as np
import pytest

from src.tracking.player_identity_scoring import best_mapping, human_labels, score
from src.tracking.player_identity_workflow import METHOD, SEED_FRAME_METHOD, run_identity_pass
from src.tracking.player_roster_discovery import (
    RosterDiscoveryParameters, RosterNotFound, assign_roster, cluster_people, discover_roster,
    merge_fragments, name_left_to_right, tracklet_chunks)

FPS = 30.
PARAMETERS = RosterDiscoveryParameters()


def looks(count, seed=0, dim=48):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return basis[:count]


def scene(length=240, seed=1):
    """Five players and a bystander at the start; the sixth player arrives after two seconds.

    Detector IDs change every 80 frames, and player 2 leaves for two seconds.
    """
    rng = np.random.default_rng(seed)
    base = looks(7)
    frames, features, truth = [], {}, {}
    for f in range(length):
        players = []
        for person in range(7):
            if person == 5 and f < 60 or person == 6 and f > 40 or person == 2 and 100 <= f < 160:
                continue
            x = 100+person*300+(900 if person == 6 else 0)
            key = f'{f}:{len(players)}'
            raw = person*100+f//80+(50 if person == 2 and f >= 160 else 0)
            players.append({'key': key, 'pose_index': len(players), 'bb': [x, 200, x+100, 500],
                            'raw_id': raw, 'occlusion': 0.})
            features[key] = base[person]+rng.normal(scale=.03, size=base.shape[1])
            truth[key] = person
        frames.append({'f': f, 'players': players})
    return frames, features, truth


def test_roster_is_the_constant_players_not_the_people_in_the_first_frame():
    frames, features, truth = scene()
    assert len(frames[0]['players']) == 6 and 6 in {truth[p['key']] for p in frames[0]['players']}
    found = discover_roster(frames, features, FPS, 6, PARAMETERS)
    seed = name_left_to_right(frames, found['roster'])
    assert [p['player_id'] for p in found['roster']] == [f'P{i}' for i in range(1, 7)]
    assert seed['frame'] == 60 and seed['verified'] is False
    predictions = assign_roster(frames, features, FPS, found['roster'], PARAMETERS)
    names = {}
    for key, person in truth.items():
        pid = predictions[key]['player_id']
        if person == 6:
            assert pid is None  # A bystander never becomes a roster player.
        else:
            assert names.setdefault(person, pid) == pid and pid is not None
    assert sorted(names.values()) == [f'P{i}' for i in range(1, 7)]
    assert names == {person: f'P{person+1}' for person in range(6)}  # Left to right.
    assert found['other_people'] and found['other_people'][0]['presence_frames'] == 41


def test_identity_survives_new_detector_ids_and_a_long_absence():
    frames, features, truth = scene()
    found = discover_roster(frames, features, FPS, 6, PARAMETERS)
    name_left_to_right(frames, found['roster'])
    predictions = assign_roster(frames, features, FPS, found['roster'], PARAMETERS)
    returned = [predictions[key]['player_id'] for key, person in truth.items()
                if person == 2 and int(key.split(':')[0]) in (99, 160, 239)]
    assert returned == ['P3', 'P3', 'P3']


def test_tracklets_visible_together_never_merge_even_when_they_look_alike():
    base = looks(1)[0]
    rng = np.random.default_rng(3)
    frames, features = [], {}
    for f in range(60):
        players = []
        for person in range(2):
            key = f'{f}:{person}'
            players.append({'key': key, 'pose_index': person, 'bb': [person*300, 0, person*300+100, 300],
                            'raw_id': person, 'occlusion': 0.})
            features[key] = base+rng.normal(scale=.01, size=base.size)
        frames.append({'f': f, 'players': players})
    chunks = tracklet_chunks(frames, features, FPS, PARAMETERS)
    groups = cluster_people(chunks, PARAMETERS)
    assert len(groups) == 2
    assert all(len({chunks[i]['raw'] for i in group}) == 1 for group in groups)


def test_detector_track_that_changes_person_gets_both_names():
    frames, features, truth = scene()
    # Players 0 and 1 swap detector IDs half-way: the track is not an identity.
    for frame in frames[120:]:
        for obs in frame['players']:
            if truth[obs['key']] in (0, 1):
                obs['raw_id'] = (1-truth[obs['key']])*100+1
    found = discover_roster(frames, features, FPS, 6, PARAMETERS)
    name_left_to_right(frames, found['roster'])
    predictions = assign_roster(frames, features, FPS, found['roster'], PARAMETERS)
    scored = [(truth[key], predictions[key]['player_id']) for key in truth if truth[key] in (0, 1)]
    assigned = [(person, pid) for person, pid in scored if pid is not None]
    assert all(pid == f'P{person+1}' for person, pid in assigned)
    assert len(assigned) >= .9*len(scored)  # Only the frames around the swap may stay unknown.


def test_two_boxes_of_one_person_share_one_id():
    frames, features, truth = scene()
    frame = frames[200]
    twin = deepcopy(frame['players'][0])
    twin.update(key='200:99', pose_index=99, raw_id=999)
    frame['players'].append(twin)
    features['200:99'] = features[frame['players'][0]['key']]
    found = discover_roster(frames, features, FPS, 6, PARAMETERS)
    name_left_to_right(frames, found['roster'])
    predictions = assign_roster(frames, features, FPS, found['roster'], PARAMETERS)
    ids = [predictions[obs['key']]['player_id'] for obs in frame['players']]
    assert len([pid for pid in ids if pid is not None]) == len({pid for pid in ids if pid is not None})
    assert ids.count('P1') == 1


def test_confirmed_anchors_choose_and_name_the_people():
    frames, features, truth = scene()
    anchors = [{'key': obs['key'], 'player_id': f'N{truth[obs["key"]]}'} for obs in frames[70]['players']]
    found = discover_roster(frames, features, FPS, 6, PARAMETERS, anchors=anchors)
    assert {p['player_id'] for p in found['roster']} == {f'N{i}' for i in range(6)}
    predictions = assign_roster(frames, features, FPS, found['roster'], PARAMETERS,
                                forced={a['key']: a['player_id'] for a in anchors},
                                available_from={'N4': 30})
    assert all(predictions[a['key']]['reason'] == 'roster_anchor' for a in anchors)
    assert all(predictions[key]['player_id'] in (f'N{person}', None) for key, person in truth.items() if person < 6)
    early = [predictions[key]['player_id'] for key, person in truth.items()
             if person == 4 and int(key.split(':')[0]) < 30]
    assert early and set(early) == {None}  # Not available before the confirmed frame.
    same_person = [{'key': frames[70]['players'][0]['key'], 'player_id': 'A'},
                   {'key': frames[75]['players'][0]['key'], 'player_id': 'B'}]
    with pytest.raises(RosterNotFound, match='one person'):
        discover_roster(frames, features, FPS, 6, PARAMETERS, anchors=same_person)


def test_too_few_constant_players_is_reported_not_padded():
    frames, features, truth = scene()
    for frame in frames:
        frame['players'] = [obs for obs in frame['players'] if truth[obs['key']] not in (4, 5)]
    with pytest.raises(RosterNotFound, match='Only 4 of 6'):
        discover_roster(frames, features, FPS, 6, PARAMETERS)


class ColourEncoder:
    provenance = {'method': 'test_encoder', 'weights_sha256': 'test'}
    crop = staticmethod(lambda image, bb: image[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])])

    def encode(self, crops):
        return np.stack([np.eye(8, dtype=np.float32)[int(c[0, 0, 0])-1] for c in crops])


def annotation_fixture(length):
    image = np.zeros((200, 700, 3), np.uint8)
    poses = []
    for i in range(6):
        bb = [i*90+5, 20, i*90+65, 180]
        image[20:180, bb[0]:bb[2]] = i+1
        poses.append({'id': i+1, 'team': 'b', 'bb': bb, 'kp': [[bb[0]+20., 40., .9]], 'source': 'yolo_pose'})
    frames = [{'f': f, 'ball': None, 'ev': ['pass'], 'poses': deepcopy(poses)} for f in range(length)]
    return frames, image


def test_workflow_uses_whole_video_roster_and_keeps_raw_fields():
    frames, image = annotation_fixture(120)
    # A seventh person stands on court in the first frames only.
    image[20:180, 600:660] = 7
    for frame in frames[:4]:
        frame['poses'].append({'id': 77, 'team': 'b', 'bb': [600, 20, 660, 180], 'kp': [], 'source': 'yolo_pose'})
    original = deepcopy(frames)
    report = run_identity_pass(frames, [(f, image) for f in range(120)], FPS, ColourEncoder())
    assert report['status'] == 'completed' and report['method'] == METHOD
    assert report['roster_discovery']['selection'] == 'longest_presence'
    assert set(report['gallery_keys']) == {f'P{i}' for i in range(1, 7)}
    assert set(report['seed']['assignments'].values()) == set(report['gallery_keys'])
    assert report['seed']['label_source'] == 'automatic_initialisation' and report['verified'] is False
    for before, after in zip(original, frames):
        assert before == {**after, 'poses': [{k: v for k, v in p.items() if k not in ('player_id', 'identity')}
                                            for p in after['poses']]}
        assert [p['player_id'] for p in after['poses'][:6]] == [f'P{i}' for i in range(1, 7)]
        assert all(p['player_id'] is None for p in after['poses'][6:])
        assert all(p['identity']['method'] == METHOD and p['identity']['verified'] is False for p in after['poses'])


def test_workflow_falls_back_to_the_seed_frame_tracker_on_a_short_clip():
    frames, image = annotation_fixture(2)
    report = run_identity_pass(frames, [(0, image), (1, image)], FPS, ColourEncoder())
    assert report['method'] == SEED_FRAME_METHOD
    assert report['roster_discovery']['status'] == 'fallback_to_seed_frame'
    forced = run_identity_pass(annotation_fixture(120)[0], [(f, image) for f in range(120)], FPS, ColourEncoder(),
                               method='seed-frame')
    assert forced['method'] == SEED_FRAME_METHOD and 'roster_discovery' not in forced


def movement():
    return {'schema': 'human_player_movement_v1', 'roster': {'players': [
        {'player_id': 'A', 'references': []}, {'player_id': 'B', 'references': []}]},
        'videos': {'clip': {'spans': [
            {'start': 0, 'end': 3, 'decision': 'player', 'player_id': 'A', 'label_source': 'human',
             'verified': True, 'revision': 1, 'pose_indices': [0, 0, 0, 0]},
            {'start': 0, 'end': 3, 'decision': 'player', 'player_id': 'B', 'label_source': 'human',
             'verified': True, 'revision': 2, 'pose_indices': [1, 1, 1, 1]},
            {'start': 2, 'end': 2, 'decision': 'player', 'player_id': 'A', 'label_source': 'human',
             'verified': True, 'revision': 3, 'pose_indices': [2], 'duplicate_frames': [2]},
            {'start': 0, 'end': 1, 'decision': 'other', 'player_id': None, 'label_source': 'human',
             'verified': True, 'revision': 4, 'pose_indices': [3, 3]},
            {'start': 3, 'end': 3, 'decision': 'unsure', 'player_id': None, 'label_source': 'human',
             'verified': False, 'revision': 5, 'pose_indices': [3]}]}}}


def test_scoring_counts_one_pose_per_player_and_names_ids_one_to_one():
    labels = human_labels(movement())
    assert len(labels['players']) == 8 and labels['duplicates'] == {'2:2': 'A'}
    assert labels['excluded'] == {'0:3': 'other', '1:3': 'other', '3:3': 'unsure'}
    predictions = {'0:0': 'P2', '1:0': 'P2', '2:0': None, '3:0': 'P1',
                   '0:1': 'P1', '1:1': 'P1', '2:1': 'P1', '3:1': 'P1',
                   '2:2': 'P2', '0:3': 'P2', '1:3': None, '3:3': None}
    assert best_mapping(predictions, labels['players']) == {'P1': 'B', 'P2': 'A'}
    result = score(predictions, labels, FPS)
    assert (result['correct'], result['wrong'], result['unknown']) == (6, 1, 1)
    assert result['per_player']['A']['predicted'] == {'A': 2, 'P?': 1, 'B': 1}
    assert result['ids_on_excluded_poses'] == {'other': 1}
    fixed = score(predictions, labels, FPS, {'P1': 'A', 'P2': 'B'})
    assert fixed['correct'] == 1
    with pytest.raises(ValueError, match='absent from predictions'):
        score({'0:0': 'P1'}, labels, FPS)


def test_stray_duplicate_box_does_not_split_a_player_in_two():
    base = looks(2)

    def chunk(look, start, raw):
        frames = list(range(start, start+30))
        return {'raw': raw, 'start': start, 'end': start+29, 'frames': frames, 'clean_frames': frames,
                'keys': [f'{f}:{raw}' for f in frames], 'mean': base[look], 'coherence': 1.}

    first_half = [chunk(0, start, 1) for start in range(0, 300, 30)]
    # A duplicate box beside the player for six frames forbade the merge of the halves.
    ghost = {**chunk(0, 60, 2), 'frames': list(range(60, 66)), 'clean_frames': list(range(60, 66))}
    second_half = [ghost]+[chunk(0, start, 3) for start in range(300, 720, 30)]
    twin = [chunk(0, start, 4) for start in range(0, 300, 30)]  # Always beside the player: somebody else.
    other = [chunk(1, start, 5) for start in range(0, 300, 30)]
    chunks = first_half+second_half+twin+other
    groups, start = [], 0
    for part in (first_half, second_half, twin, other):
        groups.append(list(range(start, start+len(part))))
        start += len(part)
    merged = merge_fragments(chunks, groups, PARAMETERS)
    assert sorted(len(group) for group in merged) == [10, 10, 25]
    assert sorted(merged, key=len)[-1] == groups[1]+groups[0]


def test_known_other_person_is_not_given_the_id_of_a_look_alike():
    frames, features, truth = scene(length=300)
    rng = np.random.default_rng(5)
    base = looks(8)
    stranger = .78*base[0]+.62*base[7]  # Looks like player 0 at cosine 0.78.
    for frame in frames:
        f = frame['f']
        if 70 <= f < 300:
            key = f'{f}:80'
            frame['players'].append({'key': key, 'pose_index': 80, 'bb': [2600, 200, 2700, 500],
                                     'raw_id': 800, 'occlusion': 0.})
            features[key] = stranger+rng.normal(scale=.03, size=stranger.size)
            truth[key] = 7
        if f >= 240:  # Player 0 leaves; only the look-alike remains.
            frame['players'] = [obs for obs in frame['players'] if truth[obs['key']] != 0]
    found = discover_roster(frames, features, FPS, 6, PARAMETERS)
    name_left_to_right(frames, found['roster'])
    assert [o['role'] for o in found['other_people']] == ['other_person', 'unresolved']
    without = assign_roster(frames, features, FPS, found['roster'], PARAMETERS)
    assert {without[k]['player_id'] for k, person in truth.items() if person == 7 and int(k.split(':')[0]) >= 260} == {'P1'}
    predictions = assign_roster(frames, features, FPS, found['roster'], PARAMETERS, cohort=found['cohort'])
    late = [predictions[k] for k, person in truth.items() if person == 7]
    assert {p['player_id'] for p in late} == {None}
    assert {p['reason'] for p in late} == {'other_person'}
    assert all(predictions[k]['player_id'] == 'P1' for k, person in truth.items() if person == 0 and k in predictions)
