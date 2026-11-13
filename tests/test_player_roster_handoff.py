"""An opponent's raw track can be hijacked by P2 after an occlusion."""
from copy import deepcopy

import numpy as np
import pytest

from src.tracking.player_roster_handoff import recover_recent_roster_handoffs
from src.tracking.player_team_membership import preserve_roster_membership, cap_saved_players


RECENT_P2 = np.array([.8, .6, 0], dtype=np.float32)
OTHER = np.array([0, 0, 1], dtype=np.float32)
FROZEN_P2 = np.array([1, 0, 0], dtype=np.float32)
GALLERIES = {'P2': FROZEN_P2[None, :], 'P4': OTHER[None, :]}


def pose(raw, *, owner=None, side='a', x=100):
    value = {'id': raw, 'team': side, 'team_id': 'observed_team' if side == 'b' else 'opponent_team',
             'bb': [x - 20, 20, x + 20, 180]}
    if owner:
        value.update(player_id=owner, identity={'reason': 'associated', 'appearance_distance': .1})
    return value


def fixture(end=10):
    annotations, features = [], {}
    for frame in range(end):
        if frame < 3:
            players = [pose(307, owner='P2', side='b'), pose(518)]
            features[f'{frame}:0'], features[f'{frame}:1'] = RECENT_P2, OTHER
        else:
            players = [pose(518, x=110)]
            features[f'{frame}:0'] = RECENT_P2
        annotations.append({'f': frame, 'poses': players})
    return annotations, features


def recover(annotations, features, galleries=None):
    return recover_recent_roster_handoffs(annotations, {'method': 'test'}, features,
                                         galleries or GALLERIES, 30, 'observed_team')


def test_reused_opponent_track_recovers_p2_without_recolouring_its_earlier_owner():
    annotations, features = fixture(150)
    before = deepcopy(annotations[:3])
    result = recover(annotations, features)
    assert annotations[:3] == before
    assert result['confirmed_handoffs'] == 1
    assert all(a['poses'][0]['team_id'] == 'observed_team' for a in annotations[3:])
    assert all(a['poses'][0]['player_id'] == 'P2' for a in annotations[3:])
    # A single frozen-gallery cutoff rejected this changed view; recent
    # confirmed observations recover it without relaxing that cutoff.
    assert float(FROZEN_P2 @ RECENT_P2) < .85


def test_two_observations_cannot_establish_a_handoff():
    annotations, features = fixture(5)
    assert not recover(annotations, features).get('confirmed_handoffs')
    assert all(a['poses'][0]['team_id'] == 'opponent_team' for a in annotations[3:])


def gallery_fixture(matches=3):
    annotations, features = fixture(150)
    for frame in range(3, 3 + matches):
        annotations[frame]['poses'][0].update(
            player_id='P2', team='b', team_id='observed_team',
            identity={'reason': 'appearance_motion_roster_recovery'})
    for frame in range(3, 150):
        features[f'{frame}:0'] = FROZEN_P2
    # No recent accepted appearance remains when the frozen-gallery matches
    # arrive. Each supplied match already passed the upstream motion gate.
    for row in reversed(annotations[3:]):
        old = row['f']
        row['f'] += 60
        features[f'{row["f"]}:0'] = features.pop(f'{old}:0')
    return annotations, features


def test_three_frozen_gallery_motion_matches_bind_recycled_track_after_recent_anchor_expires():
    annotations, features = gallery_fixture()
    before = deepcopy(annotations[:3])
    result = recover(annotations, features)
    assert result['confirmed_gallery_handoffs'] == 1
    assert annotations[:3] == before
    assert all(row['poses'][0]['player_id'] == 'P2' for row in annotations[3:])
    assert all(row['poses'][0]['team_id'] == 'observed_team' for row in annotations[3:])


@pytest.mark.parametrize('failure', ['two_matches', 'ambiguous_gallery', 'spaced_matches', 'distant_match'])
def test_frozen_gallery_binding_requires_three_distinct_unambiguous_continuous_matches(failure):
    annotations, features = gallery_fixture(2 if failure == 'two_matches' else 3)
    galleries = GALLERIES
    if failure == 'ambiguous_gallery':
        galleries = {**GALLERIES, 'P4': FROZEN_P2[None, :]}
    elif failure == 'spaced_matches':
        for row in reversed(annotations[4:]):
            old = row['f']
            row['f'] += 20
            features[f'{row["f"]}:0'] = features.pop(f'{old}:0')
    elif failure == 'distant_match':
        annotations[4]['poses'][0]['bb'] = [2000, 20, 2040, 180]
    result = recover(annotations, features, galleries)
    assert not result.get('confirmed_gallery_handoffs')
    assert all(row['poses'][0]['team_id'] == 'opponent_team' for row in annotations[6:])


@pytest.mark.parametrize('recent_owner', [True,False])
def test_rejected_side_prior_cannot_be_revived_from_roster_vacancy(recent_owner):
    annotations, features = fixture()
    if not recent_owner:
        for row in annotations[:3]:
            row['poses'][0].pop('player_id')
    for row in annotations[3:]:
        row['poses'][0].update(identity={'reason':'uncorroborated_side_prior'})
    result = recover(annotations,features)
    assert not result.get('confirmed_handoffs')
    assert all(not row['poses'][0].get('player_id') for row in annotations[3:])


@pytest.mark.parametrize('reason', ['stale', 'distant', 'occupied', 'ambiguous_person', 'ambiguous_box', 'wrong_gallery'])
def test_handoff_abstains_without_distinct_recent_person_and_motion(reason):
    annotations, features = fixture()
    galleries = GALLERIES
    if reason == 'stale':
        for a in annotations[3:]:
            old = a['f']
            a['f'] += 60
            features[f'{a["f"]}:0'] = features.pop(f'{old}:0')
    elif reason == 'distant':
        for a in annotations[3:]:
            a['poses'][0]['bb'] = [2000, 20, 2040, 180]
    elif reason == 'occupied':
        for a in annotations[3:]:
            a['poses'].append(pose(307, owner='P2', side='b'))
            features[f'{a["f"]}:1'] = RECENT_P2
    elif reason == 'ambiguous_person':
        for a in annotations[:3]:
            a['poses'].append(pose(99, owner='P4', side='b', x=110))
            features[f'{a["f"]}:2'] = RECENT_P2
    elif reason == 'ambiguous_box':
        for a in annotations[3:]:
            a['poses'].append(pose(519, x=111))
            features[f'{a["f"]}:1'] = RECENT_P2
    else:
        galleries = {**GALLERIES, 'P2': OTHER[None, :]}
    assert not recover(annotations, features, galleries).get('confirmed_handoffs')
    assert all(a['poses'][0]['team_id'] == 'opponent_team' for a in annotations[3:])


def test_another_raw_id_hijack_ends_the_binding():
    annotations, features = fixture(10)
    features['8:0'] = OTHER
    features['9:0'] = OTHER
    recover(annotations, features)
    assert annotations[7]['poses'][0]['team_id'] == 'observed_team'
    assert annotations[8]['poses'][0]['team_id'] == 'opponent_team'
    assert not annotations[8]['poses'][0].get('player_id')


def test_identity_conflict_does_not_erase_confirmed_team_or_duplicate_person_id():
    annotations, features = fixture(10)
    annotations[8]['poses'].append(pose(99, owner='P2', side='b', x=200))
    features['8:1'] = RECENT_P2
    recover(annotations, features)
    assert annotations[8]['poses'][0]['team_id'] == 'observed_team'
    assert not annotations[8]['poses'][0].get('player_id')
    assert sum(p.get('player_id') == 'P2' for p in annotations[8]['poses']) == 1


def test_membership_projection_and_cap_keep_confirmed_handoff_without_an_id():
    annotations, features = fixture(10)
    report = {'method': 'test', 'side': 'b', 'gallery_keys': {'P2': ['anchor:p2'], 'P4': ['anchor:p4']}}
    features.update({'anchor:p2': FROZEN_P2, 'anchor:p4': OTHER})
    annotations[8]['poses'].append(pose(99, owner='P2', side='b', x=200))
    features['8:1'] = RECENT_P2
    preserve_roster_membership(annotations, report, features, 30)
    assert annotations[8]['poses'][0]['team'] == 'b'
    assert annotations[8]['poses'][0]['team_id'] == 'observed_team'
    cap_saved_players(annotations, None, 1)
    assert annotations[8]['poses'][0]['id'] == 518
    assert annotations[8]['p'] == [[110.0, 100.0]]
