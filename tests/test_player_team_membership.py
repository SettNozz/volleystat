from copy import deepcopy

import numpy as np
import pytest

from src.detection.player_pose import CourtPlayerGate, CourtProjector, PlayerPose
from src.tracking.player_team_membership import (
    apply_seed_membership, cap_saved_players, preserve_roster_membership, roster_side_at_frame, team_map_at_frame,
)


def live_pose(tid, foot_y, *, x=100, height=160):
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[15:] = [[x - 3, foot_y, 1], [x + 3, foot_y, 1]]
    return PlayerPose('team_b', (x, foot_y - height / 2),
                      (x - 20, foot_y - height, x + 20, foot_y), kp, track_id=tid)


def gate():
    return CourtPlayerGate(CourtProjector(
        [[20, 0, 100], [0, 20, 100], [0, 0, 1]],
        net_floor_line=((0, 100), (200, 100))), max_per_team=None)


def test_confirmed_player_keeps_team_through_jump_occlusion_and_landing():
    g = gate()
    for frame in range(5):
        assert g.filter([live_pose(3, 160)], frame_index=frame)[0].team == 'team_b'
    for frame, y in enumerate([135, 100, 70, 40, 30, 40, 70, 100, 140, 160], 5):
        pose = live_pose(3, y)
        if frame == 9:
            pose.keypoints[15:, 2] = 0  # Missing ankles must not change membership.
        kept = g.filter([pose], frame_index=frame)
        assert kept == [pose]
        assert pose.team == 'team_b'
        assert pose.team_id == 'observed_team'


def test_initial_bad_observation_does_not_lock_wrong_team():
    g = gate()
    g.filter([live_pose(3, 40)])
    assert 3 not in g._member_team
    for _ in range(4):
        pose = g.filter([live_pose(3, 160)])[0]
    assert pose.team == 'team_b'


def test_back_facing_net_occlusion_cannot_lock_far_membership():
    g = gate()
    g.projector._net_tape_band = (0, 200, -100, 100)
    for _ in range(5):
        pose = live_pose(3, 40)
        pose.back_facing_score = .9
        g.filter([pose])
    assert 3 not in g._member_team
    for _ in range(4):
        pose = g.filter([live_pose(3, 160)])[0]
    assert pose.team_id == 'observed_team'


@pytest.mark.parametrize('frame,x', [(40, 100), (5, 1000)])
def test_reused_raw_id_cannot_transfer_team_after_gap_or_impossible_jump(frame, x):
    g = gate()
    for f in range(4):
        g.filter([live_pose(3, 160)], frame_index=f)
    g.filter([live_pose(3, 40, x=x)], frame_index=frame)
    assert 3 not in g._member_team


def test_explicit_side_switch_preserves_real_team_and_changes_court_side():
    g = gate()
    for f in range(4):
        g.filter([live_pose(3, 160)], frame_index=f)
    mapping = {'team_a': 'observed_team', 'team_b': 'opponent_team'}
    p = g.filter([live_pose(3, 160)], frame_index=4, team_identity_map=mapping)[0]
    assert p.team == 'team_a'
    assert p.team_id == 'observed_team'


def saved(tid, side='b', pid=None, x=100):
    p = {'id': tid, 'team': side, 'bb': [x - 20, 20, x + 20, 180]}
    if pid:
        p['player_id'] = pid
    return p


def report():
    return {'side': 'b', 'method': 'test', 'gallery_keys': {'P6': ['0:0'], 'P1': ['0:1']}}


def features():
    return {'0:0': np.array([1., 0.]), '0:1': np.array([0., 1.])}


def test_same_track_membership_restores_id_and_team_without_creating_missing_poses():
    anns = [{'f': f, 'poses': [saved(3, pid='P6')]} for f in range(3)]
    anns += [{'f': 3, 'poses': [saved(3, side='a')]}, {'f': 4, 'poses': []}]
    preserve_roster_membership(anns, report(), features(), 30)
    assert anns[3]['poses'][0]['player_id'] == 'P6'
    assert anns[3]['poses'][0]['team'] == 'b'
    assert anns[4]['poses'] == []


def test_jump_appearance_change_keeps_recent_identity_on_continuous_track():
    anns = [{'f': f, 'poses': [saved(3, pid='P6')]} for f in range(3)]
    anns += [{'f': f, 'poses': [saved(3)]} for f in range(3, 20)]
    vectors = {**features(), **{f'{f}:0': np.array([.6, .8]) for f in range(3, 20)}}
    preserve_roster_membership(anns, report(), vectors, 30)
    assert all(a['poses'][0]['player_id'] == 'P6' for a in anns)


def test_tentative_net_opponent_cannot_acquire_absent_roster_id_by_membership_recovery():
    anns = [{'f':f,'poses':[saved(3,pid='P6')]} for f in range(3)]
    for f in range(3,8):
        anns.append({'f':f,'poses':[{**saved(49), 'team_prior':'serve_back_facing_net_override',
                                   'team_source':'frame_geometry','team_geometry':'team_a'}]})
    vectors = {**features(), **{f'{f}:0':np.array([1.,0.]) for f in range(3,8)}}
    preserve_roster_membership(anns,report(),vectors,30)
    assert all(not a['poses'][0].get('player_id') and a['poses'][0]['team']=='a' for a in anns[3:])
    assert all(a['poses'][0]['team_id']=='opponent_team' for a in anns[3:])


def test_tentative_matches_cannot_seed_a_self_supporting_tracklet_membership():
    anns = [{'f':f,'poses':[{**saved(49,pid='P6' if f<3 else None),
                           'team_prior':'serve_back_facing_net_override','team_source':'frame_geometry',
                           'team_geometry':'team_a'}]} for f in range(6)]
    vectors = {**features(), **{f'{f}:0':np.array([.6,.8]) for f in range(1,6)}}
    preserve_roster_membership(anns,report(),vectors,30)
    assert all(not a['poses'][0].get('player_id') for a in anns[3:])


def test_three_strong_prior_only_matches_preserve_real_net_player_through_missing_id():
    anns = [{'f':f,'poses':[{**saved(49,pid='P6' if f<4 else None),
                           'team_prior':'serve_back_facing_net_override','team_source':'frame_geometry',
                           'team_geometry':'team_a'}]} for f in range(1,6)]
    vectors = {**features(), **{f'{f}:0':np.array([1.,0.]) for f in range(1,6)}}
    preserve_roster_membership(anns,report(),vectors,30)
    assert all(a['poses'][0].get('player_id')=='P6' and a['poses'][0]['team']=='b' for a in anns)


def test_roster_photo_corrects_its_continuous_track_before_side_filtering():
    anns = [{'f': 0, 'poses': [saved(3, side='a')]},
            {'f': 1, 'poses': [saved(3, side='a')]},
            {'f': 90, 'poses': [saved(3, side='a')]}]
    seed = {'assignments': {'1:0': 'P6'}}
    apply_seed_membership(anns, seed, 'b', 30)
    assert [a['poses'][0]['team'] for a in anns] == ['b', 'b', 'a']


def test_new_raw_id_needs_distinct_appearance_and_nearby_motion():
    anns = [{'f': 0, 'poses': [saved(3, pid='P6')]},
            {'f': 1, 'poses': [saved(8, side='a')]}]
    vectors = {**features(), '1:0': np.array([1., 0.])}
    preserve_roster_membership(anns, report(), vectors, 30)
    assert anns[1]['poses'][0]['player_id'] == 'P6'
    assert anns[1]['poses'][0]['team_id'] == 'observed_team'


@pytest.mark.parametrize('kind', ['ambiguous', 'distant', 'stale', 'occupied'])
def test_appearance_alone_cannot_steal_roster_identity(kind):
    anns = [{'f': 0, 'poses': [saved(3, pid='P6')]},
            {'f': 90 if kind == 'stale' else 1, 'poses': [saved(8, side='a', x=2000 if kind == 'distant' else 100)]}]
    vectors = {**features(), f'{anns[1]["f"]}:0': np.array([1., 1.] if kind == 'ambiguous' else [1., 0.])}
    if kind == 'occupied':
        anns[1]['poses'].append(saved(3, pid='P6'))
    preserve_roster_membership(anns, report(), vectors, 30)
    assert not anns[1]['poses'][0].get('player_id')
    assert anns[1]['poses'][0]['team'] == 'a'


def test_side_switch_keeps_roster_and_reorders_player_centres():
    segments = [{'start_frame': 10, 'end_frame': 20,
                 'team_identity_map': {'team_a': 'observed_team', 'team_b': 'opponent_team'}}]
    anns = [{'f': 10, 'poses': [saved(3, pid='P6'), saved(7, 'a', x=200)], 'p': [[200, 100], [100, 100]]}]
    preserve_roster_membership(anns, report(), features(), 30, segments=segments)
    assert anns[0]['poses'][0]['team'] == 'a'
    assert anns[0]['poses'][0]['team_id'] == 'observed_team'
    assert anns[0]['p'] == [[100, 100], [200, 100]]
    assert roster_side_at_frame(10, 'b', segments=segments) == 'a'
    assert team_map_at_frame(0)['team_b'] == 'observed_team'


def test_roster_members_take_priority_at_player_cap_with_centres_in_sync():
    anns = [{'f': 0, 'poses': [saved(9, x=200), saved(3, pid='P6')], 'p': [[200, 100], [100, 100]]}]
    cap_saved_players(anns, None, 1)
    assert [p['player_id'] for p in anns[0]['poses']] == ['P6']
    assert anns[0]['p'] == [[100, 100]]


def test_cap_remaps_roster_photo_keys_to_exported_pose_indices():
    anns = [{'f': 0, 'poses': [saved(9), {**saved(3, pid='P6'), 'team_id': 'observed_team'}]}]
    metadata = {'status': 'completed', 'team_id': 'observed_team',
                'seed': {'assignments': {'0:1': 'P6'}}, 'gallery_keys': {'P6': ['0:1']}}
    cap_saved_players(anns, None, 1, metadata)
    assert metadata['seed']['assignments'] == {'0:0': 'P6'}
    assert metadata['gallery_keys'] == {'P6': ['0:0']}
    assert metadata['coverage'] == 1
