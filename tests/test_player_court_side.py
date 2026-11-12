import numpy as np

from src.detection.player_pose import CourtProjector
from src.tracking.player_court_side import frame_sides, measure_poses, preserve_frame_sides, temporal_sides
from src.tracking.player_reid import ReIDParameters, associate


PROJECTOR = CourtProjector(np.eye(3), net_floor_line=((0, 100), (1200, 100)),
                           net_tape_band=(0, 1200, 20, 80))


def frame(f, y, *, x=500, back=.8):
    kp = [[0, 0, 0] for _ in range(15)] + [[x - 8, y, .9], [x + 8, y, .9]]
    return {'f': f, 'poses': [{'id': 10, 'bb': [x - 12, y - 160, x + 12, y], 'kp': kp, 'back': back}]}


def test_past_and_future_anchors_keep_side_through_net_jitter_and_rotation():
    frames = [frame(f, y, back=0.1) for f, y in enumerate([145, 144, 93, 146, 145])]
    measurements = measure_poses(frames, PROJECTOR, 2160)
    assert frame_sides(frames, measurements, 10)[(2, 0)].side == 'a'
    for offset in (-10, -5, 0, 5, 10):
        resolved = temporal_sides(frames, measurements, 30, offset)
        assert resolved[(2, 0)].side == 'b'
        assert resolved[(2, 0)].reason == 'track_history'


def test_spatial_jump_blocks_history_and_back_prior_stays_tentative():
    frames = [frame(f, 145) for f in range(4)] + [frame(4, 94, x=1100)]
    measurements = measure_poses(frames, PROJECTOR, 2160)
    without_back = temporal_sides(frames, measurements, 30, 0)
    with_back = temporal_sides(frames, measurements, 30, 0, use_back=True)
    assert without_back[(4, 0)].side is None
    assert with_back[(4, 0)].side == 'b'
    assert with_back[(4, 0)].tentative


def test_back_prior_cannot_override_far_side_track_evidence():
    frames = [frame(f, y) for f, y in enumerate([60, 63, 96, 61, 62])]
    evidence = temporal_sides(frames, measure_poses(frames, PROJECTOR, 2160), 30, 0, use_back=True)
    assert evidence[(2, 0)].side == 'a'
    assert evidence[(2, 0)].reason == 'track_history'


def test_preserving_policy_recovers_near_side_through_history():
    frames = [frame(f, y, back=.1) for f, y in enumerate([145, 144, 93, 146, 145])]
    measurements = measure_poses(frames, PROJECTOR, 2160)
    baseline = frame_sides(frames, measurements, 10)
    sides = preserve_frame_sides(temporal_sides(frames, measurements, 30, 10), baseline)
    assert sides[(2, 0)].side == 'b'
    assert sides[(2, 0)].reason == 'track_history'
    assert sides[(0, 0)] is baseline[(0, 0)]


def test_tentative_id_requires_distinct_appearance_and_cannot_extend_local_state():
    def obs(f, tentative=False):
        return {'key': f'{f}:0', 'bb': [10, 10, 45, 200], 'raw_id': 10, 'occlusion': 0.,
                'tentative_side': tentative}

    frames = [{'f': f, 'players': [obs(f, f != 0)]} for f in (0, 15, 29, 31)]
    seed = {'frame': 0, 'assignments': {'0:0': 'P1'}}
    features = {'0:0': [1., 0.], '15:0': [1., 0.], '29:0': [.6, .8], '31:0': [1., 0.]}
    parameters = ReIDParameters(motion_model='kalman', local_gap_seconds=1.)
    result = associate(frames, seed, features, 30, recover=True, parameters=parameters)
    assert result['gallery_keys'] == {'P1': ['0:0']}
    assert result['predictions']['15:0']['reason'] == 'tentative_side_corroborated'
    assert result['predictions']['29:0']['player_id'] is None
    assert result['predictions']['31:0']['player_id'] is None


def test_five_visible_roster_members_do_not_lend_absent_server_id_to_opponent():
    seed_players = [{'key':f'0:{i}', 'raw_id':i, 'bb':[i*100,0,i*100+30,100],
                     'occlusion':0.} for i in range(6)]
    frames = [{'f':0,'players':seed_players}]
    vectors = np.eye(7)
    features = {p['key']:vectors[i] for i,p in enumerate(seed_players)}
    for f in (1,2,3):
        # P3 has left the camera to serve. A similar-looking opponent at the
        # net is only a tentative side candidate. Its weak appearance would
        # pass the ordinary .55 gate, but not the stricter side-prior gate.
        visible = [p for i,p in enumerate(seed_players) if i!=2]
        players = [{**p,'key':f'{f}:{i}'} for i,p in enumerate(visible)]
        opponent = {**seed_players[2],'key':f'{f}:5','raw_id':49,'side_prior_only':True}
        players.append(opponent)
        frames.append({'f':f,'players':players})
        for p in players:features[p['key']] = .62*vectors[2]+.7846*vectors[6] if p is opponent else vectors[p['raw_id']]
    returned = [{**p,'key':f'4:{i}'} for i,p in enumerate(seed_players)]
    returned.append({**seed_players[2],'key':'4:6','raw_id':49,'side_prior_only':True})
    frames.append({'f':4,'players':returned})
    for p in returned:features[p['key']] = .62*vectors[2]+.7846*vectors[6] if p['raw_id']==49 else vectors[p['raw_id']]
    seed = {'frame':0,'assignments':{p['key']:f'P{i+1}' for i,p in enumerate(seed_players)}}
    result = associate(frames,seed,features,30,True,ReIDParameters(motion_model='kalman'))
    # Reproduce the weak ordinary match, then exercise the production veto
    # before these predictions can turn into self-supporting membership.
    assert result['predictions']['1:5']['player_id']=='P3'
    annotations=[]
    for row in frames:
        poses=[]
        for p in row['players']:
            prior=p.get('side_prior_only',False)
            prediction=result['predictions'][p['key']]
            poses.append({'id':p['raw_id'],'bb':p['bb'],'team':'b',
                          'team_geometry':'team_a' if prior else 'team_b',
                          'team_source':'frame_geometry' if prior else 'continuous_track_membership',
                          'team_prior':'serve_back_facing_net_override' if prior else None,
                          'player_id':prediction['player_id'],'identity':prediction})
        annotations.append({'f':row['f'],'poses':poses})
    from src.tracking.player_team_membership import preserve_roster_membership
    report={'side':'b','method':'test','gallery_keys':result['gallery_keys'],'parameters':result['parameters']}
    preserve_roster_membership(annotations,report,features,30)
    for f in (1,2,3):
        assigned = [p.get('player_id') for p in annotations[f]['poses']]
        assert annotations[f]['poses'][5]['player_id'] is None
        assert annotations[f]['poses'][5]['team']=='a'
        assert len(annotations[f]['poses'])==6
        assert len([pid for pid in assigned if pid]) == 5
        assert 'P3' not in assigned
    assert annotations[4]['poses'][2]['player_id']=='P3'
    assert annotations[4]['poses'][6]['player_id'] is None
    assert len([p for p in annotations[4]['poses'] if p.get('player_id')])==6
