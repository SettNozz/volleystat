"""Lost identity recovery, frozen galleries and evaluation separation."""
from argparse import Namespace
import copy
import json

import numpy as np
import pytest

from src.tracking.player_reid import ReIDParameters, associate, gallery_keys, unit
from src.tracking.player_motion import PlayerMotionKalman


def person(f, i, x, feature, raw_id=None, occlusion=0.):
    return {'key': f'{f}:{i}', 'raw_id': raw_id, 'bb': [x,0,x+30,100],
            'appearance': feature, 'occlusion': occlusion}


def recovery_case():
    frames = [{'f': 0, 'players': [person(0,0,0,[1.,0.],1), person(0,1,100,[0.,1.],2)]}]
    for f in [121,124,127]:
        frames.append({'f':f, 'players':[person(f,0,250,[1.,0.],99), person(f,1,400,[0.,1.],101)]})
    seed = {'frame':0, 'assignments':{'0:0':'P1','0:1':'P2'}, 'verified':True, 'label_source':'human'}
    features = {p['key']: p['appearance'] for f in frames for p in f['players']}
    return frames, seed, features


@pytest.mark.parametrize('motion_model', ['ema', 'kalman'])
def test_distinct_appearance_recovers_long_exit_without_recycling_id(motion_model):
    frames, seed, features = recovery_case()
    parameters = ReIDParameters(motion_model=motion_model)
    local = associate(frames,seed,features,30.,False,parameters)
    recovered = associate(frames,seed,features,30.,True,parameters)
    assert local['predictions']['121:0']['player_id'] is None
    for f in [121,124,127]:
        assert recovered['predictions'][f'{f}:0']['player_id'] == 'P1'
        assert recovered['predictions'][f'{f}:1']['player_id'] == 'P2'
        assert recovered['predictions'][f'{f}:0']['reason'] == 'recovered'


@pytest.mark.parametrize('motion_model', ['ema', 'kalman'])
def test_ambiguous_identical_uniforms_do_not_recover_by_force(motion_model):
    frames, seed, features = recovery_case()
    features = {key:[1.,0.] for key in features}
    result = associate(frames,seed,features,30.,True,ReIDParameters(motion_model=motion_model))
    assert all(result['predictions'][p['key']]['player_id'] is None
               for frame in frames[1:] for p in frame['players'])


def test_one_candidate_frame_is_not_enough_for_lost_recovery():
    frames, seed, features = recovery_case()
    result = associate(frames[:2],seed,features,30.,True)
    assert result['predictions']['121:0']['reason'] == 'pending_recovery'
    assert result['predictions']['121:0']['player_id'] is None


def test_missing_frame_breaks_recovery_confirmation():
    frames, seed, features = recovery_case()
    frames[2] = {'f':124,'players':[]}
    result = associate(frames,seed,features,30.,True)
    assert result['predictions']['121:0']['player_id'] is None
    assert result['predictions']['127:0']['player_id'] is None


def test_frozen_gallery_uses_clean_initial_context_not_later_relabelled_track():
    frames, seed, features = recovery_case()
    frames.insert(1,{'f':3,'players':[person(3,0,1,[1.,0.],1),person(3,1,100,[0.,1.],2,occlusion=.9)]})
    keys = gallery_keys(frames,seed,ReIDParameters())
    assert keys == {'P1':['0:0','3:0'],'P2':['0:1']}
    features.update({p['key']:p['appearance'] for p in frames[1]['players']})
    original = copy.deepcopy((frames,features))
    result = associate(frames,seed,features,30.,True)
    assert result['gallery_keys'] == keys
    assert (frames,features) == original


def test_recovery_stays_one_to_one_with_duplicate_detections():
    frames, seed, features = recovery_case()
    for frame in frames[1:]:
        extra = person(frame['f'],2,252,[1.,0.],None)
        frame['players'].append(extra)
        features[extra['key']] = extra['appearance']
    result = associate(frames,seed,features,30.,True)
    for frame in frames:
        ids = [result['predictions'][p['key']]['player_id'] for p in frame['players']]
        ids = [p for p in ids if p is not None]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize('motion_model', ['ema', 'kalman'])
def test_backward_recovery_uses_future_seed_and_confirms_in_reverse(motion_model):
    frames, seed, features = recovery_case()
    frames[0]['f'] = 150
    seed['frame'] = 150
    # Keys are source observation identities, independent of iteration order.
    result = associate(frames,seed,features,30.,True,ReIDParameters(local_gap_seconds=.5,motion_model=motion_model))
    assert result['predictions']['121:0']['player_id'] == 'P1'


def test_invalid_features_fail_instead_of_random_identity():
    with pytest.raises(ValueError,match='embedding'):
        unit([0.,0.])
    with pytest.raises(ValueError,match='embedding'):
        unit([1.,float('nan')])


def test_kalman_uncertainty_grows_in_gap_and_updates_remain_positive():
    motion = PlayerMotionKalman([0., 0.], 100., 0.)
    for t in np.arange(.1, 1.1, .1):
        motion.update([100*t, 0.], 100., t, 0.)
    near, near_cov = motion.predict(1.1)
    far, far_cov = motion.predict(2.)
    assert far[0]-near[0] > 70.
    assert np.trace(far_cov) > np.trace(near_cov)
    assert np.linalg.eigvalsh(motion.covariance).min() > 0


@pytest.mark.parametrize('direction', [1, -1])
def test_motion_uncertainty_does_not_depend_on_prediction_frame_rate(direction):
    direct = PlayerMotionKalman([0.,0.],100.,0.)
    stepped = PlayerMotionKalman([0.,0.],100.,0.)
    mean, covariance = direct.predict(direction*1.)
    for time in np.linspace(0., direction*1., 31)[1:]:
        stepped.mean, stepped.covariance = stepped.predict(time)
        stepped.time = time
    assert stepped.mean == pytest.approx(mean)
    assert stepped.covariance == pytest.approx(covariance)


def test_kalman_crossing_after_gap_preserves_same_uniform_roster():
    frames = []
    # Establish opposing velocities before both people disappear for one second.
    for f in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 60, 63, 66]:
        players = [person(f, 0, f*4., [1.,0.]),
                   person(f, 1, 400.-f*4., [1.,0.])]
        # Different initial views, identical uniforms after initialisation.
        if f == 0:
            players[1]['appearance'] = [.99,.01]
        frames.append({'f': f, 'players': players})
    seed = {'frame':0, 'assignments':{'0:0':'P1','0:1':'P2'}}
    features = {p['key']:p['appearance'] for f in frames for p in f['players']}
    result = associate(frames,seed,features,30.,True,ReIDParameters(motion_model='kalman'))
    assert result['predictions']['60:0']['player_id'] == 'P1'
    assert result['predictions']['60:1']['player_id'] == 'P2'
    baseline = associate(frames,seed,features,30.,True,ReIDParameters(motion_model='ema'))
    assert baseline['predictions']['60:0']['player_id'] == 'P2'
    assert baseline['predictions']['60:1']['player_id'] == 'P1'
    assert all(p['key'] in result['predictions'] for f in frames for p in f['players'])


def test_occluded_box_does_not_drag_motion_or_assign_ambiguous_crop():
    frames = [{'f': f, 'players': [person(f,0,0,[1.,0.],occlusion=0.)]}
              for f in [0,3,6,9]]
    frames += [{'f': f, 'players': [person(f,0,80,[1.,0.],occlusion=.9)]}
               for f in [12,15,18,21,24]]
    frames += [{'f': f, 'players': [person(f,0,0,[1.,0.])]}
               for f in [27,30]]
    seed = {'frame':0, 'assignments':{'0:0':'P1'}}
    features = {p['key']:p['appearance'] for f in frames for p in f['players']}
    result = associate(frames,seed,features,30.,True,ReIDParameters(motion_model='kalman'))
    assert result['predictions']['27:0']['player_id'] == 'P1'
    assert result['predictions']['30:0']['player_id'] == 'P1'
    frames[0]['players'].append(person(0,1,300,[1.,0.]))
    seed['assignments']['0:1'] = 'P2'
    features['0:1'] = [1.,0.]
    result = associate(frames,seed,features,30.,True,ReIDParameters(motion_model='kalman'))
    assert result['predictions']['24:0']['player_id'] is None


@pytest.mark.parametrize('direction', [1, -1])
def test_kalman_time_and_resolution_scaling(direction):
    a = PlayerMotionKalman([10.,20.], 100., 0.)
    b = PlayerMotionKalman([40.,80.], 400., 0.)
    for f in range(1,10):
        t = direction*f/30.
        a.update([10.+f*2,20.],100.,t,0.)
        b.update([40.+f*8,80.],400.,t,0.)
    da, ma, *_ = a.innovation([32.,20.],100.,direction*.4,.2)
    db, mb, *_ = b.innovation([128.,80.],400.,direction*.4,.2)
    assert da == pytest.approx(db)
    assert mb == pytest.approx(ma*4.)


def test_evaluation_checks_change_scores_only(tmp_path):
    from scripts.evaluation.run_player_identity_pilot import digest, write_json
    from scripts.evaluation.run_player_reid_experiment import evaluate
    frames, seed, features = recovery_case()
    project = tmp_path/'project'
    cache = tmp_path/'cache'
    project.mkdir()
    cache.mkdir()
    write_json(project/'observations.json',{'frames':frames,'fps':30.,'source':{}})
    fingerprint = digest(project/'observations.json')
    labels = {'seed':seed,'checks':[],'observations_sha256':fingerprint}
    write_json(project/'labels.json',labels)
    write_json(cache/'plan.json',{'seed':seed,'parameters':ReIDParameters().to_dict(),'observations_sha256':fingerprint})
    np.savez_compressed(cache/'embeddings.npz',keys=np.asarray(list(features)),embeddings=np.asarray(list(features.values())))
    write_json(cache/'embedding_manifest.json',{'observations_sha256':fingerprint,'embedding_sha256':digest(cache/'embeddings.npz')})
    first, second = tmp_path/'first.json', tmp_path/'second.json'
    evaluate(Namespace(project=project,cache=cache,output=first))
    labels['checks'] = [{'key':'121:0','frame':121,'player_id':'P1','verified':True,'label_source':'human'}]
    (project/'labels.json').write_text(json.dumps(labels))
    evaluate(Namespace(project=project,cache=cache,output=second))
    a,b = json.loads(first.read_text()),json.loads(second.read_text())
    assert b['methods']['osnet_fixed_recovery']['human_checks']['correct'] == 1
    for method in a['methods']:
        assert a['methods'][method]['predictions'] == b['methods'][method]['predictions']


def test_stress_blackout_excludes_masked_checks_and_preserves_sources(tmp_path):
    from scripts.evaluation.run_player_identity_pilot import digest, write_json
    from scripts.evaluation.run_player_reid_experiment import stress_gap
    frames, seed, _ = recovery_case()
    frames = frames[:1] + [{'f':f,'players':[person(f,0,0,[1.,0.],1),person(f,1,100,[0.,1.],2)]}
                          for f in [30,141,144,147]]
    features = {p['key']:p['appearance'] for f in frames for p in f['players']}
    project,cache = tmp_path/'project',tmp_path/'cache'
    project.mkdir()
    cache.mkdir()
    write_json(project/'observations.json',{'frames':frames,'fps':30.,'source':{}})
    fingerprint = digest(project/'observations.json')
    checks = [{'key':f'{f}:0','frame':f,'player_id':'P1','verified':True,'label_source':'human'} for f in [30,141]]
    write_json(project/'labels.json',{'seed':seed,'checks':checks,'observations_sha256':fingerprint})
    labels_sha = digest(project/'labels.json')
    parameters = ReIDParameters()
    write_json(cache/'plan.json',{'seed':seed,'parameters':parameters.to_dict(),'observations_sha256':fingerprint,
                                'gallery_keys':gallery_keys(frames,seed,parameters)})
    np.savez_compressed(cache/'embeddings.npz',keys=np.asarray(list(features)),embeddings=np.asarray(list(features.values())))
    write_json(cache/'embedding_manifest.json',{'embedding_sha256':digest(cache/'embeddings.npz')})
    output = tmp_path/'stress.json'
    stress_gap(Namespace(project=project,cache=cache,output=output))
    result = json.loads(output.read_text())
    assert result['post_blackout_checks'] == result['excluded_checks'] == 1
    assert result['methods']['osnet_fixed_local']['human_checks']['correct'] == 0
    assert result['methods']['osnet_fixed_recovery']['human_checks']['correct'] == 1
    assert digest(project/'observations.json') == fingerprint
    assert digest(project/'labels.json') == labels_sha
