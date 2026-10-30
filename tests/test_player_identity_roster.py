"""Confirmed substitutions and rejection-only occlusion handover checks."""
from copy import deepcopy
import json
from dataclasses import replace

import numpy as np
import pytest

from src.models.player_reid_encoder import file_sha256
from src.tracking.player_reid import ReIDParameters, associate, gallery_keys
from src.tracking.player_identity_consistency import reject_tracklet_conflicts
from src.tracking.player_identity_workflow import PRODUCTION_PARAMETERS, load_seed


def obs(f,i,x,feature,raw):
    return {'key':f'{f}:{i}','bb':[x,0,x+30,100],'raw_id':raw,
            'occlusion':0.,'feature':feature}


def introduction_case():
    frames=[{'f':0,'players':[obs(0,0,0,[1,0,0],1),obs(0,1,100,[0,1,0],2)]}]
    for f in [118,121,124,127]:
        frames.append({'f':f,'players':[obs(f,0,250,[0,0,1],9),obs(f,1,400,[0,1,0],2)]})
    for f in [240,243,246]:
        frames.append({'f':f,'players':[obs(f,0,20,[1,0,0],10),obs(f,1,250,[0,0,1],9),
                                        obs(f,2,400,[0,1,0],2)]})
    seed={'frame':0,'assignments':{'0:0':'P1','0:1':'P2'},
          'introductions':[{'frame':127,'key':'127:0','player_id':'P3','available_from_frame':121}]}
    features={p['key']:p['feature'] for f in frames for p in f['players']}
    return frames,seed,features


@pytest.mark.parametrize('parameters',[ReIDParameters(motion_model='kalman'),PRODUCTION_PARAMETERS])
def test_new_person_gets_new_id_and_returning_original_keeps_old_id(parameters):
    frames,seed,features=introduction_case()
    result=associate(frames,seed,features,30.,True,parameters)
    assert result['predictions']['118:0']['player_id'] is None
    assert result['predictions']['121:0']['player_id']=='P3'
    assert result['predictions']['127:0']['reason']=='roster_anchor'
    for f in [240,243,246]:
        assert result['predictions'][f'{f}:0']['player_id']=='P1'
        assert result['predictions'][f'{f}:1']['player_id']=='P3'
        assert result['predictions'][f'{f}:2']['player_id']=='P2'
    assert '118:0' not in gallery_keys(frames,seed,ReIDParameters())['P3']


def test_unknown_person_does_not_automatically_create_an_identity():
    frames,seed,features=introduction_case()
    seed.pop('introductions')
    result=associate(frames,seed,features,30.,True,ReIDParameters(motion_model='kalman'))
    assert result['predictions']['127:0']['player_id'] is None
    assert set(result['gallery_keys'])=={'P1','P2'}


def test_roster_file_preserves_automatic_initial_provenance_and_human_confirmation(tmp_path):
    frames,_,_=introduction_case()
    video=tmp_path/'video.webm'
    video.write_bytes(b'exact video')
    payload={'schema':'player_identity_roster_v2','side':'b','max_on_court':2,
             'video_sha256':file_sha256(video),
             'initial':{'frame':0,'label_source':'automatic_initialisation','verified':False,
                        'players':[{'player_id':f'P{i+1}','bb':p['bb']} for i,p in enumerate(frames[0]['players'])]},
             'introductions':[{'player_id':'P3','anchor':{'frame':127,'bb':frames[4]['players'][0]['bb'],
                                                        'provenance':'model detection'},
                               'available_from_frame':121,
                               'identity_confirmation':{'label_source':'human','verified':True,
                                                        'scope':'new_person','note':'User confirmed replacement'}}]}
    path=tmp_path/'roster.json'
    path.write_text(json.dumps(payload))
    seed=load_seed(path,frames,video,'b',2)
    assert seed['verified'] is False
    assert seed['introductions'][0]['player_id']=='P3'
    assert seed['introductions'][0]['identity_confirmation']['verified'] is True
    payload['introductions'][0]['identity_confirmation']['verified']=False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='human confirmation'):
        load_seed(path,frames,video,'b',2)
    payload['introductions'][0]['identity_confirmation']['verified']=True
    payload['introductions'][0]['player_id']='P1'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='distinct'):
        load_seed(path,frames,video,'b',2)


def test_visible_established_owner_prevents_borrowing_id_during_occlusion():
    frames=[]
    predictions={}
    for f in range(21):
        owner=obs(f,0,100,[1,0],10)
        owner['occlusion']=.9 if 9<=f<=11 else 0.
        players=[owner]
        predictions[owner['key']]={'player_id':None if 9<=f<=11 else 'P6','reason':'associated','cost':.1}
        if f>=9:
            other=obs(f,1,0,[0,1],20)
            players.append(other)
            predictions[other['key']]={'player_id':'P6' if f<=11 else 'P3','reason':'associated','cost':.3}
        frames.append({'f':f,'players':players})
    original=deepcopy(frames)
    report=reject_tracklet_conflicts(frames,predictions,[],fps=30.)
    assert report['rejected_observations']==3
    for f in [9,10,11]:
        assert predictions[f'{f}:0']['player_id'] is None
        assert predictions[f'{f}:1']['player_id'] is None
        assert predictions[f'{f}:1']['candidate_player_id']=='P6'
    assert predictions['12:1']['player_id']=='P3'
    assert frames==original


def test_clean_evidence_of_a_raw_track_switch_is_not_frozen_to_majority():
    frames=[{'f':f,'players':[obs(f,0,0,[1,0],10)]} for f in range(36)]
    predictions={f'{f}:0':{'player_id':'P1' if f<6 else 'P2','reason':'associated','cost':.1}
                 for f in range(36)}
    before=deepcopy(predictions)
    report=reject_tracklet_conflicts(frames,predictions,[],fps=30.)
    assert report['rejected_observations']==0
    assert predictions==before


def test_distinct_appearance_can_establish_owner_despite_bbox_overlap():
    frames=[]
    predictions={}
    for f in range(21):
        owner=obs(f,0,100,[1,0],10)
        owner['occlusion']=.8
        players=[owner]
        predictions[owner['key']]={'player_id':None if 9<=f<=11 else 'P6',
                                   'reason':'associated','cost':.45,'appearance_distance':.25}
        if f>=9:
            other=obs(f,1,0,[0,1],20)
            players.append(other)
            predictions[other['key']]={'player_id':'P6' if f<=11 else 'P3',
                                       'reason':'associated','cost':.3,'appearance_distance':.22}
        frames.append({'f':f,'players':players})
    report=reject_tracklet_conflicts(frames,predictions,[],fps=30.)
    assert report['rejected_observations']==3
    assert all(predictions[f'{f}:1']['player_id'] is None for f in [9,10,11])
    assert predictions['12:1']['player_id']=='P3'


def test_motion_cannot_borrow_an_id_against_distinct_immutable_appearance():
    frames=[{'f':0,'players':[obs(0,0,0,[1,0],1),obs(0,1,1000,[0,1],2)]},
            {'f':1,'players':[obs(1,0,0,[.65,.75],3)]}]
    features={p['key']:p['feature'] for f in frames for p in f['players']}
    seed={'frame':0,'assignments':{'0:0':'P1','0:1':'P2'}}
    baseline=ReIDParameters(motion_model='kalman')
    assert associate(frames,seed,features,30.,True,baseline)['predictions']['1:0']['player_id']=='P1'
    guarded=replace(baseline,local_appearance_regret_max=.06)
    assert associate(frames,seed,features,30.,True,guarded)['predictions']['1:0']['player_id'] is None
