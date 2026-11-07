from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts.evaluation.refine_contact_actor_cases import load_appearance


def checksum(path): return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path):
    poses=[{'id':1,'bb':[0,0,10,20]},{'id':2,'bb':[20,0,30,20]}]
    ann={'frames':[{'f':4915,'poses':poses}],
         'player_identity':{'gallery_keys':{'P4':['4915:0'],'P1':['4915:1']}}}
    observations={'frames':[{'f':4915,'players':[
        {'key':'4915:0','pose_index':0,'bb':poses[0]['bb'],'raw_id':1},
        {'key':'4915:1','pose_index':1,'bb':poses[1]['bb'],'raw_id':2}]}]}
    (tmp_path/'observations.json').write_text(json.dumps(observations),encoding='utf-8')
    np.savez(tmp_path/'embeddings.npz',keys=['4915:0','4915:1'],embeddings=np.eye(2,dtype=np.float32))
    manifest={'schema':'player_reid_embedding_cache_v1','count':2,
              'observations_sha256':checksum(tmp_path/'observations.json'),'embedding_sha256':checksum(tmp_path/'embeddings.npz')}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    review={'cases':[{'author_confirmed':'P4','identity_anchor':{'frame':4915,'pose_index':0,'raw_track_id':1,'player_id':'P4'}},
                     {'review_frame':1051,'author_excluded':'P4'}]}
    return tmp_path,ann,review


def test_positive_reference_reuses_exact_cached_crops_without_source_changes(frozen):
    directory,ann,review=frozen
    before=deepcopy(ann),{p.name:p.read_bytes() for p in directory.iterdir()}
    features,provenance=load_appearance(directory,ann,review)
    assert features['4915:0']['distance']==pytest.approx(0)
    assert features['4915:1']['distance']==pytest.approx(1)
    assert provenance['model_inference'] is False
    assert provenance['reference_keys']==['4915:0']
    assert before==(ann,{p.name:p.read_bytes() for p in directory.iterdir()})


def test_negative_case_answers_do_not_change_identity_inference(frozen):
    directory,ann,review=frozen
    first,_=load_appearance(directory,ann,review)
    changed=deepcopy(review);changed['cases'][1].update(author_excluded='P1',review_frame=999)
    second,_=load_appearance(directory,ann,changed)
    assert first==second


def test_wrong_pose_key_or_modified_embedding_cache_is_rejected(frozen):
    directory,ann,review=frozen
    changed=deepcopy(ann);changed['frames'][0]['poses'][0]['bb'][0]=1
    with pytest.raises(ValueError,match='source pose'):load_appearance(directory,changed,review)
    path=directory/'embeddings.npz';path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='cache changed'):load_appearance(directory,ann,review)
