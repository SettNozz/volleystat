from copy import deepcopy
import numpy as np

from src.tracking.contact_author_temporal_features import interpolated_pose,polynomial_contact
from src.tracking.contact_author_reid_evaluation import person_sequence
from src.tracking.contact_author_evaluation import sequence_scores


def test_joint_interpolation_needs_both_sides_and_never_changes_observations():
    pose={"bb":[0,0,100,200],"kp":[[10.,20.,.9] for _ in range(17)]}
    hidden=deepcopy(pose); hidden["kp"][9]=[0.,0.,0.]
    future=deepcopy(pose); future["kp"][9]=[30.,40.,.9]
    track=[{"frame":0,"pose":pose},{"frame":2,"pose":hidden},{"frame":4,"pose":future}]
    before=deepcopy(track); completed,fraction=interpolated_pose(track,2)
    assert completed["kp"][9][:2]==[20.,30.] and fraction==1/17
    assert track==before
    completed,_=interpolated_pose(track[:2],2)
    assert completed["kp"][9][2]==0
    assert interpolated_pose(track,20)==(None,0.)


def test_polynomial_contact_is_derived_from_observed_only_and_stops_at_gaps():
    frames={f:{"ball":[100+f,200+f*f],"ball_source":"detected"} for f in range(30)}
    fit=polynomial_contact(frames,15,-1)
    np.testing.assert_allclose(fit["xy"],[115,425],atol=1e-5)
    frames[15]["ball_out_of_frame"]=True
    assert polynomial_contact(frames,15,-1) is None
    frames[15]["ball_out_of_frame"]=False; frames[12]["ball"]=None
    assert polynomial_contact(frames,15,-1) is None


def test_person_sequence_team_marginal_is_independent_of_roster_cardinality():
    groups=np.array(["a"]*3+["b"]*3+["c"]*3)
    y=np.array([0,1,2,1,2,0,999,999,999]); train=groups!="c"
    cases=[{"contact":{"id":str(i)}} for i in range(9)]
    side=np.array([.2,.8,.7,.8,.9,.1,.3,.8,.65])
    for size in (2,6,7):
        snapshot={"videos":{v:{"fps":30.} for v in ("a","b","c")},
            "rosters":{v:{"players":[{}]*size} for v in ("a","b","c")},"queue":[
                {"id":str(i),"video_id":groups[i],"frame":30+30*(i%3),"type":("serve","reception","set")[i%3],"timing_verified":True} for i in range(9)]}
        conditional=np.zeros((9,size+1)); conditional[:,1:]=1/size
        posterior,_=person_sequence(snapshot,cases,groups,train,y,conditional,side)
        binary,_=sequence_scores(snapshot,cases,groups,train,y,side)
        np.testing.assert_allclose(1-posterior[:,0],binary,atol=1e-8)
        changed=y.copy(); changed[~train]=-20
        other,_=person_sequence(snapshot,cases,groups,train,changed,conditional,side)
        np.testing.assert_array_equal(posterior,other)
