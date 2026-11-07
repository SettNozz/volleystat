from copy import deepcopy
import numpy as np
import pytest

from src.tracking.contact_actor_context import ActorContext,action_evidence,build_action_report


def pose(raw=1,pid="P4",team="b",raised=False,x=0):
    kp = [[x+50,90,1] for _ in range(17)]
    for j in (5,6): kp[j]=[x+50+(j-5)*30,120,1]
    for j in (7,8): kp[j]=[x+50+(j-7)*30,100 if raised else 155,1]
    for j in (9,10): kp[j]=[x+50+(j-9)*30,75 if raised else 180,1]
    for j in (15,16): kp[j]=[x+45+(j-15)*35,245,1]
    return {"id":raw,"player_id":pid,"team":team,"bb":[x+20,65,x+105,250],"kp":kp,
            "identity":{"reason":"associated","appearance_distance":.2}}


def annotations(rows):
    return {"frames":rows,"fps":30,"net_y":90,"player_identity":{"gallery_keys":{"P1":[],"P4":[]}}}


def test_stationary_defender_is_not_attack_or_block_despite_nearby_head():
    p=pose();ball=[50,90]
    assert not action_evidence(p,ball,"attack",90)["supported"]
    assert not action_evidence(p,ball,"block",90)["supported"]
    assert action_evidence(p,ball,"reception",90)["supported"]


def test_backcourt_attack_and_low_underhand_contact_remain_possible():
    p=pose(raised=True)
    assert action_evidence(p,[50,75],"attack",90)["supported"]
    p=pose()
    assert action_evidence(p,[50,180],"attack",90)["supported"]
    assert not action_evidence(p,[50,180],"block",90)["supported"]


def test_raised_other_arm_cannot_validate_defending_arm_close_to_ball():
    p=pose();p['kp'][10]=[80,75,1];p['kp'][8]=[80,100,1]
    evidence=action_evidence(p,[50,180],"block",90)
    assert evidence['wrist_joint']==10
    assert evidence['distance_body_heights']>.45


def test_jump_preserves_team_and_id_from_two_sided_same_raw_track():
    rows=[]
    for f in range(71):
        p=pose(raw=29,pid='P1' if f<26 or f>40 else None,team='b' if f<26 or f>40 else 'a',raised=26<=f<=40)
        rows.append({'f':f,'poses':[p],'ball':[50,75]})
    original=deepcopy(rows);context=ActorContext(annotations(rows))
    result=context.resolve(35,0)
    assert result['side']=='b' and result['player_id']=='P1'
    assert result['identity_method']=='two_sided_raw_track_identity'
    assert rows==original


def test_identity_does_not_bridge_long_gap_or_conflicting_people():
    rows=[]
    for f in range(71):
        pid='P1' if f<25 else 'P2' if f>40 else None
        rows.append({'f':f,'poses':[pose(raw=29,pid=pid,team='b' if pid else 'a')],'ball':[50,75]})
    result=ActorContext(annotations(rows)).resolve(35,0)
    assert result['player_id'] is None and result['side']=='a'
    sparse=[r for r in rows if r['f']<=35 or r['f']>=65]
    assert ActorContext(annotations(sparse)).resolve(35,0)['player_id'] is None


def test_cached_positive_reference_rejects_wrong_P4_and_selects_supported_person():
    rows=[{'f':f,'poses':[pose(raw=1,pid='P4'),pose(raw=2,pid=None,x=200)],'ball':[250,180]} for f in range(20)]
    features={f'{f}:{i}':{'distance':.5 if i==0 else .15,'other_distance':.45 if i==0 else .4}
              for f in range(20) for i in range(2)}
    context=ActorContext(annotations(rows),features)
    states=context.frame_context(10)
    assert states[0]['player_id'] is None and states[1]['player_id']=='P4'
    assert rows[10]['poses'][0]['player_id']=='P4' and rows[10]['poses'][1]['player_id'] is None


def test_P4_overlay_is_one_to_one_and_ambiguous_matches_stay_unknown():
    rows=[{'f':f,'poses':[pose(raw=1,pid=None),pose(raw=2,pid=None,x=200)],'ball':[50,75]} for f in range(20)]
    features={f'{f}:{i}':{'distance':.1+i*.03,'other_distance':.5} for f in range(20) for i in range(2)}
    states=ActorContext(annotations(rows),features).frame_context(10)
    assert all(s['player_id'] is None for s in states)


def test_active_actor_beats_defender_despite_strong_wrong_side_prior():
    rows=[{'f':f,'poses':[pose(),pose(raw=2,pid=None,team='a',raised=True,x=10)],'ball':[60,75]} for f in range(20)]
    data=annotations(rows);original=deepcopy(data)
    report=build_action_report([{'frame':10,'type':'attack'}],ActorContext(data),[.98],data['player_identity'],'test',False,False)
    event=report['contacts'][0]
    assert event['status']=='other_side' and event['candidates'][0]['raw_track_id']==2
    assert event['player_id'] is None and event['proposed_player_id'] is None
    assert event['rejected_inactive_P4']
    assert data==original


def test_low_forearm_reception_preserved_with_derived_ball_provenance():
    rows=[{'f':f,'poses':[pose()],'ball':[50,175],'ball_interp':True} for f in range(20)]
    data=annotations(rows)
    report=build_action_report([{'frame':10,'type':'reception'}],ActorContext(data),[.8],data['player_identity'],'test',False,False)
    event=report['contacts'][0]
    assert event['player_id']=='P4' and report['players'][1]['total']==1
    assert event['candidates'][0]['ball_source']=='interpolated_short'
    assert event['verified'] is False and event['label_source']=='derived'


def test_no_action_assignment_when_ball_is_off_camera():
    rows=[{'f':f,'poses':[pose(raised=True)],'ball':[50,75],'ball_out_of_frame':True} for f in range(20)]
    data=annotations(rows)
    event=build_action_report([{'frame':10,'type':'block'}],ActorContext(data),[.8],data['player_identity'],'test')['contacts'][0]
    assert event['status']=='unresolved' and not event['candidates']


def test_arm_motion_uses_same_track_relative_to_shoulder():
    rows=[{'f':f,'poses':[pose()],'ball':[50,175]} for f in range(20)]
    for f,row in enumerate(rows):
        p=row['poses'][0]
        # Whole-body translation must not become an arm swing.
        for joint in p['kp']:joint[0]+=f*3
    context=ActorContext(annotations(rows))
    assert context.arm_speed(10,rows[10]['poses'][0],9)==pytest.approx(0.)
    rows[13]['poses'][0]['kp'][9][1]-=60
    context=ActorContext(annotations(rows))
    assert context.arm_speed(10,rows[10]['poses'][0],9)>1
    rows[13]['poses'][0]['id']=2
    assert ActorContext(annotations(rows)).arm_speed(10,rows[10]['poses'][0],9) is None


def test_p4_overlay_does_not_rewrite_other_people_identity_context():
    rows=[]
    for f in range(71):
        p=pose(raw=29,pid='P1' if f<26 or f>40 else None,team='b' if f<26 or f>40 else 'a')
        rows.append({'f':f,'poses':[p],'ball':[50,75]})
    context=ActorContext(annotations(rows))
    assert context.frame_context(35)[0]['player_id']=='P1'
    assert context.frame_context(35,p4_only=True)[0]['player_id'] is None


def test_same_block_pose_does_not_become_preceding_opponent_attack():
    rows=[{'f':f,'poses':[pose(raised=True)],'ball':[50,75]} for f in range(30)]
    data=annotations(rows)
    report=build_action_report([{'frame':10,'type':'attack'},{'frame':15,'type':'block'}],
                              ActorContext(data),[.8,.8],data['player_identity'],'test',False,False)
    assert report['contacts'][0]['player_id'] is None and not report['contacts'][0]['candidates']
    assert report['contacts'][1]['player_id']=='P4'
    # A legal follow-up by the blocker is a different ordering and is retained.
    reverse=build_action_report([{'frame':10,'type':'block'},{'frame':15,'type':'attack'}],
                               ActorContext(data),[.8,.8],data['player_identity'],'test',False,False)
    assert reverse['contacts'][1]['player_id']=='P4'


def test_raised_hands_far_from_ball_do_not_prove_a_block():
    rows=[{'f':f,'poses':[pose(raised=True)],'ball':[-10,75]} for f in range(20)]
    data=annotations(rows)
    event=build_action_report([{'frame':10,'type':'block'}],ActorContext(data),[.9],data['player_identity'],'test',False,False)['contacts'][0]
    assert event['status']=='unresolved' and event['player_id'] is None and event['proposed_player_id'] is None
    assert event['reason']=='active_actor_too_far' and event['side_accepted'] is False


@pytest.mark.parametrize('scale',[1.,2.])
def test_ground_role_uses_standing_anchors_and_is_resolution_invariant(scale):
    from src.tracking.court_setups import WORLD_CORNERS
    H=np.array([[100.,0.,450.],[5.,100.,900.],[0.,0.,1.]])
    corners=[]
    for x,y in WORLD_CORNERS:
        p=H@np.array([x,y,1.]);corners.append((p[:2]*scale).tolist())
    rows=[]
    for f in range(71):
        p=pose();dy=1155-max(0,150-15*abs(f-35))
        for joint in p['kp']:joint[0]=(joint[0]+400)*scale;joint[1]=(joint[1]+dy)*scale
        p['bb']=[(p['bb'][0]+400)*scale,(p['bb'][1]+dy)*scale,(p['bb'][2]+400)*scale,(p['bb'][3]+dy)*scale]
        rows.append({'f':f,'poses':[p],'ball':[0,0]})
    data=annotations(rows);data.update(court_corners=corners,net_floor_left=[0,877.5*scale],net_floor_right=[900*scale,922.5*scale])
    result=ActorContext(data).resolve(35,0)
    assert result['position']=='backcourt' and result['depth_m']==pytest.approx(5.,abs=.05)
    assert result['rotation_eligibility_known'] is False
