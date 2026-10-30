"""Offline team/anatomical contact features for an isolated event-model experiment.

Production 18-channel checkpoints and human labels are untouched. Raw skeletons
remain in the frame cache; the classifier uses compact contact geometry per side.
"""
from __future__ import annotations
import numpy as np
from action_recognition.event_clips.config import ALL_FEATURE_NAMES

SCHEMA='resolution_normalized_team_pose_offline_v1'
CANDIDATE_SCHEMA='resolution_normalized_team_pose_contact_candidate_v2'
TEAM_NAMES=('count','body_ball_dist','wrist_ball_dist','elbow_ball_dist',
            'forearm_ball_dist','head_ball_dist','knee_ball_dist','ankle_ball_dist',
            'wrist_above_net','arm_raise','jump','torso_confidence')
VALUE_NAMES=tuple(ALL_FEATURE_NAMES)+tuple(f'{team}_{name}' for team in ('a','b') for name in TEAM_NAMES)
FEATURE_NAMES=VALUE_NAMES+tuple('valid_'+name for name in VALUE_NAMES)


def point_segment_distance(point,a,b):
    a,b,point=map(lambda p:np.asarray(p,dtype=float),(a,b,point))
    delta=b-a;den=float(delta@delta)
    alpha=np.clip(float((point-a)@delta)/den,0,1) if den>1e-12 else 0.
    return float(np.linalg.norm(point-(a+alpha*delta)))


def team_contact_features(poses,ball,net_y,pose_extractor):
    """24 values + availability; no FSM state, clip labels or human team input."""
    values=np.zeros(24,np.float32);valid=np.zeros(24,bool)
    for team_no,team in enumerate(('team_a','team_b')):
        people=[p for p in poses if p.team==team];base=team_no*12
        values[base]=len(people)/6;valid[base]=True
        if not people:continue
        candidates={i:[] for i in range(1,12)}
        for p in people:
            height=p.bbox_h
            if ball is not None:
                x,y=ball;a,b,c,d=p.bbox
                dx=max(a-x,0,x-c);dy=max(b-y,0,y-d)
                candidates[1].append(min(float(np.hypot(dx,dy)/height),10.))
                for channel,indices in ((2,(9,10)),(3,(7,8)),(5,(0,)),(6,(13,14)),(7,(15,16))):
                    points=[p.kp(k) for k in indices if p.kp(k) is not None]
                    if points:candidates[channel].append(min(min(float(np.linalg.norm(np.array(q)-ball)/height) for q in points),10.))
                for elbow,wrist in ((7,9),(8,10)):
                    a,b=p.kp(elbow),p.kp(wrist)
                    if a is not None and b is not None:
                        candidates[4].append(min(point_segment_distance(ball,a,b)/height,10.))
            wrists=p.wrists()
            shoulders=[p.kp(k) for k in (5,6) if p.kp(k) is not None]
            if wrists:
                top=min(q[1] for q in wrists)
                candidates[8].append(float(np.clip((net_y-top)/height,-2,2)))
                if shoulders:candidates[9].append(float(np.clip((min(q[1] for q in shoulders)-top)/height,-1,2)))
            if p.track_id is not None and p.ankles() and len(pose_extractor._ankle_hist.get(p.track_id,()))>=pose_extractor.MIN_HISTORY:
                candidates[10].append(pose_extractor._jump(p))
            candidates[11].append(float(np.mean(p.keypoints[[5,6,11,12],2])))
        for i,observations in candidates.items():
            if observations:
                values[base+i]=min(observations) if i<=7 else max(observations)
                valid[base+i]=True
    return values,valid


def contact_candidate(people,ball):
    """One anatomical candidate in image pixels; no class or human contact input."""
    if ball is None:return None
    def distance(p):
        ds=[float(np.linalg.norm(np.asarray(p.kp(k))-ball))
            for k in range(17) if p.kp(k) is not None]
        for elbow,wrist in ((7,9),(8,10)):
            a,b=p.kp(elbow),p.kp(wrist)
            if a is not None and b is not None:
                ds.append(point_segment_distance(ball,a,b))
        return min(ds,default=float('inf'))
    eligible=[p for p in people if any(p.kp(k) is not None for k in range(17))]
    return min(eligible,key=distance) if eligible else None


def candidate_team_features(poses,ball,net_y,pose_extractor):
    """Counts describe each side; all remaining channels describe ONE candidate."""
    values=np.zeros(24,np.float32);valid=np.zeros(24,bool)
    for side,team in enumerate(('team_a','team_b')):
        people=[p for p in poses if p.team==team];base=side*12
        values[base]=len(people)/6;valid[base]=True
        candidate=contact_candidate(people,ball)
        if candidate is not None:
            v,m=team_contact_features([candidate],ball,net_y,pose_extractor)
            values[base+1:base+12]=v[base+1:base+12]
            valid[base+1:base+12]=m[base+1:base+12]
    return values,valid


def normalize_observations(values,valid,mean,std):
    """Missing observations become train-mean => standardized zero, not observed zero."""
    values=np.asarray(values,np.float32);valid=np.asarray(valid,bool)
    if values.shape!=valid.shape or not np.isfinite(values).all():
        raise ValueError('Invalid values/availability')
    normalized=np.where(valid,(values-mean)/std,0).astype(np.float32)
    return np.concatenate([normalized,valid.astype(np.float32)],axis=-1)


def observation_stats(sequences):
    values=np.concatenate([v for v,m in sequences]);valid=np.concatenate([m for v,m in sequences])
    n=valid.sum(0);mean=np.divide(np.where(valid,values,0).sum(0),n,out=np.zeros(values.shape[1]),where=n>0)
    variance=np.divide(np.where(valid,(values-mean)**2,0).sum(0),n,out=np.zeros(values.shape[1]),where=n>0)
    std=np.sqrt(variance);std[std<1e-6]=1
    return mean.astype(np.float32),std.astype(np.float32)
