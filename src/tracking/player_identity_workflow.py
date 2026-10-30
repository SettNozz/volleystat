"""Offline identity pass over final saved poses, preserving raw track IDs/events."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models.player_reid_encoder import file_sha256
from src.tracking.player_identity import default_seed, overlap
from src.tracking.player_reid import ReIDParameters, associate

METHOD='osnet_frozen_gallery_kalman_roster_v4'
PRODUCTION_PARAMETERS=ReIDParameters(motion_model='kalman',tracklet_consistency=True,
                                    local_appearance_regret_max=.04)


def observations_from_annotations(annotations:list[dict],side:str='b') -> list[dict]:
    frames=[]
    for ann in annotations:
        players=[]
        poses=ann.get('poses') or []
        for index,pose in enumerate(poses):
            if pose.get('team') not in (side,f'team_{side}'):
                continue
            box=pose['bb']
            if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
                raise ValueError(f'Invalid player box at frame {ann["f"]}')
            players.append({'key':f'{ann["f"]}:{index}','pose_index':index,'bb':box,'raw_id':pose.get('id'),
                            'occlusion':max((overlap(box,p['bb']) for j,p in enumerate(poses) if j!=index),default=0.)})
        frames.append({'f':int(ann['f']),'players':players})
    if len({f['f'] for f in frames})!=len(frames) or [f['f'] for f in frames]!=sorted(f['f'] for f in frames):
        raise ValueError('Identity pass requires unique ordered source frames')
    return frames


def _match_anchor_boxes(frame,players):
    if not players or len(frame['players'])<len(players):
        raise ValueError('Anchor detections missing from the processed frame')
    if any(len(p['bb'])!=4 or not np.isfinite(p['bb']).all()
           or p['bb'][2]<=p['bb'][0] or p['bb'][3]<=p['bb'][1] for p in players):
        raise ValueError('Invalid roster anchor box')
    ids=[p['player_id'] for p in players]
    if len(set(ids))!=len(ids) or any(not isinstance(pid,str) or not pid.strip() or len(pid)>40 for pid in ids):
        raise ValueError('Manual roster IDs must be distinct nonempty strings')
    def iou(a,b):
        area=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
        union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
        return area/max(union,1.)
    similarity=np.asarray([[iou(p['bb'],q['bb']) for q in frame['players']] for p in players])
    rows,columns=linear_sum_assignment(1.-similarity)
    assignments={}
    for i,j in zip(rows,columns):
        rivals=np.delete(similarity[i],j)
        if similarity[i,j]<.65 or similarity[i,j]-np.max(rivals,initial=0.)<.1:
            raise ValueError('Manual seed boxes do not uniquely match the new detections')
        assignments[frame['players'][j]['key']]=ids[i]
    return assignments


def load_seed(path:Path,frames:list[dict],video_path:Path|None,side:str,roster_size:int)->dict:
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    schema=payload.get('schema')
    if schema not in ('player_identity_seed_v1','player_identity_roster_v2'):
        raise ValueError('Identity seed must be a verified player_identity_seed_v1 file or confirmed roster v2')
    if schema=='player_identity_seed_v1':
        if payload.get('label_source')!='human' or payload.get('verified') is not True:
            raise ValueError('Identity seed must be a verified player_identity_seed_v1 file')
        initial=payload
    else:
        initial=payload.get('initial') or {}
        if not ((initial.get('label_source')=='human' and initial.get('verified') is True)
                or (initial.get('label_source')=='automatic_initialisation' and initial.get('verified') is False)):
            raise ValueError('Initial roster provenance is missing')
    if video_path is None or payload.get('video_sha256')!=file_sha256(video_path):
        raise ValueError('Identity seed belongs to a different source video')
    if (payload.get('side')!=side or len(initial.get('players',[]))!=roster_size
            or payload.get('max_on_court',roster_size)!=roster_size):
        raise ValueError('Identity seed roster/side differs from this run')
    by_frame={f['f']:f for f in frames}
    frame=by_frame.get(initial.get('frame'))
    if frame is None or len(frame['players'])!=roster_size:
        raise ValueError('Manual seed frame/complete roster missing from processed detections; check frame skipping')
    seed={'frame':frame['f'],'assignments':_match_anchor_boxes(frame,initial['players']),
          'label_source':initial['label_source'],'verified':initial['verified'],
          'seed_file':str(Path(path).resolve()),'seed_sha256':file_sha256(path),'matching':'one_to_one_bbox_iou'}
    if schema=='player_identity_roster_v2':
        introductions=[]
        for addition in payload.get('introductions',[]):
            confirmation=addition.get('identity_confirmation') or {}
            if (confirmation.get('label_source')!='human' or confirmation.get('verified') is not True
                    or confirmation.get('scope')!='new_person' or not confirmation.get('note')):
                raise ValueError('New roster identity needs explicit human confirmation')
            anchor=addition['anchor']
            target=by_frame.get(anchor['frame'])
            start=addition.get('available_from_frame',anchor['frame'])
            if target is None or not isinstance(start,int) or start<0 or start>anchor['frame']:
                raise ValueError('Invalid introduction frame/availability')
            assignment=_match_anchor_boxes(target,[{'player_id':addition['player_id'],'bb':anchor['bb']}])
            key,pid=next(iter(assignment.items()))
            introductions.append({'key':key,'player_id':pid,'frame':anchor['frame'],
                                  'available_from_frame':start,'identity_confirmation':confirmation,
                                  'anchor_provenance':anchor.get('provenance','detector box'),
                                  'availability_provenance':addition.get('availability_provenance',{})})
        seed['introductions']=introductions
        from src.tracking.player_reid import identity_anchors
        identity_anchors(seed)
    return seed


def identity_seed(frames,roster_size=6,seed_path=None,video_path=None,side='b'):
    if seed_path is not None:
        return load_seed(seed_path,frames,video_path,side,roster_size)
    return default_seed(frames,roster_size)


def apply_player_identities(annotations,frames,seed,features,fps,side='b',parameters=PRODUCTION_PARAMETERS):
    result=associate(frames,seed,features,fps,recover=True,parameters=parameters)
    return apply_identity_result(annotations,seed,result,side)


def apply_identity_result(annotations,seed,result,side='b'):
    """Export a validated association; detectors, geometry and events stay intact."""
    expected={f'{ann["f"]}:{i}' for ann in annotations for i,p in enumerate(ann.get('poses') or [])
              if p.get('team') in (side,'team_'+side)}
    if set(result['predictions'])!=expected:
        raise ValueError('Association keys differ from saved court-side observations')
    parameters=result['parameters']
    method=('osnet_frozen_gallery_kalman_roster_v3'
            if parameters.get('tracklet_consistency') or seed.get('introductions')
            else 'osnet_frozen_gallery_kalman_v2')
    if parameters.get('local_appearance_regret_max') is not None:
        method='osnet_frozen_gallery_kalman_roster_v4'
    counts=Counter()
    by_person=Counter()
    for ann in annotations:
        assigned=set()
        for index,pose in enumerate(ann.get('poses') or []):
            key=f'{ann["f"]}:{index}'
            if key not in result['predictions']:
                continue
            prediction=result['predictions'][key]
            pid=prediction['player_id']
            if pid is not None:
                if pid in assigned:
                    raise ValueError('Two simultaneous detections share a roster identity')
                assigned.add(pid)
                by_person[pid]+=1
            pose['player_id']=pid
            pose['identity']={'method':method,**{k:v for k,v in prediction.items() if k!='player_id'}}
            counts['observations']+=1
            counts['assigned']+=pid is not None
    return {'enabled':True,'status':'completed','method':method,'side':side,'roster_size':len(result['gallery_keys']),
            'max_on_court':len(seed['assignments']),'tracklet_consistency':result['tracklet_consistency'],
            'seed':seed,'gallery_keys':result['gallery_keys'],'parameters':result['parameters'],
            'observations':counts['observations'],'assigned':counts['assigned'],
            'coverage':counts['assigned']/max(counts['observations'],1),'person_observations':dict(by_person),
            'label_source':'model','verified':False,'contact_attribution':False,
            'scope':'one court side; roster grows only from confirmed new-person anchors; existing IDs are retained on return'}


def run_identity_pass(annotations,images,fps,encoder,side='b',roster_size=6,seed_path=None,video_path=None,progress=None,
                      feature_cache=None):
    frames=observations_from_annotations(annotations,side)
    if not any(f['players'] for f in frames):
        return {'enabled':True,'status':'no_players','method':METHOD,'side':side,'assigned':0,'coverage':0.}
    try:
        seed=identity_seed(frames,roster_size,seed_path,video_path,side)
    except ValueError:
        if seed_path is not None:
            raise
        for ann in annotations:
            for pose in ann.get('poses') or []:
                if pose.get('team') in (side,f'team_{side}'):
                    pose['player_id']=None
                    pose['identity']={'method':METHOD,'reason':'no_complete_roster','verified':False,'label_source':'model'}
        return {'enabled':True,'status':'no_complete_roster','method':METHOD,'side':side,'assigned':0,'coverage':0.}
    from src.models.inference_cache import collect_embeddings
    features=collect_embeddings(frames,images,encoder,feature_cache,progress=progress)
    report=apply_player_identities(annotations,frames,seed,features,fps,side)
    report['encoder']=encoder.provenance
    return report
