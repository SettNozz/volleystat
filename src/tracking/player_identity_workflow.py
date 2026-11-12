"""Offline identity pass over final saved poses, preserving raw track IDs/events."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models.player_reid_encoder import file_sha256
from src.tracking.player_identity import default_seed, overlap
from src.tracking.player_reid import ReIDParameters, associate, identity_anchors
from src.tracking.player_roster_discovery import (
    RosterDiscoveryParameters, RosterNotFound, assign_roster, discover_roster, name_left_to_right)

# Default: the roster comes from tracklets of the whole video. The sequential
# seed-frame tracker stays for short clips, jersey constraints and old replays.
METHOD='osnet_tracklet_roster_v5'
SEED_FRAME_METHOD='osnet_frozen_gallery_kalman_roster_v4'
IDENTITY_METHODS=('roster-discovery','seed-frame')
DISCOVERY_PARAMETERS=RosterDiscoveryParameters()
PRODUCTION_PARAMETERS=ReIDParameters(motion_model='kalman',tracklet_consistency=True,
                                    local_appearance_regret_max=.04)


def observations_from_annotations(annotations:list[dict],side:str | None='b', *, identity_map=None, segments=()) -> list[dict]:
    frames=[]
    for ann in annotations:
        from src.tracking.player_team_membership import roster_side_at_frame
        local_side = roster_side_at_frame(ann['f'], side, identity_map, segments) if side is not None else None
        players=[]
        poses=ann.get('poses') or []
        for index,pose in enumerate(poses):
            if local_side is not None and pose.get('team') not in (local_side,f'team_{local_side}'):
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
    if frame is None or len(frame['players'])<roster_size:
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
    return apply_identity_result(annotations,seed,result,side,
                                 expected_keys={p['key'] for f in frames for p in f['players']})


def _write_predictions(annotations,predictions,side,method,expected_keys=None):
    """Add IDs to saved poses; detectors, geometry and events stay intact."""
    expected={f'{ann["f"]}:{i}' for ann in annotations for i,p in enumerate(ann.get('poses') or [])
              if p.get('team') in (side,'team_'+side)}
    if expected_keys is not None:
        expected = expected_keys
    if set(predictions)!=expected:
        raise ValueError('Association keys differ from saved court-side observations')
    for ann in annotations:
        assigned=[predictions[key]['player_id'] for index in range(len(ann.get('poses') or []))
                  if (key:=f'{ann["f"]}:{index}') in predictions and predictions[key]['player_id'] is not None]
        if len(set(assigned))!=len(assigned):
            raise ValueError('Two simultaneous detections share a roster identity')
    counts=Counter()
    by_person=Counter()
    for ann in annotations:
        for index,pose in enumerate(ann.get('poses') or []):
            key=f'{ann["f"]}:{index}'
            if key not in predictions:
                continue
            prediction=predictions[key]
            pid=prediction['player_id']
            if pid is not None:
                by_person[pid]+=1
            pose['player_id']=pid
            pose['identity']={'method':method,**{k:v for k,v in prediction.items() if k!='player_id'}}
            counts['observations']+=1
            counts['assigned']+=pid is not None
    return counts,by_person


def apply_identity_result(annotations,seed,result,side='b',expected_keys=None):
    """Export a validated association; detectors, geometry and events stay intact."""
    parameters=result['parameters']
    method=('osnet_frozen_gallery_kalman_roster_v3'
            if parameters.get('tracklet_consistency') or seed.get('introductions')
            else 'osnet_frozen_gallery_kalman_v2')
    if parameters.get('local_appearance_regret_max') is not None:
        method='osnet_frozen_gallery_kalman_roster_v4'
    counts,by_person=_write_predictions(annotations,result['predictions'],side,method,expected_keys)
    return {'enabled':True,'status':'completed','method':method,'side':side,'roster_size':len(result['gallery_keys']),
            'max_on_court':len(seed['assignments']),'tracklet_consistency':result['tracklet_consistency'],
            'seed':seed,'gallery_keys':result['gallery_keys'],'parameters':result['parameters'],
            'observations':counts['observations'],'assigned':counts['assigned'],
            'coverage':counts['assigned']/max(counts['observations'],1),'person_observations':dict(by_person),
            'label_source':'model','verified':False,'contact_attribution':False,
            'scope':'one court side; roster grows only from confirmed new-person anchors; existing IDs are retained on return'}


def discovery_possible(frames,fps,roster_size=6,parameters=DISCOVERY_PARAMETERS):
    """Cheap check before any embedding is read: enough clean tracklets for a roster?"""
    limit=max(1,round(parameters.chunk_seconds*fps))
    gap=max(1,round(parameters.chunk_gap_seconds*fps))
    usable=0
    runs={}
    for frame in frames:
        for obs in frame['players']:
            raw=obs.get('raw_id')
            if raw is None or obs['occlusion']>=parameters.clean_overlap_max:
                continue
            start,previous,count=runs.get(raw,(frame['f'],frame['f'],0))
            if frame['f']-previous>gap or frame['f']-start>=limit:
                start,count=frame['f'],0
            runs[raw]=(start,frame['f'],count+1)
            usable+=count+1==parameters.chunk_min_clean
    return usable>=roster_size


def apply_discovered_identities(annotations,frames,features,fps,side='b',roster_size=6,seed=None,
                                parameters=DISCOVERY_PARAMETERS):
    """Roster from tracklets of the whole video; confirmed anchors name it when given.

    Raises ``RosterNotFound`` before anything is written when the video does not
    show ``roster_size`` separable constant players.
    """
    anchors=identity_anchors(seed) if seed is not None else None
    found=discover_roster(frames,features,fps,roster_size,parameters,anchors=anchors)
    roster=found['roster']
    forced,available={},{}
    if seed is None:
        seed=name_left_to_right(frames,roster)
    else:
        forced={a['key']:a['player_id'] for a in anchors}
        available={a['player_id']:a.get('available_from_frame') for a in anchors}
    predictions=assign_roster(frames,features,fps,roster,parameters,available_from=available,forced=forced,
                              cohort=found['cohort'])
    counts,by_person=_write_predictions(annotations,predictions,side,METHOD,
                                       {p['key'] for f in frames for p in f['players']})
    return {'enabled':True,'status':'completed','method':METHOD,'side':side,'roster_size':len(roster),
            'max_on_court':len(seed['assignments']),'tracklet_consistency':{'enabled':False},
            'seed':seed,'gallery_keys':{p['player_id']:p['gallery_keys'] for p in roster},
            'parameters':parameters.to_dict(),
            'roster_discovery':{
                'status':'completed','tracklets':found['tracklets'],'people':found['people'],
                'selection':'confirmed_anchors' if anchors else 'longest_presence',
                'players':[{k:p[k] for k in ('player_id','presence_frames','tracklets','first_frame','last_frame')}
                           for p in roster],
                # People seen on this side who are not in the roster. 'other_person' was
                # seen together with every roster player; 'unresolved' may be a bystander,
                # a roster player from an unusual angle, or a substitute awaiting confirmation.
                'other_people':found['other_people'],'known_other_people':len(found['cohort']),
                'label_source':'model','verified':False},
            'observations':counts['observations'],'assigned':counts['assigned'],
            'coverage':counts['assigned']/max(counts['observations'],1),'person_observations':dict(by_person),
            'label_source':'model','verified':False,'contact_attribution':False,
            'scope':'one court side; roster people are found from tracklets of the whole video; '
                    'substitutes still need a confirmed new-person anchor'}


def apply_jersey_identities(annotations,frames,seed,features,fps,reads,side='b',jersey_parameters=None,
                            parameters=PRODUCTION_PARAMETERS):
    """Two passes: the first gives roster jersey numbers, the second uses them as constraints."""
    from src.tracking.jersey_numbers import JerseyParameters, jersey_constraints
    jersey_parameters=jersey_parameters or JerseyParameters()
    first=associate(frames,seed,features,fps,recover=True,parameters=parameters)
    constraints=jersey_constraints(frames,reads['reads'],fps,seed,first['predictions'],jersey_parameters)
    result=associate(frames,seed,features,fps,recover=True,parameters=parameters,jersey=constraints)
    report=apply_identity_result(annotations,seed,result,side,
                                 expected_keys={p['key'] for f in frames for p in f['players']})
    first_assigned=sum(p['player_id'] is not None for p in first['predictions'].values())
    report['jersey_numbers']={
        'enabled':True,'roster':constraints['roster'],'roster_source':constraints['roster_source'],
        'observations_with_number':len(constraints['observations']),
        'tracklets_with_number':constraints['tracklets_with_number'],
        'assigned_without_numbers':first_assigned,'parameters':constraints['parameters'],
        'reader':reads.get('reader'),'reads_annotations_sha256':reads.get('annotations_sha256'),
        'label_source':'model','verified':False}
    return report


def run_identity_pass(annotations,images,fps,encoder,side='b',roster_size=6,seed_path=None,video_path=None,progress=None,
                      feature_cache=None,jersey_reads=None,method='roster-discovery',
                      preserve_membership=False,identity_map=None,segments=()):
    if method not in IDENTITY_METHODS:
        raise ValueError(f'Unknown identity method: {method}')
    preseed = None
    if preserve_membership and seed_path is not None:
        # Locate known people by their photo boxes before filtering by side;
        # otherwise the foot-side bug can discard its own correction anchor.
        preseed = load_seed(seed_path, observations_from_annotations(annotations, None),
                            video_path, side, roster_size)
        from src.tracking.player_team_membership import apply_seed_membership
        apply_seed_membership(annotations, preseed, side, fps, identity_map=identity_map, segments=segments)
    frames=observations_from_annotations(annotations,side,identity_map=identity_map,segments=segments)
    if not any(f['players'] for f in frames):
        return {'enabled':True,'status':'no_players','method':METHOD,'side':side,'assigned':0,'coverage':0.}
    # Jersey constraints are implemented in the sequential tracker only.
    discover=method=='roster-discovery' and jersey_reads is None
    fallback=None
    if discover and not discovery_possible(frames,fps,roster_size):
        discover,fallback=False,'too few clean tracklets for a roster'
    from src.models.inference_cache import collect_embeddings
    feature_frames = observations_from_annotations(annotations, side=None) if preserve_membership else frames
    def finish(report, features):
        if preserve_membership:
            from src.tracking.player_team_membership import preserve_roster_membership
            preserve_roster_membership(annotations, report, features, fps,
                                      identity_map=identity_map, segments=segments)
        return report
    features=None
    if discover:
        seed=preseed or (load_seed(seed_path,frames,video_path,side,roster_size) if seed_path is not None else None)
        features=collect_embeddings(feature_frames,images,encoder,feature_cache,progress=progress)
        try:
            report=apply_discovered_identities(annotations,frames,features,fps,side,roster_size,seed)
        except RosterNotFound as exc:
            fallback=str(exc)
        else:
            report['encoder']=encoder.provenance
            return finish(report, features)
    try:
        seed=preseed or identity_seed(frames,roster_size,seed_path,video_path,side)
    except ValueError:
        if seed_path is not None:
            raise
        for ann in annotations:
            for pose in ann.get('poses') or []:
                if pose.get('team') in (side,f'team_{side}'):
                    pose['player_id']=None
                    pose['identity']={'method':METHOD,'reason':'no_complete_roster','verified':False,'label_source':'model'}
        report={'enabled':True,'status':'no_complete_roster','method':METHOD,'side':side,'assigned':0,'coverage':0.}
        if fallback:
            report['roster_discovery']={'status':'not_found','reason':fallback}
        return report
    if features is None:
        features=collect_embeddings(feature_frames,images,encoder,feature_cache,progress=progress)
    if jersey_reads is not None:
        report=apply_jersey_identities(annotations,frames,seed,features,fps,jersey_reads,side)
    else:
        report=apply_player_identities(annotations,frames,seed,features,fps,side)
    if fallback:
        # The seed-frame tracker ran because the whole-video roster was not found.
        report['roster_discovery']={'status':'fallback_to_seed_frame','reason':fallback}
    report['encoder']=encoder.provenance
    return finish(report, features)
