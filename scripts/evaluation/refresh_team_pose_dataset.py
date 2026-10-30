"""Versioned human event dataset refresh; frozen balls plus fresh causal poses."""
from __future__ import annotations
import argparse,collections,copy,json,sys,time,shutil
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline.game_analyzer import GameAnalyzer,CalibrationData
from src.detection.sam_pose_recovery import file_hash
from src.tracking.pose_feature_extractor import PoseFeatureExtractor
from action_recognition.event_clips.config import collapse_labels
from action_recognition.event_clips.compute_features import compute_clip_features
from action_recognition.event_clips.feature_schema import RESOLUTION_NORMALIZED_SCHEMA
from action_recognition.event_clips.team_pose_features import SCHEMA,VALUE_NAMES,FEATURE_NAMES,team_contact_features


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)


def intervals(clips,warmup=90):
    result=[]
    for start,end in sorted((max(0,c['start_frame']-warmup),c['end_frame']) for c in clips):
        if result and start<=result[-1][1]+1:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result


def pose_validity(poses,ball,net_y,extractor):
    valid=np.zeros(8,bool)
    if not poses:return valid
    p=min(poses,key=lambda p:np.linalg.norm(np.array(p.center)-ball)) if ball is not None else min(poses,key=lambda p:abs(p.center[1]-net_y))
    wrists=p.wrists();valid[0]=bool(wrists and ball is not None);valid[1]=bool(wrists)
    valid[2]=p.track_id is not None and bool(p.ankles()) and len(extractor._ankle_hist.get(p.track_id,()))>=extractor.MIN_HISTORY
    valid[3]=bool(wrists and (p.kp(5) or p.kp(6)));valid[4:6]=True
    near=[q for q in poses if q.team=='team_b']
    valid[6]=any(q.track_id is not None and q.ankles() and len(extractor._ankle_hist.get(q.track_id,()))>=extractor.MIN_HISTORY for q in near)
    valid[7]=any(q.wrists() for q in near)
    return valid


def prepare(args):
    source=Path(args.manifest);parent=json.loads(source.read_text(encoding='utf-8-sig'))
    if any(c.get('label_source')!='human' for c in parent['clips']):raise ValueError('Only human labels authorized')
    # A fixed offline context around each candidate contact, independent of class/source.
    parent=copy.deepcopy(parent)
    for c in parent['clips']:
        ball_path=Path(args.ball_results)/(c['source_stem']+'_annotations.json')
        c['original_start_frame']=c['start_frame'];c['original_end_frame']=c['end_frame']
        c['original_num_frames']=c.get('num_frames')
        c['start_frame']=max(0,c['contact_frame']-60)
        c['end_frame']=c['contact_frame']+59
        c['feature_window']='offline_centered_120';c['contact_position']=60
        c['num_frames']=c['end_frame']-c['start_frame']+1
    # Frozen ball analyses can end before the video; unavailable edge context is padded.
    for stem in {c['source_stem'] for c in parent['clips']}:
        payload=json.loads((Path(args.ball_results)/(stem+'_annotations.json')).read_text())
        last=max(r['f'] for r in payload['frames'])
        for c in parent['clips']:
            if c['source_stem']==stem:
                c['end_frame']=min(c['end_frame'],last)
                c['num_frames']=c['end_frame']-c['start_frame']+1
    out=Path(args.output);manifest_path=out/'manifest.json' 
    if out.exists():
        if not args.resume:raise ValueError('Output exists; use --resume only for the same frozen experiment')
        meta=json.loads((out/'preflight.json').read_text())
        if meta['parent_manifest_sha256']!=file_hash(source):raise ValueError('Human manifest changed; refuse resume')
        if args.pose_imgsz!=meta['pose_imgsz']:raise ValueError('Pose settings changed; refuse mixed resume')
        return parent,meta
    out.mkdir(parents=True)
    (out/'features').mkdir();(out/'poses').mkdir();(out/'baseline_features_snapshot').mkdir()
    shutil.copy2(source,out/'parent_manifest.json');shutil.copy2(args.split_policy,out/'split_policy.json')
    for p in Path(args.baseline_features).iterdir():
        if p.suffix in ('.parquet','.npz','.json'):shutil.copy2(p,out/'baseline_features_snapshot'/p.name)
    setup_payload=json.loads(Path(args.court_setups).read_text())
    setup=next(s for s in setup_payload['setups'] if s['name']==args.setup_name)
    # Reuse the existing named setup for the matching fixed camera, not hy2S geometry.
    corners=np.asarray(setup['court_corners'],np.float32)*[3840/setup['frame_width'],2160/setup['frame_height']]
    H=cv2.getPerspectiveTransform(np.array([[-4.5,-9],[4.5,-9],[4.5,9],[-4.5,9]],np.float32),corners.astype(np.float32))
    geometry=dict(homography=H.tolist(),court_corners=corners.tolist(),frame_width=3840,frame_height=2160,
        net_y=setup['net_y']*2160/setup['frame_height'],net_x_left=setup['net_x_left']*3840/setup['frame_width'],
        net_x_right=setup['net_x_right']*3840/setup['frame_width'],calibration_source=f'existing_named_setup:{args.setup_name}',
        geometry_provenance='model_geometry_not_event_ground_truth')
    homogeneous=np.array([[-4.5,0,1],[4.5,0,1]]) @ H.T
    mid=homogeneous[:,:2]/homogeneous[:,2:]
    def floor_at(x):
        return [float(x),float(mid[0,1]+(x-mid[0,0])*(mid[1,1]-mid[0,1])/(mid[1,0]-mid[0,0]))]
    geometry['net_floor_left']=floor_at(geometry['net_x_left'])
    geometry['net_floor_right']=floor_at(geometry['net_x_right'])
    geometry['net_floor_method']='calibrated_court_midline_fallback'
    save(out/'geometry.json',geometry)
    counts=collections.Counter(c['source_stem'] for c in parent['clips'])
    core=collections.Counter(c['source_stem'] for c in parent['clips'] if collapse_labels(c.get('event_types'),c.get('fsm_event_type')) is not None)
    sources={}
    for stem in sorted(counts):
        clips=[c for c in parent['clips'] if c['source_stem']==stem]
        video=Path(clips[0]['source_video']);ball=Path(args.ball_results)/(stem+'_annotations.json')
        if not video.is_file() or not ball.is_file():raise ValueError(f'Missing source for {stem}')
        ranges=intervals(clips)
        sources[stem]=dict(video=str(video),video_sha256=file_hash(video),ball_annotations=str(ball),ball_sha256=file_hash(ball),
                           intervals=ranges,required_pose_frames=sum(b-a+1 for a,b in ranges),clips=counts[stem],core_clips=core[stem])
    meta=dict(status='prepared',parent_manifest=str(source.resolve()),parent_manifest_sha256=file_hash(source),
        manifest=str(manifest_path.resolve()),features=str((out/'features').resolve()),label_source='human',
        feature_schema=SCHEMA,value_names=VALUE_NAMES,input_features=FEATURE_NAMES,input_dim=len(FEATURE_NAMES),
        sources=sources,split_policy=json.loads(Path(args.split_policy).read_text()),
        pose_checkpoint=str(Path(args.pose_model).resolve()),pose_checkpoint_sha256=file_hash(args.pose_model),pose_imgsz=args.pose_imgsz,
        selection='fixed offline [-60,+59] contact context; causal 90-frame pose warmup; no FSM/labels as feature inputs',
        geometry=geometry,sam_masks_used=False)
    save(out/'preflight.json',meta)
    clone=copy.deepcopy(parent);clone['feature_schema']=SCHEMA
    clone['parent_manifest_sha256']=meta['parent_manifest_sha256']
    clone['window_policy']=dict(type='offline_centered',frames=120,before=60,after=59,contact_position=60,padding='explicit_unavailable_at_edges')
    for c in clone['clips']:
        c['feature_schema']=SCHEMA;c['window_policy']='offline_centered'
        c['contact_offset']=c['contact_frame']-c['start_frame']
        c['post_contact_frames']=c['end_frame']-c['contact_frame']
        c['features_path']='features/'+c['clip_id']+'.npz';c['features_format']='npz' 
    save(manifest_path,clone)
    print(json.dumps(meta,indent=2),flush=True)
    return parent,meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True);ap.add_argument('--ball-results',required=True)
    ap.add_argument('--split-policy',default='data/event_clips_4k_gold_v1/split_policy.json')
    ap.add_argument('--baseline-features',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--pose-model',default='models/pretrained/yolo11m-pose.pt')
    ap.add_argument('--pose-imgsz',type=int,default=2560);ap.add_argument('--court-setups',default='configs/court_setups.json')
    ap.add_argument('--setup-name',default='setup_1');ap.add_argument('--resume',action='store_true');ap.add_argument('--check',action='store_true')
    args=ap.parse_args();parent,meta=prepare(args);out=Path(args.output)
    if args.check:return
    if file_hash(args.pose_model)!=meta['pose_checkpoint_sha256']:raise ValueError('Pose weights changed')
    from ultralytics import YOLO
    geometry=meta['geometry']
    started=time.time()
    for stem,source in meta['sources'].items():
        clips=[c for c in parent['clips'] if c['source_stem']==stem]
        if file_hash(source['ball_annotations'])!=source['ball_sha256'] or file_hash(source['video'])!=source['video_sha256']:
            raise ValueError('Source inputs changed since preflight')
        if all((out/'features'/(c['clip_id']+'.npz')).exists() for c in clips):
            print('Already complete:',stem,flush=True);continue
        ball_payload=json.loads(Path(source['ball_annotations']).read_text());balls={r['f']:r for r in ball_payload['frames']}
        fps=ball_payload['fps'];ranges=source['intervals'];light={};coverage=collections.Counter()
        an=GameAnalyzer.__new__(GameAnalyzer)
        an.calibration=CalibrationData(net_y=geometry['net_y'],net_x_left=geometry['net_x_left'],net_x_right=geometry['net_x_right'],
            court_top=0,court_bottom=2160,court_left=0,court_right=3840,frame_width=3840,frame_height=2160,
            net_tape_top=geometry['net_y'],homography=geometry['homography'],court_corners=geometry['court_corners'],
            net_floor_left=tuple(geometry['net_floor_left']),net_floor_right=tuple(geometry['net_floor_right']),
            net_floor_method=geometry['net_floor_method'])
        an._projector=None;an._projector_calib=None;an._court_gate=None
        an.pose_model=YOLO(str(args.pose_model));an.pose_imgsz=args.pose_imgsz;an.max_players_per_team=6;an._pose_track_warned=False
        an._person_pose_recovery=None;fx=PoseFeatureExtractor()
        cap=cv2.VideoCapture(source['video'])
        if not cap.isOpened():raise ValueError(f'Cannot open {stem}')
        if (round(cap.get(3)),round(cap.get(4)))!=(3840,2160):raise ValueError('Geometry requires native 4K')
        segment=0;processed=0
        try:
            for f in range(ranges[-1][1]+1):
                if not cap.grab():raise ValueError(f'{stem}: decode ended at {f}')
                while segment<len(ranges) and f>ranges[segment][1]:segment+=1
                if segment>=len(ranges):break
                start,end=ranges[segment]
                if f<start:continue
                cache=out/'poses'/f'{stem}_s{segment:03d}.jsonl'
                if f==start:
                    fx.reset()
                    if an._court_gate is not None:an._court_gate.reset()
                    predictor=getattr(an.pose_model,'predictor',None)
                    for tracker in getattr(predictor,'trackers',[]):tracker.reset()
                    if cache.exists():
                        for line in cache.open(encoding='utf-8'):
                            record=json.loads(line);light[record['f']]=record['features_context']
                        continue
                    handle=cache.with_suffix('.partial').open('w',encoding='utf-8')
                if cache.exists():continue
                ok,frame=cap.retrieve()
                if not ok:raise ValueError(f'{stem}: frame retrieval failed {f}')
                source_row=balls[f]
                ball=None if source_row.get('ball_out_of_frame') else source_row.get('ball')
                if ball is not None and (not np.isfinite(ball).all() or not (0<=ball[0]<3840 and 0<=ball[1]<2160)):ball=None
                _,poses=an._detect_players(frame,fsm_state='')
                pf=fx.update(f,poses,ball,geometry['net_y'])
                extra,extra_valid=team_contact_features(poses,ball,geometry['net_y'],fx)
                row={k:source_row.get(k) for k in ('ball_source','ball_confidence','ball_new','ball_out_of_frame')}
                row.update(ball=ball,p=[list(p.center) for p in poses],pose_f=pf,
                    pose_valid=pose_validity(poses,ball,geometry['net_y'],fx).tolist(),
                    team_values=extra.tolist(),team_valid=extra_valid.tolist())
                light[f]=row
                record=dict(f=f,t=f/fps,segment=segment,features_context=row,poses=[dict(id=p.track_id,team=p.team,
                    bb=list(p.bbox),kp=p.keypoints.tolist(),source='yolo_pose') for p in poses])
                handle.write(json.dumps(record,allow_nan=False)+'\n');processed+=1
                coverage['frames']+=1;coverage['pose_person_frames']+=len(poses)
                if f==end:
                    handle.close();cache.with_suffix('.partial').replace(cache)
                if processed%200==0:
                    save(out/'progress.json',dict(status='extracting',source=stem,frame=f,processed_this_source=processed,
                        required_source=source['required_pose_frames'],elapsed_seconds=time.time()-started))
                    print(f'{stem} f{f}: {processed}/{source["required_pose_frames"]} pose frames',flush=True)
        finally:
            cap.release()
            if 'handle' in locals() and not handle.closed:handle.close()
        for clip in clips:
            indices=list(range(clip['start_frame'],clip['end_frame']+1))
            if any(f not in light for f in indices):raise ValueError(f'Missing exact frames: {clip["clip_id"]}')
            rows=[light[f] for f in indices]
            base=compute_clip_features(rows,geometry['net_y'],fps,frame_width=3840,frame_height=2160,feature_schema=RESOLUTION_NORMALIZED_SCHEMA).to_numpy(np.float32)
            values=np.concatenate([base,np.array([r['team_values'] for r in rows],np.float32)],axis=1)
            validity=np.zeros(values.shape,bool);previous=False
            for i,row in enumerate(rows):
                b=row['ball'] is not None;velocity=b and previous and not row.get('ball_new')
                validity[i,:10]=[b,b,velocity,velocity,b,b and bool(row['p']),b,True,velocity,b]
                validity[i,10:18]=row['pose_valid'];validity[i,18:]=row['team_valid'];previous=b
            values[~validity]=0
            np.savez_compressed(out/'features'/(clip['clip_id']+'.npz'),features=values,valid=validity,
                frames=np.array(indices,np.int64),feature_names=np.array(VALUE_NAMES),schema=np.array(SCHEMA),
                source_video=np.array(stem),ball_source=np.array([r['ball_source'] or 'unspecified' for r in rows]))
        save(out/'poses'/(stem+'_report.json'),dict(status='complete',**coverage,pose_imgsz=args.pose_imgsz,geometry=geometry,
            source_video_sha256=source['video_sha256'],ball_annotations_sha256=source['ball_sha256']))
        del an,light,balls
    if file_hash(args.manifest)!=meta['parent_manifest_sha256']:raise ValueError('Source human manifest changed')
    total=len(list((out/'features').glob('*.npz')))
    if total!=len(parent['clips']):raise ValueError('Feature-file count mismatch')
    save(out/'features/feature_schema.json',dict(feature_schema=SCHEMA,value_names=VALUE_NAMES,feature_names=FEATURE_NAMES,
        feature_files=total,input_dim=len(FEATURE_NAMES),label_source='human'))
    meta['status']='complete';meta['elapsed_seconds']=time.time()-started
    save(out/'preflight.json',meta);save(out/'progress.json',dict(status='complete',features=total,elapsed_seconds=meta['elapsed_seconds']))
    print('COMPLETE:',total,'feature files',flush=True)


if __name__=='__main__':main()
