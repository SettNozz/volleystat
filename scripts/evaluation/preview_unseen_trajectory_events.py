"""Offline, ungated per-candidate predictions on an unseen video; no spotting claims."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.evaluation.run_trajectory_contact_experiment import TrajectoryFusion
from action_recognition.event_clips.contact_trajectory import SCHEMA,OFFSETS,summarize
from action_recognition.event_clips.team_pose_features import normalize_observations
from action_recognition.event_clips.compute_features import compute_clip_features
from action_recognition.event_clips.feature_schema import RESOLUTION_NORMALIZED_SCHEMA
from action_recognition.data.event_clip_dataset import _fix_length,EventClipDataset
from src.detection.sam_pose_recovery import file_hash
from action_recognition.inference.serve_guard import ServeGuard


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations',required=True,type=Path);ap.add_argument('--baseline',required=True,type=Path)
    ap.add_argument('--new-model',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--start',type=float,default=15);ap.add_argument('--end',type=float,default=45)
    ap.add_argument('--serve-rules',action='store_true',help='Show raw new-model vs same model with serve guard')
    ap.add_argument('--serve-restart-seconds',type=float,default=3.0)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('Refuse existing output')
    payload=json.loads(args.annotations.read_text());video=Path(payload['original_video_path']);stem=video.stem
    device='cuda' if torch.cuda.is_available() else 'cpu';models=[];metadata=[]
    for path,enabled in ((args.baseline,False),(args.new_model,True)):
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['model_type']!='experimental_trajectory_late_fusion_v1' or ck['feature_schema']!=SCHEMA:
            raise ValueError('Wrong checkpoint schema')
        if bool(ck['enabled'])!=enabled or stem in ck['train_sources'] or stem==ck['holdout']:
            raise ValueError('Wrong condition or video not independent')
        model=TrajectoryFusion();model.load_state_dict(ck['state_dict']);model.to(device).eval()
        training=json.loads((path.parent.parent/'preflight.json').read_text())
        manifest=json.loads(Path(training['manifest']).read_text())
        train=EventClipDataset([c for c in manifest['clips'] if c['source_stem'] in ck['train_sources']],
            training['features'],sequence_len=120)
        mean,std=train.feature_stats()
        np.testing.assert_allclose(mean,ck['feature_mean'],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(std,ck['feature_std'],rtol=1e-5,atol=1e-6)
        ck['feature_active_mask']=train.feature_active_mask()
        models.append(model);metadata.append(ck)
    if metadata[0]['seed']!=metadata[1]['seed'] or metadata[0]['train_sources']!=metadata[1]['train_sources']:
        raise ValueError('Unmatched comparison')
    args.output.mkdir(parents=True)
    fps=payload['fps'];size=np.array([payload['frame_width'],payload['frame_height']],float)
    rows={r['f']:r for r in payload['frames']};first=max(min(rows),round(args.start*fps));last=min(max(rows),round(args.end*fps)-1)
    if first>last:raise ValueError('Empty interval')
    samples=[];batch_x=[[],[]];batch_z=[[],[]]
    for f in range(first,last+1):
        history=[rows.get(k,dict(ball=None,p=[],ball_out_of_frame=True)) for k in range(max(0,f-119),f+1)]
        raw=compute_clip_features(history,payload['net_y'],fps,frame_width=size[0],frame_height=size[1],
            feature_schema=RESOLUTION_NORMALIZED_SCHEMA).to_numpy(np.float32)
        raw=_fix_length(raw,120).copy();raw[:,7]=np.linspace(0,1,120,dtype=np.float32)
        p=np.zeros((25,2));a=np.zeros(25,bool);n=np.zeros(25,bool);r=np.zeros(25,bool)
        for i,k in enumerate(f+OFFSETS):
            row=rows.get(int(k),{});b=row.get('ball');src=row.get('ball_source') or ''
            n[i]=bool(row.get('ball_new'))
            if b is None or row.get('ball_out_of_frame'):continue
            b=np.asarray(b,float)
            if b.shape!=(2,) or not np.isfinite(b).all() or np.any(b<0) or np.any(b>=size):continue
            p[i]=b/size;a[i]=True;r[i]='reconstruct' in src or 'interpol' in src
        values,valid=summarize(p,a,n,r,fps)
        for j,ck in enumerate(metadata):
            x=raw.copy();mean=np.asarray(ck['feature_mean'],np.float32)
            x[:,~ck['feature_active_mask']]=mean[~ck['feature_active_mask']]
            batch_x[j].append((x-mean)/np.asarray(ck['feature_std'],np.float32))
            batch_z[j].append(normalize_observations(values,valid,np.asarray(ck['summary_mean']),np.asarray(ck['summary_std']))
                             if ck['enabled'] else np.zeros(34,np.float32))
        samples.append(dict(frame=f,time=f/fps,trajectory_change_available=bool(valid[6]),
            weak_fsm_events=rows.get(f,{}).get('ev',[])))
    predictions=[]
    with torch.no_grad():
        for j,model in enumerate(models):
            probs=[]
            for start in range(0,len(samples),64):
                x=torch.tensor(np.array(batch_x[j][start:start+64]),dtype=torch.float32,device=device)
                z=torch.tensor(np.array(batch_z[j][start:start+64]),dtype=torch.float32,device=device)
                probs.extend(model(x,z).softmax(1).cpu().numpy().tolist())
            predictions.append(probs)
    labels=metadata[0]['event_types']
    guard=ServeGuard(args.serve_restart_seconds)
    if args.serve_rules:
        for f in sorted(k for k in rows if k<first):guard.update(f/fps,rows[f].get('state',''))
    for i,s in enumerate(samples):
        for j,name in enumerate(('baseline','new')):
            probs=predictions[j][i];s[name]=dict(label=labels[int(np.argmax(probs))],confidence=max(probs),probabilities=probs)
        if args.serve_rules:
            guard.update(s['time'],rows[s['frame']].get('state',''))
            s['serve_rule']=guard.decision(s['new']['label'])
    report=dict(video=str(video),annotations=str(args.annotations),annotations_sha256=file_hash(args.annotations),
        checkpoints=[dict(path=str(p),sha256=file_hash(p),seed=ck['seed'],train_sources=ck['train_sources'])
            for p,ck in zip((args.baseline,args.new_model),metadata)],
        mode='offline candidate classifier on every frame; no event localization or FSM probability gating',
        warning='No human ground truth: no accuracy/F1 claims. Train-constant channels are imputed with training mean; missing observations preserved.',
        frames=len(samples),start=first/fps,end=(last+1)/fps,
        changed_class_frames=sum(s['baseline']['label']!=s['new']['label'] for s in samples),predictions=samples)
    if args.serve_rules:
        report['serve_rules']=dict(min_restart_seconds=args.serve_restart_seconds,
            state_source='saved model/FSM state, not human ground truth',history='replayed from annotation start',
            suppressed_frames=sum(bool(s['serve_rule']['suppression_reason']) for s in samples))
    (args.output/'predictions.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,first)
    header=140 if args.serve_rules else 80
    writer=cv2.VideoWriter(str(args.output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,540+header))
    if not cap.isOpened() or not writer.isOpened():raise ValueError('Video input/output unavailable')
    try:
        for s in samples:
            ok,frame=cap.read()
            if not ok:raise ValueError('Decode ended before expected frame')
            panel=cv2.resize(frame,(960,540));canvas=np.zeros((540+header,1920,3),np.uint8)
            for j,name in enumerate(('baseline','new')):
                x=j*960;canvas[header:,x:x+960]=panel;pred=s['new'] if args.serve_rules else s[name]
                color=(100,230,100) if j else (80,210,255)
                title=('NEW + TRAJECTORY' if j else 'BASELINE')+f' | {pred["label"]} {pred["confidence"]:.0%}'
                if args.serve_rules:
                    d=s['serve_rule'];out=d['label'] if j else pred['label']
                    title=('WITH SERVE RULES' if j else 'RAW NEW MODEL')+f' | {out or "BLOCKED"} | raw {pred["confidence"]:.0%}'
                cv2.putText(canvas,title,(x+15,30),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2)
                cv2.putText(canvas,f'{s["time"]:.2f}s | frame {s["frame"]} | candidate class, not confirmed event',
                    (x+15,122 if args.serve_rules else 62),cv2.FONT_HERSHEY_SIMPLEX,.48,(220,220,220),1)
                if args.serve_rules:
                    cv2.putText(canvas,f'FSM: {d["state"]} | pause remaining: {d["restart_wait_seconds"]:.2f}s',
                        (x+15,62),cv2.FONT_HERSHEY_SIMPLEX,.58,(220,220,220),1)
                    cv2.putText(canvas,('Reason: '+(d['suppression_reason'] or 'allowed')) if j else 'Unfiltered classifier output',
                        (x+15,94),cv2.FONT_HERSHEY_SIMPLEX,.57,(100,180,255),1)
                ball=rows.get(s['frame'],{}).get('ball')
                if ball is not None and not rows.get(s['frame'],{}).get('ball_out_of_frame'):
                    cv2.circle(canvas,(x+int(ball[0]*960/size[0]),header+int(ball[1]*540/size[1])),8,color,2)
            writer.write(canvas)
    finally:cap.release();writer.release()
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2),flush=True)


if __name__=='__main__':main()
