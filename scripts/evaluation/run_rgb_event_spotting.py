"""Versioned RGB/BiGRU spotting experiment on three completed human sources.

An explicit MobileNet frozen-feature adaptation of STES, without GSM/spatial
regression. No FSM inputs/labels; native frame timestamps and complete videos.
"""
from __future__ import annotations
import argparse,collections,json,random,sys,time,shutil,subprocess
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from action_recognition.rally_spotting.rgb_spotter import RGBEventSpotter,EVENTS,SCHEMA,frame_targets,predict,evaluate
from action_recognition.event_clips.config import collapse_labels
from scripts.evaluation.refresh_team_pose_dataset import save
from src.detection.sam_pose_recovery import file_hash


class Windows(Dataset):
    def __init__(self,records,mean,std,window=120,stride=60):
        self.data=[];self.items=[];self.window=window
        for i,record in enumerate(records):
            x=(np.asarray(record['features'],np.float32)-mean)/std
            y=frame_targets(len(x),record['events'],record['ignored'])
            self.data.append((x,y));starts=list(range(0,max(1,len(x)-window+1),stride))
            last=max(0,len(x)-window)
            if starts[-1]!=last:starts.append(last)
            self.items.extend((i,s) for s in starts)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        rec,start=self.items[i];x,y=self.data[rec];end=min(start+self.window,len(x))
        a=np.zeros((self.window,x.shape[1]),np.float32);b=np.full(self.window,-100,np.int64)
        a[:end-start]=x[start:end];b[:end-start]=y[start:end]
        return torch.from_numpy(a),torch.from_numpy(b)


def extract(video,path,backbone,device,progress):
    """All native frames; image geometry preserved by letterboxing, no temporal sampling."""
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open '+str(video))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
    scale=min(224/width,224/height);nw=max(2,2*round(width*scale/2));nh=max(2,2*round(height*scale/2))
    filt=f'scale_cuda={nw}:{nh},hwdownload,format=nv12,format=rgb24,pad=224:224:{(224-nw)//2}:{(224-nh)//2}:black'
    command=['ffmpeg','-hide_banner','-loglevel','error','-hwaccel','cuda','-hwaccel_output_format','cuda',
        '-i',str(video),'-vf',filt,'-an','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    log=path.with_suffix('.decode.log').open('wb')
    decoder=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    mmap=np.lib.format.open_memmap(path.with_suffix('.partial.npy'),mode='w+',dtype=np.float16,shape=(total,576))
    batch=[];count=0;started=time.time()
    def flush():
        nonlocal count,batch
        x=torch.from_numpy(np.stack(batch)).to(device).float()/255
        x=(x-torch.tensor([.485,.456,.406],device=device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        with torch.no_grad():values=backbone(x).flatten(1).cpu().numpy()
        mmap[count:count+len(batch)]=values;count+=len(batch);batch=[]
    try:
        while True:
            raw=decoder.stdout.read(224*224*3)
            if not raw:break
            if len(raw)!=224*224*3:raise ValueError('Partial decoded RGB frame')
            batch.append(np.frombuffer(raw,np.uint8).reshape(224,224,3).transpose(2,0,1))
            if len(batch)==64:flush()
            if count and count%512==0 and not batch:
                mmap.flush();save(progress,dict(status='extracting',video=str(video),frames=count,total=total,elapsed_seconds=time.time()-started))
                print(video.stem,count,'/',total,flush=True)
        if batch:flush()
        if decoder.wait()!=0:raise ValueError('GPU decode failed; see '+str(path.with_suffix('.decode.log')))
    finally:
        if decoder.poll() is None:decoder.terminate();decoder.wait()
        decoder.stdout.close();log.close()
    if count>total or total-count>2:raise ValueError(f'Decode count mismatch {count} != {total}')
    mmap.flush()
    if count!=total:
        # Container frame count may be rounded by one; preserve every actual decoded frame.
        np.save(path,np.asarray(mmap[:count]));del mmap
    else:
        del mmap;path.with_suffix('.partial.npy').replace(path)
    return count,fps


def prepare(args,device):
    manifest=json.loads(args.manifest.read_text());clips=manifest['clips']
    if any(c.get('label_source')!='human' for c in clips):raise ValueError('Only human labels accepted')
    stems=sorted({c['source_stem'] for c in clips})
    source_specs=[]
    for stem in stems:
        human=Path('data/full_video_gold')/stem/'annotations_v1.json'
        raw=json.loads(human.read_text());ann={a['annotation_id']:a for a in raw['annotations'] if a['video_id']==stem}
        selected=[c for c in clips if c['source_stem']==stem]
        for c in selected:
            for identity in c['annotation_ids']:
                if identity not in ann or ann[identity].get('label_source')!='human' or ann[identity]['frame']!=c['contact_frame']:
                    raise ValueError('Active manifest and authoritative human annotation disagree')
        active_pairs={(c['contact_frame'],collapse_labels(c['event_types'],c.get('fsm_event_type'))) for c in selected}
        for a in ann.values():
            label=collapse_labels([a['event_type']])
            if label is not None and (a['frame'],label) not in active_pairs:
                raise ValueError('Unrepresented authoritative human contact; refuse to label it background')
        video=Path(selected[0]['source_video'])
        source_specs.append(dict(stem=stem,video=str(video),human_source=str(human),human_sha256=file_hash(human),
            video_sha256=file_hash(video),clips=selected))
    preflight=dict(schema=SCHEMA,manifest=str(args.manifest.resolve()),manifest_sha256=file_hash(args.manifest),
        features=str((args.output/'features').resolve()),label_source='human',clips=len(clips),
        core_clips=sum(collapse_labels(c['event_types'],c.get('fsm_event_type')) is not None for c in clips),
        sources=[{k:v for k,v in s.items() if k!='clips'} for s in source_specs],
        split_policy='leave-one-complete-source-video-out; Khe primary; no random windows across splits',
        backbone=str(args.backbone),backbone_sha256=file_hash(args.backbone),
        adaptation='Frozen ImageNet MobileNetV3-Small 224 letterbox + 3-layer BiGRU; no GSM, no spatial head',
        temporal_mode='offline bidirectional; future frames permitted',classes=['background',*EVENTS],seed=42,
        decoder='ffmpeg_cuda_native_frames_v1',epochs=args.epochs,window=120,stride=60,
        caveats=['Three completed human sources only; negatives rely on completed annotation coverage.',
                 '410 clip labels are not 410 independent videos; no untouched final test.',
                 'Not a reproduction of VNL-STES; F1 is not comparable with clip-classification F1.'])
    print(json.dumps(preflight,indent=2),flush=True)
    if args.output.exists() and not args.resume:raise ValueError('Refuse existing output without --resume')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'features').mkdir(exist_ok=True)
    frozen=args.output/'preflight.json'
    if frozen.exists():
        previous=json.loads(frozen.read_text())
        for key in ('manifest_sha256','backbone_sha256','sources','decoder'):
            if previous[key]!=preflight[key]:raise ValueError('Resume inputs changed: '+key)
    else:
        save(frozen,preflight);shutil.copy2(args.manifest,args.output/'parent_manifest.json')
        for rel in ('scripts/evaluation/run_rgb_event_spotting.py','action_recognition/rally_spotting/rgb_spotter.py'):
            destination=args.output/'code_snapshot'/rel;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(rel,destination)
    from torchvision.models import mobilenet_v3_small
    net=mobilenet_v3_small(weights=None);net.load_state_dict(torch.load(args.backbone,map_location='cpu',weights_only=True))
    backbone=nn.Sequential(net.features,net.avgpool).to(device).eval()
    records=[]
    for source in source_specs:
        stem=source['stem'];path=args.output/'features'/(stem+'.npy');meta_path=path.with_suffix('.json')
        if path.exists() and meta_path.exists():metadata=json.loads(meta_path.read_text())
        else:
            total,fps=extract(Path(source['video']),path,backbone,device,args.output/'progress.json')
            metadata=dict(total_frames=total,fps=fps,features_sha256=file_hash(path));save(meta_path,metadata)
        if file_hash(path)!=metadata['features_sha256']:raise ValueError('Cached feature hash changed')
        events=[];ignored=[];ids={}
        for c in source['clips']:
            label=collapse_labels(c['event_types'],c.get('fsm_event_type'))
            if label is None:ignored.append(c['contact_frame']);continue
            key=(c['contact_frame'],label)
            if key in ids:
                ids[key].extend(c['annotation_ids']);continue
            ids[key]=list(c['annotation_ids']);events.append(dict(frame=c['contact_frame'],type=label,annotation_ids=ids[key]))
        human=json.loads(Path(source['human_source']).read_text())
        for a in human['annotations']:
            if a['video_id']!=stem:continue
            label=collapse_labels([a['event_type']]);key=(a['frame'],label)
            if label is None:ignored.append(a['frame'])
            elif a['annotation_id'] not in ids[key]:ids[key].append(a['annotation_id'])
        ignored=sorted(set(ignored))
        features=np.load(path,mmap_mode='r');frame_targets(len(features),events,ignored)
        record=dict(stem=stem,features=features,events=events,ignored=ignored,fps=metadata['fps'])
        save(args.output/(stem+'_labels.json'),dict(events=events,ignored_noncore_frames=ignored,label_source='human',
            exact_duplicate_policy='Only identical frame/class exported once; all annotation IDs retained'))
        records.append(record)
    del backbone,net
    return records


def fit(args,records,holdout,device):
    folder=args.output/('fold_'+holdout)
    if (folder/'metrics.json').exists():
        completed=json.loads((folder/'metrics.json').read_text())
        if completed.get('training_complete'):return completed
    if folder.exists():raise ValueError('Partial training fold; use fresh output rather than mixing checkpoints')
    folder.mkdir();random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    train=[r for r in records if r['stem']!=holdout];val=next(r for r in records if r['stem']==holdout)
    # Statistics come only from training RGB frames.
    x=np.concatenate([np.asarray(r['features'],np.float32) for r in train]);mean=x.mean(0);std=x.std(0);std[std<1e-5]=1;del x
    ds=Windows(train,mean,std);loader=DataLoader(ds,batch_size=16,shuffle=True)
    vx=(np.asarray(val['features'],np.float32)-mean)/std
    labels=np.concatenate([frame_targets(len(r['features']),r['events'],r['ignored']) for r in train])
    counts=np.bincount(labels[labels>=0],minlength=5);weight=np.sqrt(counts[0]/np.maximum(counts,1));weight=np.clip(weight,1,50)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device=device),ignore_index=-100)
    model=RGBEventSpotter().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=-1;best_epoch=-1;history=[]
    for epoch in range(args.epochs):
        model.train();losses=[]
        for a,b in loader:
            a,b=a.to(device),b.to(device)
            if not (b!=-100).any():continue
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(a).transpose(1,2),b)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        p=predict(model,vx,device)
        choices=[]
        for threshold in (.1,.2,.3,.4,.5,.6,.7,.8,.9):
            metric,events=evaluate(p,val['events'],threshold,tolerances=(10,))
            choices.append((metric['at_10']['macro_f1'],threshold))
        score,threshold=max(choices);history.append(dict(epoch=epoch,loss=float(np.mean(losses)),macro_f1_at10=score,threshold=threshold))
        print(holdout,'epoch',epoch,'F1@10',round(score,4),'threshold',threshold,flush=True)
        save(args.output/'progress.json',dict(status='training',holdout=holdout,epoch=epoch,macro_f1_at10=score))
        if score>best:
            best=score;best_epoch=epoch;metric,events=evaluate(p,val['events'],threshold)
            result=dict(holdout=holdout,train_sources=[r['stem'] for r in train],metrics=metric,threshold=threshold,
                best_epoch=epoch,predictions=events,human_events=val['events'],
                false_events_per_minute=metric['at_10']['micro']['fp']/(len(vx)/val['fps']/60),
                label_source='human',selection='validation model selection, not independent test')
            torch.save(dict(model_type=SCHEMA,state_dict=model.state_dict(),mean=mean,std=std,
                threshold=threshold,event_types=['background',*EVENTS],holdout=holdout,train_sources=result['train_sources'],
                seed=42,backbone=str(args.backbone),backbone_sha256=file_hash(args.backbone),
                input_size=224,window=120,stride=60,production_promoted=False),folder/'best.pt')
            np.save(folder/'probabilities.npy',p);save(folder/'metrics.json',result)
        elif epoch-best_epoch>=8:break
    save(folder/'history.json',history)
    result=json.loads((folder/'metrics.json').read_text());result['training_complete']=True
    save(folder/'metrics.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=Path('data/event_clips_three_full_gold_causal_120f_v1/manifest.json'))
    ap.add_argument('--backbone',type=Path,default=Path.home()/'.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth')
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records=prepare(args,device);results=[]
    ordered=sorted(records,key=lambda r:(r['stem']!='Khe4P8QcJkg_5_4k',r['stem']))
    for record in ordered:results.append(fit(args,records,record['stem'],device))
    save(args.output/'report.json',dict(status='complete',schema=SCHEMA,folds=results,
        production_promoted=False,description='STES-inspired frozen MobileNet RGB + BiGRU; no GSM or spatial head'))
    save(args.output/'progress.json',dict(status='complete'))


if __name__=='__main__':main()
