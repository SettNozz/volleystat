"""Fine-tune the pinned VNL-STES authors' RegNet-Y/GSM/deeper-GRU temporal model.

Four local contact classes + background; spatial loss is disabled because local
human XY targets are absent. This is temporal adaptation, not full paper reproduction.
"""
from __future__ import annotations
import argparse,json,math,random,subprocess,sys,time,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'.runtime/vnl_stes_deps'))
import cv2,numpy as np,torch
from torch import nn
from action_recognition.models.vnl_stes_model import build_author_stes
from action_recognition.rally_spotting.rgb_spotter import EVENTS,frame_targets,evaluate
from action_recognition.rally_spotting.metrics import match_events,_prf
from third_party.vnl_stes.util.score import compute_average_precision
from scripts.evaluation.refresh_team_pose_dataset import save
from src.detection.sam_pose_recovery import file_hash

SCHEMA='vnl_stes_author_temporal_v1'


def fast_peaks(scores,threshold=.5,radius=5):
    values=np.asarray(scores,np.float32);candidates=np.flatnonzero(values>=threshold)
    order=candidates[np.lexsort((candidates,-values[candidates]))]
    blocked=np.zeros(len(values),bool);kept=[]
    for f in order:
        if not blocked[f]:
            kept.append(int(f));blocked[max(0,f-radius):min(len(values),f+radius+1)]=True
    return sorted(kept)


def event_metrics(p,events,threshold,tolerances=(1,2,4,5,10,15)):
    predicted=[]
    for k,name in enumerate(EVENTS,1):
        scores=np.where(p[:,k]>p[:,0],p[:,k],0)
        predicted.extend(dict(frame=f,type=name,confidence=float(p[f,k])) for f in fast_peaks(scores,threshold))
    result={}
    for t in tolerances:
        classes={};totals=np.zeros(3,int)
        for name in EVENTS:
            counts=match_events([e['frame'] for e in predicted if e['type']==name],
                [e['frame'] for e in events if e['type']==name],t)
            totals+=counts;classes[name]=_prf(*counts)
        result[f'at_{t}']=dict(micro=_prf(*map(int,totals)),per_class=classes,
            macro_f1=float(np.mean([v['f1'] for v in classes.values()])))
    return result,sorted(predicted,key=lambda e:e['frame'])


def atomic_checkpoint(path,state):
    temporary=path.with_suffix('.partial');torch.save(state,temporary);temporary.replace(path)


class Clips:
    def __init__(self,records):
        self.records=records;self.caps={}
        self.targets={r['stem']:frame_targets(r['frames'],r['events'],r['ignored']) for r in records}
        self.flat=[(r,e['frame']) for r in records for e in r['events']]
        self.weights=[r['frames'] for r in records]
    def read(self,record,start):
        key=record['stem']
        if key not in self.caps:self.caps[key]=cv2.VideoCapture(record['cache'])
        cap=self.caps[key];cap.set(cv2.CAP_PROP_POS_FRAMES,int(start));images=[]
        for _ in range(64):
            ok,frame=cap.read()
            if not ok:raise ValueError(f'Cached frame decode failed: {key} at {start}')
            images.append(cv2.cvtColor(cv2.resize(frame,(224,224)),cv2.COLOR_BGR2RGB))
        return np.stack(images).transpose(0,3,1,2)
    def sample(self,batch):
        xs=[];ys=[]
        for _ in range(batch):
            if random.random()<.5:
                rec,frame=random.choice(self.flat);start=max(0,min(rec['frames']-64,frame-random.randrange(64)))
            else:
                rec=random.choices(self.records,weights=self.weights)[0];start=random.randrange(rec['frames']-63)
            x=self.read(rec,start)
            if random.random()<.5:x=x[:,:,:,::-1].copy()
            xs.append(x);ys.append(self.targets[rec['stem']][start:start+64])
        return np.stack(xs),np.stack(ys)
    def close(self):
        for cap in self.caps.values():cap.release()


def tensor_images(x,device):
    x=torch.from_numpy(x).to(device).float()/255
    mean=x.new_tensor([.485,.456,.406])[None,None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,None,:,None,None]
    return (x-mean)/std


@torch.no_grad()
def predict_video(model,reader,record,device,batch=2):
    model.eval();length=record['frames'];scores=np.zeros((length,5),np.float32);counts=np.zeros((length,1),np.float32)
    starts=list(range(0,length-63,32));last=length-64
    if starts[-1]!=last:starts.append(last)
    for i in range(0,len(starts),batch):
        chunk=starts[i:i+batch];x=tensor_images(np.stack([reader.read(record,s) for s in chunk]),device)
        with torch.autocast('cuda',dtype=torch.float16):p=model(x)['im_feat'].float().softmax(-1).cpu().numpy()
        for s,v in zip(chunk,p):scores[s:s+64]+=v;counts[s:s+64]+=1
    if np.any(counts==0):raise ValueError('Uncovered validation frame')
    return scores/counts


def temporal_ap(p,events,stem):
    # Authors' AP scorer, high-recall per-class proposals; no background-argmax gate.
    result={};ranked={}
    for k,name in enumerate(EVENTS,1):
        peaks=fast_peaks(p[:,k],.01,5)
        ranked[name]=sorted([(stem,f,float(p[f,k]),None) for f in peaks],key=lambda e:-e[2])
    for tolerance in (0,1,2,4,10):
        per_class={}
        for k,name in enumerate(EVENTS,1):
            predictions=ranked[name]
            truth={stem:[e['frame'] for e in events if e['type']==name]}
            per_class[name]=compute_average_precision(predictions,truth,tolerance=tolerance) if truth[stem] else 0.
        result[str(tolerance)]=dict(per_class=per_class,map=float(np.mean(list(per_class.values()))))
    return result


def prepare(args):
    parent=json.loads((args.data_dir/'preflight.json').read_text())
    if file_hash(parent['manifest'])!=parent['manifest_sha256']:raise ValueError('Human manifest changed')
    records=[]
    for s in parent['sources']:
        if file_hash(s['human_source'])!=s['human_sha256'] or file_hash(s['video'])!=s['video_sha256']:
            raise ValueError('Frozen source changed: '+s['stem'])
        labels_path=args.data_dir/(s['stem']+'_labels.json');labels=json.loads(labels_path.read_text())
        image_meta=json.loads((args.data_dir/'features'/(s['stem']+'.json')).read_text())
        records.append(dict(stem=s['stem'],video=s['video'],frames=image_meta['total_frames'],fps=image_meta['fps'],
            events=labels['events'],ignored=labels['ignored_noncore_frames'],labels_sha256=file_hash(labels_path),
            video_sha256=s['video_sha256'],human_sha256=s['human_sha256']))
    author=json.loads((ROOT/'third_party/vnl_stes/SOURCE_REVISION.json').read_text())
    preflight=dict(schema=SCHEMA,manifest=parent['manifest'],manifest_sha256=parent['manifest_sha256'],
        rgb_frames=str((args.output/'videos').resolve()),source_videos=[r['video'] for r in records],
        human_sources=parent['sources'],label_source='human',clips=parent['clips'],events=sum(len(r['events']) for r in records),
        classes=['background',*EVENTS],split_policy='whole-source holdout; Khe primary then j0/j3',
        source_revision=author,backbone=str(args.weights.resolve()),backbone_sha256=file_hash(args.weights),
        author_code_hashes={rel:file_hash(ROOT/'third_party/vnl_stes'/rel) for rel in
            ('train_e2e_spatial.py','model/modules.py','model/shift.py','model/common.py','util/score.py')},
        architecture='authors rny008_gsm + deeper_gru (3 bidirectional layers); all weights trainable',
        spatial_head=False,spatial_head_reason='No verified human contact XY; derived ball positions are not targets',
        compatibility='Author GSM zero padding uses input device/dtype; equivalent zero values for AMP/CPU',
        source_hashes={r['stem']:dict(video=r['video_sha256'],labels=r['labels_sha256'],human=r['human_sha256']) for r in records},
        recipe=dict(epochs=args.epochs,steps=args.steps,batch=args.batch,accumulation=args.accumulation,
            evaluation_every=args.eval_every,foreground_sample_probability=.5,foreground_loss_weight=5,
            clip_frames=64,seed=42,backbone_lr=.0001,head_lr=.001,window_stride=32,
            note='Bounded local pilot recipe; not the paper full training budget'),
        production_promoted=False)
    print(json.dumps(preflight,indent=2),flush=True)
    if args.output.exists() and not args.resume:raise ValueError('Refuse existing output')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'videos').mkdir(exist_ok=True)
    frozen=args.output/'preflight.json'
    if frozen.exists():
        previous=json.loads(frozen.read_text())
        if previous!=preflight:raise ValueError('Resume preflight differs; use a fresh version')
    else:
        save(frozen,preflight)
        for rel in ('scripts/evaluation/run_author_stes.py','action_recognition/models/vnl_stes_model.py',
                    'action_recognition/rally_spotting/rgb_spotter.py'):
            target=args.output/'code_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(rel,target)
    for r in records:
        cache=args.output/'videos'/(r['stem']+'.mp4');meta=cache.with_suffix('.json')
        if not meta.exists():
            save(args.output/'progress.json',dict(status='preparing_rgb',source=r['stem'],pid=os.getpid()))
            partial=cache.with_suffix('.partial.mp4')
            if partial.exists():raise ValueError('Partial video cache present; preserve and use fresh output')
            command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-hwaccel','cuda',
                '-hwaccel_output_format','cuda','-i',r['video'],'-vf','scale_cuda=398:224',
                '-c:v','h264_nvenc','-preset','p4','-cq','18','-g','32','-bf','0','-an',
                '-fps_mode','passthrough',str(partial)]
            subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            cap=cv2.VideoCapture(str(partial));actual=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            if actual!=r['frames']:raise ValueError('RGB cache frame alignment differs')
            partial.replace(cache);save(meta,dict(sha256=file_hash(cache),frames=actual,source_sha256=r['video_sha256']))
        metadata=json.loads(meta.read_text())
        if file_hash(cache)!=metadata['sha256'] or metadata['frames']!=r['frames']:raise ValueError('RGB cache changed')
        r['cache']=str(cache)
    return records


def train_fold(args,records,holdout):
    folder=args.output/('fold_'+holdout);folder.mkdir(exist_ok=True)
    done=folder/'complete.json'
    if done.exists():return json.loads(done.read_text())
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    device=torch.device('cuda');model=build_author_stes(args.weights).to(device)
    groups=[dict(params=[p for n,p in model.named_parameters() if n.startswith('_features')],lr=.0001),
            dict(params=[p for n,p in model.named_parameters() if not n.startswith('_features')],lr=.001)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.0001);scaler=torch.amp.GradScaler('cuda')
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:min(1.,(e+1)/3)*.5*(1+math.cos(math.pi*e/args.epochs)))
    criterion=nn.CrossEntropyLoss(weight=torch.tensor([1,5,5,5,5],device=device,dtype=torch.float32),ignore_index=-100)
    train=[r for r in records if r['stem']!=holdout];val=next(r for r in records if r['stem']==holdout)
    reader=Clips(records);start_epoch=0;best=-1;history=[]
    latest=folder/'last.pt'
    if latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=False);model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);scheduler.load_state_dict(ck['scheduler'])
        start_epoch=ck['next_epoch'];best=ck['best'];history=ck['history']
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng']])
    sampler=Clips(train)
    try:
        for epoch in range(start_epoch,args.epochs):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[];started=time.time()
            for step in range(args.steps):
                if (args.output/'cancel.request').exists():raise KeyboardInterrupt('cancel.request')
                x,y=sampler.sample(args.batch);x=tensor_images(x,device);y=torch.from_numpy(y).to(device)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(x)['im_feat'];loss=criterion(logits.transpose(1,2),y)/args.accumulation
                scaler.scale(loss).backward()
                if (step+1)%args.accumulation==0 or step+1==args.steps:
                    scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),5)
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach())*args.accumulation)
                if step%8==0:
                    save(args.output/'progress.json',dict(status='training',holdout=holdout,epoch=epoch,step=step,
                        steps=args.steps,pid=os.getpid(),train_loss=float(np.mean(losses)),elapsed_seconds=time.time()-started))
            scheduler.step();entry=dict(epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.time()-started)
            if (epoch+1)%args.eval_every==0 or epoch==args.epochs-1:
                save(args.output/'progress.json',dict(status='validating',holdout=holdout,epoch=epoch,pid=os.getpid()))
                probabilities=predict_video(model,reader,val,device,args.batch)
                maps=temporal_ap(probabilities,val['events'],holdout)
                score=float(np.mean([maps[str(t)]['map'] for t in (0,1,2,4)]));entry['map_0_1_2_4']=score
                choices=[]
                for threshold in (.1,.2,.3,.4,.5,.6,.7,.8,.9):
                    metric,_=event_metrics(probabilities,val['events'],threshold,tolerances=(10,))
                    choices.append((metric['at_10']['macro_f1'],threshold))
                f1,threshold=max(choices);entry.update(macro_f1_at10=f1,threshold=threshold)
                print(holdout,'epoch',epoch,'mAP',round(score,4),'F1@10',round(f1,4),flush=True)
                if score>best:
                    best=score;metric,events=event_metrics(probabilities,val['events'],threshold)
                    result=dict(holdout=holdout,epoch=epoch,train_sources=[r['stem'] for r in train],
                        temporal_AP=maps,map_0_1_2_4=score,metrics=metric,threshold=threshold,
                        false_events_per_minute=metric['at_10']['micro']['fp']/(val['frames']/val['fps']/60),
                        predictions=events,human_events=val['events'],selection='validation model selection, not independent test')
                    save(folder/'best_metrics.json',result);np.save(folder/'probabilities.npy',probabilities)
                    atomic_checkpoint(folder/'best.pt',dict(model_type=SCHEMA,model=model.state_dict(),
                        backbone_sha256=file_hash(args.weights),threshold=threshold,classes=['background',*EVENTS],
                        clip_len=64,spatial_head=False,source_revision=json.loads((ROOT/'third_party/vnl_stes/SOURCE_REVISION.json').read_text()),
                        train_sources=result['train_sources'],holdout=holdout,production_promoted=False))
            history.append(entry);save(folder/'history.json',history)
            atomic_checkpoint(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                scheduler=scheduler.state_dict(),next_epoch=epoch+1,best=best,history=history,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()))
    finally:reader.close();sampler.close()
    result=json.loads((folder/'best_metrics.json').read_text());save(done,result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,default=Path('results/rgb_event_spotting_20261027_v2'))
    ap.add_argument('--weights',type=Path,default=Path('models/pretrained/vnl_stes_regnety_008.safetensors'))
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--batch',type=int,default=2);ap.add_argument('--accumulation',type=int,default=4)
    ap.add_argument('--eval-every',type=int,default=2);args=ap.parse_args()
    if not torch.cuda.is_available():raise ValueError('CUDA required for this author model run')
    if min(args.epochs,args.steps,args.batch,args.accumulation,args.eval_every)<1:raise ValueError('Invalid training budget')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    try:
        records=prepare(args);results=[]
        for rec in sorted(records,key=lambda r:(r['stem']!='Khe4P8QcJkg_5_4k',r['stem'])):
            results.append(train_fold(args,records,rec['stem']))
        save(args.output/'report.json',dict(status='complete',schema=SCHEMA,folds=results,production_promoted=False,
            note='Author temporal architecture on local four-class data; no XY head; bounded pilot recipe'))
        save(args.output/'progress.json',dict(status='complete',pid=os.getpid()))
    except BaseException as exc:
        if args.output.exists():save(args.output/'progress.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),pid=os.getpid()))
        raise


if __name__=='__main__':main()
