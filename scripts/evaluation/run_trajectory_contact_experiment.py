"""Train matched legacy-input controls with/without offline trajectory summaries."""
from __future__ import annotations
import argparse,collections,copy,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from action_recognition.event_clips.contact_trajectory import SCHEMA,NAMES,OFFSETS,summarize
from action_recognition.event_clips.team_pose_features import observation_stats,normalize_observations
from action_recognition.data.event_clip_dataset import EventClipDataset
from action_recognition.models.sequence_event_model import build_model
from action_recognition.training.train_event_sequence import run_epoch,per_class_f1
from action_recognition.event_clips.config import EVENT_TYPES,collapse_labels
from src.detection.sam_pose_recovery import file_hash
from scripts.evaluation.refresh_team_pose_dataset import save


class TrajectoryFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=build_model('multiscale',18,sequence_len=120,short_sequence_len=60)
        # Preserve base initialization and RNG stream in the matched control.
        with torch.random.fork_rng(devices=[]):
            self.delta=nn.Linear(len(NAMES)*2,4,bias=False)
            nn.init.zeros_(self.delta.weight)
    def forward(self,x,z):return self.base(x)+self.delta(z)


class SummaryDataset(Dataset):
    def __init__(self,base,features,stats,enabled):
        self.base=base;self.enabled=enabled;self.mean,self.std=stats
        self.z=[normalize_observations(features[c['clip_id']][0],features[c['clip_id']][1],
                self.mean,self.std) for c,_ in base.items]
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i];z=self.z[i] if self.enabled else np.zeros(len(NAMES)*2,np.float32)
        return x,torch.from_numpy(z),y


def prepare(args):
    manifest=json.loads(args.manifest.read_text());meta=json.loads(args.frozen_preflight.read_text())
    if meta['status']!='complete' or file_hash(args.manifest)!=meta['parent_manifest_sha256']:
        raise ValueError('Frozen manifest mismatch or incomplete parent')
    sidecar=json.loads((args.features/'feature_schema.json').read_text())
    if manifest['feature_schema']!='resolution_normalized_v2' or sidecar['feature_schema']!=manifest['feature_schema']:
        raise ValueError('Baseline feature schema mismatch')
    clips=manifest['clips']
    if any(c.get('label_source')!='human' for c in clips):raise ValueError('Human labels required')
    if len({c['clip_id'] for c in clips})!=len(clips):raise ValueError('Duplicate clip IDs')
    core=[c for c in clips if collapse_labels(c.get('event_types'),c.get('fsm_event_type')) is not None]
    preflight=dict(manifest=str(args.manifest.resolve()),features=str(args.features.resolve()),
        manifest_sha256=file_hash(args.manifest),feature_schema=manifest['feature_schema'],summary_schema=SCHEMA,
        summary_names=NAMES,label_source='human',clips=len(clips),core_clips=len(core),
        sources=sorted({c['source_stem'] for c in clips}),source_counts=dict(collections.Counter(c['source_stem'] for c in core)),
        split_policy=meta['split_policy'],evaluation='leave_one_complete_source_video_out',seeds=args.seeds,
        base_context='unchanged original causal 120/60',extra_context='[-12,+12] source frames; no interpolation',
        epochs=args.epochs,patience=15,production_promoted=False)
    print(json.dumps(preflight,indent=2),flush=True)
    if args.output.exists():raise ValueError('Refuse existing experiment output')
    args.output.mkdir(parents=True);(args.output/'features').mkdir()
    save(args.output/'manifest.json',manifest)  # Exact human labels, not a rewritten dataset.
    import shutil
    snapshot=args.output/'code_snapshot';snapshot.mkdir()
    for rel in ('scripts/evaluation/run_trajectory_contact_experiment.py',
                'action_recognition/event_clips/contact_trajectory.py',
                'action_recognition/models/sequence_event_model.py',
                'action_recognition/data/event_clip_dataset.py'):
        p=snapshot/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p)
    preflight['code_hashes']={str(p.relative_to(snapshot)):file_hash(p) for p in snapshot.rglob('*.py')}
    result={};provenance={};baseline_hashes={}
    for stem in preflight['sources']:
        source=meta['sources'][stem];path=Path(source['ball_annotations'])
        if file_hash(path)!=source['ball_sha256']:raise ValueError('Frozen ball source changed')
        payload=json.loads(path.read_text());rows={r['f']:r for r in payload['frames']}
        size=np.array([payload['frame_width'],payload['frame_height']],float)
        if not np.isfinite(size).all() or np.any(size<=0):raise ValueError('Invalid frame size')
        provenance[stem]=dict(path=str(path),sha256=source['ball_sha256'],fps=payload['fps'],frame_size=size.tolist())
        for c in [c for c in clips if c['source_stem']==stem]:
            indices=c['contact_frame']+OFFSETS;p=np.zeros((25,2));a=np.zeros(25,bool)
            n=np.zeros(25,bool);recon=np.zeros(25,bool);sources=[];confidence=[];off=[]
            for i,f in enumerate(indices):
                row=rows.get(int(f),{});ball=row.get('ball');out=bool(row.get('ball_out_of_frame'))
                src=row.get('ball_source') or 'unspecified';sources.append(src)
                confidence.append(float(row.get('ball_confidence') or 0));off.append(out)
                n[i]=bool(row.get('ball_new'))
                if ball is None or out:continue
                ball=np.asarray(ball,float)
                if ball.shape!=(2,) or not np.isfinite(ball).all() or np.any(ball<0) or np.any(ball>=size):continue
                p[i]=ball/size;a[i]=True
                recon[i]=('reconstruct' in src or 'interpol' in src)
            v,m=summarize(p,a,n,recon,payload['fps']);result[c['clip_id']]=(v,m)
            np.savez_compressed(args.output/'features'/(c['clip_id']+'.npz'),values=v,valid=m,frames=indices,
                normalized_ball=p,ball_available=a,ball_new=n,ball_source=np.array(sources),
                ball_confidence=np.array(confidence),ball_out_of_frame=np.array(off),reconstructed=recon,
                schema=np.array(SCHEMA),source_video=np.array(stem))
            baseline=args.features/(c['clip_id']+'.parquet')
            if not baseline.exists():baseline=args.features/(c['clip_id']+'.npz')
            baseline_hashes[c['clip_id']]=file_hash(baseline)
    preflight.update(ball_sources=provenance,baseline_hashes=baseline_hashes,
        coverage={name:float(np.mean([m[i] for v,m in result.values()])) for i,name in enumerate(NAMES)})
    save(args.output/'preflight.json',preflight)
    return core,result,preflight


def fit(args,clips,features,holdout,seed,enabled,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    condition='trajectory_delta' if enabled else 'baseline'
    folder=args.output/f'{holdout}_seed{seed}_{condition}';folder.mkdir()
    tr=EventClipDataset([c for c in clips if c['source_stem']!=holdout],str(args.features),sequence_len=120)
    va=EventClipDataset([c for c in clips if c['source_stem']==holdout],str(args.features),sequence_len=120)
    if len(tr)+len(va)!=len(clips):raise ValueError('Missing baseline features')
    mean,std=tr.feature_stats();tr.set_normalization(mean,std);va.set_normalization(mean,std)
    stats=observation_stats([(features[c['clip_id']][0][None,:],features[c['clip_id']][1][None,:]) for c,_ in tr.items])
    train=SummaryDataset(tr,features,stats,enabled);val=SummaryDataset(va,features,stats,enabled)
    train_dl=DataLoader(train,batch_size=32,shuffle=True);val_dl=DataLoader(val,batch_size=32)
    counts=np.bincount(tr.class_indices(),minlength=4)
    weights=torch.tensor(len(tr)/np.maximum(counts*4,1),dtype=torch.float32).clamp(max=10).to(device)
    model=TrajectoryFusion().to(device);criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    best=-1;best_epoch=-1;history=[];best_result=None
    for epoch in range(args.epochs):
        loss,_,_=run_epoch(model,train_dl,criterion,device,optimizer,dual=True)
        _,pred,true=run_epoch(model,val_dl,criterion,device,dual=True);schedule.step()
        f1,macro=per_class_f1(true,pred)
        history.append(dict(epoch=epoch,train_loss=loss,macro_f1=macro))
        print(f'{holdout} seed={seed} {condition} epoch={epoch} F1={macro:.4f}',flush=True)
        if macro>best:
            best=macro;best_epoch=epoch
            confusion=np.zeros((4,4),int)
            for a,b in zip(true,pred):confusion[a,b]+=1
            best_result=dict(holdout=holdout,seed=seed,condition=condition,macro_f1=macro,
                accuracy=float((pred==true).mean()),per_class_f1=f1,truth=true.tolist(),predictions=pred.tolist(),
                confusion_matrix=confusion.tolist(),clip_ids=[c['clip_id'] for c,_ in va.items],best_epoch=epoch,
                train_sources=sorted({c['source_stem'] for c,_ in tr.items}),train_clips=len(tr),val_clips=len(va))
            torch.save(dict(model_type='experimental_trajectory_late_fusion_v1',state_dict=model.state_dict(),
                feature_schema=SCHEMA,base_feature_schema='resolution_normalized_v2',feature_mean=mean.tolist(),
                feature_std=std.tolist(),summary_mean=stats[0].tolist(),summary_std=stats[1].tolist(),
                summary_names=NAMES,enabled=enabled,sequence_len=120,short_sequence_len=60,
                summary_offsets=OFFSETS.tolist(),event_types=EVENT_TYPES,production_promoted=False,
                inference_mode='offline_contact_candidate',**best_result),folder/'best.pt')
        elif epoch-best_epoch>=15:break
    save(folder/'history.json',history);save(folder/'metrics.json',best_result)
    return best_result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=Path('data/event_clips_three_full_gold_causal_120f_v1/manifest.json'))
    ap.add_argument('--features',type=Path,default=Path('data/event_clips_three_full_gold_causal_120f_v1/features'))
    ap.add_argument('--frozen-preflight',type=Path,default=Path('data/event_clips_three_gold_team_pose_offline_20261027_v3/preflight.json'))
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    ap.add_argument('--epochs',type=int,default=50)
    args=ap.parse_args();clips,features,meta=prepare(args);results=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for seed in args.seeds:
        for holdout in meta['sources']:
            for enabled in (False,True):
                save(args.output/'progress.json',dict(status='training',holdout=holdout,seed=seed,enabled=enabled))
                results.append(fit(args,clips,features,holdout,seed,enabled,device))
                save(args.output/'fold_metrics.json',results)
    pooled={}
    for seed in args.seeds:
        pooled[str(seed)]={}
        for condition in ('baseline','trajectory_delta'):
            folds=[f for f in results if f['seed']==seed and f['condition']==condition]
            truth=np.concatenate([f['truth'] for f in folds]);pred=np.concatenate([f['predictions'] for f in folds])
            f1,macro=per_class_f1(truth,pred)
            pooled[str(seed)][condition]=dict(macro_f1=macro,accuracy=float((truth==pred).mean()),per_class_f1=f1)
        pooled[str(seed)]['delta']=pooled[str(seed)]['trajectory_delta']['macro_f1']-pooled[str(seed)]['baseline']['macro_f1']
    save(args.output/'report.json',dict(status='complete',pooled_by_seed=pooled,folds=results,
        production_promoted=False,caveat='Best-epoch whole-source model-selection estimates, not an independent test'))
    save(args.output/'progress.json',dict(status='complete'))
    print(json.dumps(pooled,indent=2),flush=True)


if __name__=='__main__':main()
