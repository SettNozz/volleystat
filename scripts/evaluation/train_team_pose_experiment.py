"""Same-split, same-capacity offline event feature ablation; no production promotion."""
from __future__ import annotations
import argparse,copy,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from action_recognition.event_clips.config import EVENT_TYPES,collapse_labels,label_index
from action_recognition.event_clips.team_pose_features import SCHEMA,CANDIDATE_SCHEMA,VALUE_NAMES,FEATURE_NAMES,normalize_observations,observation_stats
from action_recognition.data.event_clip_dataset import _load_features,_fix_length
from action_recognition.models.sequence_event_model import build_model,save_checkpoint,load_checkpoint
from action_recognition.training.train_event_sequence import run_epoch,per_class_f1
from scripts.evaluation.refresh_team_pose_dataset import save
from src.detection.sam_pose_recovery import file_hash

CONDITIONS=('trajectory','pose','team_pose')


def offline_sequence(clip,values,valid,frames,length=120):
    if not np.array_equal(frames,np.arange(clip['start_frame'],clip['end_frame']+1)):
        raise ValueError('Feature rows do not align with absolute source frames')
    if length!=120:raise ValueError('This schema requires fixed 120-frame offline context')
    result=np.zeros((length,values.shape[1]),np.float32);availability=np.zeros_like(result,dtype=bool)
    slots=frames-(clip['contact_frame']-60)
    if np.any(slots<0) or np.any(slots>=length):raise ValueError('Frames outside offline contact context')
    result[slots]=values;availability[slots]=valid
    result[:,7]=np.linspace(0,1,length,dtype=np.float32);availability[:,7]=True
    return result,availability


def condition_sequence(sequence,condition):
    values,valid=sequence;values=values.copy();valid=valid.copy()
    enabled=np.zeros(len(VALUE_NAMES),bool)
    if condition=='trajectory':enabled[[0,1,2,3,4,6,7,8,9]]=True
    elif condition=='pose':enabled[:18]=True
    elif condition=='team_pose':enabled[:]=True
    else:raise ValueError(condition)
    values[:,~enabled]=0;valid[:,~enabled]=False
    return values,valid


class TeamDataset(Dataset):
    def __init__(self,items,condition,stats=None):
        self.items=items;self.condition=condition
        self.sequences=[condition_sequence((item['values'],item['valid']),condition) for item in items]
        self.mean,self.std=stats if stats is not None else observation_stats(self.sequences)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        v,m=self.sequences[i];x=normalize_observations(v,m,self.mean,self.std)
        return torch.from_numpy(x),torch.tensor(self.items[i]['target'],dtype=torch.long)


def load_items(dataset):
    manifest=json.loads((dataset/'manifest.json').read_text())
    schema=manifest['feature_schema']
    if schema not in (SCHEMA,CANDIDATE_SCHEMA):raise ValueError('Incorrect new feature schema')
    if len({c['clip_id'] for c in manifest['clips']})!=len(manifest['clips']):raise ValueError('Duplicate clip IDs')
    items=[];excluded=[];hashes={}
    for c in manifest['clips']:
        if c.get('label_source')!='human':raise ValueError('Non-human class labels are not authorized')
        path=dataset/'features'/(c['clip_id']+'.npz')
        with np.load(path,allow_pickle=False) as data:
            v=np.asarray(data['features'],np.float32);valid=data['valid'];frames=data['frames']
            if (data['schema'].item()!=schema or data['source_video'].item()!=c['source_stem']
                or data['feature_names'].tolist()!=list(VALUE_NAMES) or valid.dtype!=bool
                or v.shape!=valid.shape or v.shape[1]!=len(VALUE_NAMES) or not np.isfinite(v).all()
                or np.any(v[~valid]!=0)):
                raise ValueError(f'Invalid feature contract: {path}')
            v,valid=offline_sequence(c,v,valid,frames)
        hashes[c['clip_id']]=file_hash(path)
        if collapse_labels(c.get('event_types'),c.get('fsm_event_type')) is None:
            excluded.append(c['clip_id']);continue
        items.append(dict(clip=c,values=v,valid=valid,target=label_index(c['event_types'],c.get('fsm_event_type'))))
    return manifest,items,excluded,hashes


def evaluate(model,loader,device):
    model.eval();true=[];pred=[];prob=[]
    with torch.no_grad():
        for x,y in loader:
            scores=model(x.to(device));prob.extend(scores.softmax(1).cpu().numpy().tolist())
            true.extend(y.numpy().tolist());pred.extend(scores.argmax(1).cpu().numpy().tolist())
    true=np.asarray(true);pred=np.asarray(pred);f1,macro=per_class_f1(true,pred)
    confusion=np.zeros((4,4),int)
    for a,b in zip(true,pred):confusion[a,b]+=1
    return dict(accuracy=float((true==pred).mean()),macro_f1=macro,per_class_f1=f1,
                confusion_matrix=confusion.tolist(),truth=true.tolist(),predictions=pred.tolist(),probabilities=prob)


def fit(items,holdout,condition,destination,args,device):
    destination.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    train=[x for x in items if x['clip']['source_stem']!=holdout];validation=[x for x in items if x['clip']['source_stem']==holdout]
    if not train or not validation:raise ValueError('Empty whole-source split')
    tr=TeamDataset(train,condition);va=TeamDataset(validation,condition,(tr.mean,tr.std))
    train_dl=DataLoader(tr,batch_size=32,shuffle=True);val_dl=DataLoader(va,batch_size=32,shuffle=False)
    model=build_model('multiscale',len(FEATURE_NAMES),short_sequence_len=60,sequence_len=120,short_window_centered=True,
                      contact_center_pool=args.contact_center_pool).to(device)
    counts=np.bincount([x['target'] for x in train],minlength=4)
    class_weights=torch.tensor(len(train)/np.maximum(counts*4,1),dtype=torch.float32).clamp(max=10).to(device)
    criterion=torch.nn.CrossEntropyLoss(weight=class_weights,label_smoothing=.05)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=-1;best_epoch=-1;history=[];checkpoint=destination/'best_event_sequence.pt'
    for epoch in range(args.epochs):
        loss,_,_=run_epoch(model,train_dl,criterion,device,opt)
        metric=evaluate(model,val_dl,device);schedule.step()
        record=dict(epoch=epoch,train_loss=loss,accuracy=metric['accuracy'],macro_f1=metric['macro_f1'],per_class_f1=metric['per_class_f1'])
        history.append(record)
        print(f'{holdout} {condition} epoch={epoch} F1={metric["macro_f1"]:.4f}',flush=True)
        if metric['macro_f1']>best:
            best=metric['macro_f1'];best_epoch=epoch
            save_checkpoint(checkpoint,model,'multiscale',len(FEATURE_NAMES),extra=dict(
                feature_schema=args.feature_schema,feature_names=list(FEATURE_NAMES),value_names=list(VALUE_NAMES),
                contact_center_pool=args.contact_center_pool,
                feature_mean=np.r_[tr.mean,np.zeros(len(VALUE_NAMES))].tolist(),
                feature_std=np.r_[tr.std,np.ones(len(VALUE_NAMES))].tolist(),
                normalization='train_only_valid_values; impute standardized zero; validity bits unscaled',
                seed=args.seed,epoch=epoch,sequence_len=120,short_sequence_len=60,short_window_centered=True,condition=condition,
                val_macro_f1=best,val_acc=metric['accuracy'],per_class_f1=metric['per_class_f1'],
                train_clip_ids=[x['clip']['clip_id'] for x in train],val_clip_ids=[x['clip']['clip_id'] for x in validation],
                train_sources=sorted({x['clip']['source_stem'] for x in train}),val_sources=[holdout],
                taxonomy='v4_core_4class',inference_mode='offline_centered_context',contact_position=60,context_before=60,context_after=59,
                final_fit=False,production_promoted=False))
        elif epoch-best_epoch>=args.patience:break
    model,metadata=load_checkpoint(checkpoint,device);model=model.to(device)
    result=evaluate(model,val_dl,device)
    result.update(condition=condition,holdout=holdout,best_epoch=best_epoch,epochs_run=len(history),
        train_clips=len(train),val_clips=len(validation),train_class_counts=counts.tolist(),
        clip_ids=[x['clip']['clip_id'] for x in validation],checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint))
    save(destination/'history.json',history);save(destination/'metrics.json',result)
    del model;return result


class HistoricalDataset(Dataset):
    def __init__(self,items,features,metadata,causal):
        self.items=items;self.arrays=[]
        mean=np.asarray(metadata['feature_mean'],np.float32);std=np.asarray(metadata['feature_std'],np.float32)
        active=np.asarray(metadata.get('feature_active_mask',np.ones(18)),bool)
        for item in items:
            c=copy.deepcopy(item['clip'])
            c['start_frame']=c.get('original_start_frame',c['start_frame'])
            c['end_frame']=c.get('original_end_frame',c['end_frame'])
            arr=_load_features(features,c['clip_id'])
            if arr is None:raise ValueError('Missing historical feature file')
            if causal:
                expected=c['end_frame']-c['start_frame']+1
                if len(arr)!=expected:raise ValueError('Cannot infer source-frame alignment for historical features')
                arr=arr[:max(0,min(c['contact_frame'],c['end_frame'])-c['start_frame']+1)]
            x=_fix_length(arr,int(metadata.get('sequence_len',120))).copy()
            x[:,7]=np.linspace(0,1,len(x),dtype=np.float32);x[:,~active]=mean[~active]
            self.arrays.append((x-mean)/std)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return torch.from_numpy(self.arrays[i]),torch.tensor(self.items[i]['target'],dtype=torch.long)


def historical_evaluation(items,holdout,root,dataset,device):
    names={'Khe4P8QcJkg_5_4k':'khe','j0gRhgFCgIs_4_4k':'j0','j3BO9GgUFBI_3_4k':'j3'}
    path=root/f'multiscale_val_{names[holdout]}_seed42/best_event_sequence.pt'
    model,metadata=load_checkpoint(path,device)
    model=model.to(device)
    if metadata.get('final_fit'):raise ValueError('Do not evaluate an all-source final-fit model as held-out validation')
    selected=[x for x in items if x['clip']['source_stem']==holdout]
    result=dict(checkpoint=str(path),checkpoint_sha256=file_hash(path),published_macro_f1=metadata.get('val_macro_f1'),
        published_accuracy=metadata.get('val_acc'),current_val_clips=len(selected),
        warning='Historical checkpoint keeps its original past-only window; new models use matched offline windows. Feature conclusions use retrained controls.')
    for mode in ('historical_window','causal_window'):
        ds=HistoricalDataset(selected,dataset/'baseline_features_snapshot',metadata,mode=='causal_window')
        result[mode]=evaluate(model,DataLoader(ds,batch_size=32),device)
        result[mode]['clip_ids']=[x['clip']['clip_id'] for x in selected]
    result['published_macro_f1_reproduced']=abs(result['historical_window']['macro_f1']-result['published_macro_f1'])<1e-8
    del model;return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--historical',required=True,type=Path);ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--patience',type=int,default=15);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--contact-center-pool',action='store_true')
    ap.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    args=ap.parse_args();meta=json.loads((args.dataset/'preflight.json').read_text())
    if meta['status']!='complete':raise ValueError('Extraction must finish before training')
    if file_hash(meta['parent_manifest'])!=meta['parent_manifest_sha256']:raise ValueError('Human labels changed since freeze')
    if file_hash(meta['pose_checkpoint'])!=meta['pose_checkpoint_sha256']:
        raise ValueError('Pose weights changed during extraction')
    for source in meta['sources'].values():
        if file_hash(source['ball_annotations'])!=source['ball_sha256'] or file_hash(source['video'])!=source['video_sha256']:
            raise ValueError('Frozen source inputs changed during extraction')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest,items,excluded,hashes=load_items(args.dataset)
    args.feature_schema=manifest['feature_schema']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    sources=sorted({x['clip']['source_stem'] for x in items})
    main_holdout=meta['split_policy']['validation_source_stems'][0]
    sources=[main_holdout]+[s for s in sources if s!=main_holdout]
    preflight=dict(dataset=str(args.dataset.resolve()),manifest_sha256=file_hash(args.dataset/'manifest.json'),
        parent_manifest_sha256=meta['parent_manifest_sha256'],label_source='human',core_clips=len(items),
        excluded_noncore=len(excluded),excluded_clip_ids=excluded,feature_hashes=hashes,feature_schema=args.feature_schema,
        contact_center_pool=args.contact_center_pool,
        input_dim=len(FEATURE_NAMES),conditions=args.conditions,primary_holdout=main_holdout,
        cross_validation_unit='complete_source_video',seed=args.seed,epochs=args.epochs,patience=args.patience,
        model='multiscale',long_frames=120,short_frames=60,short_window_centered=True,inference_mode='offline',device=device,
        caveats=['One seed, three related-camera sources; no independent untouched test set.',
        'Best-epoch validation metrics are model selection estimates.',
        'Offline future frames are authorized; no human team/contact/FSM fields are feature inputs.',
        'Full skeleton cache is model-derived, not human keypoint ground truth.'])
    save(args.output/'preflight.json',preflight);print(json.dumps({k:v for k,v in preflight.items() if k not in ('feature_hashes','excluded_clip_ids')},indent=2),flush=True)
    results=[];historic={}
    for source in sources:
        historic[source]=historical_evaluation(items,source,args.historical,args.dataset,device)
        save(args.output/'historical_reevaluation.json',historic)
        for condition in args.conditions:
            save(args.output/'progress.json',dict(status='training',holdout=source,condition=condition))
            result=fit(items,source,condition,args.output/f'{source}_{condition}',args,device)
            results.append(result);save(args.output/'fold_metrics.json',results)
    aggregate={}
    for condition in args.conditions:
        folds=[r for r in results if r['condition']==condition]
        truth=np.concatenate([r['truth'] for r in folds]);pred=np.concatenate([r['predictions'] for r in folds])
        f1,macro=per_class_f1(truth,pred)
        aggregate[condition]=dict(oof_macro_f1=macro,oof_accuracy=float((truth==pred).mean()),per_class_f1=f1,
            mean_fold_macro_f1=float(np.mean([r['macro_f1'] for r in folds])),
            worst_fold_macro_f1=float(min(r['macro_f1'] for r in folds)),core_clips=len(truth))
    historic_true=np.concatenate([h['historical_window']['truth'] for h in historic.values()])
    historic_pred=np.concatenate([h['historical_window']['predictions'] for h in historic.values()])
    historic_f1,historic_macro=per_class_f1(historic_true,historic_pred)
    aggregate['historical_past_only']=dict(oof_macro_f1=historic_macro,oof_accuracy=float((historic_true==historic_pred).mean()),
        per_class_f1=historic_f1,mean_fold_macro_f1=float(np.mean([h['historical_window']['macro_f1'] for h in historic.values()])),
        worst_fold_macro_f1=float(min(h['historical_window']['macro_f1'] for h in historic.values())),core_clips=len(historic_true),
        context='original past-only 120 frames',input_dim=18)
    report=dict(status='complete',experiment=preflight,aggregate=aggregate,folds=results,historical=historic,
        production_promoted=False,final_fit_performed=False,
        team_pose_minus_trajectory_oof_f1=(aggregate['team_pose']['oof_macro_f1']-aggregate['trajectory']['oof_macro_f1']
            if {'team_pose','trajectory'}<=aggregate.keys() else None),
        team_pose_minus_pose_oof_f1=(aggregate['team_pose']['oof_macro_f1']-aggregate['pose']['oof_macro_f1']
            if {'team_pose','pose'}<=aggregate.keys() else None))
    save(args.output/'report.json',report)
    rows=''.join(f'<tr><td>{name}</td><td>{r["oof_macro_f1"]:.3f}</td><td>{r["oof_accuracy"]:.3f}</td><td>{r["worst_fold_macro_f1"]:.3f}</td></tr>' for name,r in aggregate.items())
    page=f'''<!doctype html><meta charset="utf-8"><title>Event-model comparison</title><style>body{{font:17px/1.5 system-ui;max-width:1000px;margin:40px auto}}td,th{{padding:12px;border-bottom:1px solid #ddd}}</style><h1>Події: траєкторія, пози та команди</h1><p>{len(items)} кліпів чотирьох класів, {len(excluded)} інших збережено й виключено з класифікатора. Розділення за повними відео, з однаковим контекстом 60 кадрів до та 59 після контакту. Seed {args.seed}.</p><table><tr><th>Ознаки</th><th>OOF macro F1</th><th>Accuracy</th><th>Найгірше відео, F1</th></tr>{rows}</table><p>trajectory — контроль тільки з м’ячем; pose — також найближчий скелет; team_pose — також геометрія контакту окремо для обох сторін. Всі три мають однакову архітектуру й розмір входу; вимкнені канали заповнені нулями.</p><p>Це оцінювання вибору моделі на трьох відео й одному seed, не незалежний фінальний тест. Старі метрики мають інші мітки/вікна; повторні оцінки є в JSON. Чинний checkpoint не замінено.</p><a href="report.json">Повний звіт</a>'''
    (args.output/'comparison.html').write_text(page,encoding='utf-8')
    save(args.output/'progress.json',dict(status='complete',report=str(args.output/'report.json')))
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':main()
