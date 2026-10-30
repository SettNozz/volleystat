"""Joint author STES temporal + spatial pilot on canonical human contacts.
Native-frame split: two complete videos train, one complete video validation.
"""
from __future__ import annotations
import argparse,json,random,time,os,sys,shutil,math,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/".runtime/vnl_stes_deps"))
import cv2,numpy as np,torch
from torch.nn import functional as F
from action_recognition.models.vnl_stes_model import build_author_stes
from action_recognition.rally_spotting.spatial_labels import canonicalize,spatial_targets,file_sha
from action_recognition.rally_spotting.rgb_spotter import frame_targets,EVENTS
from scripts.evaluation.run_author_stes import Clips,tensor_images,atomic_checkpoint,temporal_ap,event_metrics
from scripts.evaluation.refresh_team_pose_dataset import save
from action_recognition.rally_spotting.stes_continuation import validate_source,extend_scheduler,lr_factor

SCHEMA="vnl_stes_author_spatial_v1"

class SpatialClips(Clips):
    def __init__(self,records,input_width=224,input_height=224,clip_frames=64):
        super().__init__(records)
        self.input_width=input_width;self.input_height=input_height;self.clip_frames=clip_frames
        self.xy={};self.masks={}
        for r in records:
            self.targets[r["stem"]][r["ignored_exact"]]=-100
            # An explicit valid event takes priority over nearby ignore masks.
            for e in r["events"]:self.targets[r["stem"]][e["frame"]]=EVENTS.index(e["type"])+1
            self.xy[r["stem"]],self.masks[r["stem"]]=spatial_targets(r["frames"],r["events"])
    def read(self,record,start):
        key=record["stem"]
        if key not in self.caps:
            cap=cv2.VideoCapture(record["cache"])
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH)<self.input_width or cap.get(cv2.CAP_PROP_FRAME_HEIGHT)<self.input_height:
                cap.release();raise ValueError("Refuse to upscale low-resolution cache; prepare from originals")
            self.caps[key]=cap
        cap=self.caps[key];cap.set(cv2.CAP_PROP_POS_FRAMES,int(start));images=[]
        for _ in range(self.clip_frames):
            ok,image=cap.read()
            if not ok:raise ValueError("Cached frame decode failed")
            image=cv2.resize(image,(self.input_width,self.input_height))
            images.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        return np.stack(images).transpose(0,3,1,2)
    def sample(self,batch):
        xs=[];ys=[];points=[];masks=[]
        for _ in range(batch):
            if random.random()<.5:
                rec,frame=random.choice(self.flat);start=max(0,min(rec["frames"]-self.clip_frames,frame-random.randrange(self.clip_frames)))
            else:
                rec=random.choices(self.records,weights=self.weights)[0];start=random.randrange(rec["frames"]-self.clip_frames+1)
            x=self.read(rec,start);stem=rec["stem"]
            xy=self.xy[stem][start:start+self.clip_frames].copy();mask=self.masks[stem][start:start+self.clip_frames].copy()
            if random.random()<.5:
                x=x[:,:,:,::-1].copy();xy[mask,0]=1-xy[mask,0]
            xs.append(x);ys.append(self.targets[stem][start:start+self.clip_frames]);points.append(xy);masks.append(mask)
        return np.stack(xs),np.stack(ys),np.stack(points),np.stack(masks)


def losses(output,y,xy,mask):
    weight=y.new_tensor([1,5,5,5,5],dtype=torch.float32)
    total=F.cross_entropy(output["im_feat"].float().transpose(1,2),y,weight=weight,ignore_index=-100,reduction="sum")
    valid=y!=-100
    denominator=weight[y[valid]].sum().clamp_min(1)
    cls=total/denominator
    loc=output["loc_feat"].float().sigmoid()
    spatial=(loc[mask]-xy[mask]).abs().sum(-1).mean() if mask.any() else loc.sum()*0
    return cls,spatial


@torch.no_grad()
def infer(model,reader,record,device,batch):
    model.eval();n=record["frames"];scores=np.zeros((n,5),np.float32);coords=np.zeros((n,2),np.float32);counts=np.zeros((n,1),np.float32)
    clip=reader.clip_frames;stride=max(1,clip//2)
    starts=list(range(0,n-clip+1,stride));last=n-clip
    if starts[-1]!=last:starts.append(last)
    for i in range(0,len(starts),batch):
        chunk=starts[i:i+batch];x=tensor_images(np.stack([reader.read(record,s) for s in chunk]),device)
        with torch.autocast("cuda",dtype=torch.float16):out=model(x)
        p=out["im_feat"].float().softmax(-1).cpu().numpy();xy=out["loc_feat"].float().sigmoid().cpu().numpy()
        for s,a,b in zip(chunk,p,xy):scores[s:s+clip]+=a;coords[s:s+clip]+=b;counts[s:s+clip]+=1
    if (counts==0).any():raise ValueError("Uncovered validation frame")
    return scores/counts,coords/counts


def spatial_metrics(coords,events,width,height,input_width=224,input_height=224):
    valid=[e for e in events if e["spatial_valid"]]
    truth=np.asarray([e["xy_normalized"] for e in valid]);pred=coords[[e["frame"] for e in valid]]
    errors=np.linalg.norm((pred-truth)*224,axis=-1)
    native=np.linalg.norm((pred-truth)*np.array([width,height]),axis=-1)
    actual=np.linalg.norm((pred-truth)*np.array([input_width,input_height]),axis=-1)
    return dict(count=len(valid),input_size=[input_width,input_height],mean_error_input_px=float(actual.mean()),within_6px_input=float(np.mean(actual<=6)),mean_error_224=float(errors.mean()),median_error_224=float(np.median(errors)),
        p90_error_224=float(np.quantile(errors,.9)),mean_error_native_px=float(native.mean()),
        within_2px_224=float(np.mean(errors<=2)),within_4px_224=float(np.mean(errors<=4)),within_6px_224=float(np.mean(errors<=6)),
        note="Conditional localization at human contact frames; *_224 are fixed legacy reference units; *_input use actual input dimensions; not end-to-end spatial AP")


def masked_probabilities(p,reader,record):
    p=p.copy();ignore=reader.targets[record["stem"]]==-100;p[ignore]=[1,0,0,0,0]
    return p


def prepare(args):
    canonical=canonicalize(args.reviews)
    if args.output.exists() and not args.resume:raise ValueError("Existing output; use --resume or a new path")
    parent=json.loads((args.cache_dir/"preflight.json").read_text())
    if canonical["manifest_sha256"]!=parent["manifest_sha256"]:raise ValueError("Cache manifest differs")
    records=[]
    for source in parent["human_sources"]:
        stem=source["stem"];video=canonical["videos"][stem]
        if args.input_width>video["width"] or args.input_height>video["height"]:raise ValueError("Requested input exceeds original resolution")
        if file_sha(source["video"])!=source["video_sha256"]:raise ValueError("Original video changed")
        cache=args.cache_dir/"videos"/(stem+".mp4");meta=json.loads(cache.with_suffix(".json").read_text())
        if file_sha(cache)!=meta["sha256"] or meta["source_sha256"]!=source["video_sha256"]:raise ValueError("RGB cache changed")
        old=json.loads((args.temporal_data_dir/(stem+"_labels.json")).read_text())
        frames=meta["frames"]
        if frames<args.clip_frames:raise ValueError("Video shorter than model temporal window")
        ignored_exact=canonical["ignored_exact_frames"].get(stem,[])
        if any(f>=frames for f in ignored_exact):raise ValueError("Unknown region exceeds cache")
        spatial_targets(frames,canonical["events"][stem])
        records.append(dict(stem=stem,cache=str(cache),video=source["video"],frames=frames,fps=video["fps"],
            width=video["width"],height=video["height"],events=canonical["events"][stem],ignored=old["ignored_noncore_frames"],ignored_exact=ignored_exact))
    if args.holdout not in {r["stem"] for r in records}:raise ValueError("Unknown holdout")
    preflight=dict(schema=SCHEMA,review_sha256=canonical["review_sha256"],canonical_audit=canonical["audit"],
        source_revision=parent["source_revision"],author_code_hashes=parent["author_code_hashes"],
        backbone_sha256=file_sha(args.weights),cache_preflight_sha256=file_sha(args.cache_dir/"preflight.json"),
        holdout=args.holdout,train_sources=[r["stem"] for r in records if r["stem"]!=args.holdout],
        recipe=dict(epochs=args.epochs,steps=args.steps,batch=args.batch,accumulation=args.accumulation,eval_every=args.eval_every,
            input_width=args.input_width,input_height=args.input_height,cudnn_benchmark=(args.input_width<=398),activation_checkpointing=args.activation_checkpointing,backbone_chunk_frames=args.backbone_chunk_frames,
            seed=42,backbone_lr=1e-4,heads_lr=1e-3,loss="weighted CE + 1.0 masked L1(sum xy)",clip_frames=args.clip_frames,
            foreground_sampling=.5,initialization="ImageNet backbone; random temporal/spatial heads; all trainable"),
        implementation_hashes={rel:file_sha(ROOT/rel) for rel in ("scripts/evaluation/run_spatial_stes.py","action_recognition/rally_spotting/spatial_labels.py","action_recognition/models/vnl_stes_model.py","action_recognition/models/activation_checkpointing.py","scripts/evaluation/run_author_stes.py")},
        selection="minimum validation mean Euclidean coordinate error at verified contact frames; not independent test",
        highres_cache_dir=str((args.rgb_cache_dir or args.output/f"rgb_{args.input_width}x{args.input_height}").resolve()),
        production_promoted=False)
    if args.resume_from:
        preflight["continuation"]=validate_source(args.resume_from,preflight,args.epochs)
        preflight["recipe"]["initialization"]="Full checkpoint restore; retain AdamW/RNG; extend cosine schedule"
    preflight["implementation_hashes"]["action_recognition/rally_spotting/stes_continuation.py"]=file_sha(ROOT/"action_recognition/rally_spotting/stes_continuation.py")
    for rel,expected in parent["author_code_hashes"].items():
        if file_sha(ROOT/"third_party/vnl_stes"/rel)!=expected:raise ValueError("Author source changed")
    args.output.mkdir(parents=True,exist_ok=True)
    frozen=args.output/"preflight.json"
    if frozen.exists():
        if json.loads(frozen.read_text())!=preflight:raise ValueError("Resume preflight changed")
    else:
        save(frozen,preflight);save(args.output/"canonical_labels.json",canonical)
        for rel in preflight["implementation_hashes"]:
            target=args.output/"code_snapshot"/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    if (args.input_width,args.input_height)!=(224,224):prepare_highres_caches(records,args)
    return records,canonical



def prepare_highres_caches(records,args):
    # Always decode the original source, never magnify the 398x224 legacy cache.
    folder=args.rgb_cache_dir or args.output/f"rgb_{args.input_width}x{args.input_height}"
    if args.rgb_cache_dir and not folder.is_dir():raise ValueError("Explicit shared RGB cache must already exist")
    folder.mkdir(exist_ok=True)
    for record in records:
        cache=folder/(record["stem"]+".mp4");metadata=cache.with_suffix(".json")
        if not metadata.exists():
            if args.rgb_cache_dir:raise ValueError("Shared RGB cache is incomplete; do not modify it")
            save(args.output/"progress.json",dict(status="preparing_high_resolution",source=record["stem"],input_size=[args.input_width,args.input_height],pid=os.getpid()))
            partial=cache.with_suffix(".partial.mp4")
            if partial.exists() or cache.exists():raise ValueError("Incomplete high-res cache; preserve it and use a new output folder")
            subprocess.run(["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-n","-hwaccel","cuda","-hwaccel_output_format","cuda","-i",record["video"],"-vf",f"scale_cuda={args.input_width}:{args.input_height}","-c:v","h264_nvenc","-preset","p4","-cq","18","-g","32","-bf","0","-an","-fps_mode","passthrough",str(partial)],check=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            cap=cv2.VideoCapture(str(partial));size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            if frames!=record["frames"] or size!=[args.input_width,args.input_height]:raise ValueError("High-res cache alignment mismatch")
            partial.replace(cache);save(metadata,dict(sha256=file_sha(cache),frames=frames,size=size,source=str(Path(record["video"]).resolve()),source_sha256=file_sha(record["video"])))
        meta=json.loads(metadata.read_text())
        if file_sha(cache)!=meta["sha256"] or meta["size"]!=[args.input_width,args.input_height] or meta["frames"]!=record["frames"] or meta["source_sha256"]!=file_sha(record["video"]):raise ValueError("High-res cache changed")
        record["cache"]=str(cache)


def train(args,records,canonical):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=(args.input_width<=398)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    device=torch.device("cuda");model=build_author_stes(args.weights,clip_len=args.clip_frames,predict_location=True,activation_checkpointing=args.activation_checkpointing,backbone_chunk_frames=args.backbone_chunk_frames).to(device)
    optimizer=torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if n.startswith("_features")],lr=1e-4),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith("_features")],lr=1e-3)],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:lr_factor(e,args.epochs))
    scaler=torch.amp.GradScaler("cuda")
    training=[r for r in records if r["stem"]!=args.holdout];validation=next(r for r in records if r["stem"]==args.holdout)
    sampler=SpatialClips(training,args.input_width,args.input_height,args.clip_frames);reader=SpatialClips([validation],args.input_width,args.input_height,args.clip_frames);best=float("inf");history=[];start=0
    latest=args.output/"last.pt"
    restore=latest if latest.exists() else args.resume_from
    if restore:
        ck=torch.load(restore,map_location=device,weights_only=False);model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"]);scheduler.load_state_dict(ck["scheduler"]);scaler.load_state_dict(ck["scaler"])
        start=ck["next_epoch"];best=ck["best"];history=ck["history"]
        random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"]);torch.set_rng_state(ck["torch_rng"].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in ck["cuda_rng"]])
    if restore and not latest.exists():
        extend_scheduler(optimizer,scheduler,start,args.epochs)
        source=Path(args.resume_from).parent
        for name in ("best.pt","best_metrics.json","best_predictions.npz"):
            if not (source/name).is_file():raise ValueError("Missing source best artifact: "+name)
            shutil.copy2(source/name,args.output/name)
        save(args.output/"history.json",history)
        save(args.output/"continuation_restore.json",dict(start_epoch=start+1,optimizer_restored=True,scaler_restored=True,rng_restored=True,lr=[g["lr"] for g in optimizer.param_groups]))
    train_points=np.asarray([e["xy_normalized"] for r in training for e in r["events"] if e["spatial_valid"]])
    centre=train_points.mean(0);constant=np.tile(centre,(validation["frames"],1))
    save(args.output/"constant_baseline.json",dict(train_mean_xy=centre.tolist(),metrics=spatial_metrics(constant,validation["events"],validation["width"],validation["height"],args.input_width,args.input_height),train_only=True))
    try:
        for epoch in range(start,args.epochs):
            model.train();optimizer.zero_grad(set_to_none=True);cls_values=[];xy_values=[];started=time.time();valid_seen=0;cycle_valid=0
            for step in range(args.steps):
                if (args.output/"cancel.request").exists():raise KeyboardInterrupt("cancel.request")
                x,y,xy,mask=sampler.sample(args.batch);x=tensor_images(x,device)
                y=torch.from_numpy(y).to(device);xy=torch.from_numpy(xy).to(device);mask=torch.from_numpy(mask).to(device)
                with torch.autocast("cuda",dtype=torch.float16):
                    out=model(x);cls,loc=losses(out,y,xy,mask);loss=(cls+loc)/args.accumulation
                if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
                scaler.scale(loss).backward();valid_seen+=int(mask.sum());cycle_valid+=int(mask.sum())
                if (step+1)%args.accumulation==0 or step+1==args.steps:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                    if cycle_valid and not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model._pred_loc.parameters()):
                        raise ValueError("Spatial head received no finite nonzero gradient")
                    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);cycle_valid=0
                cls_values.append(float(cls.detach()));xy_values.append(float(loc.detach()))
                if step%8==0:save(args.output/"progress.json",dict(status="training",epoch=epoch+1,epochs=args.epochs,step=step+1,steps=args.steps,pid=os.getpid(),classification_loss=float(np.mean(cls_values)),spatial_loss=float(np.mean(xy_values)),verified_contacts_seen=valid_seen,elapsed_seconds=time.time()-started))
            scheduler.step();entry=dict(epoch=epoch+1,classification_loss=float(np.mean(cls_values)),spatial_loss=float(np.mean(xy_values)),verified_contacts_seen=valid_seen,train_seconds=time.time()-started)
            if (epoch+1)%args.eval_every==0 or epoch+1==args.epochs:
                save(args.output/"progress.json",dict(status="validating",epoch=epoch+1,epochs=args.epochs,pid=os.getpid()))
                validation_started=time.perf_counter()
                p,coords=infer(model,reader,validation,device,args.batch);p=masked_probabilities(p,reader,validation)
                metrics=spatial_metrics(coords,validation["events"],validation["width"],validation["height"],args.input_width,args.input_height)
                maps=temporal_ap(p,validation["events"],args.holdout);map_score=float(np.mean([maps[str(t)]["map"] for t in (0,1,2,4)]))
                choices=[]
                for threshold in (.1,.2,.3,.4,.5,.6,.7,.8,.9):
                    metric,_=event_metrics(p,validation["events"],threshold,tolerances=(10,));choices.append((metric["at_10"]["macro_f1"],threshold))
                f1,threshold=max(choices);entry.update(spatial=metrics,map_0_1_2_4=map_score,macro_f1_at10=f1,threshold=threshold)
                entry["validation_seconds"]=time.perf_counter()-validation_started
                print(json.dumps(entry),flush=True)
                if metrics["mean_error_224"]<best:
                    best=metrics["mean_error_224"];metric,predictions=event_metrics(p,validation["events"],threshold)
                    result=dict(schema=SCHEMA,epoch=epoch+1,holdout=args.holdout,train_sources=[r["stem"] for r in training],spatial=metrics,temporal_AP=maps,map_0_1_2_4=map_score,metrics=metric,threshold=threshold,predictions=predictions,
                        constant_baseline=spatial_metrics(constant,validation["events"],validation["width"],validation["height"],args.input_width,args.input_height),
                        audit=canonical["audit"],selection="validation coordinate-error model selection, not independent test",production_promoted=False)
                    save(args.output/"best_metrics.json",result);np.savez_compressed(args.output/"best_predictions.npz",probabilities=p,xy=coords)
                    atomic_checkpoint(args.output/"best.pt",dict(model_type=SCHEMA,model=model.state_dict(),spatial_head=True,clip_len=args.clip_frames,input_size=[args.input_width,args.input_height],activation_checkpointing=args.activation_checkpointing,backbone_chunk_frames=args.backbone_chunk_frames,classes=["background",*EVENTS],threshold=threshold,holdout=args.holdout,train_sources=result["train_sources"],review_sha256=canonical["review_sha256"],production_promoted=False))
            history.append(entry);save(args.output/"history.json",history)
            atomic_checkpoint(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),next_epoch=epoch+1,best=best,history=history,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    finally:sampler.close();reader.close()
    result=json.loads((args.output/"best_metrics.json").read_text());result.update(status="complete",completed_epochs=args.epochs)
    save(args.output/"report.json",result);save(args.output/"progress.json",dict(status="complete",epochs=args.epochs,pid=os.getpid()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews",type=Path,default=Path("data/contact_spatial_gold_v1/reviews.json"))
    parser.add_argument("--cache-dir",type=Path,default=Path("results/vnl_stes_author_20261027_v1"))
    parser.add_argument("--temporal-data-dir",type=Path,default=Path("results/rgb_event_spotting_20261027_v2"))
    parser.add_argument("--weights",type=Path,default=Path("models/pretrained/vnl_stes_regnety_008.safetensors"))
    parser.add_argument("--holdout",default="Khe4P8QcJkg_5_4k")
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--resume-from",type=Path,help="Import full checkpoint from an older experiment")
    parser.add_argument("--rgb-cache-dir",type=Path,help="Read-only existing high-resolution cache")
    parser.add_argument("--prepare-only",action="store_true",help="Validate inputs and write plan; never construct/train a model")
    parser.add_argument("--input-width",type=int,default=224);parser.add_argument("--input-height",type=int,default=224)
    parser.add_argument("--backbone-chunk-frames",type=int)
    parser.add_argument("--clip-frames",type=int,default=64);parser.add_argument("--activation-checkpointing",action="store_true")
    parser.add_argument("--epochs",type=int,default=20);parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--batch",type=int,default=2);parser.add_argument("--accumulation",type=int,default=4);parser.add_argument("--eval-every",type=int,default=2)
    args=parser.parse_args()
    if min(args.epochs,args.steps,args.batch,args.accumulation,args.eval_every,args.clip_frames,args.input_width,args.input_height)<1:raise ValueError("Invalid training budget")
    if args.backbone_chunk_frames is not None and (args.backbone_chunk_frames<1 or args.clip_frames%args.backbone_chunk_frames):raise ValueError("Backbone chunk must divide clip frames")
    if args.steps%args.accumulation:raise ValueError("Steps must be divisible by accumulation")
    if args.resume_from and args.output.resolve()==args.resume_from.parent.resolve():raise ValueError("Continuation must use a separate output directory")
    if not args.prepare_only and not torch.cuda.is_available():raise ValueError("CUDA required")
    try:
        started=time.perf_counter()
        records,canonical=prepare(args);preparation_seconds=time.perf_counter()-started
        save(args.output/"timing.json",dict(status="running",preparation_seconds=preparation_seconds))
        print(json.dumps(canonical["audit"]),flush=True)
        if args.prepare_only:
            save(args.output/"timing.json",dict(status="prepared_not_started",preparation_seconds=preparation_seconds))
            save(args.output/"progress.json",dict(status="prepared_not_started",training_started=False,epochs=args.epochs))
            save(args.output/"readiness.json",dict(status="prepared_not_started",training_started=False,continuation=json.loads((args.output/"preflight.json").read_text()).get("continuation"),preflight_sha256=file_sha(args.output/"preflight.json"),preparation_seconds=preparation_seconds))
            return
        train(args,records,canonical)
        history=json.loads((args.output/"history.json").read_text())
        imported=json.loads((args.output/"preflight.json").read_text()).get("continuation",{}).get("completed_epochs",0)
        current_history=[e for e in history if e["epoch"]>imported]
        save(args.output/"timing.json",dict(status="complete",preparation_seconds=preparation_seconds,
            total_wall_seconds=time.perf_counter()-started,training_seconds=sum(e["train_seconds"] for e in current_history),imported_epochs=imported,
            validation_seconds=sum(e.get("validation_seconds",0) for e in current_history),epochs=history,
            note="Wall time includes preparation, model initialization, checkpoints and evaluation; prepare time is one-time cache generation/hash checking"))
    except BaseException as exc:
        if args.output.exists():save(args.output/"progress.json",dict(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",error=repr(exc),pid=os.getpid()))
        raise

if __name__=="__main__":main()
