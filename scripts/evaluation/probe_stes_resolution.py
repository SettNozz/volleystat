"""Measure actual STES forward/backward memory at a chosen input resolution."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/".runtime/vnl_stes_deps"))
import torch
from action_recognition.models.vnl_stes_model import build_author_stes
from scripts.evaluation.run_spatial_stes import losses

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width",type=int,default=1248);p.add_argument("--height",type=int,default=702)
    p.add_argument("--backbone-chunk-frames",type=int)
    p.add_argument("--activation-checkpointing",action="store_true")
    p.add_argument("--clip-frames",type=int,default=8);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    result=dict(width=args.width,height=args.height,clip_frames=args.clip_frames,batch=1,gpu=torch.cuda.get_device_name(),amp=True,activation_checkpointing=args.activation_checkpointing,backbone_chunk_frames=args.backbone_chunk_frames)
    started=time.time()
    try:
        model=build_author_stes(pretrained=False,clip_len=args.clip_frames,predict_location=True,activation_checkpointing=args.activation_checkpointing,backbone_chunk_frames=args.backbone_chunk_frames).cuda().train()
        x=torch.randn(1,args.clip_frames,3,args.height,args.width,device="cuda",dtype=torch.float16)
        y=torch.zeros(1,args.clip_frames,device="cuda",dtype=torch.long);y[0,args.clip_frames//2]=1
        xy=torch.zeros(1,args.clip_frames,2,device="cuda");xy[0,args.clip_frames//2]=torch.tensor([.8,.25],device="cuda")
        mask=y>0;torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda",dtype=torch.float16):
            out=model(x);cls,loc=losses(out,y,xy,mask);loss=cls+loc
        loss.backward();torch.cuda.synchronize()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model._pred_loc.parameters())
        result.update(status="passed",logits_shape=list(out["im_feat"].shape),xy_shape=list(out["loc_feat"].shape),loss=float(loss.detach()))
    except torch.OutOfMemoryError as exc:result.update(status="out_of_memory",error=str(exc).splitlines()[0])
    finally:
        result.update(seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=="__main__":main()
