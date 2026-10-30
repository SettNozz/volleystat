"""Validate and extend a completed-epoch STES checkpoint without resetting Adam."""
import json,math
from pathlib import Path
import torch
from .spatial_labels import file_sha


def lr_factor(epoch,total_epochs):
    return min(1.,(epoch+1)/3)*.5*(1+math.cos(math.pi*epoch/total_epochs))


def validate_source(checkpoint,expected,total_epochs):
    checkpoint=Path(checkpoint).resolve()
    parent=json.loads(checkpoint.with_name("preflight.json").read_text())
    for key in ("schema","review_sha256","canonical_audit","source_revision","author_code_hashes","backbone_sha256","cache_preflight_sha256","holdout","train_sources"):
        if parent[key]!=expected[key]:raise ValueError("Continuation source mismatch: "+key)
    old=parent["recipe"];new=expected["recipe"]
    for key,value in old.items():
        if key not in ("epochs","initialization") and new.get(key)!=value:raise ValueError("Continuation recipe mismatch: "+key)
    for rel,sha in parent["implementation_hashes"].items():
        if rel.startswith("action_recognition/") and expected["implementation_hashes"].get(rel)!=sha:
            raise ValueError("Continuation architecture/labels changed: "+rel)
    for name in ("best.pt","best_metrics.json","best_predictions.npz"):
        if not checkpoint.with_name(name).is_file():raise ValueError("Missing source best artifact: "+name)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    required={"model","optimizer","scheduler","scaler","next_epoch","best","history","python_rng","numpy_rng","torch_rng","cuda_rng"}
    if not required<=set(ck):raise ValueError("Need a full last.pt checkpoint, not best.pt")
    start=ck["next_epoch"]
    if not 0<start<total_epochs:raise ValueError("Checkpoint must precede the requested final epoch")
    if [e["epoch"] for e in ck["history"]]!=list(range(1,start+1)):raise ValueError("Checkpoint history is inconsistent")
    if ck["scheduler"]["last_epoch"]!=start:raise ValueError("Checkpoint scheduler epoch differs")
    result=dict(path=str(checkpoint),sha256=file_sha(checkpoint),source_preflight_sha256=file_sha(checkpoint.with_name("preflight.json")),
        completed_epochs=start,start_epoch=start+1,final_epoch=total_epochs,new_epochs=total_epochs-start,
        preserved=["model","AdamW moments and step counters","AMP scaler","Python/NumPy/CPU/CUDA RNG","completed history","best checkpoint"],
        schedule_policy="Keep completed epochs; rebase the current and future cosine learning rates to the extended total epoch budget. Not identical to a fresh long-budget run.",
        best_artifact_hashes={name:file_sha(checkpoint.with_name(name)) for name in ("best.pt","best_metrics.json","best_predictions.npz")},
        previous_lr=[g["lr"] for g in ck["optimizer"]["param_groups"]],
        next_lr=[b*lr_factor(start,total_epochs) for b in ck["scheduler"]["base_lrs"]])
    del ck
    return result


def extend_scheduler(optimizer,scheduler,start,total_epochs):
    scheduler.last_epoch=start
    for group,base in zip(optimizer.param_groups,scheduler.base_lrs):group["lr"]=base*lr_factor(start,total_epochs)
    scheduler._last_lr=[g["lr"] for g in optimizer.param_groups]
