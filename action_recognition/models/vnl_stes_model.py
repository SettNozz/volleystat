"""Load the pinned authors' STES temporal implementation without their CLI/telemetry.

Author code and BSD notices remain unchanged in third_party/vnl_stes. Only
GSM zero padding gets device/dtype-safe allocation (same mathematical zeros).
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]


def build_author_stes(weights=None,clip_len=64,pretrained=True,predict_location=False,activation_checkpointing=False,backbone_chunk_frames=None):
    import timm,torchvision
    from third_party.vnl_stes.model import modules,shift,common
    if pretrained and (weights is None or not Path(weights).is_file()):
        raise ValueError('Explicit local ImageNet weights required; no implicit download')
    # The upstream implementation hardcodes CUDA float32 zeros; AMP and CPU
    # need zeros on the actual tensor device and dtype. No learned operation changes.
    def left(self,x):return torch.cat((x[:,:,1:],x.new_zeros(x.size(0),x.size(1),1,x.size(3),x.size(4))),dim=2)
    def right(self,x):return torch.cat((x.new_zeros(x.size(0),x.size(1),1,x.size(3),x.size(4)),x[:,:,:-1]),dim=2)
    shift._GSM.lshift_zeroPad=left;shift._GSM.rshift_zeroPad=right
    load_pretrained=bool(pretrained)
    def create(name,pretrained=False):
        if name!='regnety_008':raise ValueError('Unexpected author backbone')
        pretrained=bool(pretrained and load_pretrained)
        extra={'pretrained_cfg_overlay':{'file':str(Path(weights).resolve()),'hf_hub_id':None,'url':None}} if pretrained else {}
        return timm.create_model(name,pretrained=pretrained,**extra)
    path=ROOT/'third_party/vnl_stes/train_e2e_spatial.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    outer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='E2EModel')
    impl=next(n for n in outer.body if isinstance(n,ast.ClassDef) and n.name=='Impl')
    constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_GRU_HIDDEN_DIM' for t in n.targets)]
    namespace=dict(nn=nn,F=F,torch=torch,torchvision=torchvision,timm=SimpleNamespace(create_model=create),
        GRUPrediction=modules.GRUPrediction,FCPrediction=modules.FCPrediction,MLP=common.MLP,
        make_temporal_shift=shift.make_temporal_shift,__name__=__name__)
    exec(compile(ast.Module(body=constants+[impl],type_ignores=[]),str(path),'exec'),namespace)
    model=namespace['Impl'](num_classes=5,feature_arch='rny008_gsm',temporal_arch='deeper_gru',
        clip_len=clip_len,modality='rgb',predict_location=predict_location)
    if activation_checkpointing:
        from .activation_checkpointing import enable_backbone_checkpointing
        enable_backbone_checkpointing(model._features)
    if backbone_chunk_frames is not None:
        from .activation_checkpointing import enable_backbone_chunking
        enable_backbone_chunking(model._features,clip_len,backbone_chunk_frames,activation_checkpointing)
    return model
