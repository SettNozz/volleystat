"""Non-reentrant block checkpoints without double-updating BatchNorm statistics."""
from contextlib import contextmanager,nullcontext
from types import MethodType
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def isolated_batchnorm_buffers(block):
    saved=[]
    for module in block.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):
            for name in ("running_mean","running_var","num_batches_tracked"):
                buffer=getattr(module,name,None)
                if buffer is not None:
                    saved.append((module,name,buffer))
                    setattr(module,name,buffer.clone())
    try:yield
    finally:
        for module,name,buffer in saved:setattr(module,name,buffer)


def enable_backbone_checkpointing(features):
    blocks=[]
    for name,stage in features.named_children():
        if name.startswith("s") and name[1:].isdigit():blocks.extend(stage.children())
        elif name in ("stem","final_conv"):blocks.append(stage)
    if not blocks:raise ValueError("Unsupported RegNet stage layout")
    for block in blocks:
        if getattr(block,"_stes_checkpointed",False):continue
        original=block.forward
        def forward(self,x,_original=original):
            if not self.training or not torch.is_grad_enabled():return _original(x)
            return checkpoint(_original,x,use_reentrant=False,context_fn=lambda:(nullcontext(),isolated_batchnorm_buffers(self)))
        block.forward=MethodType(forward,block);block._stes_checkpointed=True
    return len(blocks)


def enable_backbone_chunking(features,clip_frames,chunk_frames,checkpoint_chunks):
    """GSM uses local chunks; the original BiGRU still receives the full clip."""
    from third_party.vnl_stes.model.shift import _GSM,GatedShift
    if chunk_frames<1 or clip_frames%chunk_frames:
        raise ValueError("Backbone chunk must divide the full temporal clip")
    for module in features.modules():
        if isinstance(module,_GSM):module.num_segments=chunk_frames
        if isinstance(module,GatedShift):module.n_segment=chunk_frames
    original=features.forward
    def forward(self,x):
        if x.shape[0]%clip_frames:raise ValueError("Flattened frames must contain complete temporal clips")
        outputs=[]
        for part in x.split(chunk_frames,dim=0):
            if checkpoint_chunks and self.training and torch.is_grad_enabled():
                value=checkpoint(original,part,use_reentrant=False,context_fn=lambda:(nullcontext(),isolated_batchnorm_buffers(self)))
            else:value=original(part)
            outputs.append(value)
        return torch.cat(outputs,dim=0)
    features.forward=MethodType(forward,features)
