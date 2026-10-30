"""STES-inspired temporal RGB baseline, not an exact VNL-STES reproduction.

Frozen ImageNet MobileNetV3-Small replaces RegNet-Y/GSM. Three-layer BiGRU
classifies continuous frames including background. No spatial regression head.
"""
import numpy as np
import torch
from torch import nn
from action_recognition.rally_spotting.metrics import nms_peaks,match_events,_prf

EVENTS=('serve','reception','pass','attack')
SCHEMA='rgb_mobilenet_bigru_spotting_v1'


class RGBEventSpotter(nn.Module):
    def __init__(self,input_dim=576,hidden=128,layers=3):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.ReLU())
        self.temporal=nn.GRU(hidden,hidden,num_layers=layers,batch_first=True,
                             bidirectional=True,dropout=.2 if layers>1 else 0)
        self.head=nn.Linear(hidden*2,5)
    def forward(self,x):
        h,_=self.temporal(self.project(x));return self.head(h)


def frame_targets(length,events,ignored=(),ignore_radius=6):
    """0=background; 1..4=core contacts. Preserve ambiguous/unsupported labels."""
    target=np.zeros(length,np.int64);seen={};conflicts=set()
    for event in events:
        f=int(event['frame']);k=EVENTS.index(event['type'])+1
        if not 0<=f<length:raise ValueError('Human event outside decoded video')
        if f in seen and seen[f]!=k:conflicts.add(f)
        seen[f]=k;target[f]=k
    for f in ignored:
        left=max(0,int(f)-ignore_radius);right=min(length,int(f)+ignore_radius+1)
        region=target[left:right]
        region[region==0]=-100  # Never erase a nearby authoritative core contact.
    for f in conflicts:target[f]=-100
    return target


def decode(probabilities,threshold=.5,radius=5):
    # Only local maxima can be contacts; a non-event frame can remain background.
    p=np.asarray(probabilities)
    if p.ndim!=2 or p.shape[1]!=5:raise ValueError('Expected background + four classes')
    result=[]
    for k,name in enumerate(EVENTS,1):
        scores=np.where(p[:,k]>p[:,0],p[:,k],0)
        for f in nms_peaks(scores,threshold,radius):
            result.append(dict(frame=f,type=name,confidence=float(p[f,k])))
    return sorted(result,key=lambda e:e['frame'])


def evaluate(probabilities,events,threshold,tolerances=(1,2,4,5,10,15)):
    predicted=decode(probabilities,threshold);result={}
    for tolerance in tolerances:
        classes={};totals=np.zeros(3,int)
        for name in EVENTS:
            counts=match_events([e['frame'] for e in predicted if e['type']==name],
                                [e['frame'] for e in events if e['type']==name],tolerance)
            classes[name]=_prf(*counts);totals+=counts
        result[f'at_{tolerance}']=dict(micro=_prf(*map(int,totals)),per_class=classes,
            macro_f1=float(np.mean([v['f1'] for v in classes.values()])))
    return result,predicted


@torch.no_grad()
def predict(model,features,device,window=120,stride=60):
    model.eval();length=len(features);sums=np.zeros((length,5),np.float32);counts=np.zeros((length,1))
    starts=list(range(0,max(1,length-window+1),stride))
    last=max(0,length-window)
    if starts[-1]!=last:starts.append(last)
    for start in starts:
        end=min(start+window,length);x=np.zeros((window,features.shape[1]),np.float32)
        x[:end-start]=features[start:end]
        p=model(torch.from_numpy(x[None]).to(device)).softmax(-1)[0,:end-start].cpu().numpy()
        sums[start:end]+=p;counts[start:end]+=1
    return sums/np.maximum(counts,1)
