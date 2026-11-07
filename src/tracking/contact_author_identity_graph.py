"""Offline appearance/motion identity propagation from confirmed photo setup.

No contact-author decision is an input. Short temporal chunks avoid assuming
that one tracker ID remains the same person for an entire match.
"""
from collections import defaultdict,Counter
import numpy as np
from scipy.special import softmax
from src.tracking.contact_author_benchmark import track_segments,unit
from src.tracking.player_contacts import side


class IdentityGraph:
    def __init__(self,annotations,plan,vectors,chunk_frames=45):
        _,tracks=track_segments(annotations["frames"],annotations["fps"])
        people=list(plan["galleries"]);self.people=people;self.chunk_frames=chunk_frames
        near={s for s,obs in tracks.items() if Counter(side(o["pose"].get("team")) for o in obs).most_common(1)[0][0]=="b"}
        buckets=defaultdict(list)
        for obs in plan["observations"]:
            if obs["segment"] in near:buckets[(obs["segment"],obs["f"]//chunk_frames)].append(obs)
        self.keys=list(buckets);self.lookup={k:i for i,k in enumerate(self.keys)};node_vectors=[];bounds=[];centres=[]
        for segment,chunk in self.keys:
            observations=buckets[segment,chunk];clean=[o for o in observations if o["occlusion"]<.3]or observations
            node_vectors.append(unit(np.mean([vectors[o["key"]] for o in clean],axis=0)))
            full=[o for o in tracks[segment] if o["frame"]//chunk_frames==chunk]
            bounds.append([full[0]["frame"],full[-1]["frame"]])
            centres.append([np.array([(full[j]["pose"]["bb"][0]+full[j]["pose"]["bb"][2])/2,(full[j]["pose"]["bb"][1]+full[j]["pose"]["bb"][3])/2]) for j in (0,-1)])
        matrix=np.stack(node_vectors);bounds=np.asarray(bounds);self.bounds=bounds
        overlap=(bounds[:,None,0]<=bounds[None,:,1])&(bounds[:,None,1]>=bounds[None,:,0]);np.fill_diagonal(overlap,False)
        similarities=matrix@matrix.T;np.fill_diagonal(similarities,-1)
        similarities[overlap]=-1;weight=np.zeros_like(similarities)
        for i,row in enumerate(similarities):
            for j in np.argsort(row)[-8:]:
                if row[j]>.7:weight[i,j]=np.exp((row[j]-1)/.08)
        weight=np.minimum(weight,weight.T)
        # Adjacent chunks/track fragments get motion support only when observed
        # endpoints and appearance agree; simultaneous tracks cannot link.
        for i in range(len(self.keys)):
            candidates=np.flatnonzero((bounds[:,0]>bounds[i,1])&(bounds[:,0]-bounds[i,1]<=15))
            for j in candidates:
                same=self.keys[i][0]==self.keys[j][0]
                height=max(1,tracks[self.keys[i][0]][-1]["pose"]["bb"][3]-tracks[self.keys[i][0]][-1]["pose"]["bb"][1])
                shift=float(np.linalg.norm(centres[j][0]-centres[i][1]))/height
                if similarities[i,j]>.65 and shift<.4:
                    value=(2. if same else 1.)*np.exp(-shift*3)
                    weight[i,j]=weight[j,i]=max(weight[i,j],value)
        weight/=np.maximum(weight.sum(axis=1,keepdims=True),1e-8)
        cannot=overlap.astype(np.float32);cannot/=np.maximum(cannot.sum(axis=1,keepdims=True),1)
        seeds={};refs={pid:np.stack([vectors[k] for k in keys]) for pid,keys in plan["galleries"].items()}
        for pid,keys in plan["galleries"].items():
            for key in keys:
                obs=next((o for o in plan["observations"] if o["key"]==key),None)
                if obs is not None and obs["segment"] is not None:
                    node=self.lookup.get((obs["segment"],obs["f"]//chunk_frames))
                    if node is not None:seeds[node]=people.index(pid)
        unary=np.column_stack([np.max(matrix@g.T,axis=1) for g in refs.values()])
        unary=(unary-unary.max(axis=1,keepdims=True))/.05
        self.unary=unary;self.weight=weight;self.cannot=cannot;self.seeds=seeds

    def solve(self,attraction=2.,repulsion=2.,iterations=40):
        q=softmax(self.unary,axis=1)
        for _ in range(iterations):
            logits=self.unary+attraction*(self.weight@q)-repulsion*(self.cannot@q)
            updated=softmax(logits,axis=1)
            for i,label in self.seeds.items():updated[i]=0.;updated[i,label]=1.
            q=.5*q+.5*updated
        return q

    def distribution(self,q,segment,frame):
        index=self.lookup.get((segment,frame//self.chunk_frames))
        if index is None:
            candidates=[i for i,key in enumerate(self.keys) if key[0]==segment]
            if not candidates:return None
            index=min(candidates,key=lambda i:min(abs(frame-f) for f in self.bounds[i]))
        return q[index]
