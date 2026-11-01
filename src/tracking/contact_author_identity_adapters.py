"""Identity enrollment uses reference setup only, never contact-author targets."""
from collections import defaultdict
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


class PhotoEnrollment:
    def __init__(self,plan,vectors):
        self.vectors=vectors; self.by_segment=defaultdict(list)
        self.people=list(plan["galleries"]); self.plan=plan
        references={pid:[o["f"] for o in plan["observations"] if o["key"].startswith("reference:"+pid+":")] for pid in self.people}
        tx=[];ty=[]
        for o in plan["observations"]:
            if o["segment"] is not None:self.by_segment[o["segment"]].append(o)
            ids=plan["anchor_segments"].get(o["segment"],[])
            pid=ids[0] if len(ids)==1 else o["key"].split(":")[1] if o["key"].startswith("reference:") else None
            if pid and o["occlusion"]<.3 and min(abs(o["f"]-f) for f in references[pid])<=15:
                tx.append(vectors[o["key"]]);ty.append(pid)
        self.models={
            "lda_auto":LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto",priors=np.ones(len(self.people))/len(self.people)),
            "lda_05":LinearDiscriminantAnalysis(solver="lsqr",shrinkage=.5,priors=np.ones(len(self.people))/len(self.people)),
            "logistic_10":LogisticRegression(C=10,class_weight="balanced",max_iter=1000),
            "logistic_100":LogisticRegression(C=100,class_weight="balanced",max_iter=1000)}
        for model in self.models.values():model.fit(np.stack(tx),ty)
        self.enrollment_count=len(tx)
        mean=np.stack(list(vectors.values())).mean(axis=0)
        centred={k:(v-mean)/max(1e-8,np.linalg.norm(v-mean)) for k,v in vectors.items()}
        residual=[]
        for observations in self.by_segment.values():
            values=np.stack([vectors[o["key"]] for o in observations]);residual.extend(values-values.mean(axis=0))
        residual=np.stack(residual);cov=residual.T@residual/max(1,len(residual));scale=np.trace(cov)/512
        cov=.8*cov+.2*scale*np.eye(512);eig,U=np.linalg.eigh(cov);W=(U/np.sqrt(np.maximum(eig,1e-7)))@U.T
        whitened={k:((v-mean)@W)/max(1e-8,np.linalg.norm((v-mean)@W)) for k,v in vectors.items()}
        self.transforms={"centred":centred,"whitened":whitened}

    def distances(self,segment,frame):
        observations=self.by_segment[segment]
        nearest=min(observations,key=lambda o:abs(o["f"]-frame))
        clean=sorted([o for o in observations if o["occlusion"]<.3 and abs(o["f"]-frame)<=45],key=lambda o:abs(o["f"]-frame))[:8]or[nearest]
        q=np.stack([self.vectors[o["key"]] for o in clean]);result={}
        for name,model in self.models.items():
            probs=model.predict_proba(q).mean(axis=0)
            result[name]=[-float(np.log(max(1e-9,probs[list(model.classes_).index(pid)]))) for pid in self.people]
        for name,vectors in self.transforms.items():
            queries=np.stack([vectors[o["key"]] for o in clean]);row=[]
            for pid in self.people:
                gallery=np.stack([vectors[k] for k in self.plan["galleries"][pid]])
                scores=np.sort(np.max(queries@gallery.T,axis=1));row.append(1-float(scores[-min(3,len(scores)):].mean()))
            result[name]=row
        return result
