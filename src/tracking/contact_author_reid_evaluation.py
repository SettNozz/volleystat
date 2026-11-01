"""Richer person attribution with held-out source labels kept out of fitting."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from action_recognition.annotator.contact_author_dataset import now,sha,write_new
from src.tracking.contact_author_evaluation import metrics,prediction,fit_rankers,ranker_scores,sequence_scores,PairRanker
from src.tracking.volleyball_contact_lab import kind,transition_key,learn_transitions

BASE_METHODS=("osnet_fixed","osnet_expanded","osnet_hsv","actor_trees","actor_hist","mil_near_linear","mil_near_mlp","mil_joint")
EXTRA_METHODS=("actor_ensemble","actor_ensemble_sequence","actor_trees_soft_sequence",
               "actor_trees_richside","actor_trees_richside_sequence","actor_trees_hybrid","actor_trees_hybrid_sequence",
               "actor_joint_mix","actor_joint_geomean","class_fusion","class_fusion_sequence")


def actors(case):
    result=defaultdict(list)
    for n,p in enumerate(case["pairs"]): result[p["actor_index"]].append((n,p))
    return result


def assignments(case,mode="osnet_hsv"):
    groups=actors(case); people=case["options"][1:]; selected=[]; costs=[]
    for index,pairs in groups.items():
        valid=[p for _,p in pairs if p["class_index"]>0]
        distance=[p["reid_expanded"] if mode=="osnet_expanded" else p["reid_distance"] for p in valid]
        if not any(p["track_near"] for p in valid) and min((d for d in distance if d is not None),default=1)>.25: continue
        row=[]
        for p,d in zip(valid,distance):
            value=1. if d is None else d
            if mode=="osnet_hsv" and p["hsv_distance"] is not None: value+=.35*p["hsv_distance"]
            row.append(value)
        selected.append(index); costs.append(row)
    if not selected: return {}
    matrix=np.concatenate([np.array(costs),np.ones((len(selected),len(selected)))],axis=1)
    rr,cc=linear_sum_assignment(matrix)
    return {selected[r]:people[c] if c<len(people) else "unknown" for r,c in zip(rr,cc)}


def conditional_from_actors(case,scores,identity):
    result=np.zeros(len(case["options"])); assigned=assignments(case,identity); groups=actors(case); selected={}
    for actor_index,pid in assigned.items():
        if pid=="unknown": continue
        c=case["options"].index(pid); n,p=next((n,p) for n,p in groups[actor_index] if p["player_id"]==pid)
        value=float(scores.get(actor_index,0.)); result[c]=max(result[c],value); selected[c]=p
    if result[1:].sum()>0: result[1:]/=result[1:].sum()
    else: result[1:]=1/max(1,len(result)-1)
    return result,selected


def predict_composed(cases,conditional,side,selected):
    result=[]
    for i,case in enumerate(cases):
        if side[i]<.5: result.append(prediction(case,"other_team",1-float(side[i]))); continue
        c=int(np.argmax(conditional[i,1:len(case["options"])]))+1
        actor=selected[i].get(c)
        result.append(prediction(case,case["options"][c] if actor is not None else "unknown",float(side[i]*conditional[i,c]),actor,"b"))
    return result


def scene_features(cases,x,names,gx):
    """Fixed per-side pooling, without learned/in-sample actor scores or IDs."""
    selected=[i for i,name in enumerate(names) if name in {
        "contact_geometry_0","contact_geometry_1","contact_geometry_3","contact_geometry_4","contact_geometry_13",
        "active_hand_distance","active_arm_raise","arm_9_speed","arm_10_speed","arm_9_acceleration","arm_10_acceleration",
        "jump_from_ground","body_image_height","foot_image_y","incoming_hand_distance","outgoing_hand_distance",
        "incoming_forearm_distance","outgoing_forearm_distance","ball_confidence","estimated_joint_fraction"}]
    rows=[]
    for i,case in enumerate(cases):
        groups=actors(case); extra=[]
        for team in (False,True):
            indices=[v[0][0] for v in groups.values() if bool(v[0][1]["track_near"])==team]
            extra.append(float(len(indices)))
            for column in selected:
                valid=[float(x[i,j,column*2]) for j in indices if x[i,j,column*2+1]>.5]
                for operation in (np.min,np.max,np.mean): extra.extend((float(operation(valid)) if valid else 0.,float(bool(valid))))
            identity=[p["reid_distance"] for p in case["pairs"] if p["class_index"]>0 and bool(p["track_near"])==team and p["reid_distance"] is not None]
            extra.extend((min(identity) if identity else 1.,float(bool(identity))))
        rows.append(extra)
    return np.concatenate([gx,np.asarray(rows,np.float32)],axis=1)


def fusion_features(case,index,conditionals,selections):
    heads=("actor_trees","actor_hist","mil_near_mlp","mil_joint","osnet_expanded")
    raw=set(case["contact"]["type"].split(" / "))
    types=[float(bool(raw & aliases)) for aliases in ({"serve"},{"reception"},{"pass","dig","set","free_ball"},{"attack"},{"block"})]
    tree_best=int(np.argmax(conditionals["actor_trees"][index,1:len(case["options"])]))+1
    primary=selections["actor_trees"][index].get(tree_best)
    rows=[]
    for c in range(1,len(case["options"])):
        features=[]
        for head in heads:
            values=conditionals[head][index,1:len(case["options"])];p=float(values[c-1]);lp=float(np.log(max(p,1e-6)))
            other=max((float(v) for j,v in enumerate(values,1) if j!=c),default=0.)
            features.extend([p,lp,lp-float(np.log(max(other,1e-6)))])
            features.extend(lp*t for t in types)
            actor=selections[head][index].get(c)
            features.extend([float(actor is not None),float(actor is not None and primary is not None and actor["actor_index"]==primary["actor_index"]),
                             float(actor["reid_distance"]) if actor is not None and actor.get("reid_distance") is not None else 1.])
        rows.append(features)
    return np.asarray(rows,np.float32)


def person_sequence(snapshot,cases,groups,train,y,conditional,side):
    """Generic same-person transition counts; no per-ID parameters or test targets."""
    lookup={c["contact"]["id"]:i for i,c in enumerate(cases)}; fps=next(iter(snapshot["videos"].values()))["fps"]
    train_sequences=[]; counts=defaultdict(Counter); total=Counter()
    for video in sorted(set(groups[train])):
        previous=None; sequence=[]
        for row in sorted((r for r in snapshot["queue"] if r["video_id"]==video),key=lambda r:r["frame"]):
            i=lookup.get(row["id"]); target=int(y[i]) if i is not None and train[i] else None
            item={"frame":row["frame"],"type":row["type"],"team_label":int(target>0) if target is not None else None}
            sequence.append(item)
            if previous is not None and target is not None and previous[1] is not None and target>0 and previous[1]>0 and kind(item)!="serve" and (item["frame"]-previous[0]["frame"])/fps<=4:
                same=target==previous[1]; counts[transition_key(previous[0],item,fps)][same]+=1; total[same]+=1
            previous=(item,target)
        train_sequences.append(sequence)
    team_prior=learn_transitions(train_sequences,fps); default_same=(total[True]+1)/(sum(total.values())+2)
    result=np.zeros_like(conditional)
    for video in sorted(set(groups)):
        queue=sorted((r for r in snapshot["queue"] if r["video_id"]==video),key=lambda r:r["frame"])
        size=len(snapshot["rosters"][video]["players"])+1; near=size-1
        initial=np.array([.5]+[.5/near]*near); emissions=[]; transitions=[]
        for n,row in enumerate(queue):
            i=lookup.get(row["id"])
            # Convert per-ID posterior under uniform identity prior to likelihood.
            emission=np.ones(size) if i is None else np.concatenate([[1-side[i]],near*side[i]*conditional[i,1:size]])
            emissions.append(np.clip(emission,1e-8,None))
            if n==0: continue
            previous=queue[n-1]
            if kind(row)=="serve" or not row["timing_verified"] or not previous["timing_verified"] or (row["frame"]-previous["frame"])/fps>4:
                transitions.append(np.tile(initial,(size,1))); continue
            key=transition_key(previous,row,fps); same_team=team_prior.get(key,{}).get("same",.5)
            values=counts.get(key,{})
            same_person=(values.get(True,0)+1)/(sum(values.values())+2) if sum(values.values())>=3 else .5 if "block" in (kind(previous),kind(row)) else default_same
            # Temper team and person probabilities separately. Tempering the
            # expanded matrix would bias a team merely because it has more IDs.
            same_team=same_team**.7/(same_team**.7+(1-same_team)**.7)
            diagonal=max(1e-8,same_person)**.7
            off_diagonal=max(1e-8,(1-same_person)/max(1,near-1))**.7
            denominator=diagonal+max(1,near-1)*off_diagonal
            diagonal/=denominator; off_diagonal/=denominator
            matrix=np.zeros((size,size)); matrix[0,0]=same_team; matrix[0,1:]=(1-same_team)/near
            for a in range(1,size):
                matrix[a,0]=1-same_team
                for b in range(1,size): matrix[a,b]=same_team*(diagonal if a==b else off_diagonal)
            transitions.append(matrix)
        emission=np.stack(emissions); forward=emission.copy(); forward[0]*=initial; forward[0]/=forward[0].sum()
        for n in range(1,len(queue)):
            forward[n]*=forward[n-1]@transitions[n-1]; forward[n]/=forward[n].sum()
        backward=np.ones_like(forward)
        for n in range(len(queue)-2,-1,-1):
            backward[n]=transitions[n]@(emission[n+1]*backward[n+1]); backward[n]/=backward[n].sum()
        posterior=forward*backward; posterior/=posterior.sum(axis=1,keepdims=True)
        for row,p in zip(queue,posterior):
            if row["id"] in lookup: result[lookup[row["id"]],:size]=p
    return result,{"same_person_counts":{k:{str(s):n for s,n in v.items()} for k,v in counts.items()},"training_sources":sorted(set(groups[train]))}


def evaluate(dataset,side_models,output,seeds=(42,43,44),reuse_trained=None):
    dataset,side_models,output=Path(dataset),Path(side_models),Path(output)
    if output.exists(): raise ValueError("Evaluation output must be fresh")
    manifest=json.loads((dataset/"manifest.json").read_bytes())
    if manifest["schema"]!="contact_author_reid_temporal_v1": raise ValueError("Unknown rich feature schema")
    for name,key in (("features.npz","features_sha256"),("cases.json","cases_sha256"),("human_labels.json","snapshot_sha256")):
        if sha(dataset/name)!=manifest[key]: raise ValueError("Rich inputs changed")
    data=np.load(dataset/"features.npz",allow_pickle=False); x,classes,y,groups,strict,gx=(data[k] for k in ("pairs","classes","targets","groups","strict","global_features"))
    cases=json.loads((dataset/"cases.json").read_bytes()); snapshot=json.loads((dataset/"human_labels.json").read_bytes())
    videos=sorted(set(groups)); names=manifest["fields"]; maximum=max(map(lambda c:len(c["options"]),cases))
    print(json.dumps({"manifest":str((dataset/"manifest.json").resolve()),"features":str((dataset/"features.npz").resolve()),
        "schema":manifest["schema"],"source_videos":videos,"label_provenance":"frozen human actor IDs; derived pose-target alignment explicitly weak",
        "counts":manifest["counts"],"split":"whole-source outer folds and inner method selection; no random clip split",
        "seeds":seeds,"training":"CPU near-person rankers/weak actor classifiers; only new features and fold artifacts"},indent=2),flush=True)
    output.mkdir(parents=True); (output/"models").mkdir()
    assignments_all=[assignments(c) for c in cases]
    physical_cols=[j for j in range(x.shape[-1]) if names[j//2] not in {"reference_option","reid_local","reid_track","reid_mean","reid_expanded","reid_regret","reid_margin","hsv_track","hsv_box","reference_track"}]
    scene=scene_features(cases,x,names,gx)
    cache={}; auxiliary={}; audit=[]
    def fit_sources(sources):
        sources=tuple(sorted(sources))
        if sources in cache: return cache[sources]
        train=np.isin(groups,sources)&strict; key="_".join(str(videos.index(v)) for v in sources)
        print(f"Fit rich sources {sources}: {int(train.sum())} contacts",flush=True)
        old=joblib.load(side_models/(key+"_extra_trees.joblib"))
        if old["manifest_sha256"]!=manifest["base_manifest_sha256"] or tuple(old["train_sources"])!=sources: raise ValueError("Incompatible side fold")
        base_side=old["classifier"].predict_proba(gx[:,old["columns"]])[:,1]
        side,priors=sequence_scores(snapshot,cases,groups,train,y,base_side)
        conditionals={}; selections={}
        for mode in BASE_METHODS[:3]:
            cond=np.zeros((len(cases),maximum)); selected=[]
            for i,c in enumerate(cases):
                scores={a:float(np.exp(-max(0,p[0][1]["score"])/.12)) for a,p in actors(c).items()}
                row,chosen=conditional_from_actors(c,scores,mode); cond[i,:len(row)]=row; selected.append(chosen)
            conditionals[mode]=cond; selections[mode]=selected
        # Weak actor boxes are derived from human person ID + frozen appearance,
        # never exported as human box/keypoint ground truth.
        train_x=[]; train_y=[]; weights=[]
        for i in np.flatnonzero(train):
            c=cases[i]; assigned=assignments_all[i]; target=c["options"][y[i]]; grouped=actors(c)
            positive=[a for a,pid in assigned.items() if pid==target] if y[i]>0 else []
            if y[i]>0 and len(positive)!=1: continue
            for a,pid in assigned.items():
                if pid=="unknown": continue
                n,_=grouped[a][0]; train_x.append(x[i,n,physical_cols]); train_y.append(int(a in positive)); weights.append(1/max(1,len(assigned)))
        tx=np.stack(train_x); ty=np.array(train_y)
        for family in ("trees","hist"):
            filename=key+"_actor_"+family+".joblib"
            if reuse_trained is not None:
                artifact=joblib.load(Path(reuse_trained)/filename)
                if artifact["manifest_sha256"]!=sha(dataset/"manifest.json") or tuple(artifact["training_sources"])!=sources or artifact["columns"]!=physical_cols:
                    raise ValueError("Incompatible saved physical actor model")
                classifier=artifact["classifier"]
            else:
                classifier=ExtraTreesClassifier(n_estimators=350,min_samples_leaf=3,max_features=.8,class_weight="balanced",random_state=42,n_jobs=2) if family=="trees" else HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=9,min_samples_leaf=12,l2_regularization=2.,class_weight="balanced",early_stopping=False,random_state=42)
                classifier.fit(tx,ty,sample_weight=np.array(weights))
            cond=np.zeros((len(cases),maximum)); selected=[]
            for i,c in enumerate(cases):
                grouped=actors(c); indices=[v[0][0] for v in grouped.values()]
                probs=classifier.predict_proba(x[i,indices][:,physical_cols])[:,1] if indices else []
                scores={a:float(np.exp(np.clip(np.log(p+1e-6)-np.log(1-p+1e-6),-10,10))) for a,p in zip(grouped,probs)}
                row,chosen=conditional_from_actors(c,scores,"osnet_hsv"); cond[i,:len(row)]=row; selected.append(chosen)
            conditionals["actor_"+family]=cond; selections["actor_"+family]=selected
            joblib.dump({"classifier":classifier,"columns":physical_cols,"training_sources":sources,"manifest_sha256":sha(dataset/"manifest.json"),
                         "target_provenance":"weak pose alignment from human author ID and frozen ReID"},output/"models"/filename)
        for hidden,name,near_only in ((0,"mil_near_linear",True),(24,"mil_near_mlp",True),(24,"mil_joint",False)):
            mask=classes.copy()
            if near_only: mask[mask==0]=-1
            filename=key+"_"+name+".pt"
            if reuse_trained is not None:
                artifact=torch.load(Path(reuse_trained)/filename,map_location="cpu",weights_only=False)
                if artifact["manifest_sha256"]!=sha(dataset/"manifest.json") or tuple(artifact["training_sources"])!=sources or tuple(m["seed"] for m in artifact["members"])!=tuple(seeds):
                    raise ValueError("Incompatible saved near-person ranker")
                fit={k:v for k,v in artifact.items() if k!="members"}; fit["members"]=[]
                for m in artifact["members"]:
                    network=PairRanker(x.shape[-1],hidden); network.load_state_dict(m["state_dict"]); network.eval()
                    fit["members"].append({"seed":m["seed"],"network":network})
            else:
                fit=fit_rankers(x,mask,y,train&(y>0) if near_only else train,hidden,seeds=seeds)
            probabilities,instance=ranker_scores(fit,x,mask,maximum)
            conditional=probabilities.copy(); conditional[:,0]=0; conditional/=np.maximum(conditional.sum(axis=1,keepdims=True),1e-8)
            selected=[]
            for i,c in enumerate(cases):
                selected.append({j:max([(n,p) for n,p in enumerate(c["pairs"]) if p["class_index"]==j],key=lambda q:instance[i,q[0]])[1]
                                 for j in range(1,len(c["options"])) if any(p["class_index"]==j for p in c["pairs"])})
            conditionals[name]=conditional; selections[name]=selected
            artifact={k:v for k,v in fit.items() if k!="members"}; artifact.update(members=[{"seed":m["seed"],"state_dict":m["network"].state_dict()} for m in fit["members"]],
                training_sources=sources,manifest_sha256=sha(dataset/"manifest.json"),schema=manifest["schema"],near_only=near_only)
            torch.save(artifact,output/"models"/filename)
        result={}
        for name,conditional in conditionals.items():
            result[name]=predict_composed(cases,conditional,side,selections[name])
            full,person_prior=person_sequence(snapshot,cases,groups,train,y,conditional,base_side)
            pside=1-full[:,0]; pcond=full.copy(); pcond[:,0]=0; pcond/=np.maximum(pcond.sum(axis=1,keepdims=True),1e-8)
            result[name+"_sequence"]=predict_composed(cases,pcond,pside,selections[name])
        ensemble=(conditionals["actor_trees"]+conditionals["actor_hist"])/2
        result["actor_ensemble"]=predict_composed(cases,ensemble,side,selections["actor_trees"])
        full,_=person_sequence(snapshot,cases,groups,train,y,ensemble,base_side)
        pside=1-full[:,0];pc=full.copy();pc[:,0]=0;pc/=np.maximum(pc.sum(axis=1,keepdims=True),1e-8)
        result["actor_ensemble_sequence"]=predict_composed(cases,pc,pside,selections["actor_trees"])
        softer=np.sqrt(conditionals["actor_trees"]);softer/=np.maximum(softer.sum(axis=1,keepdims=True),1e-8)
        full,_=person_sequence(snapshot,cases,groups,train,y,softer,base_side)
        pside=1-full[:,0];pc=full.copy();pc[:,0]=0;pc/=np.maximum(pc.sum(axis=1,keepdims=True),1e-8)
        result["actor_trees_soft_sequence"]=predict_composed(cases,pc,pside,selections["actor_trees"])
        saved_side=Path(reuse_trained)/(key+"_rich_side.joblib") if reuse_trained else None
        if saved_side is not None and saved_side.exists():
            saved=joblib.load(saved_side)
            if saved["manifest_sha256"]!=sha(dataset/"manifest.json") or tuple(saved["training_sources"])!=sources: raise ValueError("Rich side model changed sources")
            side_classifier=saved["classifier"]
        else:
            side_classifier=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=3,max_features=.5,class_weight="balanced",random_state=42,n_jobs=2)
            side_classifier.fit(scene[train],(y[train]>0).astype(int))
        rich_side=side_classifier.predict_proba(scene)[:,1]
        joblib.dump({"classifier":side_classifier,"training_sources":sources,"manifest_sha256":sha(dataset/"manifest.json"),"pooling":"fixed_side_pose_temporal_v1"},output/"models"/(key+"_rich_side.joblib"))
        for label,probability in (("richside",rich_side),("hybrid",(rich_side+base_side)/2)):
            seq,_=sequence_scores(snapshot,cases,groups,train,y,probability)
            result["actor_trees_"+label]=predict_composed(cases,conditionals["actor_trees"],seq,selections["actor_trees"])
            full,_=person_sequence(snapshot,cases,groups,train,y,conditionals["actor_trees"],probability)
            pside=1-full[:,0];pc=full.copy();pc[:,0]=0;pc/=np.maximum(pc.sum(axis=1,keepdims=True),1e-8)
            result["actor_trees_"+label+"_sequence"]=predict_composed(cases,pc,pside,selections["actor_trees"])
        auxiliary[sources]={"conditionals":conditionals,"selections":selections}
        for label,conditional in (("actor_joint_mix",.7*conditionals["actor_trees"]+.3*conditionals["mil_joint"]),
                                  ("actor_joint_geomean",np.maximum(conditionals["actor_trees"],1e-8)**.7*np.maximum(conditionals["mil_joint"],1e-8)**.3)):
            conditional=conditional.copy();conditional[:,0]=0;conditional/=np.maximum(conditional.sum(axis=1,keepdims=True),1e-8)
            selected=[{c:(selections["actor_trees"][i].get(c) if conditionals["actor_trees"][i,c]>=conditionals["mil_joint"][i,c]
                          else selections["mil_joint"][i].get(c)) for c in range(1,len(case["options"]))} for i,case in enumerate(cases)]
            seq,_=sequence_scores(snapshot,cases,groups,train,y,rich_side)
            result[label]=predict_composed(cases,conditional,seq,selected)
        fusion=.7*conditionals["actor_trees"]+.3*conditionals["mil_joint"]
        if len(sources)>=2:
            fusion_x=[];fusion_y=[];sample_weights=[]
            for validation in sources:
                child=tuple(v for v in sources if v!=validation); fit_sources(child)
                bundle=auxiliary[child]
                for i in np.flatnonzero(train&(groups==validation)&(y>0)):
                    values=fusion_features(cases[i],i,bundle["conditionals"],bundle["selections"])
                    fusion_x.extend(values);fusion_y.extend(int(j==y[i]) for j in range(1,len(cases[i]["options"])));sample_weights.extend([1/len(values)]*len(values))
            classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight="balanced",max_iter=1500,random_state=42))
            classifier.fit(np.stack(fusion_x),np.asarray(fusion_y),logisticregression__sample_weight=np.asarray(sample_weights))
            fusion=np.zeros_like(fusion)
            for i,case in enumerate(cases):
                values=fusion_features(case,i,conditionals,selections);scores=classifier.decision_function(values)
                probability=np.exp(scores-scores.max());probability/=probability.sum();fusion[i,1:len(case["options"])]=probability
            joblib.dump({"pipeline":classifier,"training_sources":sources,"training_predictions":"whole-source cross-fitted children",
                         "manifest_sha256":sha(dataset/"manifest.json")},output/"models"/(key+"_fusion.joblib"))
        fusion[:,0]=0;fusion/=np.maximum(fusion.sum(axis=1,keepdims=True),1e-8)
        selected=[{c:(selections["actor_trees"][i].get(c) if conditionals["actor_trees"][i,c]>=conditionals["mil_joint"][i,c]
                      else selections["mil_joint"][i].get(c)) for c in range(1,len(case["options"]))} for i,case in enumerate(cases)]
        seq,_=sequence_scores(snapshot,cases,groups,train,y,rich_side)
        result["class_fusion"]=predict_composed(cases,fusion,seq,selected)
        full,_=person_sequence(snapshot,cases,groups,train,y,fusion,rich_side)
        pside=1-full[:,0];pc=full.copy();pc[:,0]=0;pc/=np.maximum(pc.sum(axis=1,keepdims=True),1e-8)
        result["class_fusion_sequence"]=predict_composed(cases,pc,pside,selected)
        audit.append({"training_sources":sources,"strict_contacts":int(train.sum()),"weak_pose_rows":len(ty),"weak_positive_rows":int(ty.sum()),"person_prior":person_prior})
        cache[sources]=result; return result
    methods=BASE_METHODS+tuple(m+"_sequence" for m in BASE_METHODS)+EXTRA_METHODS
    oof={m:[None]*len(cases) for m in methods}; nested=[None]*len(cases); choices=[]
    for held_out in videos:
        training=[v for v in videos if v!=held_out]; inner={m:[None]*len(cases) for m in methods}
        for validation in training:
            fit=fit_sources([v for v in training if v!=validation])
            for i in np.flatnonzero(groups==validation):
                for name in methods: inner[name][i]=fit[name][i]
        mask=strict&np.isin(groups,training); scores={m:metrics(cases,p,mask) for m,p in inner.items()}
        chosen=max(methods,key=lambda m:scores[m]["near_author_f1"])
        fitted=fit_sources(training)
        for i in np.flatnonzero(groups==held_out):
            for name in methods: oof[name][i]=fitted[name][i]
            nested[i]=fitted[chosen][i]
        choices.append({"held_out":held_out,"selected":chosen,"inner_metrics":scores,"test_targets_used":False})
        print(f"Held out {held_out}: {chosen}",flush=True)
    oof["nested_selected"]=nested
    results={m:{"strict":metrics(cases,p,strict),"all_eligible":metrics(cases,p,np.ones(len(y),bool)),
                "per_video":{v:metrics(cases,p,strict&(groups==v)) for v in videos}} for m,p in oof.items()}
    best=max(methods,key=lambda m:results[m]["strict"]["near_author_f1"])
    records=[{"id":c["contact"]["id"],"video_id":c["contact"]["video_id"],"frame":c["contact"]["frame"],"type":c["contact"]["type"],
              "truth":c["options"][c["target"]],"strict":c["strict"],"predictions":{m:p[i] for m,p in oof.items()}} for i,c in enumerate(cases)]
    write_new(output/"predictions.json",records)
    report={"schema":"human_contact_author_evaluation_v2","created_at":now(),"dataset":str(dataset.resolve()),"dataset_manifest_sha256":sha(dataset/"manifest.json"),
            "predictions_sha256":sha(output/"predictions.json"),"best_fixed_comparison":best,"results":results,"nested_selection":choices,"fit_audit":audit,
            "production_promoted":False,"selection":"near-person F1; fixed-method result remains development selection","goal":{"near_recall":.8,"near_precision":.8},
            "reused_trained_models":str(reuse_trained) if reuse_trained else None,"person_decoder":"cardinality_neutral_v2"}
    write_new(output/"comparison.json",report)
    print(json.dumps({"best":best,"strict":results[best]["strict"],"nested":results["nested_selected"]["strict"]},indent=2),flush=True)
    return report
