"""Person-level, source-held-out comparisons; no prediction becomes human gold."""
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from action_recognition.annotator.contact_author_dataset import now, sha, write_new
from scripts.evaluation.run_contact_author_learning import feature_indices, model
from src.tracking.contact_author_benchmark import PAIR_FIELDS, SCHEMA
from src.tracking.volleyball_contact_lab import learn_transitions, sequence_posteriors

METHODS = ("other_team_control","nearest_raw_anchor","nearest_box_hsv","nearest_pose_hsv",
           "nearest_track_hsv","nearest_anchor_hsv","action_anchor_hsv","logistic_side_action",
           "trees_side_action","joint_linear","joint_mlp","logistic_side_joint",
           "nearest_hungarian","logistic_side_hungarian","trees_side_nearest","trees_side_hungarian",
           "trees_side_action_hungarian","logistic_sequence_hungarian","trees_sequence_hungarian")


def prediction(case,label,confidence=.5,actor=None,team=None):
    return {"label":label,"team":team or ("a" if label=="other_team" else "b" if label!="unknown" else None),
            "score":float(confidence),"score_calibrated":False,"actor":actor,
            "label_source":"derived","verified":False}


def physical_score(pair,action=False):
    cost = pair["score"]
    if action:
        cost = pair.get("action_distance") if pair.get("action_distance") is not None else cost+.3
        cost += 0 if pair["action_supported"] else .8
        cost += .2*abs(pair.get("offset_seconds",0.))
    return cost


def identity_distance(pair,kind):
    keys = {"box":("colour_box",),"pose":("colour_pose","colour_box"),
            "track":("track_colour_pose","track_colour_box","colour_pose","colour_box")}[kind]
    for key in keys:
        if pair.get(key) is not None: return pair[key]
    return 2.


def roster_assignment(case,kind="track",anchor=True):
    """One-to-one local track/reference hypotheses, with explicit unknown slots.

    Nearby track fragments may represent one person; this is an experiment,
    not human identity truth. Assignments never update reference galleries.
    """
    pairs = [p for p in case["pairs"] if p["class_index"]>0]
    segments = sorted({p["segment"] for p in pairs}); people = case["options"][1:]
    if not segments: return {}
    costs = np.full((len(segments),len(people)+len(segments)),.75)
    costs[:,:len(people)] = 2.
    for p in pairs:
        row = segments.index(p["segment"]); column = people.index(p["player_id"])
        value = -.5 if anchor and p["anchor_track"] else identity_distance(p,kind)
        costs[row,column] = min(costs[row,column],value)
    rows,columns = linear_sum_assignment(costs)
    return {segments[i]:people[j] if j<len(people) else "unknown" for i,j in zip(rows,columns)}


def fixed_predict(case,kind="track",anchor=True,action=False,force_near=False,raw=False,one_to_one=False):
    pairs = [p for p in case["pairs"] if not force_near or p["class_index"]>0]
    if not pairs: return prediction(case,"unknown",0.,team="b" if force_near else None)
    # First identify observed action, then compare immutable person references.
    selected = min(pairs,key=lambda p:physical_score(p,action))
    if selected["class_index"]==0: return prediction(case,"other_team",.5,selected)
    siblings = [p for p in pairs if (p["frame"],p["pose_index"])==(selected["frame"],selected["pose_index"])]
    anchors = [p for p in siblings if p["anchor_track"]]
    if raw and not anchors: return prediction(case,"unknown",0.,selected,"b")
    ranked = sorted(siblings,key=lambda p:identity_distance(p,kind))
    chosen = anchors[0] if anchor and anchors else ranked[0]
    if not anchors and identity_distance(chosen,kind)>=2: return prediction(case,"unknown",0.,selected,"b")
    margin = identity_distance(ranked[1],kind)-identity_distance(ranked[0],kind) if len(ranked)>1 else 0.
    confidence = .95 if anchor and anchors else min(.95,.5+2*max(0,margin))
    if one_to_one:
        pid = roster_assignment(case,kind,anchor).get(chosen["segment"],"unknown")
        if pid=="unknown": return prediction(case,pid,0.,selected,"b")
        chosen = next(p for p in siblings if p["player_id"]==pid)
    return prediction(case,chosen["player_id"],confidence,chosen)


def metrics(cases,predictions,mask):
    indices = np.flatnonzero(mask)
    if not len(indices): return {"count":0}
    truth = [cases[i]["options"][cases[i]["target"]] for i in indices]
    labels = [predictions[i]["label"] for i in indices]
    correct = np.array([a==b for a,b in zip(truth,labels)])
    near = np.array([a!="other_team" for a in truth]); other = ~near
    teams = np.array([predictions[i]["team"]==("b" if n else "a") for i,n in zip(indices,near)])
    offered = np.array([label not in ("other_team","unknown") for label in labels])
    coverage = np.array([label!="unknown" for label in labels])
    selective = np.array([predictions[i]["score"]>=.7 and predictions[i]["label"]!="unknown" for i in indices])
    scoped = lambda label,i: label if label in ("other_team","unknown") else cases[i]["contact"]["video_id"]+":"+label
    yt = [scoped(label,i) for label,i in zip(truth,indices)]
    yp = [scoped(label,i) for label,i in zip(labels,indices)]
    average = lambda values,where: float(values[where].mean()) if where.any() else None
    near_given_team = near & np.array([predictions[i]["team"]=="b" for i in indices])
    true_near = int((correct & near).sum()); proposed_near = int(offered.sum())
    selected_near = selective & offered
    near_errors = {"wrong_team":int(sum(n and not ok and predictions[i]["team"]=="a" for i,n,ok in zip(indices,near,correct))),
                   "unknown":int(sum(n and not ok and predictions[i]["label"]=="unknown" for i,n,ok in zip(indices,near,correct)))}
    near_errors["wrong_person"] = int((near & ~correct).sum())-sum(near_errors.values())
    return {"count":len(indices),"near_count":int(near.sum()),"other_count":int(other.sum()),
        "correct":int(correct.sum()),"author_accuracy":float(correct.mean()),
        "balanced_author_accuracy":float((correct[near].mean()+correct[other].mean())/2) if near.any() and other.any() else None,
        "near_author_recall":average(correct,near),"other_recall":average(correct,other),
        "team_accuracy":float(teams.mean()),"near_team_recall":average(teams,near),
        "near_identity_accuracy_given_near_team":average(correct,near_given_team),
        "near_assignment_precision":average(correct,offered),"near_assignments":int(offered.sum()),
        "near_author_f1":2*true_near/max(1,int(near.sum())+proposed_near),
        "macro_f1_people_and_other":float(f1_score(yt,yp,labels=sorted(set(yt)),average="macro",zero_division=0)),
        "coverage":float(coverage.mean()),"selective_coverage_score_07":float(selective.mean()),
        "selective_accuracy_score_07":average(correct,selective),
        "selective_near_precision_score_07":average(correct,selected_near),
        "selective_near_recall_score_07":int((correct & near & selective).sum())/max(1,int(near.sum())),
        "near_errors":near_errors}


class PairRanker(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Dropout(.1),nn.Linear(hidden,1)) if hidden else nn.Linear(dim,1)

    def forward(self,x,classes,class_count):
        scores = self.score(x).squeeze(-1); pooled = []
        for c in range(class_count):
            mask = classes==c
            value = torch.logsumexp((scores/.25).masked_fill(~mask,-10000.),dim=1)*.25
            value -= .25*mask.sum(dim=1).clamp_min(1).float().log()
            pooled.append(value.masked_fill(~mask.any(dim=1),-10000.))
        return torch.stack(pooled,dim=1),scores


def fit_rankers(x,classes,y,train,hidden,epochs=None,seeds=(42,43,44)):
    torch.set_num_threads(2)
    supported = np.zeros(len(y),bool)
    for i in np.flatnonzero(train): supported[i] = np.any(classes[i]==y[i])
    used = train & supported
    if used.sum()<5: raise ValueError("Too few training contacts with a supported target class")
    values = x[used][classes[used]>=0]
    mean = values.mean(axis=0); scale = values.std(axis=0); scale[scale<1e-4] = 1.
    data = torch.tensor((x[used]-mean)/scale,dtype=torch.float32)
    groups = torch.tensor(classes[used],dtype=torch.long); target = torch.tensor(y[used],dtype=torch.long)
    weights = np.where(y[used]>0,1.,max(1,int((y[used]>0).sum()))/max(1,int((y[used]==0).sum())))
    weights = torch.tensor(weights,dtype=torch.float32)
    members = []
    epochs = (100 if hidden else 120) if epochs is None else epochs
    for seed in seeds:
        torch.manual_seed(seed); network = PairRanker(x.shape[-1],hidden)
        optimizer = torch.optim.AdamW(network.parameters(),lr=.006 if hidden else .025,weight_decay=.1)
        network.train()
        for _ in range(epochs):
            optimizer.zero_grad(); logits,_ = network(data,groups,int(classes.max()+1))
            loss = (nn.functional.cross_entropy(logits,target,reduction="none")*weights).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),2.); optimizer.step()
        network.eval(); members.append({"network":network,"seed":seed})
    return {"members":members,"mean":mean,"scale":scale,"training_count":int(train.sum()),
            "training_supported":int(used.sum()),"epochs":epochs,"hidden":hidden}


def ranker_scores(fitted,x,classes,class_count):
    data = torch.tensor((x-fitted["mean"])/fitted["scale"],dtype=torch.float32)
    ids = torch.tensor(classes,dtype=torch.long); values = []; actor_values = []
    with torch.inference_mode():
        for member in fitted["members"]:
            logits,actors = member["network"](data,ids,class_count)
            values.append(torch.softmax(logits,dim=1).numpy()); actor_values.append(actors.numpy())
    return np.mean(values,axis=0),np.mean(actor_values,axis=0)


def learned_predictions(cases,probabilities,actor_scores,force_near=False):
    predictions = []
    for i,case in enumerate(cases):
        valid = probabilities[i,:len(case["options"])].copy()
        if force_near: valid[0] = -1
        if not case["pairs"] or not any(p["class_index"]>0 for p in case["pairs"]) and force_near:
            predictions.append(prediction(case,"unknown",0.,team="b" if force_near else None)); continue
        chosen = int(np.argmax(valid)); options = [(j,p) for j,p in enumerate(case["pairs"]) if p["class_index"]==chosen]
        actor = max(options,key=lambda pair:actor_scores[i,pair[0]])[1] if options else None
        predictions.append(prediction(case,case["options"][chosen],float(valid[chosen]),actor))
    return predictions


def sequence_scores(snapshot,cases,groups,train,y,probabilities):
    """Learn priors on training labels only; test event nodes contain no author."""
    index = {c["contact"]["id"]:i for i,c in enumerate(cases)}
    sequences = []; fps_values = {v["fps"] for v in snapshot["videos"].values()}
    if len(fps_values)!=1: raise ValueError("Sequence experiment requires a common source FPS")
    fps = next(iter(fps_values))
    for video in sorted(set(groups[train])):
        sequence = []
        for row in snapshot["queue"]:
            if row["video_id"]!=video: continue
            i = index.get(row["id"])
            label = int(y[i]>0) if i is not None and train[i] else None
            sequence.append({"frame":row["frame"],"type":row["type"],"team_label":label})
        sequences.append(sequence)
    priors = learn_transitions(sequences,fps); scores = probabilities.copy()
    for video in sorted(set(groups)):
        rows = [r for r in snapshot["queue"] if r["video_id"]==video]
        # Availability uses only audited time, never the held-out author decision.
        events = [{"frame":r["frame"],"type":r["type"],"team_label":"available" if r["timing_verified"] else None} for r in rows]
        emission = np.array([probabilities[index[r["id"]]] if r["id"] in index else .5 for r in rows])
        posteriors = sequence_posteriors(events,emission,priors,fps)
        for row,value in zip(rows,posteriors):
            if row["id"] in index: scores[index[row["id"]]] = value
    return scores,priors


def evaluate(dataset,output,reuse_models=None,actor_context=None):
    dataset,output = Path(dataset),Path(output)
    if output.exists(): raise ValueError("Evaluation output must be fresh")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    if manifest["schema"]!=SCHEMA: raise ValueError("Unknown benchmark schema")
    for name,key in (("features.npz","features_sha256"),("cases.json","cases_sha256"),("human_labels.json","snapshot_sha256")):
        if sha(dataset/name)!=manifest[key]: raise ValueError("Frozen benchmark changed: "+name)
    data = np.load(dataset/"features.npz",allow_pickle=False)
    x,classes,y,groups,strict,gx = (data[k] for k in ("pairs","classes","targets","groups","strict","global_features"))
    cases = json.loads((dataset/"cases.json").read_bytes()); videos = sorted(set(groups))
    snapshot = json.loads((dataset/"human_labels.json").read_bytes())
    contextual = None; methods = METHODS
    if actor_context is not None:
        context = json.loads(Path(actor_context).read_bytes())
        if (context.get("schema")!="contact_author_central_actor_v1" or context["source_manifest_sha256"]!=sha(dataset/"manifest.json")
                or context["source_cases_sha256"]!=manifest["cases_sha256"] or context["source_features_sha256"]!=manifest["features_sha256"]
                or context.get("targets_used_for_selection") is not False):
            raise ValueError("Incompatible actor-time ablation")
        lookup = {r["id"]:r["pairs"] for r in context["cases"]}
        if set(lookup)!={c["contact"]["id"] for c in cases}: raise ValueError("Actor context changed contact queue")
        contextual = [{**c,"pairs":lookup[c["contact"]["id"]]} for c in cases]
        methods += ("central_hungarian","trees_sequence_central","trees_sequence_central_action")
    if len(videos)!=3 or len(cases)!=len(y): raise ValueError("Expected three whole-source folds")
    if x.shape[-1]!=2*len(PAIR_FIELDS): raise ValueError("Pair feature schema mismatch")
    print(json.dumps({"manifest":str((dataset/"manifest.json").resolve()),"feature_path":str((dataset/"features.npz").resolve()),
        "manifest_sha256":sha(dataset/"manifest.json"),"source_videos":videos,"labels":"human contact authors",
        "schema":SCHEMA,"counts":manifest["counts"],"split":"nested whole-source leave-one-video-out",
        "training":"reuse compatible saved fold models" if reuse_models else "CPU experimental pair rankers and team classifiers; no source/model overwrite"},indent=2),flush=True)
    output.mkdir(parents=True); (output/"fold_models").mkdir()
    fixed = {"other_team_control":[prediction(c,"other_team",.5) for c in cases],
        "nearest_raw_anchor":[fixed_predict(c,raw=True) for c in cases],
        "nearest_box_hsv":[fixed_predict(c,kind="box",anchor=False) for c in cases],
        "nearest_pose_hsv":[fixed_predict(c,kind="pose",anchor=False) for c in cases],
        "nearest_track_hsv":[fixed_predict(c,anchor=False) for c in cases],
        "nearest_anchor_hsv":[fixed_predict(c) for c in cases],
        "action_anchor_hsv":[fixed_predict(c,action=True) for c in cases],
        "nearest_hungarian":[fixed_predict(c,one_to_one=True) for c in cases]}
    if contextual is not None:
        fixed["central_hungarian"] = [fixed_predict(c,one_to_one=True) for c in contextual]
    cache = {}; fit_audit = []
    columns = feature_indices(manifest["global_fields"],"typed")

    def predictions_for(train_videos):
        key = tuple(sorted(train_videos))
        if key in cache: return cache[key]
        train = np.isin(groups,key) & strict
        if len(set(groups[train]))!=len(key): raise ValueError("Training fold lost a whole source")
        result = dict(fixed); side_scores = {}
        print("Fit sources: "+", ".join(key)+f" ({int(train.sum())} strict contacts)",flush=True)
        for family,name in (("logistic","logistic_side_action"),("extra_trees","trees_side_action")):
            filename = "_".join(str(videos.index(v)) for v in key)+"_"+family+".joblib"
            if reuse_models is not None:
                artifact = joblib.load(Path(reuse_models)/filename)
                if tuple(artifact["train_sources"])!=key or artifact["manifest_sha256"]!=sha(dataset/"manifest.json") or artifact["columns"]!=columns:
                    raise ValueError("Incompatible saved team fold")
                classifier = artifact["classifier"]
            else:
                classifier = model(family); classifier.fit(gx[train][:,columns],(y[train]>0).astype(int))
            scores = classifier.predict_proba(gx[:,columns])[:,1]; side_scores[family] = scores
            variants = [(name,True,False)]
            if family=="logistic": variants += [("logistic_side_hungarian",False,True)]
            else: variants += [("trees_side_nearest",False,False),("trees_side_hungarian",False,True),
                               ("trees_side_action_hungarian",True,True)]
            for variant,action,unique in variants:
                base = [fixed_predict(c,force_near=True,action=action,one_to_one=unique) for c in cases]
                result[variant] = [prediction(c,"other_team",1-float(s)) if s<.5 else
                                {**p,"score":float(s)*p["score"]} for c,p,s in zip(cases,base,scores)]
            seq,priors = sequence_scores(snapshot,cases,groups,train,y,scores)
            seq_name = "logistic_sequence_hungarian" if family=="logistic" else "trees_sequence_hungarian"
            base = [fixed_predict(c,force_near=True,one_to_one=True) for c in cases]
            result[seq_name] = [prediction(c,"other_team",1-float(s)) if s<.5 else
                               {**p,"score":float(s)*p["score"]} for c,p,s in zip(cases,base,seq)]
            if family=="extra_trees":
                if contextual is not None:
                    for action,name in ((False,"trees_sequence_central"),(True,"trees_sequence_central_action")):
                        base = [fixed_predict(c,force_near=True,one_to_one=True,action=action) for c in contextual]
                        result[name] = [prediction(c,"other_team",1-float(s)) if s<.5 else
                            {**p,"score":float(s)*p["score"]} for c,p,s in zip(cases,base,seq)]
                write_new(output/"fold_models"/(filename+"_sequence.json"),{"train_sources":key,"priors":priors,
                    "strength":.7,"manifest_sha256":sha(dataset/"manifest.json"),"test_authors_read":False})
            joblib.dump({"classifier":classifier,"columns":columns,"train_sources":key,"manifest_sha256":sha(dataset/"manifest.json")},
                        output/"fold_models"/filename)
        for hidden,name in ((0,"joint_linear"),(24,"joint_mlp")):
            filename = "_".join(str(videos.index(v)) for v in key)+"_"+name+".pt"
            if reuse_models is not None:
                artifact = torch.load(Path(reuse_models)/filename,map_location="cpu",weights_only=False)
                if tuple(artifact["train_sources"])!=key or artifact["manifest_sha256"]!=sha(dataset/"manifest.json") or tuple(artifact["pair_fields"])!=PAIR_FIELDS:
                    raise ValueError("Incompatible saved person fold")
                fitted = {k:v for k,v in artifact.items() if k!="members"}; fitted["members"] = []
                for member in artifact["members"]:
                    network = PairRanker(x.shape[-1],hidden); network.load_state_dict(member["state_dict"]); network.eval()
                    fitted["members"].append({"seed":member["seed"],"network":network})
            else:
                fitted = fit_rankers(x,classes,y,train,hidden)
            scores,actor_scores = ranker_scores(fitted,x,classes,int(classes.max()+1))
            result[name] = learned_predictions(cases,scores,actor_scores)
            if hidden:
                near = learned_predictions(cases,scores,actor_scores,force_near=True)
                result["logistic_side_joint"] = [prediction(c,"other_team",1-float(s)) if s<.5 else
                    {**p,"score":float(s)*p["score"]} for c,p,s in zip(cases,near,side_scores["logistic"])]
            artifact = {k:v for k,v in fitted.items() if k!="members"}
            artifact.update(schema="experimental_contact_author_ranker_v1",pair_fields=PAIR_FIELDS,train_sources=key,
                members=[{"seed":m["seed"],"state_dict":m["network"].state_dict()} for m in fitted["members"]],
                manifest_sha256=sha(dataset/"manifest.json"),production_compatible=False)
            torch.save(artifact,output/"fold_models"/filename)
            fit_audit.append({"method":name,"train_sources":key,"training_contacts":int(train.sum()),
                              "training_supported":fitted["training_supported"],"seeds":[42,43,44],"epochs":fitted["epochs"]})
        cache[key] = result
        return result

    oof = {name:[None]*len(cases) for name in methods}; nested = [None]*len(cases); selection = []
    for held_out in videos:
        train_videos = [v for v in videos if v!=held_out]
        inner = {name:[None]*len(cases) for name in methods}
        for validation in train_videos:
            fitted = predictions_for([v for v in train_videos if v!=validation])
            for name in methods:
                for i in np.flatnonzero(groups==validation): inner[name][i] = fitted[name][i]
        inner_mask = np.isin(groups,train_videos) & strict
        inner_metrics = {name:metrics(cases,p,inner_mask) for name,p in inner.items()}
        selected = max(methods,key=lambda name:inner_metrics[name]["balanced_author_accuracy"])
        outer = predictions_for(train_videos)
        for i in np.flatnonzero(groups==held_out):
            for name in methods: oof[name][i] = outer[name][i]
            nested[i] = outer[selected][i]
        selection.append({"held_out":held_out,"training_sources":train_videos,"selected":selected,
                          "inner_metrics":inner_metrics,"test_labels_used_for_selection":False})
        print(f"Held out {held_out}: inner selected {selected}",flush=True)
    oof["nested_selected"] = nested
    results = {name:{"strict":metrics(cases,p,strict),"all_eligible":metrics(cases,p,np.ones(len(y),bool)),
                     "per_video":{v:metrics(cases,p,strict & (groups==v)) for v in videos}} for name,p in oof.items()}
    best = max(methods,key=lambda name:results[name]["strict"]["balanced_author_accuracy"])
    records = [{"id":c["contact"]["id"],"video_id":c["contact"]["video_id"],"frame":c["contact"]["frame"],
        "type":c["contact"]["type"],"truth":c["options"][c["target"]],"strict":c["strict"],
        "predictions":{name:p[i] for name,p in oof.items()}} for i,c in enumerate(cases)]
    report = {"schema":"human_contact_author_evaluation_v1","created_at":now(),"dataset":str(dataset.resolve()),
        "dataset_manifest_sha256":sha(dataset/"manifest.json"),"results":results,"nested_selection":selection,
        "best_fixed_comparison":best,"best_selection_is_development":True,"fit_audit":fit_audit,
        "quality_scope":manifest["contract"]["evaluation_scope"],"independent_videos":len(videos),
        "person_labels_manufactured":False,"production_promoted":False,"reused_models":str(reuse_models) if reuse_models else None,
        "actor_context":str(Path(actor_context).resolve()) if actor_context else None,
        "actor_context_sha256":sha(actor_context) if actor_context else None,
        "limits":["Only three sources; comparisons are model selection, not an untouched final test",
                  "Reference galleries and unlabeled track colour pooling are allowed offline setup inputs",
                  "Known human contact times/types; no detection precision/recall or missed-contact claim",
                  "No author-box gold: identity and contact-actor localization errors cannot be separated conclusively"]}
    write_new(output/"predictions.json",records); report["predictions_sha256"] = sha(output/"predictions.json")
    write_new(output/"comparison.json",report)
    lines = ["# Human contact-author benchmark","",f"Frozen human labels: revision snapshot; {len(y)} eligible contacts, {int(strict.sum())} strict contacts.",
        "Known contact times/types. Whole-video held-out folds; reference photos allowed. No model inference or production promotion.","",
        "| Method | Exact author | Near-person recall | Team | Balanced author |","|---|---:|---:|---:|---:|"]
    for name in methods+("nested_selected",):
        m = results[name]["strict"]
        lines.append(f"| {name} | {m['author_accuracy']:.1%} | {m['near_author_recall']:.1%} | {m['team_accuracy']:.1%} | {m['balanced_author_accuracy']:.1%} |")
    lines += ["",f"Best fixed comparison: **{best}** (development selection).",
              "Nested selected uses only inner training-source validation to choose an algorithm for each outer video.","",*report["limits"]]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"best":best,"strict":results[best]["strict"],"nested":results["nested_selected"]["strict"]},indent=2),flush=True)
    return report
