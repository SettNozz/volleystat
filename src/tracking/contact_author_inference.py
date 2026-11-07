"""Apply frozen cross-video author models to an unlabelled saved analysis.

No training, detector replay, fabricated targets or source-result edits. Roster
IDs retain the saved seed's meaning and its automatic/unverified provenance.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from action_recognition.annotator.contact_author_dataset import fingerprint_file, now, sha, write_new
from src.tracking.contact_author_benchmark import capture_colours, distance, track_segments, unit
from src.tracking.contact_author_evaluation import PairRanker, prediction, ranker_scores
from src.tracking.contact_author_features import extract_contact
from src.tracking.contact_author_reid_evaluation import actors, conditional_from_actors, fusion_features, scene_features
from src.tracking.player_contacts import CONTACT_TYPES, side
from src.tracking.player_identity import overlap
from src.tracking.volleyball_contact_lab import learn_transitions, sequence_posteriors


def model_roster(annotations):
    """Reuse saved initialization, without presenting automatic IDs as human gold."""
    identity = annotations.get("player_identity") or {}
    if not identity.get("enabled") or side(identity.get("side")) != "b":
        raise ValueError("A saved near-side roster is required")
    seed = identity["seed"]; frames = {r["f"]:r for r in annotations["frames"]}
    players = []
    for key,pid in sorted(seed["assignments"].items(),key=lambda item:item[1]):
        f,index = map(int,key.split(":")); pose = frames[f]["poses"][index]
        players.append({"player_id":pid,"label_source":seed.get("label_source","model"),
                        "verified":False,"references":[{"frame":f,"bbox":pose["bb"],"pose_index":index,
                        "raw_track_id":pose.get("id"),"label_source":seed.get("label_source","model"),"verified":False}]})
    if len(players)!=len({p["player_id"] for p in players}):
        raise ValueError("Duplicate seed roster IDs")
    if set(identity.get("gallery_keys",{})) != {p["player_id"] for p in players}:
        raise ValueError("Roster has introductions; explicit reference support is required")
    return {"players":players,"label_source":seed.get("label_source","model"),"verified":False,
            "limitations":"Saved automatic roster, not human-confirmed; substitutions are not established"}


def strip_model_identities(frames):
    """Match training feature semantics: identity comes from photos, never old IDs."""
    return [{**r,"poses":[{k:v for k,v in p.items() if k not in ("player_id","identity")}
                          for p in r.get("poses") or []]} for r in frames]


def prepare_input(video,annotations_path,stats_path,output):
    output = Path(output)
    if output.exists(): raise ValueError("Inference input output must be fresh")
    sources = {name:fingerprint_file(path) for name,path in
               (("video",video),("annotations",annotations_path),("stats",stats_path))}
    annotations = json.loads(Path(annotations_path).read_bytes()); stats = json.loads(Path(stats_path).read_bytes())
    if Path(annotations.get("original_video_path","")).resolve()!=Path(video).resolve():
        raise ValueError("Saved analysis belongs to another video")
    if sources["video"]["sha256"] != annotations["model_cache"]["source"]["sha256"]:
        raise ValueError("Original video differs from the cached model source")
    stem = Path(video).stem; fps = annotations["fps"]; roster = model_roster(annotations)
    metadata = {"video_id":stem,"fps":fps,"width":annotations["frame_width"],"height":annotations["frame_height"],
                "num_frames":len(annotations["frames"]),"video_source":sources["video"],
                "annotations_source":sources["annotations"],"stats_source":sources["stats"]}
    contacts = []
    for index,event in enumerate(stats["all_events"]):
        if event.get("type") not in CONTACT_TYPES: continue
        contacts.append({"id":f"{stem}:model:{index}:f{event['frame']}","video_id":stem,"frame":int(event["frame"]),
                         "type":event["type"],"event_index":index,"event_label_source":"model",
                         "event_source":event.get("source"),"event_confidence":event.get("confidence"),
                         "timing_verified":False,"author_label_source":None})
    if not contacts: raise ValueError("Saved result has no contact events")
    if any(c["frame"]<0 or c["frame"]>=metadata["num_frames"] for c in contacts):
        raise ValueError("Contact outside saved analysis")
    snapshot = {"schema":"contact_author_unlabelled_inputs_v1","created_at":now(),"videos":{stem:metadata},
                "rosters":{stem:roster},"contacts":contacts,"queue":contacts,"decisions":[],"human_labels":False}
    contract = {"source_videos":[stem],"sources":sources,"contacts":len(contacts),"schema":"contact_author_unlabelled_base_v1",
                "label_provenance":"Model contact times/types and automatic roster; no human author labels",
                "split_policy":"Unseen-source inference with frozen models; no training or feature refresh of training data",
                "features":str((output/"features.npz").resolve()),"training":False,"detector_inference":False}
    print(json.dumps(contract,indent=2),flush=True); output.mkdir(parents=True)
    write_new(output/"inference_inputs.json",snapshot)
    frames = strip_model_identities(annotations["frames"]); by_frame = {r["f"]:r for r in frames}
    observations,tracks = track_segments(frames,fps)
    requests = defaultdict(list); reference_keys = defaultdict(list)
    for player in roster["players"]:
        for n,ref in enumerate(player["references"]):
            key = f"ref:{player['player_id']}:{n}"; pose = deepcopy(by_frame[ref["frame"]]["poses"][ref["pose_index"]])
            pose["bb"] = ref["bbox"]; requests[ref["frame"]].append((key,pose)); reference_keys[player["player_id"]].append(key)
    global_features = []; actor_rows = []; global_names = None
    times = [c["frame"] for c in contacts]
    for c in contacts:
        neighbour = min((abs(f-c["frame"]) for f in times if f!=c["frame"]),default=100)
        gx,names,candidates = extract_contact(c,by_frame,fps,(metadata["width"],metadata["height"]),neighbour)
        if global_names is not None and global_names!=names: raise ValueError("Global feature order changed")
        global_names = names; global_features.append(gx)
        for actor in candidates:
            obs = observations[actor["frame"],actor["pose_index"]]; actor["segment"] = obs["segment"]
            nearest = min(tracks[obs["segment"]],key=lambda o:abs(o["frame"]-c["frame"]))
            actor["colour_key"] = f"obs:{nearest['frame']}:{nearest['index']}"
            requests[nearest["frame"]].append((actor["colour_key"],nearest["pose"]))
        actor_rows.append(candidates)
    requests = {f:list(dict(items).items()) for f,items in requests.items()}
    colours = capture_colours(metadata,requests,output/f"{stem}_colours.npz")
    galleries = {pid:[[colours[k][v] for k in keys if colours[k][v] is not None] for v in (0,1)]
                 for pid,keys in reference_keys.items()}
    track_colours = defaultdict(lambda:[[],[]])
    for key,values in colours.items():
        if not key.startswith("obs:"): continue
        _,f,index = key.split(":"); f,index = int(f),int(index); obs = observations[f,index]
        contamination = max((overlap(obs["pose"]["bb"],p["bb"]) for j,p in enumerate(by_frame[f]["poses"]) if j!=index),default=0.)
        if contamination>.35: continue
        for v,value in enumerate(values):
            if value is not None: track_colours[obs["segment"]][v].append(value)
    pooled = {segment:[unit(np.mean(v,axis=0)) if v else None for v in values] for segment,values in track_colours.items()}
    votes = {segment:Counter(side(o["pose"].get("team")) for o in track).most_common(1)[0][0]
             for segment,track in tracks.items()}
    options = ["other_team"]+[p["player_id"] for p in roster["players"]]; cases = []
    for c,candidates in zip(contacts,actor_rows):
        pairs = []
        for actor in candidates:
            near = votes[actor["segment"]]=="b"; average = pooled.get(actor["segment"],[None,None])
            for class_index,pid in enumerate(options):
                if (class_index==0 and near) or (class_index>0 and not near): continue
                track_app = [distance(average[v],galleries[pid][v]) for v in (0,1)] if class_index else [None,None]
                pairs.append({k:v for k,v in actor.items() if k!="colour_key"}|{
                    "class_index":class_index,"player_id":pid,"track_colour_box":track_app[0],"track_colour_pose":track_app[1]})
        cases.append({"contact":c,"options":options,"target":-1,"strict":False,"pairs":pairs})
    np.savez_compressed(output/"features.npz",global_features=np.stack(global_features),targets=np.full(len(cases),-1,np.int64),
                        strict=np.zeros(len(cases),bool),groups=np.array([stem]*len(cases)))
    write_new(output/"cases.json",cases)
    write_new(output/"manifest.json",{"schema":contract["schema"],"contract":contract,"snapshot_file":"inference_inputs.json",
              "label_provenance":contract["label_provenance"],"snapshot_sha256":sha(output/"inference_inputs.json"),
              "features_sha256":sha(output/"features.npz"),"cases_sha256":sha(output/"cases.json"),
              "global_fields":global_names,"counts":{"contacts":len(cases),"strict":0}})
    return output


def load_ranker(path,dimension,manifest_hash,sources):
    artifact = torch.load(path,map_location="cpu",weights_only=False)
    if artifact["manifest_sha256"]!=manifest_hash or tuple(artifact["training_sources"])!=tuple(sources):
        raise ValueError("Ranker provenance differs from the frozen training dataset")
    fit = {k:v for k,v in artifact.items() if k!="members"}; fit["members"] = []
    for member in artifact["members"]:
        network = PairRanker(dimension,artifact["hidden"]); network.load_state_dict(member["state_dict"]); network.eval()
        fit["members"].append({"network":network,"seed":member["seed"]})
    return fit


def predict(dataset,training_dataset,evaluation,output):
    """Ensemble three frozen pair-source class_fusion models, without final-fit."""
    dataset,training_dataset,evaluation,output = map(Path,(dataset,training_dataset,evaluation,output))
    if output.exists(): raise ValueError("Prediction output must be fresh")
    manifest = json.loads((dataset/"manifest.json").read_bytes()); training = json.loads((training_dataset/"manifest.json").read_bytes())
    if manifest["schema"]!=training["schema"] or manifest["fields"]!=training["fields"] or manifest["global_fields"]!=training["global_fields"]:
        raise ValueError("Unlabelled inference feature schema differs from trained models")
    snapshot_file = manifest["snapshot_file"]
    for file,key in ((snapshot_file,"snapshot_sha256"),("cases.json","cases_sha256"),("features.npz","features_sha256")):
        if sha(dataset/file)!=manifest[key]: raise ValueError("Inference dataset changed")
    snapshot = json.loads((dataset/snapshot_file).read_bytes()); cases = json.loads((dataset/"cases.json").read_bytes())
    frozen = json.loads((training_dataset/"human_labels.json").read_bytes())
    if sha(training_dataset/"human_labels.json")!=training["snapshot_sha256"]: raise ValueError("Training snapshot changed")
    for file,key in (("cases.json","cases_sha256"),("features.npz","features_sha256")):
        if sha(training_dataset/file)!=training[key]: raise ValueError("Frozen training dataset changed")
    train_cases=json.loads((training_dataset/"cases.json").read_bytes())
    strict_ids={c["contact"]["id"] for c in train_cases if c["strict"]}
    report = json.loads((evaluation/"comparison.json").read_bytes())
    if report["dataset_manifest_sha256"]!=sha(training_dataset/"manifest.json"): raise ValueError("Model dataset mismatch")
    training_videos = sorted(frozen["videos"])
    if set(snapshot["videos"]) & set(training_videos): raise ValueError("Inference source occurs in training data")
    if any(c["target"]!=-1 or c["contact"].get("author") for c in cases): raise ValueError("Inference must have no author targets")
    print(json.dumps({"manifest":str((dataset/"manifest.json").resolve()),"features":str((dataset/"features.npz").resolve()),
                      "training_manifest":str((training_dataset/"manifest.json").resolve()),"training_sources":training_videos,
                      "test_sources":sorted(snapshot["videos"]),"contacts":len(cases),"schema":manifest["schema"],
                      "label_provenance":"unlabelled model contacts; no accuracy metric available",
                      "split":"test source disjoint from all training sources","training":False,"model_inference":"frozen CPU author heads only",
                      "deployment":"equal ensemble of three existing two-source class_fusion members; no measured ensemble accuracy"},indent=2),flush=True)
    torch.set_num_threads(2)
    arrays = np.load(dataset/"features.npz",allow_pickle=False); x,classes,gx = (arrays[k] for k in ("pairs","classes","global_features"))
    maximum = max(len(c["options"]) for c in cases); scene = scene_features(cases,x,manifest["fields"],gx)
    member_results = []; member_sides = []; model_hashes = {}; actor_selections = []; member_labels = []
    for key in ("0_1","0_2","1_2"):
        sources = [training_videos[int(i)] for i in key.split("_")]
        def artifact(name):
            path = evaluation/"models"/(key+"_"+name+".joblib"); value = joblib.load(path); model_hashes[path.name]=sha(path)
            if value["manifest_sha256"]!=sha(training_dataset/"manifest.json") or tuple(value["training_sources"])!=tuple(sources):
                raise ValueError("Frozen model provenance mismatch")
            return value
        conditionals = {}; selections = {}
        cond = np.zeros((len(cases),maximum)); selected = []
        for i,c in enumerate(cases):
            scores = {a:float(np.exp(-max(0,p[0][1]["score"])/.12)) for a,p in actors(c).items()}
            row,chosen = conditional_from_actors(c,scores,"osnet_expanded"); cond[i,:len(row)]=row; selected.append(chosen)
        conditionals["osnet_expanded"]=cond; selections["osnet_expanded"]=selected
        for family in ("trees","hist"):
            saved = artifact("actor_"+family); cond=np.zeros((len(cases),maximum)); selected=[]
            for i,c in enumerate(cases):
                grouped=actors(c); indices=[v[0][0] for v in grouped.values()]
                probs=saved["classifier"].predict_proba(x[i,indices][:,saved["columns"]])[:,1] if indices else []
                scores={a:float(np.exp(np.clip(np.log(p+1e-6)-np.log(1-p+1e-6),-10,10))) for a,p in zip(grouped,probs)}
                row,chosen=conditional_from_actors(c,scores,"osnet_hsv");cond[i,:len(row)]=row;selected.append(chosen)
            conditionals["actor_"+family]=cond;selections["actor_"+family]=selected
        for name,near_only in (("mil_near_mlp",True),("mil_joint",False)):
            path=evaluation/"models"/(key+"_"+name+".pt");model_hashes[path.name]=sha(path)
            fit=load_ranker(path,x.shape[-1],sha(training_dataset/"manifest.json"),sources)
            mask=classes.copy()
            if near_only:mask[mask==0]=-1
            probabilities,instance=ranker_scores(fit,x,mask,maximum)
            probabilities[:,0]=0;probabilities/=np.maximum(probabilities.sum(axis=1,keepdims=True),1e-8)
            conditionals[name]=probabilities;selections[name]=[]
            for i,c in enumerate(cases):
                selections[name].append({j:max([(n,p) for n,p in enumerate(c["pairs"]) if p["class_index"]==j],key=lambda q:instance[i,q[0]])[1]
                                         for j in range(1,len(c["options"])) if any(p["class_index"]==j for p in c["pairs"])})
        classifier=artifact("fusion")["pipeline"];fusion=np.zeros((len(cases),maximum))
        for i,c in enumerate(cases):
            values=fusion_features(c,i,conditionals,selections);scores=classifier.decision_function(values)
            probability=np.exp(scores-scores.max());probability/=probability.sum();fusion[i,1:len(c["options"])]=probability
        rich_side=artifact("rich_side")["classifier"].predict_proba(scene)[:,1]
        sequences=[];fps=next(iter(snapshot["videos"].values()))["fps"]
        for v in sources:
            truth={c["id"]:c for c in frozen["contacts"] if c["video_id"]==v}
            sequences.append([{**r,"team_label":int(truth[r["id"]]["author"]["decision"]=="player")
                               if r["id"] in truth and r["id"] in strict_ids else None}
                              for r in frozen["queue"] if r["video_id"]==v])
        priors=learn_transitions(sequences,fps)
        # Model event times are usable sequence nodes, explicitly not human-verified.
        events=[{"frame":c["contact"]["frame"],"type":c["contact"]["type"],"team_label":"model_available"} for c in cases]
        seq=sequence_posteriors(events,rich_side,priors,fps)
        chosen=[{c:(selections["actor_trees"][i].get(c) if conditionals["actor_trees"][i,c]>=conditionals["mil_joint"][i,c]
                    else selections["mil_joint"][i].get(c)) for c in range(1,len(case["options"]))} for i,case in enumerate(cases)]
        member_results.append(fusion);member_sides.append(seq);actor_selections.append(chosen)
        member_labels.append(["other_team" if seq[i]<.5 else c["options"][int(np.argmax(fusion[i,1:len(c["options"])]))+1] for i,c in enumerate(cases)])
        print(f"Frozen class_fusion member {key}: {len(cases)} contacts",flush=True)
    conditional=np.mean(member_results,axis=0);team=np.mean(member_sides,axis=0);records=[]
    for i,c in enumerate(cases):
        selected_class=int(np.argmax(conditional[i,1:len(c["options"])]))+1
        label="other_team" if team[i]<.5 else c["options"][selected_class]
        best=max(range(len(member_results)),key=lambda m:member_results[m][i,selected_class])
        actor=actor_selections[best][i].get(selected_class) if label!="other_team" else None
        if label!="other_team" and actor is None:label="unknown"
        score=1-team[i] if label=="other_team" else team[i]*conditional[i,selected_class]
        p=prediction(c,label,score,actor);p.update(team_score=float(team[i]),ensemble_agreement=sum(m[i]==label for m in member_labels)/3)
        records.append({"id":c["contact"]["id"],"video_id":c["contact"]["video_id"],"frame":c["contact"]["frame"],
                        "type":c["contact"]["type"],"event_index":c["contact"]["event_index"],"event_label_source":"model",
                        "truth":None,"predictions":{"class_fusion_ensemble":p}})
    preservation={v:{k:{"expected":m[k]["sha256"],"actual":sha(m[k]["path"])}
                  for k in ("annotations_source","stats_source","video_source")} for v,m in snapshot["videos"].items()}
    if any(p["expected"]!=p["actual"] for sources in preservation.values() for p in sources.values()):
        raise ValueError("Original source changed during inference")
    output.mkdir(parents=True);write_new(output/"predictions.json",records)
    counts=Counter(r["predictions"]["class_fusion_ensemble"]["label"] for r in records)
    write_new(output/"report.json",{"schema":"contact_author_unlabelled_inference_v1","created_at":now(),"dataset":str(dataset.resolve()),
              "dataset_manifest_sha256":sha(dataset/"manifest.json"),"predictions_sha256":sha(output/"predictions.json"),
              "method":"class_fusion_ensemble","training_dataset":str(training_dataset.resolve()),"training_sources":training_videos,
              "training_manifest_sha256":sha(training_dataset/"manifest.json"),"model_files_sha256":model_hashes,
              "training":False,"source_labels_used":False,"metrics":None,"counts":dict(counts),"contacts":len(records),
              "limitations":["No human author labels: accuracy cannot be measured","Model contact timing and type, automatic unverified roster",
                              "Three pair-source frozen members; deployment ensemble differs from leave-one-video-out evaluation",
                              "Same-camera generalization is unmeasured; no confirmed substitution roster"],"production_promoted":False})
    write_new(output/"source_preservation.json",preservation)
    return output
