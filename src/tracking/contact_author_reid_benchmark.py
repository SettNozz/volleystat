"""Source-bound appearance/temporal follow-up to the human author benchmark.

Only previously missing OSNet crop features may be captured. Detections, human
labels and prior feature datasets stay frozen. Reference crop roles are kept
separate from observed detector boxes; no crop becomes detector ground truth.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

from action_recognition.annotator.contact_author_dataset import now, sha, write_new
from src.models.inference_cache import ModelOutputCache, FrozenOSNet, collect_embeddings, osnet_feature_store
from src.tracking.contact_author_benchmark import iou, track_segments, encode, unit
from src.tracking.contact_author_features import geometry, ball
from src.tracking.contact_author_temporal_features import temporal_actor
from src.tracking.player_contacts import side
from src.tracking.player_identity import overlap


def sparse_images(video,needed,size):
    cap = cv2.VideoCapture(str(video)); current = -1
    try:
        if not cap.isOpened() or (int(cap.get(3)),int(cap.get(4)))!=tuple(size):
            raise ValueError("Original video geometry differs from saved crop plan")
        for f in sorted(needed):
            if f-current>32:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,image = cap.read(); current = f
            else:
                while current<f:
                    if not cap.grab(): raise ValueError("Source ended before requested crop")
                    current += 1
                ok,image = cap.retrieve()
            if not ok: raise ValueError(f"Cannot decode source frame {f}")
            yield f,image
    finally: cap.release()


def crop_plan(annotations,cases,roster):
    frames = annotations["frames"]; by_frame = {r["f"]:r for r in frames}
    observations,tracks = track_segments(frames,annotations["fps"])
    requests = {}; galleries = {}; anchor_segments = defaultdict(set)
    def add(obs):
        f,i = obs["frame"],obs["index"]; p = obs["pose"]; key = f"{f}:{i}"
        peers = by_frame[f].get("poses") or []
        contamination = max((overlap(p["bb"],q["bb"]) for j,q in enumerate(peers) if j!=i),default=0.)
        requests[key] = {"key":key,"f":f,"pose_index":i,"bb":p["bb"],"raw_id":p.get("id"),
                         "segment":obs["segment"],"side":side(p.get("team")),"occlusion":contamination,
                         "role":"saved_detector_pose"}
        return key
    for player in roster:
        keys = []
        for n,ref in enumerate(player["references"]):
            f = ref["frame"]; key = f"reference:{player['player_id']}:{n}"
            requests[key] = {"key":key,"f":f,"pose_index":None,"bb":ref["bbox"],"raw_id":None,
                             "segment":None,"side":"b","occlusion":0.,"role":
                             "automatic_identity_reference" if player.get("verified") is False else "confirmed_identity_reference"}
            keys.append(key)
            ranked = sorted([(iou(ref["bbox"],p["bb"]),i) for i,p in enumerate(by_frame.get(f,{}).get("poses") or [])],reverse=True)
            if not ranked or ranked[0][0]<.35 or len(ranked)>1 and ranked[0][0]-ranked[1][0]<.1: continue
            obs = observations[f,ranked[0][1]]; segment = obs["segment"]
            anchor_segments[segment].add(player["player_id"])
            clean = []
            for o in tracks[segment]:
                if abs(o["frame"]-f)>15: continue
                k = add(o); item = requests[k]
                if item["occlusion"]<.3: clean.append((abs(o["frame"]-f),k))
            keys += [k for _,k in sorted(clean)[::3][:5]]
        galleries[player["player_id"]] = list(dict.fromkeys(keys))
    event_tracks = defaultdict(set)
    for case in cases:
        f = case["contact"]["frame"]
        for i,_ in enumerate(by_frame[f].get("poses") or []): add(observations[f,i])
        for pair in case["pairs"]:
            obs = observations[pair["frame"],pair["pose_index"]]; add(obs)
            event_tracks[obs["segment"]].add(f)
            nearest = min(tracks[obs["segment"]],key=lambda o:abs(o["frame"]-f)); add(nearest)
    # Clean views are chosen from saved poses, without reading contact targets.
    for segment,times in event_tracks.items():
        track = tracks[segment]
        if Counter(side(o["pose"].get("team")) for o in track).most_common(1)[0][0]!="b": continue
        options = []
        for obs in track:
            gap = min(abs(obs["frame"]-t) for t in times)
            if gap>45: continue
            p = obs["pose"]; peers = by_frame[obs["frame"]].get("poses") or []
            contaminated = max((overlap(p["bb"],q["bb"]) for j,q in enumerate(peers) if j!=obs["index"]),default=0.)
            quality = np.mean([p["kp"][j][2] for j in (5,6,11,12)])
            if contaminated>.25 or quality<.6: continue
            options.append((contaminated-.2*quality+.001*gap,obs))
        chosen = []
        for _,obs in sorted(options,key=lambda x:x[0]):
            if any(abs(obs["frame"]-o["frame"])<12 for o in chosen): continue
            chosen.append(obs); add(obs)
            if len(chosen)==3: break
    return {"observations":list(requests.values()),"galleries":galleries,
            "anchor_segments":{k:sorted(v) for k,v in anchor_segments.items()}}


def capture(dataset,output,assets,cache_dir,device="cuda",require=False):
    dataset,output = Path(dataset),Path(output)
    if output.exists(): raise ValueError("ReID capture output must be fresh")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    snapshot_file = manifest.get("snapshot_file","human_labels.json")
    if sha(dataset/snapshot_file)!=manifest["snapshot_sha256"] or sha(dataset/"cases.json")!=manifest["cases_sha256"]:
        raise ValueError("Frozen human benchmark changed")
    snapshot = json.loads((dataset/snapshot_file).read_bytes()); cases = json.loads((dataset/"cases.json").read_bytes())
    contract = {"schema":"contact_author_osnet_capture_v1","source_manifest":str((dataset/"manifest.json").resolve()),
        "source_manifest_sha256":sha(dataset/"manifest.json"),"source_videos":sorted(snapshot["videos"]),
        "input_snapshot_sha256":manifest["snapshot_sha256"],"feature_output":str(output.resolve()),
        "label_provenance":manifest.get("label_provenance","human author targets excluded from crop/view selection; detector crops and identity references separate"),
        "split":"not training; source-bound crop capture",
        "missing_stage":"OSNet crop embeddings only; no YOLO/SAM/spotter inference", "require_model_cache":require,
        "training":False,"device":device}
    print(json.dumps(contract,indent=2),flush=True); output.mkdir(parents=True)
    write_new(output/"plan.json",contract)
    encoder = FrozenOSNet(Path(assets),device=device,threads=2,cuda_memory_fraction=.1 if device=="cuda" else None)
    outputs = {}
    for video,metadata in sorted(snapshot["videos"].items()):
        source = metadata["annotations_source"]
        if sha(source["path"])!=source["sha256"]: raise ValueError("Saved source poses changed")
        stat = Path(metadata["video_source"]["path"]).stat()
        if stat.st_size!=metadata["video_source"]["size"] or stat.st_mtime_ns!=metadata["video_source"]["mtime_ns"]:
            raise ValueError("Original video changed")
        annotations = json.loads(Path(source["path"]).read_bytes())
        plan = crop_plan(annotations,[c for c in cases if c["contact"]["video_id"]==video],snapshot["rosters"][video]["players"])
        write_new(output/f"{video}_observations.json",plan)
        by_frame = defaultdict(list)
        for obs in plan["observations"]: by_frame[obs["f"]].append(obs)
        frames = [{"f":f,"players":players} for f,players in sorted(by_frame.items())]
        print(f"OSNet {video}: {len(plan['observations'])} crops / {len(frames)} saved frames",flush=True)
        cache = ModelOutputCache(metadata["video_source"]["path"],Path(cache_dir),require,
                                 source_identity={"kind":"video","sha256":metadata["video_source"]["sha256"]})
        store = osnet_feature_store(cache,Path(assets),Path(assets)/"osnet_msmt17.pth",metadata["width"],metadata["height"])
        def progress(done,total):
            if done%25==0 or done==total: print(f"OSNet {video}: {done}/{total} missing frames",flush=True)
        vectors = collect_embeddings(frames,lambda needed:sparse_images(metadata["video_source"]["path"],needed,(metadata["width"],metadata["height"])),
                                     encoder,store,batch_size=8,progress=progress)
        keys = list(vectors); path = output/f"{video}_embeddings.npz"
        np.savez_compressed(path,keys=np.array(keys),embeddings=np.stack([vectors[k] for k in keys]))
        outputs[video] = {"embeddings_sha256":sha(path),"observations_sha256":sha(output/f"{video}_observations.json"),
                          "cache":cache.summary(),"count":len(keys)}
        cache.db.close(); del annotations,plan,frames,by_frame,vectors
    write_new(output/"manifest.json",{**contract,"created_at":now(),"videos":outputs,"encoder":encoder.provenance})
    return output


def appearance_context(plan,vectors):
    by_segment = defaultdict(list)
    for obs in plan["observations"]:
        if obs["segment"] is not None: by_segment[obs["segment"]].append(obs)
    gallery = {pid:np.stack([vectors[k] for k in keys]) for pid,keys in plan["galleries"].items()}
    refined = {pid:list(keys) for pid,keys in plan["galleries"].items()}
    people = list(gallery); candidates = []
    for obs in plan["observations"]:
        if obs["segment"] is None or obs["occlusion"]>.2: continue
        vector = vectors[obs["key"]]
        scores = sorted([(float(np.max(g@vector)),pid) for pid,g in gallery.items()],reverse=True)
        if scores[0][0]>.75 and scores[0][0]-scores[1][0]>.08:
            candidates.append((scores[0][0]-scores[1][0],scores[0][1],obs["key"]))
    for _,pid,key in sorted(candidates,reverse=True):
        if len(refined[pid])>=len(plan["galleries"][pid])+8: continue
        if max(float(vectors[key]@vectors[k]) for k in refined[pid])>.985: continue
        refined[pid].append(key)
    expanded = {pid:np.stack([vectors[k] for k in keys]) for pid,keys in refined.items()}
    def compare(segment,frame):
        observations = by_segment.get(segment,[])
        if not observations: return {pid:[None]*4 for pid in people}
        nearest = min(observations,key=lambda o:abs(o["f"]-frame))
        clean = sorted([o for o in observations if o["occlusion"]<.3 and abs(o["f"]-frame)<=45],key=lambda o:abs(o["f"]-frame))[:8] or [nearest]
        queries = np.stack([vectors[o["key"]] for o in clean]); local = vectors[nearest["key"]]; mean = unit(queries.mean(axis=0))
        result = {}
        for pid,g in gallery.items():
            scores = np.sort(np.max(queries@g.T,axis=1)); extra = np.sort(np.max(queries@expanded[pid].T,axis=1))
            result[pid] = [1-float(np.max(g@local)),1-float(scores[-min(3,len(scores)):].mean()),
                           1-float(unit(g.mean(axis=0))@mean),1-float(extra[-min(3,len(extra)):].mean())]
        return result
    return compare,refined


def prepare(dataset,reid_dir,output):
    dataset,reid_dir,output = Path(dataset),Path(reid_dir),Path(output)
    if output.exists(): raise ValueError("Rich feature output must be fresh")
    base_manifest = json.loads((dataset/"manifest.json").read_bytes()); reid_manifest = json.loads((reid_dir/"manifest.json").read_bytes())
    if reid_manifest["source_manifest_sha256"]!=sha(dataset/"manifest.json"): raise ValueError("ReID source dataset differs")
    snapshot_file = base_manifest.get("snapshot_file","human_labels.json")
    inference = snapshot_file != "human_labels.json"
    for file,key in ((snapshot_file,"snapshot_sha256"),("cases.json","cases_sha256"),("features.npz","features_sha256")):
        if sha(dataset/file)!=base_manifest[key]: raise ValueError("Frozen input changed")
    snapshot = json.loads((dataset/snapshot_file).read_bytes()); cases = json.loads((dataset/"cases.json").read_bytes())
    parent = np.load(dataset/"features.npz",allow_pickle=False)
    contract = {"schema":"contact_author_reid_temporal_v1","manifest":str((output/"manifest.json").resolve()),
        "features":str((output/"features.npz").resolve()),"source_videos":sorted(snapshot["videos"]),
        "label_provenance":base_manifest.get("label_provenance","frozen human person-level decisions; photo references are setup inputs"),
        "contacts":len(cases),"strict_contacts":int(parent["strict"].sum()),
        "person_contacts":None if inference else int((parent["targets"]>0).sum()),
        "split":"unlabelled inference; no fitting" if inference else "complete-source held-out; normalization and learned parameters from training sources only",
        "inference":"cached pinned OSNet only; no detector/pose/ball inference","future_frames":True,
        "interpolation":"derived feature-only joint estimates with masks; source poses remain observed and unchanged"}
    print(json.dumps(contract,indent=2),flush=True); output.mkdir(parents=True)
    (output/snapshot_file).write_bytes((dataset/snapshot_file).read_bytes())
    all_cases = []; tensors = []; names = None; refine_audit = {}
    for video,metadata in sorted(snapshot["videos"].items()):
        vector_path = reid_dir/f"{video}_embeddings.npz"; plan_path = reid_dir/f"{video}_observations.json"
        provenance = reid_manifest["videos"][video]
        if sha(vector_path)!=provenance["embeddings_sha256"] or sha(plan_path)!=provenance["observations_sha256"]:
            raise ValueError("ReID capture changed")
        plan = json.loads(plan_path.read_bytes()); arrays = np.load(vector_path,allow_pickle=False)
        vectors = dict(zip(arrays["keys"],arrays["embeddings"])); compare,refined = appearance_context(plan,vectors)
        refine_audit[video] = {"source":"derived_unlabelled_gallery_expansion","original":plan["galleries"],"expanded":refined,"contact_targets_used":False}
        annotations_path = Path(metadata["annotations_source"]["path"])
        if sha(annotations_path)!=metadata["annotations_source"]["sha256"]: raise ValueError("Source pose cache changed")
        annotations = json.loads(annotations_path.read_bytes()); by_frame = {r["f"]:r for r in annotations["frames"]}
        observations,tracks = track_segments(annotations["frames"],metadata["fps"])
        for case in (c for c in cases if c["contact"]["video_id"]==video):
            frame = case["contact"]["frame"]; options = case["options"]
            actor_obs = {}
            for pair in case["pairs"]:
                segment = pair["segment"]; actor_obs[segment] = min(tracks[segment],key=lambda o:abs(o["frame"]-frame))
            for index,_ in enumerate(by_frame[frame].get("poses") or []):
                obs = observations[frame,index]; actor_obs[obs["segment"]] = obs
            pairs = []; values = []
            for actor_index,(segment,obs) in enumerate(actor_obs.items()):
                f,index,pose = obs["frame"],obs["index"],obs["pose"]
                actor = {"frame":f,"pose_index":index,"raw_track_id":pose.get("id"),"segment":segment,"bbox":pose["bb"],
                         "side":side(pose.get("team")),"ball":ball(by_frame[f]),"source_team":side(pose.get("team")),
                         "actor_index":actor_index,"label_source":"derived","verified":False}
                physical = temporal_actor(actor,tracks[segment],by_frame,frame,case["contact"]["type"],metadata,annotations.get("net_y"))
                distances = compare(segment,frame); available = [v[1] for v in distances.values() if v[1] is not None]
                minimum = min(available,default=None); second = sorted(available)[1] if len(available)>1 else None
                track_near = Counter(side(o["pose"].get("team")) for o in tracks[segment]).most_common(1)[0][0]=="b"
                physical["track_side_near"] = float(track_near)
                actor["score"] = physical["contact_geometry_0"] if physical["contact_geometry_0"] is not None else physical["observed_body_distance"] if physical["observed_body_distance"] is not None else 5.
                actor["uses_estimated_joints"] = physical["estimated_joint_fraction"]>0
                actor["ball_source"] = by_frame[f].get("ball_source") or ("interpolated_short" if by_frame[f].get("ball_interp") else "detected")
                for class_index,pid in enumerate(options):
                    d = distances[pid] if class_index else [None]*4
                    hsv = next((p for p in case["pairs"] if p["segment"]==segment and p["player_id"]==pid),{})
                    features = {"reference_option":float(class_index>0),"reid_local":d[0],"reid_track":d[1],"reid_mean":d[2],"reid_expanded":d[3],
                        "reid_regret":d[1]-minimum if d[1] is not None and minimum is not None else None,
                        "reid_margin":second-d[1] if d[1] is not None and second is not None else None,
                        "hsv_track":hsv.get("track_colour_pose"),"hsv_box":hsv.get("track_colour_box"),
                        "reference_track":float(plan["anchor_segments"].get(segment)==[pid]) if class_index else None,**physical}
                    if names is None: names = list(features)
                    if list(features)!=names: raise ValueError("Temporal feature order drift")
                    values.append(encode(features.values()))
                    pairs.append({**actor,"class_index":class_index,"player_id":pid,"reid_distance":d[1],"reid_expanded":d[3],
                                  "hsv_distance":hsv.get("track_colour_pose"),"track_near":bool(track_near),
                                  "appearance_margin":second-d[1] if d[1] is not None and second is not None else None})
            all_cases.append({**case,"pairs":pairs}); tensors.append(values)
        print(f"Rich feature windows {video}: complete",flush=True)
        del annotations,by_frame,observations,tracks,plan,vectors,arrays
    if [c["contact"]["id"] for c in all_cases]!=[c["contact"]["id"] for c in cases]: raise ValueError("Contact order drift")
    width = max(map(len,tensors)); x = np.zeros((len(tensors),width,len(names)*2),np.float32)
    classes = np.full((len(tensors),width),-1,np.int64)
    for n,(rows,case) in enumerate(zip(tensors,all_cases)):
        if rows: x[n,:len(rows)] = rows
        classes[n,:len(rows)] = [p["class_index"] for p in case["pairs"]]
    np.savez_compressed(output/"features.npz",pairs=x,classes=classes,targets=parent["targets"],strict=parent["strict"],
                        groups=parent["groups"],global_features=parent["global_features"])
    write_new(output/"cases.json",all_cases); write_new(output/"gallery_expansion.json",refine_audit)
    write_new(output/"manifest.json",{"schema":contract["schema"],"contract":contract,"created_at":now(),
        "base_manifest_sha256":sha(dataset/"manifest.json"),"reid_manifest_sha256":sha(reid_dir/"manifest.json"),
        "snapshot_file":snapshot_file,"snapshot_sha256":sha(output/snapshot_file),"features_sha256":sha(output/"features.npz"),"cases_sha256":sha(output/"cases.json"),
        "fields":names,"global_fields":base_manifest["global_fields"],"source_dataset":str(dataset.resolve()),
        "source_reid":str(reid_dir.resolve()),"counts":{"contacts":len(cases),"strict":int(parent["strict"].sum()),"max_pairs":width,"feature_channels":len(names)*2}})
    return output
