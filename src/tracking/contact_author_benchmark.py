"""Frozen person-author benchmark from human decisions and cached observations.

Reference photos are setup inputs, never contact targets. Numeric features omit
roster numbers, raw track IDs, source names and absolute contact times. Colour
descriptors decode existing video pixels; no learned model runs during capture.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

from action_recognition.annotator.contact_author_dataset import AnnotationStore, now, sha, write_new
from src.tracking.contact_author_features import ACTOR_FIELDS, _temporal_metrics, extract_contact, geometry
from src.tracking.contact_actor_context import action_evidence
from src.tracking.player_contacts import side
from src.tracking.player_identity import appearance, overlap

SCHEMA = "human_contact_author_benchmark_v1"
PAIR_FIELDS = ("colour_box", "colour_pose", "track_colour_box", "track_colour_pose", "anchor_track",
               "source_side_near", "track_side_near", "action_supported", "action_distance", "arm_raise",
               "arm_speed", "occlusion") + ACTOR_FIELDS


def iou(a,b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def unit(value):
    value = np.asarray(value,dtype=np.float32)
    return value/max(float(np.linalg.norm(value)),1e-8)


def pose_colour(image,pose):
    """Three HSV regions inside observed torso/shorts, with explicit missingness."""
    kp = pose.get("kp") or []
    if len(kp)!=17 or any(kp[j][2]<.4 for j in (5,6,11,12,13,14)): return None
    h,w = image.shape[:2]
    top = np.array([kp[j][:2] for j in (5,6)],np.float32)
    hip = np.array([kp[j][:2] for j in (11,12)],np.float32)
    knee = np.array([kp[j][:2] for j in (13,14)],np.float32)
    # Shrink sideways to reject arms/background; shorts use only upper thighs.
    def shrink(pair): return pair.mean(axis=0)+(pair-pair.mean(axis=0))*.72
    top,hip,knee = map(shrink,(top,hip,knee))
    mid = (top+hip)/2; bottom = hip+(knee-hip)*.45
    values = []
    for first,second in ((top,mid),(mid,hip),(hip,bottom)):
        polygon = np.array([first[0],first[1],second[1],second[0]],np.int32)
        x1,y1 = np.maximum(polygon.min(axis=0),0); x2,y2 = np.minimum(polygon.max(axis=0)+1,[w,h])
        if min(x2-x1,y2-y1)<3: return None
        mask = np.zeros((y2-y1,x2-x1),np.uint8)
        cv2.fillConvexPoly(mask,polygon-[x1,y1],255)
        if cv2.countNonZero(mask)<15: return None
        hsv = cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],mask,[8,4,4],[0,180,0,256,0,256]).ravel()
        values.extend(np.sqrt(hist/max(float(hist.sum()),1.)).tolist())
    return unit(values)


def track_segments(frames,fps):
    """Split local IDs at gaps/teleports; never equate raw ID with a person."""
    last = {}; observations = {}; tracks = defaultdict(list)
    for row in frames:
        f = row["f"]
        for index,p in enumerate(row.get("poses") or []):
            raw = p.get("id")
            if raw is None: key = f"single:{f}:{index}"
            else:
                previous = last.get(raw); bb = p["bb"]
                centre = np.array([(bb[0]+bb[2])/2,(bb[1]+bb[3])/2])
                continuous = previous and 0<f-previous[0]<=round(.30*fps) and np.linalg.norm(centre-previous[1])<max(1,bb[3]-bb[1])
                key = previous[2] if continuous else f"{raw}:{f}"
                last[raw] = (f,centre,key)
            obs = {"frame":f,"index":index,"pose":p,"segment":key}
            observations[(f,index)] = obs; tracks[key].append(obs)
    return observations,tracks


def distance(vector,gallery):
    return float(1-max(float(np.dot(vector,g)) for g in gallery)) if vector is not None and gallery else None


def encode(values):
    return [v for value in values for v in
            (float(np.clip(value,-5,5)) if value is not None and np.isfinite(value) else 0.,
             float(value is not None and np.isfinite(value)))]


def candidate_features(actor,by_frame,fps,event_type):
    row = by_frame[actor["frame"]]; pose = row["poses"][actor["pose_index"]]
    evidence = action_evidence(pose,row["ball"],event_type)
    speed = None; wrist = evidence.get("wrist_joint")
    if wrist in (9,10):
        shoulder = wrist-4; samples = []
        for d in (-3,3):
            others = [p for p in by_frame.get(actor["frame"]+d,{}).get("poses",[]) if p.get("id")==pose.get("id")]
            if len(others)!=1: break
            kp = others[0]["kp"]
            if min(kp[wrist][2],kp[shoulder][2])<.25: break
            samples.append(np.array(kp[wrist][:2])-kp[shoulder][:2])
        if len(samples)==2: speed = float(np.linalg.norm(samples[1]-samples[0])*fps/(6*max(1,pose["bb"][3]-pose["bb"][1])))
    contamination = max((overlap(pose["bb"],p["bb"]) for i,p in enumerate(row["poses"]) if i!=actor["pose_index"]),default=0.)
    return [float(evidence["supported"]),evidence.get("distance_body_heights"),evidence.get("arm_raise"),speed,contamination]


def capture_colours(metadata,requests,cache_path):
    """Read only requested saved-pose frames; colour features, not model inference."""
    cap = cv2.VideoCapture(metadata["video_source"]["path"]); descriptors = {}
    if not cap.isOpened(): raise ValueError("Original source unavailable")
    try:
        previous = -100
        for number,f in enumerate(sorted(requests)):
            if 0<f-previous<=20:
                for _ in range(f-previous): ok,image = cap.read()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,image = cap.read()
            if not ok: raise ValueError(f"Cannot decode source frame {f}")
            previous = f
            # Histograms need colour/body shape, not 4K pixel count.
            image = cv2.resize(image,(metadata["width"]//2,metadata["height"]//2))
            for key,pose in requests[f]:
                scaled = {**pose,"bb":[v/2 for v in pose["bb"]],
                          "kp":[[p[0]/2,p[1]/2,p[2]] for p in pose.get("kp",[])]}
                box = appearance(image,scaled["bb"]); guided = pose_colour(image,scaled)
                descriptors[key] = (unit(box) if box is not None else None,guided)
            if number%25==0 or number==len(requests)-1:
                print(f"Colour cache {metadata['video_id']}: {number+1}/{len(requests)} frames",flush=True)
    finally: cap.release()
    keys = list(descriptors)
    np.savez_compressed(cache_path,keys=np.array(keys),
        box=np.stack([descriptors[k][0] if descriptors[k][0] is not None else np.zeros(384) for k in keys]),
        pose=np.stack([descriptors[k][1] if descriptors[k][1] is not None else np.zeros(384) for k in keys]),
        valid=np.array([[v is not None for v in descriptors[k]] for k in keys]),
        source_sha256=np.array(metadata["video_source"]["sha256"]),schema=np.array("hsv_box_pose_v1"))
    return descriptors


def prepare(project_dir,output):
    output = Path(output)
    if output.exists(): raise ValueError("Benchmark output must be a fresh directory")
    store = AnnotationStore(project_dir); store.check_sources(full=True)
    snapshot = store.export(); output.mkdir(parents=True)
    write_new(output/"human_labels.json",snapshot)
    rows = snapshot["contacts"]; all_x = []; all_global = []; all_meta = []; global_names = None
    contract = {"manifest":str((output/"human_labels.json").resolve()),"features":str((output/"features.npz").resolve()),
        "source_videos":sorted(snapshot["videos"]),"human_decisions":len(snapshot["decisions"]),
        "eligible_contacts":len(rows),"near_person_contacts":sum(c["author"]["decision"]=="player" for c in rows),
        "schema":SCHEMA,"label_provenance":"human contact-author annotations, immutable revision snapshot",
        "split_policy":"leave one complete source_video out; no contact-label access in test-source training",
        "model_inference":False,"photo_setup_anchors_allowed":True,
        "strict_score":"verified contact timing and farther than 60 frames from every roster reference",
        "evaluation_scope":"known contact timing/type; offline future frames; not end-to-end contact spotting"}
    print(json.dumps(contract,ensure_ascii=False,indent=2),flush=True)
    for video,metadata in sorted(snapshot["videos"].items()):
        annotations = json.loads(Path(metadata["annotations_source"]["path"]).read_bytes())
        frames = annotations["frames"]; by_frame = {r["f"]:r for r in frames}
        observations,tracks = track_segments(frames,metadata["fps"])
        contacts = [c for c in rows if c["video_id"]==video]
        roster = snapshot["rosters"][video]["players"]
        requests = defaultdict(list); anchors = defaultdict(set); reference_keys = defaultdict(list)
        for player in roster:
            for n,ref in enumerate(player["references"]):
                key = f"ref:{player['player_id']}:{n}"
                poses = by_frame.get(ref["frame"],{}).get("poses") or []
                ranked = sorted([(iou(ref["bbox"],p["bb"]),i,p) for i,p in enumerate(poses)],reverse=True,key=lambda x:x[0])
                matched = ranked and ranked[0][0]>=.35 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>=.1)
                pose = deepcopy(ranked[0][2]) if matched else {"kp":[]}
                pose["bb"] = ref["bbox"]
                requests[ref["frame"]].append((key,pose)); reference_keys[player["player_id"]].append(key)
                if matched: anchors[observations[(ref["frame"],ranked[0][1])]["segment"]].add(player["player_id"])
        times = sorted(c["frame"] for c in snapshot["queue"] if c["video_id"]==video)
        actor_rows = []; globals_video = []
        for c in contacts:
            neighbour = min((abs(f-c["frame"]) for f in times if f!=c["frame"]),default=100)
            global_x,names,actors = extract_contact({"frame":c["frame"],"type":c["type"]},by_frame,metadata["fps"],
                                                   (metadata["width"],metadata["height"]),neighbour)
            if global_names is not None and names!=global_names: raise ValueError("Feature schema drift")
            global_names = names; globals_video.append(global_x)
            for actor in actors:
                obs = observations[(actor["frame"],actor["pose_index"])]; actor["segment"] = obs["segment"]
                track = tracks[obs["segment"]]
                nearest = min(track,key=lambda o:abs(o["frame"]-c["frame"]))
                actor["colour_key"] = f"obs:{nearest['frame']}:{nearest['index']}"
                requests[nearest["frame"]].append((actor["colour_key"],nearest["pose"]))
            # Recover existing actor vector without manufacturing identities.
            for actor in actors:
                row = by_frame[actor["frame"]]; p = row["poses"][actor["pose_index"]]
                heights = [q["bb"][3]-q["bb"][1] for q in row["poses"]]
                g = geometry(p,row["ball"],float(np.median(heights)))
                actor["vector"] = g["values"]+[(actor["frame"]-c["frame"])/metadata["fps"]]+_temporal_metrics(by_frame,actor["frame"],p,g,float(np.median(heights)),metadata["fps"])
                actor["action"] = candidate_features(actor,by_frame,metadata["fps"],c["type"])
            actor_rows.append(actors)
        # Repeated candidates request the same source observation only once.
        requests = {f:list(dict(items).items()) for f,items in requests.items()}
        colours = capture_colours(metadata,requests,output/f"{video}_colours.npz")
        galleries = {pid:[[colours[k][v] for k in keys if colours[k][v] is not None] for v in (0,1)] for pid,keys in reference_keys.items()}
        track_colours = defaultdict(lambda:[[],[]])
        for key,(a,b) in colours.items():
            if not key.startswith("obs:"): continue
            _,f,i = key.split(":"); obs = observations[(int(f),int(i))]
            contamination = max((overlap(obs["pose"]["bb"],p["bb"]) for j,p in enumerate(by_frame[int(f)]["poses"]) if j!=int(i)),default=0.)
            if contamination>.35: continue
            for v,d in enumerate((a,b)):
                if d is not None: track_colours[obs["segment"]][v].append(d)
        pooled = {key:[unit(np.mean(v,axis=0)) if v else None for v in views] for key,views in track_colours.items()}
        side_votes = {key:Counter(side(o["pose"].get("team")) for o in track).most_common(1)[0][0] for key,track in tracks.items()}
        for c,actors,gx in zip(contacts,actor_rows,globals_video):
            # One class for the opposite team, then actual roster IDs. Padded later.
            options = ["other_team"]+[p["player_id"] for p in roster]
            pair_rows = []; pair_meta = []
            for actor in actors:
                colour = colours[actor["colour_key"]]; average = pooled.get(actor["segment"],[None,None])
                ids = anchors.get(actor["segment"],set())
                for class_index,pid in enumerate(options):
                    # Observed opposite-team actors explain only the other-team
                    # class; all near-side reference people compete for near poses.
                    near = side_votes[actor["segment"]]=="b"
                    if (class_index==0 and near) or (class_index>0 and not near): continue
                    app = [distance(colour[v],galleries[pid][v]) for v in (0,1)] if class_index else [None,None]
                    track_app = [distance(average[v],galleries[pid][v]) for v in (0,1)] if class_index else [None,None]
                    values = app+track_app+[float(ids=={pid}) if ids else None,
                             float(actor["side"]=="b"),float(near)]+actor["action"]+actor["vector"]
                    assert len(values)==len(PAIR_FIELDS)
                    pair_rows.append(encode(values)); pair_meta.append({k:v for k,v in actor.items() if k not in ("vector","action","colour_key")}|{
                        "class_index":class_index,"player_id":pid,"colour_box":app[0],"colour_pose":app[1],
                        "track_colour_box":track_app[0],"track_colour_pose":track_app[1],"anchor_track":bool(ids=={pid}),
                        "action_supported":bool(actor["action"][0]),"action_distance":actor["action"][1]})
            target = c["author"]["player_id"] or "other_team"
            strict = c["timing_verified"] and all(abs(c["frame"]-r["frame"])>60 for p in roster for r in p["references"])
            all_x.append(pair_rows); all_global.append(gx); all_meta.append({"contact":c,"options":options,"target":options.index(target),
                "strict":bool(strict),"pairs":pair_meta})
        print(f"Prepared {video}: {len(contacts)} labelled contacts; {sum(bool(a) for a in actor_rows)} with ball/pose candidates",flush=True)
        del annotations,observations,tracks,by_frame,frames
    max_pairs = max(map(len,all_x)); dim = len(PAIR_FIELDS)*2
    x = np.zeros((len(all_x),max_pairs,dim),np.float32); classes = np.full((len(all_x),max_pairs),-1,np.int64)
    for n,(values,meta) in enumerate(zip(all_x,all_meta)):
        if values: x[n,:len(values)] = values
        classes[n,:len(values)] = [p["class_index"] for p in meta["pairs"]]
    np.savez_compressed(output/"features.npz",pairs=x,classes=classes,global_features=np.stack(all_global),
                        targets=np.array([m["target"] for m in all_meta]),groups=np.array([m["contact"]["video_id"] for m in all_meta]),
                        strict=np.array([m["strict"] for m in all_meta]))
    write_new(output/"cases.json",all_meta)
    write_new(output/"manifest.json",{"schema":SCHEMA,"created_at":now(),"contract":contract,
        "snapshot_sha256":sha(output/"human_labels.json"),"features_sha256":sha(output/"features.npz"),
        "cases_sha256":sha(output/"cases.json"),"pair_fields":PAIR_FIELDS,"global_fields":global_names,
        "counts":{"contacts":len(all_meta),"strict":sum(m["strict"] for m in all_meta),"pairs_max":max_pairs},
        "appearance":"HSV bounding-box vs keypoint-guided torso/shorts, plus clean raw-track aggregation; no OSNet available for these sources",
        "numeric_ids_used":False,"future_frames":True})
    return output


def prepare_central(dataset,output):
    """Known-time actor ablation, keeping all learned numeric features frozen."""
    dataset,output = Path(dataset),Path(output)
    if output.exists(): raise ValueError("Actor context output must be fresh")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    if sha(dataset/"cases.json")!=manifest["cases_sha256"] or sha(dataset/"human_labels.json")!=manifest["snapshot_sha256"]:
        raise ValueError("Frozen inputs changed")
    cases = json.loads((dataset/"cases.json").read_bytes()); snapshot = json.loads((dataset/"human_labels.json").read_bytes())
    result = []
    print(json.dumps({"source_manifest":str((dataset/"manifest.json").resolve()),"source_features":str((dataset/"features.npz").resolve()),
        "source_videos":sorted(snapshot["videos"]),"schema":"contact_author_central_actor_v1",
        "label_provenance":"same frozen human snapshot; author targets never enter candidate selection",
        "change":"closest observed pose to audited contact time per continuous track; neighbouring frames only for missing pose/ball",
        "numeric_feature_refresh":False,"model_inference":False,"training":False},indent=2),flush=True)
    for video,metadata in sorted(snapshot["videos"].items()):
        path = Path(metadata["annotations_source"]["path"])
        if sha(path)!=metadata["annotations_source"]["sha256"]: raise ValueError("Source observations changed")
        annotations = json.loads(path.read_bytes()); frames = annotations["frames"]; by_frame = {r["f"]:r for r in frames}
        _,tracks = track_segments(frames,metadata["fps"])
        times = [c["frame"] for c in snapshot["queue"] if c["video_id"]==video]
        for case in (c for c in cases if c["contact"]["video_id"]==video):
            c = case["contact"]; radius = round(.2*metadata["fps"])
            neighbour = min((abs(f-c["frame"]) for f in times if f!=c["frame"]),default=100)
            radius = min(radius,max(0,(neighbour-1)//2)); chosen = {}; pairs = []
            for pair in case["pairs"]:
                segment = pair["segment"]
                if segment not in chosen:
                    valid = []
                    for obs in tracks[segment]:
                        f = obs["frame"]; row = by_frame[f]
                        if abs(f-c["frame"])>radius or row.get("ball_out_of_frame") or row.get("ball") is None: continue
                        if any(t not in by_frame or by_frame[t].get("ball_out_of_frame") or
                               t>min(f,c["frame"]) and by_frame[t].get("ball_new") for t in range(min(f,c["frame"]),max(f,c["frame"])+1)): continue
                        valid.append(obs)
                    chosen[segment] = min(valid,key=lambda o:(abs(o["frame"]-c["frame"]),o["frame"])) if valid else None
                obs = chosen[segment]
                if obs is None: continue
                f = obs["frame"]; row = by_frame[f]; pose = obs["pose"]
                heights = [p["bb"][3]-p["bb"][1] for p in row["poses"]]
                g = geometry(pose,row["ball"],float(np.median(heights)))
                if g is None: continue
                actor = {**deepcopy(pair),"frame":f,"pose_index":obs["index"],"raw_track_id":pose.get("id"),
                         "score":float(g["cost"]+.10*abs(f-c["frame"])/max(1,radius)),"bbox":pose["bb"],"ball":row["ball"],
                         "distance_px":g["distance_px"],"distance_body_heights":g["distance_px"]/max(1,pose["bb"][3]-pose["bb"][1]),
                         "contact_part":g["part"],"offset_seconds":(f-c["frame"])/metadata["fps"],
                         "ball_source":row.get("ball_source") or ("interpolated_short" if row.get("ball_interp") else "detected"),
                         "actor_time_method":"closest_observed_to_audited_time"}
                action = candidate_features(actor,by_frame,metadata["fps"],c["type"])
                actor.update(action_supported=bool(action[0]),action_distance=action[1])
                pairs.append(actor)
            result.append({"id":c["id"],"pairs":pairs})
        print(f"Central-time context {video} complete",flush=True)
        del annotations,frames,by_frame,tracks
    write_new(output,{"schema":"contact_author_central_actor_v1","source_manifest_sha256":sha(dataset/"manifest.json"),
        "source_cases_sha256":manifest["cases_sha256"],"source_features_sha256":manifest["features_sha256"],
        "created_at":now(),"cases":result,"targets_used_for_selection":False})
    return output
