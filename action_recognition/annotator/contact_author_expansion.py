"""Prepare blind author annotation of new sources from saved model analyses.

No inference, source rewrite, training, or automatic author decisions. Model
contact time/type remain unverified even after a human assigns the author.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2

from .contact_author_dataset import SCHEMA,AnnotationStore,fingerprint_file,initial_roster,now,sha,write_new
from src.tracking.player_contacts import CONTACT_TYPES,side


def excluded_sources(manifest,author_snapshot):
    """Collect source names, file names and hashes; not adjacent-clip IDs."""
    names=set();hashes=set()
    for clip in manifest.get("clips",[]):
        for field in ("source_stem","source_video"):
            value=clip.get(field)
            if isinstance(value,str):names.add(Path(value.replace("\\","/")).stem)
    for video,metadata in author_snapshot.get("videos",{}).items():
        names.add(video);source=metadata.get("video_source") or {}
        if source.get("sha256"):hashes.add(source["sha256"])
        if source.get("path"):names.add(Path(source["path"].replace("\\","/")).stem)
    if not names:raise ValueError("Dataset exclusions contain no source videos")
    return names,hashes


def model_contacts(stem,events,available_frames):
    """Merge exact-frame model duplicates, retain all source event indices/types."""
    groups={};excluded=[]
    for index,event in enumerate(events):
        kind=event.get("type")
        if kind not in CONTACT_TYPES:
            excluded.append({"video_id":stem,"event_index":index,"reason":"not_a_contact_type","type":kind});continue
        frame=event.get("frame")
        if type(frame) is not int or frame not in available_frames:raise ValueError("Model contact outside saved source frames")
        key=(stem,frame)
        if key not in groups:
            groups[key]={"id":f"{stem}:model:f{frame}:"+hashlib.sha256(str(key).encode()).hexdigest()[:10],
                         "video_id":stem,"frame":frame,"type":kind,"raw_types":[],"event_indices":[],"model_predictions":[],
                         "clip_ids":[],"annotation_ids":[],"original_frames":[frame],"review_ids":[],
                         "timing_verified":False,"timing_source":"saved_contact_spotter_proposal",
                         "event_label_source":"model","event_verified":False,"author_label_source":None}
        row=groups[key];row["event_indices"].append(index);row["raw_types"].append(kind)
        row["model_predictions"].append({k:deepcopy(event.get(k)) for k in ("type","core_type","confidence","source")})
    result=sorted(groups.values(),key=lambda row:row["frame"])
    for row in result:
        row["raw_types"]=sorted(set(row["raw_types"]));row["type"]=" / ".join(row["raw_types"])
    return result,excluded


def proposed_roster(annotations,first_contact):
    """Retain saved roster numbering/substitute references as unconfirmed proposals."""
    identity=annotations.get("player_identity") or {};galleries=identity.get("gallery_keys") or {}
    by_frame={r["f"]:r for r in annotations["frames"]};players=[]
    if side(identity.get("side"))=="b" and len(galleries)>=6:
        seen=set()
        for pid,keys in galleries.items():
            if not keys or not pid.startswith("P") or not pid[1:].isdigit():break
            try:
                frame,index=map(int,keys[0].split(":"));pose=by_frame[frame]["poses"][index]
            except (KeyError,IndexError,ValueError):break
            if side(pose.get("team"))!="b" or (frame,index) in seen:break
            seen.add((frame,index));players.append({"player_id":pid,"identity_confirmation":None,
                "references":[{"frame":frame,"bbox":pose["bb"],"raw_track_id":pose.get("id"),"label_source":"model",
                               "scope":"identity_reference_only","reference_source":"saved_roster_gallery"}]})
        else:
            return {"confirmed":False,"players":sorted(players,key=lambda p:int(p["player_id"][1:])),
                    "proposal_source":"saved_roster; identities still require human confirmation"}
    # No prior persistent IDs are needed: six observed boxes supply photo cards.
    try:return initial_roster(annotations,first_contact)
    except ValueError:
        complete=[r["f"] for r in annotations["frames"] if sum(side(p.get("team"))=="b" for p in r.get("poses") or [])==6]
        if not complete:raise ValueError("No complete near-side cached roster frame; cannot fabricate player references")
        return initial_roster(annotations,min(complete,key=lambda f:abs(f-first_contact)))


def prepare_expansion(stats_paths,exclude_manifest,exclude_authors,output):
    output=Path(output)
    if output.exists():raise ValueError("Choose a fresh expansion project directory")
    exclude_manifest,exclude_authors=Path(exclude_manifest),Path(exclude_authors)
    manifest=json.loads(exclude_manifest.read_bytes());authors=json.loads(exclude_authors.read_bytes())
    names,hashes=excluded_sources(manifest,authors)
    stats_paths=[Path(p).resolve() for p in stats_paths]
    if not stats_paths or len(set(stats_paths))!=len(stats_paths):raise ValueError("Supply distinct saved stats files")
    exclusion_files=[fingerprint_file(exclude_manifest),fingerprint_file(exclude_authors)]
    contract={"preparation":"contact_author_new_sources_v1","source_results":[str(p) for p in stats_paths],
              "excluded_event_manifest":str(exclude_manifest.resolve()),"excluded_author_snapshot":str(exclude_authors.resolve()),
              "excluded_source_names":sorted(names),"label_provenance":"Unverified saved model contact proposals; human author labels start empty",
              "split_policy":"Keep complete source videos separate; no train/test assignment or training promotion now",
              "new_feature_or_model_training":False,"model_inference":False,"source_rewrite":False,
              "identity_proposals":"Saved roster galleries when available; otherwise six cached near-side poses; all require confirmation"}
    print(json.dumps(contract,indent=2),flush=True)
    sources=list(exclusion_files);videos={};contacts=[];excluded=[];summary={};seen_hashes=set()
    for path in stats_paths:
        if not path.name.endswith("_stats.json"):raise ValueError("Expected a saved *_stats.json result")
        annotation=path.with_name(path.name.replace("_stats.json","_annotations.json"))
        stats=json.loads(path.read_bytes());ann=json.loads(annotation.read_bytes())
        if not (stats.get("contact_spotter") or {}).get("enabled"):
            raise ValueError("Use an existing contact-spotter result, not silent FSM fallback")
        video=Path(ann["original_video_path"]).resolve();stem=video.stem
        if stem in names:raise ValueError("Source is already in an excluded dataset: "+stem)
        if stem in videos:raise ValueError("Duplicate source video: "+stem)
        video_source=fingerprint_file(video)
        if video_source["sha256"] in hashes:raise ValueError("Same excluded source content under another name: "+stem)
        if video_source["sha256"] in seen_hashes:raise ValueError("Duplicate video content under another name: "+stem)
        seen_hashes.add(video_source["sha256"])
        frames=ann["frames"];available={r["f"] for r in frames}
        rows,dropped=model_contacts(stem,stats["all_events"],available)
        if not rows:raise ValueError("No saved contact proposals: "+stem)
        if not any(r.get("poses") for r in frames):raise ValueError("Saved player poses are missing: "+stem)
        cap=cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():raise ValueError("Cannot open original video: "+stem)
            width,height=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (width,height)!=(ann["frame_width"],ann["frame_height"]) or abs(cap.get(cv2.CAP_PROP_FPS)-ann["fps"])>.05:
                raise ValueError("Cached geometry differs from original video: "+stem)
        finally:cap.release()
        roster=proposed_roster(ann,rows[0]["frame"]);ann_source=fingerprint_file(annotation);stats_source=fingerprint_file(path)
        sources.extend((ann_source,stats_source));contacts.extend(rows);excluded.extend(dropped)
        videos[stem]={"video_id":stem,"name":stem,"fps":ann["fps"],"width":width,"height":height,"num_frames":max(available)+1,
                      "video_source":video_source,"annotations_source":ann_source,"stats_source":stats_source,
                      "target_side":"b","initial_roster":roster,"contact_count":len(rows)}
        summary[stem]={"contacts":len(rows),"proposed_people":len(roster["players"]),"duration_minutes":round((max(available)+1)/ann["fps"]/60,2),
                       "per_type":dict(Counter(r["type"] for r in rows))}
        print(f"Prepared {stem}: {len(rows)} contacts / {len(roster['players'])} unconfirmed player cards",flush=True)
        del ann,stats,frames,available
    contract["contacts"]=len(contacts);contract["videos"]=summary
    output.mkdir(parents=True)
    project={"schema":SCHEMA,"created_at":now(),"source_manifest_sha256":sha(exclude_manifest),"source_scope":contract,
             "sources":sources,"videos":videos,"contacts":contacts,"excluded":excluded,"initial_author_labels":0,"training":False,
             "evaluation_scope":"new_author_sources_outside_active_event_training_and_author_gold; contact timing/type remain model proposals",
             "roster_policy":"per-video confirmed people; automatic photo proposals, explicit substitutes; returning people retain their IDs"}
    write_new(output/"project.json",project);AnnotationStore(output).initialize()
    write_new(output/"preparation_report.json",{"created_at":now(),"contacts":len(contacts),"videos":summary,
              "excluded_training_source_names":sorted(names),"model_inference":False,"human_labels":0,
              "provenance_note":"Author decisions do not verify model event time/type. No training promotion."})
    print(json.dumps({"project":str(output.resolve()),"videos":len(videos),"contacts":len(contacts),"human_labels":0},indent=2),flush=True)
    return output
