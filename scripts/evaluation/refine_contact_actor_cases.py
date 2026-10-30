"""Replay observed actor/action context and cached ReID against human P4 cases.

Only a positive identity reference enters the algorithm. Negative author/ID
cases are evaluation checks, never frame-specific vetoes in the scorer.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from scripts.evaluation.run_contact_author_learning import digest,write
from src.tracking.contact_actor_context import ActorContext,ActorContextConfig,build_action_report
from src.tracking.player_contacts import source_fingerprint


def load_appearance(directory,annotations,review):
    manifest = json.loads((directory/"manifest.json").read_bytes())
    if manifest.get("schema") != "player_reid_embedding_cache_v1":
        raise ValueError("Unknown cached embedding schema")
    for name,key in (("embeddings.npz","embedding_sha256"),("observations.json","observations_sha256")):
        if digest((directory/name).read_bytes()) != manifest[key]:
            raise ValueError("Frozen ReID cache changed: "+name)
    observations = json.loads((directory/"observations.json").read_bytes())
    by_frame = {row["f"]:row for row in annotations["frames"]}
    boxes = {}
    for row in observations["frames"]:
        current = by_frame[row["f"]].get("poses") or []
        for p in row["players"]:
            i = p["pose_index"]
            if i >= len(current) or current[i]["id"] != p["raw_id"] or current[i]["bb"] != p["bb"]:
                raise ValueError("ReID observation key no longer describes this source pose")
            boxes[p["key"]] = p
    with np.load(directory/"embeddings.npz",allow_pickle=False) as saved:
        keys = saved["keys"].tolist(); matrix = saved["embeddings"].copy()
    if len(set(keys)) != len(keys) or set(keys) != set(boxes) or len(matrix) != manifest["count"]:
        raise ValueError("ReID keys/count changed")
    norms = np.linalg.norm(matrix,axis=1,keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms<=1e-8): raise ValueError("Invalid cached embedding")
    matrix /= norms
    lookup = {key:i for i,key in enumerate(keys)}
    anchors = [case["identity_anchor"] for case in review["cases"] if case.get("author_confirmed")=="P4"]
    if len(anchors) != 1: raise ValueError("Exactly one explicit positive P4 identity anchor required")
    anchor = anchors[0]; row = by_frame[anchor["frame"]]; pose = row["poses"][anchor["pose_index"]]
    if pose["id"] != anchor["raw_track_id"]: raise ValueError("Human anchor bound to another pose")
    # Nearby views inherit only model raw-track continuity, not human labels.
    references = []
    for f in range(anchor["frame"]-6,anchor["frame"]+7):
        for i,p in enumerate(by_frame.get(f,{}).get("poses") or []):
            key = f"{f}:{i}"
            if p["id"]==anchor["raw_track_id"] and key in lookup:
                references.append((abs(f-anchor["frame"]),key))
    reference_keys = [key for _,key in sorted(references)[:8]]
    if f'{anchor["frame"]}:{anchor["pose_index"]}' not in reference_keys: raise ValueError("Anchor embedding missing")
    gallery = matrix[[lookup[k] for k in reference_keys]]
    distance = np.min(np.clip(1-matrix@gallery.T,0,2),axis=1)
    other_keys = [k for pid,values in annotations["player_identity"]["gallery_keys"].items() if pid != "P4" for k in values]
    if not all(k in lookup for k in other_keys): raise ValueError("Competing roster galleries missing")
    other_distance = np.min(np.clip(1-matrix@matrix[[lookup[k] for k in other_keys]].T,0,2),axis=1)
    evidence = {key:{"distance":float(d),"other_distance":float(o)} for key,d,o in zip(keys,distance,other_distance)}
    return evidence,{"cache_manifest_sha256":digest((directory/"manifest.json").read_bytes()),
                     "embedding_sha256":manifest["embedding_sha256"],"observations_sha256":manifest["observations_sha256"],
                     "count":len(keys),"positive_anchor":anchor,"reference_keys":reference_keys,
                     "gallery_label_source":"human anchor plus derived adjacent views","model_inference":False}


def overlays(context,use_appearance):
    result = {}; changed = 0
    for f,row in context.frames.items():
        updates = []
        resolved = context.frame_context(f,use_appearance,p4_only=True)
        for i,(pose,value) in enumerate(zip(row.get("poses") or [],resolved)):
            original_side = {"team_a":"a","team_b":"b","a":"a","b":"b"}.get(pose.get("team"))
            if pose.get("player_id") != value["player_id"] or original_side != value["side"]:
                updates.append({"pose_index":i,"raw_track_id":pose.get("id"),"player_id":value["player_id"],"side":value["side"],
                                "source_player_id":pose.get("player_id"),"source_side":original_side,
                                "method":value["identity_method"],"label_source":"derived","verified":False})
        if updates: result[str(f)] = updates; changed += len(updates)
        if f and f%4000==0: print(f"Derived overlay {f}/{len(context.frames)} frames",flush=True)
    return result,changed


def case_scores(cases,events,context,use_context,use_appearance):
    rows = []
    for case in cases:
        event = events[case["event_index"]]
        verdicts = {}
        if case.get("author_excluded"):
            forbidden = case["author_excluded"]
            verdicts["wrong_author_removed"] = event.get("player_id") != forbidden and event.get("proposed_player_id") != forbidden
        if case.get("author_confirmed"):
            verdicts["positive_contact_preserved"] = event.get("player_id") == case["author_confirmed"]
        if case.get("team_confirmed"):
            verdicts["opponent_action_recognized"] = event.get("status")=="other_side" and event.get("side_proposal")==case["team_confirmed"]
        if case.get("identity_excluded"):
            label = case["identity_excluded"]
            state = (context.frame_context(label["frame"],use_appearance)[label["pose_index"]] if use_context
                     else context.resolve(label["frame"],label["pose_index"],False,False))
            verdicts["wrong_skeleton_identity_removed"] = state["player_id"] != label["player_id"]
        rows.append({**case,"passed":all(verdicts.values()),"checks":verdicts,
                     "actual_player_id":event.get("player_id"),"actual_status":event.get("status"),
                     "actual_raw_track_id":event["candidates"][0]["raw_track_id"] if event.get("candidates") else None,
                     "actual_contact_frame":event.get("contact_frame"),"reason":event["reason"],
                     "identity_attribution_is_human_verified":False})
    return rows


def run(args):
    if args.output.exists(): raise ValueError("Use a fresh output; old results remain intact")
    stem = args.annotations.name.removesuffix("_annotations.json")
    stat_path = args.annotations.with_name(stem+"_stats.json")
    ann_raw,stat_raw = args.annotations.read_bytes(),stat_path.read_bytes()
    source = source_fingerprint(stat_raw,ann_raw)
    review = json.loads(args.reviews.read_bytes())
    bundle = json.loads(args.base_lab.read_bytes())
    if source != review["source"] or source != bundle["source"]: raise ValueError("Case/lab analysis source changed")
    if review.get("schema") != "player_contact_case_review_v1" or review.get("label_source") != "human":
        raise ValueError("Expected explicit human case review")
    if not all(c.get("verified") is True for c in review["cases"]): raise ValueError("Unverified case review")
    ann,stats = json.loads(ann_raw),json.loads(stat_raw)
    for case in review["cases"]:
        event = stats["all_events"][case["event_index"]]
        if event["frame"] != case["source_frame"] or event["type"] != case["type"]:
            raise ValueError("Reviewed event changed")
    config = ActorContextConfig()
    contract = {"schema":"P4_actor_context_experiment_v1","manifest_path":str((args.output/"manifest.json").resolve()),
                "source":source,"reviews":str(args.reviews.resolve()),"reviews_sha256":digest(args.reviews.read_bytes()),
                "base_lab_sha256":digest(args.base_lab.read_bytes()),"feature_path":str(args.reid_cache.resolve()),
                "source_video":ann["video"],"case_count":len(review["cases"]),"label_provenance":"explicit user feedback",
                "positive_identity_references":1,"negative_cases_used_for_inference":False,
                "split_policy":"five development examples; not held-out accuracy or a training dataset",
                "configuration":asdict(config),"video_inference":False,"training":False,
                "code_sha256":digest((ROOT/"src/tracking/contact_actor_context.py").read_bytes())}
    print("VERIFIED FROZEN-REPLAY CONTRACT\n"+json.dumps(contract,indent=2),flush=True)
    args.output.mkdir(parents=True); write(args.output/"manifest.json",contract)
    appearance,provenance = load_appearance(args.reid_cache,ann,review)
    write(args.output/"cached_appearance_provenance.json",provenance)
    context = ActorContext(ann,appearance,config)
    prior = [e["near_side_score"] for e in bundle["methods"]["spotter_sequence"]["events"]]
    variants = (("action_only",False,False),("action_temporal",True,False),("action_P4_reference",True,True))
    reports = {}; case_results = {}; pose_contexts = {}
    for name,temporal,anchored in variants:
        report = build_action_report(stats["all_events"],context,prior,ann["player_identity"],name,temporal,anchored)
        cases = case_scores(review["cases"],report["contacts"],context,temporal,anchored)
        reports[name] = report; case_results[name] = cases
        print(name,report["totals"],[(c["id"],c["passed"],c["actual_player_id"],c["actual_raw_track_id"],c["reason"]) for c in cases],flush=True)
        bundle["methods"][name] = {"events":report["contacts"],"totals":report["totals"],
            "detection_lovo":bundle["methods"]["spotter"]["detection_lovo"],
            "author_method":"observed action + bounded ground/identity context"+(" + human P4 reference" if anchored else ""),
            "person_accuracy":None,"validation_scope":"P4_cases","case_results":cases,
            "pose_context_key":name if temporal else None}
    # Only the positive reference can influence these overlays; exclusion cases
    # above are read after inference and never become frame-specific patches.
    for name,temporal,anchored in variants:
        if temporal:
            pose_contexts[name],count = overlays(context,anchored)
            print("Pose context",name,count,"derived changes; source JSON untouched",flush=True)
    bundle["pose_contexts"] = pose_contexts
    bundle["case_review"] = {"schema":review["schema"],"label_source":"human","cases":review["cases"],
                             "source_sha256":digest(args.reviews.read_bytes()),"scope":review["scope"]}
    selected = "action_P4_reference"
    bundle["selected_method"] = selected
    bundle["actor_context_experiment"] = {"manifest":str((args.output/"manifest.json").resolve()),
         "appearance_provenance":provenance,"source_poses_changed":False,"human_cases_are_development_checks":True}
    result = args.output/"viewer_result"; result.mkdir()
    (result/args.annotations.name).write_bytes(ann_raw);(result/stat_path.name).write_bytes(stat_raw)
    write(result/f"{stem}_contact_lab.json",bundle)
    write(args.output/"case_comparison.json",case_results)
    write(args.output/"summary.json",{"selected_method":selected,"case_results":case_results,
          "counts":{name:r["totals"] for name,r in reports.items()},"viewer_result":str(result),
          "limitation":"case-targeted development validation, not general contact-author accuracy"})
    if not all(c["passed"] for c in case_results[selected]):
        raise RuntimeError("One or more requested P4 development checks failed; inspect comparison, do not claim completion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--base-lab",type=Path,required=True)
    parser.add_argument("--reviews",type=Path,required=True)
    parser.add_argument("--reid-cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if not args.annotations.name.endswith("_annotations.json"): parser.error("Expected saved analysis annotations")
    run(args)


if __name__=="__main__":main()
