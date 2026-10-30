"""Compare volleyball flight, limb-interaction and sequence hypotheses on caches.

No video/model inference. New features/results/models are written only to a
fresh output. Existing human labels, analyses and checkpoints remain read-only.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.metrics import _prf, match_events
from scripts.evaluation.run_contact_author_learning import digest, feature_indices, metrics, model, write
from src.tracking.contact_author_features import contact_report, extract_events, feature_fingerprint
from src.tracking.player_contacts import source_fingerprint
from src.tracking.volleyball_contact_lab import (
    SCHEMA, EVIDENCE_NAMES, consensus, dense_evidence, joint_rank,
    learn_transitions, peaks, sequence_posteriors,
)

THRESHOLDS = (.05, .10, .15, .20, .30, .40, .50, .65)
RADII = (2, 5)


def detection(predictions, truth, valid, non_game):
    result = {"raw_events": len(predictions)}
    for name, include in (("all", valid), ("excluding_non_game", valid & ~non_game)):
        predicted = [f for f in predictions if 0 <= f < len(include) and include[f]]
        targets = [f for f in truth if include[f]]
        result[name] = {f"at_{t}": _prf(*match_events(predicted, targets, t)) for t in (2, 5, 10)}
        result[name]["scored_events"] = len(predicted)
    return result


def pooled(rows):
    return {scope: {f"at_{t}": _prf(*(sum(row[scope][f"at_{t}"][k] for row in rows)
                                       for k in ("tp", "fp", "fn"))) for t in (2, 5, 10)}
            for scope in ("all", "excluding_non_game")}


def tune(videos, column):
    trials = []
    for threshold in THRESHOLDS:
        for radius in RADII:
            rows = [detection(peaks(v["evidence"][:, column], threshold, radius),
                              v["truth"], v["valid"], v["non_game"]) for v in videos]
            quality = pooled(rows)["all"]["at_5"]
            trials.append((quality["f1"], quality["precision"], threshold, radius))
    _, _, threshold, radius = max(trials)
    return {"threshold": threshold, "radius": radius}


def retime(events, evidence):
    """Ablation: shift <=3 frames; preserve close attack/block source events."""
    result = []
    for i, event in enumerate(events):
        f = int(event["frame"])
        gap = min([abs(int(other["frame"])-f) for j, other in enumerate(events) if j != i] or [100])
        radius = min(3, max(0, (gap-1)//2))
        left, right = max(0, f-radius), min(len(evidence), f+radius+1)
        shift = int(np.argmax(evidence[left:right, 2]))+left
        chosen = shift if evidence[shift, 2] > .10 else f
        result.append({**event, "frame": chosen, "original_frame": f})
    return result


def prepare_gold(shared, base, output):
    manifest = json.loads((base/"dataset/manifest.json").read_bytes())
    if manifest["feature_function_sha256"] != feature_fingerprint():
        raise ValueError("Frozen author feature contract changed")
    if digest((base/"dataset/features.npz").read_bytes()) != manifest["feature_sha256"]:
        raise ValueError("Frozen numeric features changed")
    labels_path = Path(manifest["label_provenance"]["source"])
    labels_raw = labels_path.read_bytes()
    if digest(labels_raw) != manifest["label_provenance"]["sha256"]:
        raise ValueError("Team labels changed since frozen run")
    labels = json.loads(labels_raw)
    reviews = dc.reviewed_contacts(labels["reviews_path"])
    if reviews["review_sha256"] != labels["review_sha256"]:
        raise ValueError("Audited timing review changed")
    reference = shared/"results/dense_contact_spotting_20261028_v2"
    preflight = json.loads((reference/"preflight.json").read_bytes())
    human_source = Path(preflight["human_source"])
    if dc.file_sha256(human_source) != preflight["human_source_sha256"]:
        raise ValueError("Human annotation source changed")
    contract = {"schema": SCHEMA, "feature_path": str((output/"features").resolve()),
                "manifest_path": str((output/"manifest.json").resolve()),
                "source_videos": manifest["source_videos"], "team_labels": manifest["label_provenance"],
                "timing_review_sha256": reviews["review_sha256"], "human_source_sha256": preflight["human_source_sha256"],
                "split_policy": "leave one complete source video out; thresholds and transition counts use train videos only",
                "feature_schema": "observed_ball_piecewise_quadratic_and_limb_interaction_v1",
                "counts": {"team_labels": 397, "person_labels": 0,
                           "audited_contact_frames": sum(len(v) for v in reviews["contacts"].values())},
                "label_provenance": "human timing/team/phase; never model/FSM labels",
                "training": "existing 366D frozen features, CPU two threads, final geometry-side ExtraTrees only",
                "production_mutation": False, "video_inference": False,
                "code_sha256":{"runner":dc.file_sha256(__file__),
                               "core":dc.file_sha256(ROOT/"src/tracking/volleyball_contact_lab.py")}}
    print("VERIFIED EXPERIMENT CONTRACT\n"+json.dumps(contract, indent=2), flush=True)
    write(output/"manifest.json", contract)
    videos = {}
    for stem, source in manifest["source_videos"].items():
        path = Path(source["annotations"])
        if dc.file_sha256(path) != source["sha256"]:
            raise ValueError("Frozen pose/ball analysis changed: "+stem)
        if preflight["sources"][stem]["annotation_sha256"] != source["sha256"]:
            raise ValueError("Spotter control used a different ball/pose analysis")
        ann = json.loads(path.read_bytes())
        length = max(a["f"] for a in ann["frames"])+1
        evidence, support = dense_evidence(ann["frames"], length)
        dense_path = shared/"data/full_video_gold"/stem/"derived_7d09206_v2/dense_targets.npz"
        with np.load(dense_path) as arrays:
            phase = arrays["phase_targets"]
        if dc.file_sha256(dense_path) != preflight["sources"][stem]["dense_targets_sha256"]:
            raise ValueError("Human phase export changed")
        if abs(len(phase)-length) > 2:
            raise ValueError("Human phase/analysis frame mismatch")
        length = min(length, len(phase)); evidence = evidence[:length]
        contacts = reviews["contacts"][stem]
        valid = dc.phase_labels(phase[:length]) >= 0
        covered = {i for c in contacts for i in c["annotation_ids"]}
        masked = reviews["masked"][stem]+dc.uncovered_human_frames(human_source, stem, covered)
        for f in masked:
            valid[max(0, f-8):min(length, f+9)] = False
        # Known timing remains valid even where its raw type is ambiguous.
        truth = sorted(c["frame"] for c in contacts if c["frame"] < length)
        for f in truth:
            valid[max(0, f-5):min(length, f+6)] = True
        non_game = phase[:length] == dc.EXPORTED_PHASES.index("non_game")
        source_events_path = reference/f"fold_{stem}/events.json"
        source_events = json.loads(source_events_path.read_bytes())["constrained"]
        # Same source analysis hashes were checked above; preserve this linkage.
        fold_metrics = json.loads((source_events_path.parent/"metrics.json").read_bytes())
        if fold_metrics.get("holdout") != stem:
            raise ValueError("Not a held-out spotter fold")
        items = sorted([i for i in labels["items"] if i["video_id"] == stem], key=lambda i:i["frame"])
        sequence = grouped_sequence(items, labels["labels"], contacts)
        videos[stem] = {"evidence": evidence, "valid": valid, "non_game": non_game,
                        "truth": truth, "events": source_events, "sequence": sequence, "fps": ann["fps"]}
        feature_path = output/"features"/f"{stem}.npz"
        feature_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(feature_path, evidence=evidence, observed_flight_support=support[:length], valid=valid,
                            non_game=non_game, truth=truth)
        write(feature_path.with_suffix(".json"), {"annotations_sha256": source["sha256"],
              "phase_sha256": dc.file_sha256(dense_path), "reference_events_sha256": dc.file_sha256(source_events_path),
              "feature_sha256":dc.file_sha256(feature_path),
              "observed_flight_support_frames": int(support.sum()), "frames": length, "truth_count": len(truth)})
        print(f"Frozen evidence {stem}: {length} frames; {len(truth)} audited contacts", flush=True)
        del ann
    return videos, contract


def grouped_sequence(items, labels, contacts):
    """One derived context node per audited frame; retain every source ID.

    Unsure sides and uncertain timing reset context. Exact duplicate source
    labels cannot become repeated team hits. Source labels remain unchanged.
    """
    by_frame = defaultdict(list)
    canonical = {c["frame"]:c for c in contacts}
    for item in items:
        by_frame[item["frame"]].append(item)
    result = []
    for f, rows in sorted(by_frame.items()):
        known = {labels.get(row["id"],{}).get("team") for row in rows} & {"team_a","team_b"}
        audited = canonical.get(f)
        result.append({"frame":f, "type":(audited or {}).get("contact_type") or "contact",
                       "case_ids":[row["id"] for row in rows],
                       "team_label":next(iter(known)) if len(known)==1 and audited is not None else None})
    return result


def compare(videos, base, output):
    results = {}; parameters = {}
    for holdout, video in videos.items():
        training = [v for name, v in videos.items() if name != holdout]
        params = {name:tune(training, col) for col, name in enumerate(EVIDENCE_NAMES)}
        parameters[holdout] = params
        original = [e["frame"] for e in video["events"]]
        proposals = {name: peaks(video["evidence"][:, col], **params[name]) for col, name in enumerate(EVIDENCE_NAMES)}
        proposals.update(spotter=original,
                         spotter_joint=[e["frame"] for e in retime(video["events"], video["evidence"])],
                         consensus=consensus(original, proposals["joint"]))
        results[holdout] = {name:detection(rows, video["truth"], video["valid"], video["non_game"])
                            for name, rows in proposals.items()}
        write(output/"folds"/f"{holdout}.json", {"parameters_train_only":params,
               "detection":results[holdout], "predictions":proposals, "truth":video["truth"]})
    summary = {name: pooled([rows[name] for rows in results.values()]) for name in next(iter(results.values()))}
    data = np.load(base/"dataset/features.npz")
    with np.load(base/"oof_predictions.npz") as saved:
        side_scores = saved["extra_trees_typed"]
        if not np.array_equal(saved["y"],data["y"]) or not np.array_equal(saved["groups"],data["groups"]):
            raise ValueError("Frozen side predictions/targets are misaligned")
    cases = json.loads((base/"dataset/cases.json").read_bytes())
    scores_by_id = {case["id"]: float(score) for case, score in zip(cases, side_scores)}
    sequence_scores = {}; chain_folds = {}; priors_by_fold = {}
    for stem, video in videos.items():
        train = [v["sequence"] for name,v in videos.items() if name != stem]
        prior = learn_transitions(train, video["fps"]); priors_by_fold[stem] = prior
        seq = video["sequence"]
        emissions = [float(np.mean([scores_by_id[i] for i in e["case_ids"] if i in scores_by_id]))
                     if any(i in scores_by_id for i in e["case_ids"]) else .5 for e in seq]
        posterior = sequence_posteriors(seq, emissions, prior, video["fps"])
        sequence_scores.update({i:float(p) for e,p in zip(seq, posterior) for i in e["case_ids"] if i in scores_by_id})
        selected = [c for c in cases if c["source_video"]==stem]
        yy = np.array([int(e["team_label"]=="team_b") for e in selected])
        chain_folds[stem] = metrics(yy, np.array([sequence_scores[e["id"]] for e in selected]))
    chain = np.array([sequence_scores[c["id"]] for c in cases])
    team = {"flat_extra_trees":metrics(data["y"], side_scores), "sequence":metrics(data["y"], chain),
            "sequence_folds": chain_folds, "person_accuracy":None,
            "scope":"human timing/type; team labels only; no personal/contact-author accuracy"}
    selected_timing = max(summary, key=lambda name:summary[name]["all"]["at_5"]["f1"])
    selected_author = "sequence" if team["sequence"]["macro_f1"] > team["flat_extra_trees"]["macro_f1"] else "flat_extra_trees"
    comparison = {"schema":SCHEMA, "detection":summary, "folds":results, "thresholds_train_only":parameters,
                  "team":team, "selected_timing":selected_timing, "selected_author":selected_author,
                  "quality_scope":"model-selection estimates on three videos; not independent final-test or JIZ3/person accuracy",
                  "timing_scope":"397 audited timestamps including ambiguous action types; uncertainty/uncovered-human masks excluded",
                  "no_non_game_gate":True}
    write(output/"comparison.json", comparison)
    write(output/"team_priors_by_fold.json", priors_by_fold)
    np.savez_compressed(output/"team_oof.npz", flat=side_scores, sequence=chain, y=data["y"], groups=data["groups"])
    return comparison, data, cases


def matched_indices(events, truth, tolerance=5):
    """Maximum-cardinality one-to-one matching; never match across videos."""
    order = sorted(range(len(events)),key=lambda i:events[i]["frame"])
    truth = sorted(truth)
    i = j = 0; pairs = []
    while i < len(order) and j < len(truth):
        f = events[order[i]]["frame"]
        if f < truth[j]-tolerance: i += 1
        elif truth[j] < f-tolerance: j += 1
        else:
            pairs.append((order[i],truth[j])); i += 1; j += 1
    return pairs


def pipeline_team_scores(events, scores, video):
    teams = {e["frame"]:e["team_label"] for e in video["sequence"]}
    result = {}
    for scope,include in (("all",video["valid"]),("excluding_non_game",video["valid"] & ~video["non_game"])):
        kept = [i for i,e in enumerate(events) if 0 <= e["frame"] < len(include) and include[e["frame"]]]
        actual = [events[i] for i in kept]
        truth = [f for f in video["truth"] if include[f]]
        pairs = matched_indices(actual,truth)
        comparable = [(kept[i],f) for i,f in pairs if teams.get(f) in ("team_a","team_b")]
        correct = sum(int((scores[i]>=.5)==(teams[f]=="team_b")) for i,f in comparable)
        unknown = sum(teams.get(f) not in ("team_a","team_b") for _,f in pairs)
        known_truth = sum(teams.get(f) in ("team_a","team_b") for f in truth)
        result[scope] = {"matched_known_team":len(comparable),"correct_team":correct,
                         "team_accuracy_on_matched":correct/len(comparable) if comparable else None,
                         "joint_detection_team":_prf(correct,len(kept)-unknown-correct,known_truth-correct),
                         "matched_unknown_team_excluded":unknown}
    return result


def compare_detected_team(videos, base, data, output):
    """End-to-end control: use predicted event times/types, not human times/types."""
    manifest = json.loads((base/"dataset/manifest.json").read_bytes())
    names = manifest["feature_names"]; columns = feature_indices(names,"typed")
    folds = {}
    for stem,video in videos.items():
        source = Path(manifest["source_videos"][stem]["annotations"])
        ann = json.loads(source.read_bytes())
        x,fields,actors,indices = extract_events(video["events"],ann["frames"],ann["fps"],(ann["frame_width"],ann["frame_height"]))
        if fields != names or indices != list(range(len(video["events"]))):
            raise ValueError("Held-out predicted event feature order mismatch")
        train = data["groups"] != stem
        estimator = model("extra_trees")
        print("DETECTED-EVENT FOLD",stem,"training sources", sorted(set(data["groups"][train])),
              "labels",int(train.sum()),"features",str((base/"dataset/features.npz").resolve()),flush=True)
        estimator.fit(data["x"][train][:,columns],data["y"][train])
        local = estimator.predict_proba(x[:,columns])[:,1]
        prior = learn_transitions([v["sequence"] for name,v in videos.items() if name != stem],ann["fps"])
        events = [{**e,"team_label":"available"} for e in video["events"]]
        sequence = sequence_posteriors(events,local,prior,ann["fps"])
        folds[stem] = {"local":pipeline_team_scores(events,local,video),
                       "sequence":pipeline_team_scores(events,sequence,video)}
        write(output/"detected_team"/f"{stem}.json", {"scores":folds[stem],"events":events,
              "local_scores":local.tolist(),"sequence_scores":sequence.tolist(),
              "training_sources":sorted(set(data["groups"][train])),"training_labels":int(train.sum())})
        del ann
    aggregate = {}
    for method in ("local","sequence"):
        aggregate[method] = {}
        for scope in ("all","excluding_non_game"):
            rows = [f[method][scope] for f in folds.values()]
            matched = sum(r["matched_known_team"] for r in rows)
            correct = sum(r["correct_team"] for r in rows)
            aggregate[method][scope] = {"matched_known_team":matched,"correct_team":correct,
                "team_accuracy_on_matched":correct/matched if matched else None,
                "joint_detection_team":_prf(*(sum(r["joint_detection_team"][key] for r in rows) for key in ("tp","fp","fn")))}
    result = {"folds":folds,"pooled":aggregate,
              "scope":"cached OOF spotter predicted times AND types; side classifier/prior train on other two videos only"}
    write(output/"detected_team_comparison.json",result)
    return result


def target_export(args, videos, comparison, data, output):
    ann_raw = args.annotations.read_bytes()
    stem = args.annotations.name.removesuffix("_annotations.json")
    stats_path = args.annotations.with_name(stem+"_stats.json")
    stats_raw = stats_path.read_bytes()
    ann, stats = json.loads(ann_raw), json.loads(stats_raw)
    if ann["video"] != stats["video"] or ann["fps"] != stats["fps"]:
        raise ValueError("Review analysis pair mismatch")
    evidence, support = dense_evidence(ann["frames"])
    known = [{**e, "source_event_index":i} for i,e in enumerate(stats["all_events"])]
    final_params = {name:tune(list(videos.values()), col) for col,name in enumerate(EVIDENCE_NAMES)}
    proposals = {name:peaks(evidence[:,col], **final_params[name]) for col,name in enumerate(EVIDENCE_NAMES)}
    original = [e["frame"] for e in known]
    frames_by_method = {**proposals, "spotter":original, "spotter_sequence":original,
                        "spotter_joint":[e["frame"] for e in retime(known, evidence)],
                        "consensus":consensus(original, proposals["joint"])}
    manifest = json.loads((args.base_run/"dataset/manifest.json").read_bytes())
    names = manifest["feature_names"]
    geometry_columns = feature_indices(names, "geometry")
    # Final fit is NOT a held-out measurement and is used only for review transfer.
    print("FINAL FIT: frozen manifest", manifest["manifest_path"], "397 human team labels; zero person labels; geometry view; CPU", flush=True)
    geometry_model = model("extra_trees")
    geometry_model.fit(data["x"][:,geometry_columns], data["y"])
    (output/"models").mkdir()
    joblib.dump({"schema":"contact_author_side_model_v1", "pipeline":geometry_model,"columns":geometry_columns,
                 "feature_names":names,"feature_function_sha256":feature_fingerprint(),
                 "person_labels":0,"final_fit_has_held_out_score":False}, output/"models/geometry_side.joblib")
    typed = joblib.load(args.base_run/"models/extra_trees_typed.joblib")
    if typed["feature_function_sha256"] != feature_fingerprint():
        raise ValueError("Typed model feature contract mismatch")
    prior = learn_transitions([v["sequence"] for v in videos.values()], ann["fps"])
    methods = {}
    known_by_frame = {int(e["frame"]):e for e in known}
    source_retimed = retime(known, evidence)
    for name, times in frames_by_method.items():
        events = []
        for number,f in enumerate(times):
            source = known[number] if name in ("spotter", "spotter_sequence") else source_retimed[number] if name=="spotter_joint" else known_by_frame.get(f)
            # Autonomous proposals remain untyped. Never invent a label from
            # human gold or nearest-in-time source events.
            events.append({**(source or {}), "frame":int(f), "type":source["type"] if source else "followup_contact",
                           "lab_type":source["type"] if source else "contact", "source":"frozen_contact_lab",
                           "team_label":"available"})
        x, fields, actors, indices = extract_events(events, ann["frames"], ann["fps"], (ann["frame_width"],ann["frame_height"]))
        if events and fields != names:
            raise ValueError("Review numeric feature order changed")
        probabilities = geometry_model.predict_proba(x[:,geometry_columns])[:,1] if events else np.empty(0)
        # Known source actions use the identical previous typed model.
        typed_rows = [i for i,e in enumerate(events) if e.get("source_event_index") is not None]
        if typed_rows:
            probabilities[typed_rows] = typed["pipeline"].predict_proba(x[typed_rows][:,typed["columns"]])[:,1]
        use_sequence = name in ("spotter_sequence", "spotter_joint", "consensus")
        if use_sequence:
            probabilities = sequence_posteriors(events, probabilities, prior, ann["fps"])
        if name not in ("spotter", "spotter_sequence"):
            actors = [joint_rank(row,evidence) for row in actors]
        report = contact_report(events, actors, indices, probabilities, ann["fps"], ann.get("player_identity"), name)
        for event, contact in zip(events,report["contacts"]):
            contact["type"] = event["lab_type"]
            contact["source_event_index"] = event.get("source_event_index")
            contact["original_frame"] = event.get("original_frame", event["frame"])
            contact["detection_evidence"] = {field:float(evidence[event["frame"],i]) for i,field in enumerate(EVIDENCE_NAMES)}
        timing_name = "spotter" if name=="spotter_sequence" else name
        methods[name] = {"events":report["contacts"], "totals":report["totals"],
                         "detection_lovo":comparison["detection"][timing_name],
                         "author_method":("sequence side + unchanged local actor" if name=="spotter_sequence"
                                          else "sequence + joint observed evidence" if use_sequence
                                          else "observed geometry + learned side"),
                         "person_accuracy":None}
        print("Review", name, len(events), report["totals"], flush=True)
    selected = comparison["selected_timing"]
    if selected == "spotter" and comparison["selected_author"] == "sequence":
        selected = "spotter_sequence"
    bundle = {"schema":SCHEMA, "source":source_fingerprint(stats_raw,ann_raw),
              "selected_method":selected,"methods":methods,"comparison":comparison,
              "roster_ids":report["roster_ids"],"fps":ann["fps"],"video":ann["video"],
              "parameters_final_fit":final_params,"team_priors_final_fit":prior,
              "provenance":{"annotations":str(args.annotations.resolve()),"stats":str(stats_path.resolve()),
                            "source_events_overwritten":False,"observed_flight_support_frames":int(support.sum()),
                            "person_labels":0,"video_inference":False}}
    result = output/"viewer_result"; result.mkdir()
    (result/args.annotations.name).write_bytes(ann_raw)
    (result/stats_path.name).write_bytes(stats_raw)
    write(result/f"{stem}_contact_lab.json",bundle)
    write(output/"summary.json", {"selected_method":selected,"viewer_result":str(result),
          "detection":comparison["detection"],"team":comparison["team"],
          "quality_scope":comparison["quality_scope"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root",type=Path,required=True)
    parser.add_argument("--base-run",type=Path,required=True)
    parser.add_argument("--annotations",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh versioned output; existing artifacts are preserved")
    if not args.annotations.name.endswith("_annotations.json"):
        parser.error("Expected a saved analysis annotations file")
    args.output.mkdir(parents=True)
    videos, contract = prepare_gold(args.shared_root,args.base_run,args.output)
    comparison, data, cases = compare(videos,args.base_run,args.output)
    comparison["detected_team"] = compare_detected_team(videos,args.base_run,data,args.output)
    real = comparison["detected_team"]["pooled"]
    comparison["selected_author"] = "sequence" if real["sequence"]["all"]["joint_detection_team"]["f1"] > real["local"]["all"]["joint_detection_team"]["f1"] else "flat_extra_trees"
    # The first file is immutable; the updated decision has a separate name.
    write(args.output/"comparison_final.json",comparison)
    target_export(args,videos,comparison,data,args.output)
    print(json.dumps({"selected_timing":comparison["selected_timing"],"selected_author":comparison["selected_author"],
                      "detection_f1":{k:v["all"]["at_5"]["f1"] for k,v in comparison["detection"].items()},
                      "team":comparison["team"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
