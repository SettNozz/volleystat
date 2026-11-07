"""Audit committed author reports and compare them on identical held-out contacts.

Reuses frozen class_fusion predictions and saved identity/ball/pose outputs.
Refits only Claude's small CPU classifiers with strict-only training and proper
inner-fold team models. Original methods, human labels and reports stay intact.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable,"2")
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from action_recognition.annotator.contact_author_dataset import now,sha,write_new
from src.tracking.contact_author_evaluation import metrics,prediction
from src.tracking.contact_author_person_evaluation import _ranker,candidate_rows,load_inputs
from src.tracking.contact_author_person_ranker import base_type,chain_posterior,decide,rally_chains


def align_predictions(cases,records,label_getter):
    lookup={r["id"]:r for r in records}
    if len(lookup)!=len(records) or set(lookup)!={c["contact"]["id"] for c in cases}:
        raise ValueError("Prediction queue has missing, extra or duplicated contacts")
    output=[]
    for c in cases:
        row=lookup[c["contact"]["id"]];contact=c["contact"];truth=c["options"][c["target"]]
        if (row["video_id"],row["frame"],row["type"],row["truth"],row["strict"])!=(contact["video_id"],contact["frame"],contact["type"],truth,c["strict"]):
            raise ValueError("Prediction truth, source, time, type or strict membership differs")
        label=label_getter(row)
        if label not in c["options"] and label!="unknown":raise ValueError("Prediction is outside the source roster")
        output.append(prediction(c,label))
    return output


def strict_transitions(contacts,near,strict,training):
    """Original rally-chain counts, with excluded author labels never accessed."""
    counts=defaultdict(lambda:[1.,1.])
    for chain in rally_chains(contacts):
        if contacts[chain[0]]["video_id"] not in training:continue
        for a,b in zip(chain,chain[1:]):
            if strict[a] and strict[b]:
                counts[base_type(contacts[a]["type"]),base_type(contacts[b]["type"])][int(near[a]==near[b])]+=1
    return {key:value[1]/sum(value) for key,value in counts.items()}


def paired(cases,left,right,mask):
    result=Counter()
    for i in np.flatnonzero(mask):
        truth=cases[i]["options"][cases[i]["target"]]
        a=left[i]["label"]==truth;b=right[i]["label"]==truth
        result["both_correct" if a and b else "codex_only" if a else "claude_only" if b else "both_wrong"]+=1
    return dict(result)


def run(benchmark,identity_dir,codex_evaluation,claude_evaluation,output):
    benchmark,identity_dir,codex_evaluation,claude_evaluation,output=map(Path,(benchmark,identity_dir,codex_evaluation,claude_evaluation,output))
    if output.exists():raise ValueError("Comparison output must be fresh")
    bm=json.loads((benchmark/"manifest.json").read_bytes())
    codex_report=json.loads((codex_evaluation/"comparison.json").read_bytes())
    claude_report=json.loads((claude_evaluation/"comparison.json").read_bytes())
    rich_dataset=Path(codex_report["dataset"]);rich_manifest=json.loads((rich_dataset/"manifest.json").read_bytes())
    if sha(benchmark/"manifest.json")!=claude_report["benchmark_manifest_sha256"]:
        raise ValueError("Claude report uses a different benchmark")
    if rich_manifest["base_manifest_sha256"]!=sha(benchmark/"manifest.json") or sha(rich_dataset/"manifest.json")!=codex_report["dataset_manifest_sha256"]:
        raise ValueError("Codex report uses a different benchmark")
    if rich_manifest["snapshot_sha256"]!=bm["snapshot_sha256"]:raise ValueError("Methods use different human snapshots")
    for directory,report in ((codex_evaluation,codex_report),(claude_evaluation,claude_report)):
        if sha(directory/"predictions.json")!=report["predictions_sha256"]:raise ValueError("Stored predictions changed")
    snapshot=json.loads((benchmark/"human_labels.json").read_bytes())
    preservation={str(benchmark/name):sha(benchmark/name) for name in ("manifest.json","human_labels.json","cases.json","features.npz")}
    for video,m in snapshot["videos"].items():
        preservation[m["annotations_source"]["path"]]=m["annotations_source"]["sha256"]
        expected=claude_report["identity"][video]
        if expected["input_annotations_sha256"]!=m["annotations_source"]["sha256"] or sha(expected["path"])!=expected["sha256"]:
            raise ValueError("Claude identity source/provenance differs from frozen poses")
        preservation[expected["path"]]=expected["sha256"]
    contract={"schema":"paired_contact_author_comparison_v1","created_at":now(),
              "commits":{"claude_person_ranker":"f7a86eb712c71f2ca192b3e328ae937aecb32bcb","codex_class_fusion":"0a1e7ed49ad97077200af749e80144b2743b04ba"},
              "manifest":str((benchmark/"manifest.json").resolve()),"features":str((benchmark/"features.npz").resolve()),
              "comparison_feature_output":str((output/"candidate_features.npz").resolve()),
              "human_snapshot_sha256":bm["snapshot_sha256"],"source_videos":sorted(snapshot["videos"]),
              "label_provenance":"Frozen human contact-author revision 421; photo roster is setup, not a contact target",
              "clip_counts":{"eligible":394,"strict":382,"strict_near":184,"strict_other":198},
              "schema_inputs":{"codex":rich_manifest["schema"],"claude":claude_report["schema"]},
              "split_policy":"One complete source held out; strict contacts only for fitting and scoring; inner models exclude both outer and inner held-out sources",
              "event_scope":"Given human contact timing/type; no end-to-end spotting metric; X6 excluded",
              "training":"Claude CPU classifiers only, unchanged fixed budgets; Codex frozen strict-fold predictions reused",
              "seeds":{"person_ranker":[42],"claude_team":[42,43,44]},"raw_model_inference":False,
              "original_protocol_difference":"Claude committed report fits on all 394, scores strict 382; inner team predictions include outer test-source labels",
              "independent_final_test":False}
    print(json.dumps(contract,indent=2),flush=True);output.mkdir(parents=True);write_new(output/"contract.json",contract)
    manifest,cases,arrays,identity=load_inputs(benchmark,identity_dir)
    rich_cases=json.loads((rich_dataset/"cases.json").read_bytes())
    if [(c["contact"],c["target"],c["strict"],c["options"]) for c in cases]!=[(c["contact"],c["target"],c["strict"],c["options"]) for c in rich_cases]:
        raise ValueError("Methods use different contact targets or rosters")
    for video,data in identity.items():
        if data["provenance"]["sha256"]!=claude_report["identity"][video]["sha256"]:raise ValueError("Identity input differs from committed report")
    old_codex=json.loads((codex_evaluation/"predictions.json").read_bytes());old_claude=json.loads((claude_evaluation/"predictions.json").read_bytes())
    stored={"codex_class_fusion":align_predictions(cases,old_codex,lambda r:r["predictions"]["class_fusion"]["label"]),
            "codex_nested_selected":align_predictions(cases,old_codex,lambda r:r["predictions"]["nested_selected"]["label"]),
            "claude_committed_nested":align_predictions(cases,old_claude,lambda r:r["prediction"])}
    strict=np.array([c["strict"] for c in cases]);near=np.array([c["target"]>0 for c in cases]);contacts=[c["contact"] for c in cases]
    case_groups=np.array([c["video_id"] for c in contacts]);videos=sorted(set(case_groups))
    rows=candidate_rows(cases,identity);columns=list(rows[0]["x"])
    X=np.array([[r["x"][c] for c in columns] for r in rows],np.float32);y=np.array([r["y"] for r in rows]);groups=np.array([r["video"] for r in rows])
    case_index=np.array([r["n"] for r in rows]);row_strict=strict[case_index]
    np.savez_compressed(output/"candidate_features.npz",X=X,y=y,groups=groups,case_index=case_index,strict=row_strict,
                        global_features=arrays["global_features"],columns=np.array(columns))
    fit_cache={};audit=[]
    def fit_sources(sources):
        sources=tuple(sorted(sources))
        if sources in fit_cache:return fit_cache[sources]
        train=strict&np.isin(case_groups,sources);train_rows=row_strict&np.isin(groups,sources)
        models={name:_ranker(name,42).fit(X[train_rows],y[train_rows]) for name in ("hgb","trees")}
        scores={name:model.predict_proba(X)[:,1] for name,model in models.items()}
        emission=np.zeros(len(cases))
        for seed in (42,43,44):
            classifier=ExtraTreesClassifier(500,min_samples_leaf=2,max_features=.3,random_state=seed,n_jobs=2)
            classifier.fit(arrays["global_features"][train],near[train].astype(int))
            emission+=classifier.predict_proba(arrays["global_features"])[:,1]/3
        transitions=strict_transitions(contacts,near,strict,set(sources))
        posterior=chain_posterior(contacts,emission,transitions,set(videos))
        results={}
        for name in models:
            per=defaultdict(dict)
            for row,score in zip(rows,scores[name]):per[row["n"]][row["key"]]=float(score)
            results[name]=[prediction(c,decide(per[i],posterior[i])) for i,c in enumerate(cases)]
        audit.append({"training_sources":sources,"strict_contacts":int(train.sum()),"candidate_rows":int(train_rows.sum()),
                      "training_contact_ids":[cases[i]["contact"]["id"] for i in np.flatnonzero(train)],
                      "team_transitions":[{"types":list(k),"same_probability":v} for k,v in transitions.items()]})
        fit_cache[sources]=results
        print(f"Strict fit {sources}: {int(train.sum())} contacts / {int(train_rows.sum())} candidates",flush=True)
        return results
    matched={"claude_hgb_strict":[None]*len(cases),"claude_trees_strict":[None]*len(cases),"claude_nested_strict":[None]*len(cases)}
    selection=[]
    for held_out in videos:
        training=[v for v in videos if v!=held_out];inner_scores={name:[] for name in ("hgb","trees")}
        for validation in training:
            child=[v for v in training if v!=validation];fitted=fit_sources(child)
            mask=strict&(case_groups==validation)
            for name in inner_scores:inner_scores[name].append(metrics(cases,fitted[name],mask)["near_author_recall"])
        mean={name:float(np.mean(values)) for name,values in inner_scores.items()};chosen=max(mean,key=mean.get)
        fitted=fit_sources(training)
        for i in np.flatnonzero(case_groups==held_out):
            matched["claude_hgb_strict"][i]=fitted["hgb"][i];matched["claude_trees_strict"][i]=fitted["trees"][i]
            matched["claude_nested_strict"][i]=fitted[chosen][i]
        selection.append({"outer_test_source":held_out,"outer_training_sources":training,"inner_scores":mean,"chosen":chosen,
                          "inner_training_sources":{v:[s for s in training if s!=v] for v in training},"outer_test_labels_used":False})
        print(f"Held out {held_out}: {chosen}",flush=True)
    all_methods={**stored,**matched};results={}
    for name,preds in all_methods.items():
        results[name]={"strict":metrics(cases,preds,strict),"per_video":{v:metrics(cases,preds,strict&(case_groups==v)) for v in videos},
                       "near_by_type":{t:metrics(cases,preds,strict&near&np.array([c["type"]==t for c in contacts]))
                                       for t in sorted({c["type"] for c in contacts})}}
    for name,old_report,old_key in (("codex_class_fusion",codex_report,"class_fusion"),("codex_nested_selected",codex_report,"nested_selected"),
                                  ("claude_committed_nested",claude_report,"nested_selected")):
        for key in ("count","near_count","other_count","near_author_recall","near_assignment_precision","author_accuracy","team_accuracy"):
            if not np.isclose(results[name]["strict"][key],old_report["results"][old_key]["strict"][key]):
                raise ValueError("Re-scored stored report differs: "+name+" "+key)
    records=[{"id":c["contact"]["id"],"video_id":c["contact"]["video_id"],"frame":c["contact"]["frame"],"type":c["contact"]["type"],
              "truth":c["options"][c["target"]],"strict":c["strict"],"predictions":{name:p[i] for name,p in all_methods.items()}}
             for i,c in enumerate(cases)]
    write_new(output/"predictions.json",records)
    paired_results={name:{"strict_near":paired(cases,stored["codex_class_fusion"],preds,strict&near),
                          "strict_all":paired(cases,stored["codex_class_fusion"],preds,strict)} for name,preds in matched.items()}
    changed={path:{"expected":expected,"actual":sha(path)} for path,expected in preservation.items()}
    if any(v["expected"]!=v["actual"] for v in changed.values()):raise ValueError("Frozen source changed")
    write_new(output/"source_preservation.json",changed)
    report={**contract,"results":results,"paired_vs_codex_fixed":paired_results,"selection":selection,"fit_audit":audit,
            "predictions_sha256":sha(output/"predictions.json"),"candidate_features_sha256":sha(output/"candidate_features.npz"),
            "claude_original_report_sha256":sha(claude_evaluation/"comparison.json"),"codex_original_report_sha256":sha(codex_evaluation/"comparison.json"),
            "identity_input_difference":"Claude uses full-video production OSNet identity outputs; Codex uses short-window OSNet gallery matching. These are complete approaches, not isolated rankers.",
            "production_promoted":False}
    write_new(output/"comparison.json",report)
    lines=["# Paired contact-author comparison","", "382 identical strict contacts, 184 near-person / 198 other-team; whole-source held-out.","",
           "| Method | Exact near person | Near precision | Near F1 | Team accuracy | Exact author or other |",
           "|---|---:|---:|---:|---:|---:|"]
    for name,value in results.items():
        m=value["strict"];correct=round(m["near_author_recall"]*m["near_count"])
        lines.append(f"| {name} | {correct}/{m['near_count']} ({m['near_author_recall']:.2%}) | {m['near_assignment_precision']:.2%} | {m['near_author_f1']:.2%} | {m['team_accuracy']:.2%} | {m['author_accuracy']:.2%} |")
    lines += ["","Original Claude report trained on 394 and has an outer-test dependency in inner team predictions.",
              "Matched strict runs exclude all 12 non-strict labels from fitting and keep inner team models independent of the outer test source.",
              "Codex fixed-method selection used development results; no untouched final test exists. Both methods receive human contact times/types and per-video photo setup.",
              "Claude also uses full-video persistent identity outputs; this comparison cannot attribute the gain to ranker architecture alone.","","## Per-source exact near person","",
              "| Held-out source | Targets | Codex class_fusion | Claude HGB strict | Claude nested strict |","|---|---:|---:|---:|---:|"]
    for v in videos:
        values=[results[n]["per_video"][v] for n in ("codex_class_fusion","claude_hgb_strict","claude_nested_strict")]
        cells=[f"{round(m['near_author_recall']*m['near_count'])}/{m['near_count']} ({m['near_author_recall']:.2%})" for m in values]
        lines.append(f"| {v} | {values[0]['near_count']} | "+" | ".join(cells)+" |")
    lines += ["","## Paired outcomes vs Codex fixed","",json.dumps(paired_results,indent=2),""]
    with (output/"REPORT.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines))
    print("\n".join(lines[:14]),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark","identity-dir","codex-evaluation","claude-evaluation","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();run(args.benchmark,args.identity_dir,args.codex_evaluation,args.claude_evaluation,args.output)


if __name__=="__main__":main()
