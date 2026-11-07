"""Isolated CPU contact-side learning with whole-video held-out folds.

Prepare an immutable feature/label manifest, print/verify it, compare fixed
models, then final-fit experimental artifacts. Never change production models,
source annotations, datasets, event detections or another worktree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.contact_author_features import SCHEMA, contact_report, extract_events, feature_fingerprint
from src.tracking.player_contacts import source_fingerprint


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def metrics(y, score):
    predicted = score >= .5
    selective = (score >= .65) | (score <= .35)
    return {"count": len(y), "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "near_precision": float(precision_score(y, predicted, zero_division=0)),
            "near_recall": float(recall_score(y, predicted, zero_division=0)),
            "brier": float(brier_score_loss(y, score)), "log_loss": float(log_loss(y, score, labels=[0, 1])),
            "selective_coverage": float(np.mean(selective)),
            "selective_accuracy": float(accuracy_score(y[selective], predicted[selective])) if selective.any() else None}


def model(family):
    if family == "logistic":
        estimator = LogisticRegression(C=.3, class_weight="balanced", max_iter=1500, random_state=42)
    elif family == "extra_trees":
        estimator = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4, max_features=.7,
                                         class_weight="balanced", random_state=42, n_jobs=2)
    elif family == "hist_gradient":
        estimator = HistGradientBoostingClassifier(max_iter=160, learning_rate=.05, max_leaf_nodes=7,
                                                  min_samples_leaf=12, l2_regularization=1.,
                                                  class_weight="balanced", early_stopping=False, random_state=42)
    else:
        raise ValueError("Unknown fixed model family")
    return make_pipeline(StandardScaler(), estimator)


def feature_indices(names, view):
    return [i for i, name in enumerate(names) if
            (view == "layout" or not name.startswith("layout_")) and
            (view != "geometry" or not name.startswith("type_"))]


def prepare(labels_path, output):
    raw = labels_path.read_bytes()
    labels = json.loads(raw)
    if labels.get("schema") != "contact_team_labels_v1":
        raise ValueError("Unknown label schema")
    if digest(Path(labels["reviews_path"]).read_bytes()) != labels["review_sha256"]:
        raise ValueError("Human review fingerprint changed")
    all_x, all_y, groups, cases, sources = [], [], [], [], {}
    shifted = {-2: [], 2: []}
    names = None
    for stem, video in sorted(labels["videos"].items()):
        source = Path(labels["annotations_dir"]) / f"{stem}_annotations.json"
        content = source.read_bytes()
        data = json.loads(content)
        selected = [item for item in labels["items"] if item["video_id"] == stem
                    and labels["labels"].get(item["id"], {}).get("team") in ("team_a", "team_b")]
        events = [{"frame": item["frame"], "type": item["contact_type"], "source": "human_reviewed_contact_time"} for item in selected]
        needed = {f for e in events for f in range(e["frame"]-16, e["frame"]+17)}
        frames = [a for a in data["frames"] if a["f"] in needed]
        x, fields, _, indices = extract_events(events, frames, video["fps"], (video["width"], video["height"]))
        if names is not None and names != fields:
            raise ValueError("Feature schema drift between sources")
        if indices != list(range(len(selected))):
            raise ValueError("Some reviewed contact types were silently excluded")
        names = fields
        all_x.append(x)
        y = [int(labels["labels"][item["id"]]["team"] == "team_b") for item in selected]
        all_y += y; groups += [stem]*len(y)
        cases += [{"id": item["id"], "source_video": stem, "frame": item["frame"], "type": item["contact_type"],
                   "team_label": labels["labels"][item["id"]]["team"], "label_source": "human"} for item in selected]
        for offset in (-2, 2):
            shifted[offset].append(extract_events([{**e, "frame": e["frame"]+offset} for e in events], frames,
                                                  video["fps"], (video["width"], video["height"]))[0])
        sources[stem] = {"annotations": str(source), "sha256": digest(content), "count": len(y), "team_counts": dict(Counter(y))}
        print(f"Prepared frozen features: {stem}: {len(y)} human team labels", flush=True)
        del data, frames, content
    x, y, groups = np.concatenate(all_x), np.asarray(all_y, dtype=np.int64), np.asarray(groups)
    shifted = {k: np.concatenate(v) for k, v in shifted.items()}
    dataset = output / "dataset"
    dataset.mkdir(parents=True)
    features_path = dataset / "features.npz"
    np.savez_compressed(features_path, x=x, y=y, groups=groups, shifted_minus2=shifted[-2], shifted_plus2=shifted[2])
    manifest = {"schema": SCHEMA, "manifest_path": str((dataset/"manifest.json").resolve()),
                "feature_path": str(features_path.resolve()), "source_videos": sources,
                "label_provenance": {"source": str(labels_path.resolve()), "sha256": digest(raw),
                                     "kind": "human team_a/team_b labels; unsure excluded", "training_person_labels": 0},
                "split_policy": "leave one complete source_video out; no random clip split",
                "clip_count": len(y), "feature_count": x.shape[1], "feature_names": names,
                "feature_function_sha256": feature_fingerprint(),
                "feature_sha256": digest(features_path.read_bytes()), "cases": str((dataset/"cases.json").resolve()),
                "ball_input": "frozen pipeline ball; human reviewed xy never used",
                "context": "audited contact timing/type; JIZ3 supplies model-predicted timing/type",
                "production_mutation": False, "video_inference": False, "cpu_threads": 2,
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__}}
    write(dataset/"manifest.json", manifest); write(dataset/"cases.json", cases)
    if len(sources) != 3 or len(y) != 397 or set(y) != {0, 1} or not np.isfinite(x).all():
        raise ValueError("Unexpected authoritative counts/schema/sources; training not started")
    if any(set(y[groups == video]) != {0, 1} for video in sources):
        raise ValueError("Each held-out video must retain both human team classes")
    print("VERIFIED TRAINING CONTRACT:\n"+json.dumps({k: v for k, v in manifest.items() if k != "feature_names"}, ensure_ascii=False, indent=2), flush=True)
    return x, y, groups, shifted, names, manifest


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output; old datasets/models are preserved")
    output.mkdir(parents=True, exist_ok=False)
    x, y, groups, shifted, names, manifest = prepare(args.team_labels.resolve(), output)
    experiments, oof = {}, {}
    for view in ("geometry", "typed", "layout"):
        columns = feature_indices(names, view)
        for family in ("logistic", "extra_trees", "hist_gradient"):
            key = f"{family}_{view}"
            probabilities = np.zeros(len(y)); shifted_oof = {offset: np.zeros(len(y)) for offset in (-2, 2)}
            folds = {}
            for source_video in sorted(set(groups)):
                train, test = groups != source_video, groups == source_video
                assert not set(groups[train]).intersection(set(groups[test]))
                fitted = model(family)
                fitted.fit(x[train][:, columns], y[train])
                probabilities[test] = fitted.predict_proba(x[test][:, columns])[:, 1]
                for offset in (-2, 2):
                    shifted_oof[offset][test] = fitted.predict_proba(shifted[offset][test][:, columns])[:, 1]
                folds[source_video] = {"train_videos": sorted(set(groups[train])), "test_video": str(source_video),
                                       "train_count": int(train.sum()), "test_count": int(test.sum()),
                                       **metrics(y[test], probabilities[test])}
            experiments[key] = {"family": family, "view": view, "feature_count": len(columns),
                                "pooled": metrics(y, probabilities), "folds": folds,
                                "timing_perturbation": {str(offset): metrics(y, p) for offset, p in shifted_oof.items()}}
            oof[key] = probabilities
            print(key+": "+json.dumps(experiments[key]["pooled"]), flush=True)
    eligible = [key for key in experiments if experiments[key]["view"] != "layout"]
    rank = lambda key: (experiments[key]["pooled"]["macro_f1"], min(f["macro_f1"] for f in experiments[key]["folds"].values()),
                        -experiments[key]["pooled"]["log_loss"])
    selected = max(eligible, key=rank)
    comparison = {"schema": "contact_author_side_lovo_v1", "experiments": experiments, "selected": selected,
                  "selection": "OOF macro F1; worst source-video F1 and log loss break ties. Layout is diagnostic, excluded for camera-transfer risk.",
                  "quality_scope": "model-selection estimates on human team labels; no untouched test set or person-author accuracy",
                  "final_fit_has_held_out_score": False, "manifest_sha256": digest((output/"dataset/manifest.json").read_bytes())}
    write(output/"comparison.json", comparison)
    np.savez_compressed(output/"oof_predictions.npz", y=y, groups=groups, **oof)

    ann_raw = args.annotations.read_bytes()
    stem = args.annotations.name.removesuffix("_annotations.json")
    stats_path = args.annotations.with_name(f"{stem}_stats.json")
    stats_raw = stats_path.read_bytes()
    annotations, stats = json.loads(ann_raw), json.loads(stats_raw)
    events = stats["all_events"]
    target_x, target_names, actors, indices = extract_events(events, annotations["frames"], annotations["fps"],
                                                            (annotations["frame_width"], annotations["frame_height"]))
    if target_names != names or annotations["video"] != stats["video"] or annotations["fps"] != stats["fps"]:
        raise ValueError("JIZ3 source feature/pair mismatch")
    source = source_fingerprint(stats_raw, ann_raw)
    baseline_raw = args.heuristic_bundle.read_bytes()
    baseline = json.loads(baseline_raw)
    if baseline["source"] != source:
        raise ValueError("Heuristic comparison belongs to different JIZ3 source files")
    reports, methods = baseline["reports"], baseline["methods"]
    model_records = {}
    for family in ("logistic", "extra_trees", "hist_gradient"):
        key = max([k for k in eligible if experiments[k]["family"] == family], key=rank)
        columns = feature_indices(names, experiments[key]["view"])
        fitted = model(family)
        fitted.fit(x[:, columns], y)
        artifact = {"schema": "contact_author_side_model_v1", "feature_schema": SCHEMA, "feature_names": names,
                    "feature_function_sha256": feature_fingerprint(),
                    "columns": columns, "pipeline": fitted, "parameters": fitted.get_params(),
                    "training_manifest": str((output/"dataset/manifest.json").resolve()),
                    "training_manifest_sha256": comparison["manifest_sha256"], "selection_estimates": experiments[key],
                    "final_fit_has_held_out_score": False, "person_labels": 0}
        artifact_path = output/"models"/f"{key}.joblib"
        artifact_path.parent.mkdir(exist_ok=True)
        joblib.dump(artifact, artifact_path)
        scores = fitted.predict_proba(target_x[:, columns])[:, 1]
        method_id = "learned_"+key
        reports[method_id] = contact_report(events, actors, indices, scores, annotations["fps"], annotations.get("player_identity"), method_id)
        methods[method_id] = {"label": f"Learned side: {family} ({experiments[key]['view']})",
                              "team_lovo": experiments[key]["pooled"], "person_accuracy": None}
        model_records[method_id] = {"path": str(artifact_path), "sha256": digest(artifact_path.read_bytes())}
    chosen_method = "learned_"+selected
    bundle = {"schema": "player_contact_experiment_v1", "source": source, "selected_method": chosen_method,
              "methods": methods, "reports": reports,
              "selection": {"basis": comparison["selection"], "team_lovo": experiments[selected]["pooled"],
                            "person_accuracy": None, "caveat": comparison["quality_scope"]},
              "training": {"comparison": str((output/"comparison.json").resolve()), "manifest": str((output/"dataset/manifest.json").resolve()),
                           "manifest_sha256": comparison["manifest_sha256"], "models": model_records,
                           "heuristic_bundle_sha256": digest(baseline_raw)},
              "provenance": {"annotations": str(args.annotations.resolve()), "stats": str(stats_path.resolve()),
                             "video_inference": False, "source_events_changed": False, "person_identities_changed": False}}
    result = output/"viewer_result"
    result.mkdir()
    (result/args.annotations.name).write_bytes(ann_raw); (result/stats_path.name).write_bytes(stats_raw)
    write(result/f"{stem}_player_contact_experiment.json", bundle)
    summary = {"output": str(output), "selected_method": chosen_method, "selected_team_lovo": experiments[selected],
               "viewer_result": str(result), "counts": {m: r["totals"] for m, r in reports.items()},
               "quality_scope": comparison["quality_scope"]}
    write(output/"summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--team-labels", type=Path, required=True)
    parser.add_argument("--heuristic-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.annotations.name.endswith("_annotations.json"):
        parser.error("--annotations must name a saved analysis")
    run(args)


if __name__ == "__main__":
    main()
