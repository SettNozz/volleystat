"""Leave-one-source-out evaluation of the person-level contact-author ranker."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from src.tracking.contact_author_person_ranker import (chain_posterior, count_transitions, decide,
                                                       person_candidates, roster_seed)

SCHEMA = "contact_author_person_ranker_v1"
RANKERS = ("hgb", "trees")


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, payload):
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def write_seeds(benchmark, output):
    """Identity roster files from the human photo rosters, one per source video."""
    benchmark, output = Path(benchmark), Path(output)
    manifest = json.loads((benchmark/"manifest.json").read_text(encoding="utf-8"))
    if sha(benchmark/"human_labels.json") != manifest["snapshot_sha256"]:
        raise ValueError("Frozen human snapshot changed")
    snapshot = json.loads((benchmark/"human_labels.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    for video, meta in sorted(snapshot["videos"].items()):
        source = Path(meta["annotations_source"]["path"])
        if sha(source) != meta["annotations_source"]["sha256"]:
            raise ValueError(f"{video}: saved source poses changed")
        payload = roster_seed(snapshot, video, json.loads(source.read_text(encoding="utf-8")))
        write_new(output/f"{video}_roster.json", payload)
        print(json.dumps({"video": video, "initial_frame": payload["initial"]["frame"],
                          "people": len(payload["initial"]["players"])+len(payload["introductions"])}), flush=True)


def _ranker(name, seed):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15,
                                              l2_regularization=1., min_samples_leaf=10, random_state=seed)
    return ExtraTreesClassifier(800, min_samples_leaf=2, max_features=.4, random_state=seed, n_jobs=4)


def load_inputs(benchmark, identity_dir):
    benchmark, identity_dir = Path(benchmark), Path(identity_dir)
    manifest = json.loads((benchmark/"manifest.json").read_text(encoding="utf-8"))
    for name, key in (("human_labels.json", "snapshot_sha256"), ("cases.json", "cases_sha256"),
                      ("features.npz", "features_sha256")):
        if sha(benchmark/name) != manifest[key]:
            raise ValueError(f"Frozen benchmark input changed: {name}")
    cases = json.loads((benchmark/"cases.json").read_text(encoding="utf-8"))
    arrays = np.load(benchmark/"features.npz", allow_pickle=False)
    identity = {}
    for video in sorted({c["contact"]["video_id"] for c in cases}):
        path = identity_dir/video/f"{video}_annotations.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        report = data.get("player_identity") or {}
        if report.get("status") != "completed" or (report.get("seed") or {}).get("label_source") != "human":
            raise ValueError(f"{video}: identity pass must be completed from the human photo roster")
        identity[video] = {"by_frame": {f["f"]: f for f in data["frames"]}, "height": data["frame_height"],
                           "provenance": {"path": str(path.resolve()), "sha256": sha(path),
                                          "method": report.get("method"), "coverage": report.get("coverage"),
                                          "seed_sha256": report["seed"].get("seed_sha256"),
                                          "input_annotations_sha256": report.get("input_annotations_sha256")}}
    return manifest, cases, arrays, identity


def candidate_rows(cases, identity, window=15, vote_share=.6):
    rows = []
    for n, case in enumerate(cases):
        contact = case["contact"]; video = identity[contact["video_id"]]
        people = [o for o in case["options"] if o != "other_team"]; truth = case["options"][case["target"]]
        features = person_candidates(video["by_frame"], contact, people, video["height"], window, vote_share)
        for key, x in features.items():
            rows.append({"n": n, "key": key, "y": int(key == truth), "x": x, "video": contact["video_id"]})
    return rows


def team_posterior(cases, global_features, seeds=(42, 43, 44)):
    """ExtraTrees near/far emissions plus training-source rally-chain transitions."""
    contacts = [c["contact"] for c in cases]; videos = sorted({c["video_id"] for c in contacts})
    near = np.array([int(c["options"][c["target"]] != "other_team") for c in cases])
    groups = np.array([c["video_id"] for c in contacts]); emission = np.zeros(len(cases))
    posterior = np.zeros(len(cases))
    for video in videos:
        train, test = groups != video, groups == video
        for seed in seeds:
            model = ExtraTreesClassifier(500, min_samples_leaf=2, max_features=.3, random_state=seed, n_jobs=4)
            emission[test] += model.fit(global_features[train], near[train]).predict_proba(global_features[test])[:, 1]/len(seeds)
        transitions = count_transitions(contacts, near, {v for v in videos if v != video})
        posterior[test] = chain_posterior(contacts, emission, transitions, {video})[test]
    return emission, posterior


def _fit_scores(X, y, groups, train_mask, test_mask, name, seeds):
    scores = np.zeros(int(test_mask.sum()))
    for seed in seeds:
        scores += _ranker(name, seed).fit(X[train_mask], y[train_mask]).predict_proba(X[test_mask])[:, 1]/len(seeds)
    return scores


def _predictions(rows, scores, p_near, indices):
    per = defaultdict(dict)
    for i, s in zip(indices, scores):
        per[rows[i]["n"]][rows[i]["key"]] = float(s)
    return {n: (decide(d, p_near[n]), d) for n, d in per.items()}


def metrics(cases, predicted, contacts=None):
    selected = [n for n in (contacts if contacts is not None else range(len(cases))) if n in predicted]
    truth = {n: cases[n]["options"][cases[n]["target"]] for n in selected}
    near = [n for n in selected if truth[n] != "other_team"]; other = [n for n in selected if truth[n] == "other_team"]
    named = [n for n in selected if predicted[n] != "other_team"]
    ok = lambda n: predicted[n] == truth[n]
    recall = sum(map(ok, near))/max(len(near), 1); precision = sum(map(ok, named))/max(len(named), 1)
    return {"count": len(selected), "near_count": len(near), "other_count": len(other),
            "near_correct": sum(map(ok, near)), "near_author_recall": recall,
            "near_assignment_precision": precision, "near_author_f1": 2*recall*precision/max(recall+precision, 1e-9),
            "other_recall": sum(map(ok, other))/max(len(other), 1), "author_accuracy": sum(map(ok, selected))/max(len(selected), 1),
            "team_accuracy": sum((predicted[n] == "other_team") == (truth[n] == "other_team") for n in selected)/max(len(selected), 1)}


def predict_unlabelled(benchmark, identity_dir, inputs, output, comparison=None, seeds=(42,)):
    """Fit on every labelled source, then rank authors of an unlabelled source's model events.

    ``inputs`` is an unlabelled author-input directory (``inference_inputs.json``,
    ``cases.json`` with targets -1, 366 global features). Its saved annotations
    must already carry production roster IDs. No metric exists for this source.
    """
    inputs, output = Path(inputs), Path(output)
    if output.exists():
        raise ValueError("Prediction output must be fresh")
    manifest, cases, arrays, identity = load_inputs(benchmark, identity_dir)
    unlabelled = json.loads((inputs/"manifest.json").read_text(encoding="utf-8"))
    snapshot_path = inputs/unlabelled["snapshot_file"]
    if sha(snapshot_path) != unlabelled["snapshot_sha256"] or sha(inputs/"cases.json") != unlabelled["cases_sha256"]:
        raise ValueError("Unlabelled inputs changed")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    targets = json.loads((inputs/"cases.json").read_text(encoding="utf-8")); new = np.load(inputs/"features.npz", allow_pickle=False)
    if any(c["target"] != -1 for c in targets) or snapshot.get("decisions"):
        raise ValueError("Unlabelled source must not contain author targets")
    if set(snapshot["videos"]) & set(identity):
        raise ValueError("Unlabelled source overlaps the labelled training sources")
    videos = {}
    for video, meta in snapshot["videos"].items():
        path = Path(meta["annotations_source"]["path"])
        if sha(path) != meta["annotations_source"]["sha256"]:
            raise ValueError(f"{video}: saved source annotations changed")
        data = json.loads(path.read_text(encoding="utf-8")); report = data.get("player_identity") or {}
        if report.get("status") != "completed":
            raise ValueError(f"{video}: production roster IDs are missing")
        videos[video] = {"by_frame": {f["f"]: f for f in data["frames"]}, "height": data["frame_height"],
                         "identity": {k: report.get(k) for k in ("method", "coverage", "roster_size")}
                         | {"seed_label_source": (report.get("seed") or {}).get("label_source"),
                            "seed_verified": (report.get("seed") or {}).get("verified")}}
        del data
    contract = {"schema": "contact_author_person_inference_v1", "training_benchmark": str(Path(benchmark).resolve()),
                "training_manifest_sha256": sha(Path(benchmark)/"manifest.json"), "training_sources": sorted(identity),
                "training_identity": {v: d["provenance"] for v, d in identity.items()},
                "dataset": str(inputs.resolve()), "dataset_manifest_sha256": sha(inputs/"manifest.json"),
                "label_provenance": unlabelled["contract"]["label_provenance"], "metrics": None,
                "source_identity": {v: d["identity"] for v, d in videos.items()},
                "scope": "final fit on all labelled sources; this source has no measured accuracy"}
    print(json.dumps(contract, indent=2), flush=True)
    rows = candidate_rows(cases, identity); columns = list(rows[0]["x"])
    X = np.array([[r["x"][c] for c in columns] for r in rows], np.float32); y = np.array([r["y"] for r in rows])
    ranker = [_ranker("hgb", seed).fit(X, y) for seed in seeds]
    near = np.array([int(c["options"][c["target"]] != "other_team") for c in cases])
    team = [ExtraTreesClassifier(500, min_samples_leaf=2, max_features=.3, random_state=seed, n_jobs=4)
            .fit(arrays["global_features"], near) for seed in (42, 43, 44)]
    transitions = count_transitions([c["contact"] for c in cases], near, set(identity))
    contacts = [c["contact"] for c in targets]
    emission = np.mean([m.predict_proba(new["global_features"])[:, 1] for m in team], axis=0)
    p_near = chain_posterior(contacts, emission, transitions, set(videos))
    previous = {}
    if comparison:
        for row in json.loads(Path(comparison).read_text(encoding="utf-8")):
            previous[row["id"]] = row["predictions"]
    records = []
    for n, case in enumerate(targets):
        contact = case["contact"]; video = videos[contact["video_id"]]; evidence = {}
        people = [o for o in case["options"] if o != "other_team"]
        features = person_candidates(video["by_frame"], contact, people, video["height"], evidence=evidence)
        keys = list(features); Z = np.array([[features[k][c] for c in columns] for k in keys], np.float32)
        scores = dict(zip(keys, np.mean([m.predict_proba(Z)[:, 1] for m in ranker], axis=0).tolist()))
        label = decide(scores, p_near[n]); best_near = max((k for k in keys if k != "other_team"), key=scores.get)
        records.append({"id": contact["id"], "video_id": contact["video_id"], "frame": contact["frame"], "type": contact["type"],
                        "event_index": contact.get("event_index"), "event_label_source": contact.get("event_label_source"),
                        "truth": None, "predictions": {"person_ranker": {
                            "label": label, "team": "a" if label == "other_team" else "b",
                            "team_near_probability": float(p_near[n]), "team_emission": float(emission[n]),
                            "best_near_person": best_near, "person_scores": scores, "score_calibrated": False,
                            "actor": evidence.get(label), "label_source": "derived", "verified": False},
                            **previous.get(contact["id"], {})}})
    output.mkdir(parents=True)
    write_new(output/"predictions.json", records)
    counts = Counter(r["predictions"]["person_ranker"]["label"] for r in records)
    agreement = None
    if previous:
        pairs = [(r["predictions"]["person_ranker"]["label"], next(iter(previous[r["id"]].values()))["label"])
                 for r in records if r["id"] in previous]
        agreement = {"compared": len(pairs), "same_label": sum(a == b for a, b in pairs)}
    write_new(output/"report.json", {**contract, "method": "person_ranker", "comparison_predictions": str(comparison) if comparison else None,
                                     "created_at": datetime.now(timezone.utc).isoformat(), "counts": dict(counts),
                                     "agreement_with_comparison": agreement, "feature_columns": columns,
                                     "predictions_sha256": sha(output/"predictions.json"), "production_promoted": False})
    print(json.dumps({"contacts": len(records), "counts": dict(counts), "agreement_with_comparison": agreement}), flush=True)
    return output


def evaluate(benchmark, identity_dir, output, seeds=(42,)):
    output = Path(output)
    if output.exists():
        raise ValueError("Evaluation output must be fresh")
    manifest, cases, arrays, identity = load_inputs(benchmark, identity_dir)
    strict = [n for n, c in enumerate(cases) if c["strict"]]
    contract = {"schema": SCHEMA, "benchmark_manifest": str((Path(benchmark)/"manifest.json").resolve()),
                "benchmark_manifest_sha256": sha(Path(benchmark)/"manifest.json"),
                "source_videos": sorted(identity), "contacts": len(cases), "strict_contacts": len(strict),
                "label_provenance": manifest["contract"]["label_provenance"],
                "identity": {v: d["provenance"] for v, d in identity.items()},
                "split": "outer: one complete source video held out; inner: whole-source folds of the other two choose the ranker",
                "inputs": "saved ball/poses + production roster IDs from human photo rosters; known contact time/type",
                "training": "CPU sklearn, fixed budgets; no neural training, no inference"}
    print(json.dumps(contract, indent=2), flush=True)
    rows = candidate_rows(cases, identity)
    columns = list(rows[0]["x"]); X = np.array([[r["x"][c] for c in columns] for r in rows], np.float32)
    y = np.array([r["y"] for r in rows]); groups = np.array([r["video"] for r in rows])
    emission, p_near = team_posterior(cases, arrays["global_features"])
    videos = sorted(identity); results = {name: {} for name in RANKERS}; nested = {}; selection = {}
    for video in videos:
        test = groups == video; index = np.where(test)[0]
        for name in RANKERS:
            results[name].update(_predictions(rows, _fit_scores(X, y, groups, ~test, test, name, seeds), p_near, index))
        inner = {}
        for name in RANKERS:
            scores = []
            for held in (v for v in videos if v != video):
                inner_test = groups == held
                predicted = _predictions(rows, _fit_scores(X, y, groups, ~test & ~inner_test, inner_test, name, seeds),
                                         p_near, np.where(inner_test)[0])
                held_strict = [n for n in strict if cases[n]["contact"]["video_id"] == held]
                scores.append(metrics(cases, {n: v[0] for n, v in predicted.items()}, held_strict)["near_author_recall"])
            inner[name] = float(np.mean(scores))
        chosen = max(RANKERS, key=lambda name: (inner[name], -RANKERS.index(name)))
        selection[video] = {"chosen": chosen, "inner_near_recall": inner}
        nested.update({n: results[chosen][n] for n in {rows[i]["n"] for i in index}})
    comparisons = {**results, "nested_selected": nested,
                   "team_only_emission": {n: ("other_team" if emission[n] < .5 else "near_team", {}) for n in range(len(cases))}}
    report = {}
    for name, predicted in comparisons.items():
        labels = {n: v[0] for n, v in predicted.items()}
        report[name] = {"strict": metrics(cases, labels, strict), "all": metrics(cases, labels),
                        "per_video_strict": {v: metrics(cases, labels, [n for n in strict if cases[n]["contact"]["video_id"] == v]) for v in videos}}
    by_type = defaultdict(Counter)
    for n in strict:
        truth = cases[n]["options"][cases[n]["target"]]
        if truth != "other_team":
            kind = cases[n]["contact"]["type"].split(" / ")[0]
            by_type[kind]["count"] += 1; by_type[kind]["correct"] += int(nested[n][0] == truth)
    output.mkdir(parents=True)
    predictions = [{"id": c["contact"]["id"], "video_id": c["contact"]["video_id"], "frame": c["contact"]["frame"],
                    "type": c["contact"]["type"], "truth": c["options"][c["target"]], "strict": c["strict"],
                    "team_near_probability": float(p_near[n]), "team_emission": float(emission[n]),
                    "prediction": nested[n][0], "person_scores": nested[n][1],
                    "label_source": "model", "verified": False} for n, c in enumerate(cases)]
    write_new(output/"predictions.json", predictions)
    comparison = {**contract, "created_at": datetime.now(timezone.utc).isoformat(), "feature_columns": columns,
                  "selection": selection, "results": report,
                  "nested_by_type_strict": {k: dict(v) for k, v in by_type.items()},
                  "predictions_sha256": sha(output/"predictions.json"), "production_promoted": False}
    write_new(output/"comparison.json", comparison)
    lines = ["# Person-level contact-author ranker", "", f"Schema `{SCHEMA}`. Strict set: {len(strict)} contacts.", "",
             "| Method | Team | Exact near-person recall | Near precision | Exact author or other team |", "|---|---:|---:|---:|---:|"]
    for name in (*RANKERS, "nested_selected"):
        m = report[name]["strict"]
        lines.append(f"| {name} | {m['team_accuracy']:.1%} | {m['near_author_recall']:.1%} ({m['near_correct']}/{m['near_count']}) "
                     f"| {m['near_assignment_precision']:.1%} | {m['author_accuracy']:.1%} |")
    lines += ["", "Per held-out video (nested):", ""]
    lines += [f"- {v}: near recall {report['nested_selected']['per_video_strict'][v]['near_author_recall']:.1%}, "
              f"chosen `{selection[v]['chosen']}`" for v in videos]
    (output/"REPORT.md").open("x", encoding="utf-8").write("\n".join(lines)+"\n")
    print("\n".join(lines), flush=True)
    return comparison
