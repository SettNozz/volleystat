"""Weak team-supervised actor ranking, CPU only, complete-video held-out folds.

The latent actor is never a labelled person. Compare two fixed feature views
with three seeds each; keep all source data/models immutable and unpromoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_contact_author_learning import digest, metrics, write
from src.tracking.contact_author_features import contact_report, extract_events, feature_fingerprint
from src.tracking.contact_author_mil import SCHEMA, fit_ranker, pack_instances, predict_rankers, ranked_actors


def prepare(base, output):
    manifest_bytes = (base/"dataset/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["feature_function_sha256"] != feature_fingerprint():
        raise ValueError("Base feature code changed; use a fresh validated feature dataset")
    raw_features = Path(manifest["feature_path"]).read_bytes()
    if digest(raw_features) != manifest["feature_sha256"]:
        raise ValueError("Base features changed")
    with np.load(manifest["feature_path"], allow_pickle=False) as stored:
        reference = {k: stored[k].copy() for k in stored.files}
    cases = json.loads(Path(manifest["cases"]).read_bytes())
    packs, teams, groups, labels = [], [], [], []
    shifted_x, shifted_teams = {-2: [], 2: []}, {-2: [], 2: []}
    instance_names = None
    for video, source in sorted(manifest["source_videos"].items()):
        content = Path(source["annotations"]).read_bytes()
        if digest(content) != source["sha256"]:
            raise ValueError("Frozen source analysis changed")
        payload = json.loads(content)
        rows = [i for i, c in enumerate(cases) if c["source_video"] == video]
        events = [{"frame": cases[i]["frame"], "type": cases[i]["type"]} for i in rows]
        needed = {f for e in events for f in range(e["frame"]-16, e["frame"]+17)}
        frames = [a for a in payload["frames"] if a["f"] in needed]
        fps, size = payload["fps"], (payload["frame_width"], payload["frame_height"])
        features, names, actors, indices = extract_events(events, frames, fps, size)
        np.testing.assert_allclose(features, reference["x"][rows], atol=1e-6, rtol=0)
        x, t, _, fields = pack_instances(events, frames, fps, features, names, actors, indices)
        if instance_names is not None and fields != instance_names:
            raise ValueError("MIL feature order changed")
        instance_names = fields
        packs.append(x); teams.append(t); groups += [video]*len(rows); labels += [int(cases[i]["team_label"] == "team_b") for i in rows]
        for offset in (-2, 2):
            shifted_events = [{**e, "frame": e["frame"]+offset} for e in events]
            features, names, actors, indices = extract_events(shifted_events, frames, fps, size)
            key = "shifted_minus2" if offset < 0 else "shifted_plus2"
            np.testing.assert_allclose(features, reference[key][rows], atol=1e-6, rtol=0)
            values, team, _, _ = pack_instances(shifted_events, frames, fps, features, names, actors, indices)
            shifted_x[offset].append(values); shifted_teams[offset].append(team)
        print(f"Prepared weak bags: {video}: {len(rows)} human team labels, zero person labels", flush=True)
        del frames, payload, content
    x, t, y, groups = np.concatenate(packs), np.concatenate(teams), np.asarray(labels), np.asarray(groups)
    shifted_x = {k: np.concatenate(v) for k, v in shifted_x.items()}
    shifted_teams = {k: np.concatenate(v) for k, v in shifted_teams.items()}
    if len(y) != 397 or len(set(groups)) != 3 or not np.isfinite(x).all():
        raise ValueError("Invalid weak-supervision source/count contract")
    dataset = output/"dataset"
    dataset.mkdir()
    feature_path = dataset/"instances.npz"
    np.savez_compressed(feature_path, x=x, teams=t, y=y, groups=groups,
                        x_minus2=shifted_x[-2], teams_minus2=shifted_teams[-2], x_plus2=shifted_x[2], teams_plus2=shifted_teams[2])
    contract = {"schema": SCHEMA, "manifest_path": str((dataset/"manifest.json").resolve()),
                "feature_path": str(feature_path.resolve()), "feature_sha256": digest(feature_path.read_bytes()),
                "feature_names": instance_names, "feature_shape": list(x.shape), "clip_count": len(y),
                "source_videos": manifest["source_videos"], "label_provenance": manifest["label_provenance"],
                "split_policy": "leave one complete source_video out; no random clip split",
                "training_person_labels": 0, "latent_actor_is_not_ground_truth": True,
                "base_manifest": str((base/"dataset/manifest.json").resolve()), "base_manifest_sha256": digest(manifest_bytes),
                "feature_function_sha256": feature_fingerprint(), "epochs": 140, "seeds": [42, 43, 44],
                "epoch_selection": "fixed engineering budget; no held-out epoch selection", "device": "cpu", "threads": 2}
    write(dataset/"manifest.json", contract)
    print("VERIFIED MIL TRAINING CONTRACT:\n"+json.dumps({k: v for k, v in contract.items() if k != "feature_names"}, indent=2), flush=True)
    return x, t, y, groups, shifted_x, shifted_teams, instance_names, contract


def run(args):
    output, base = args.output.resolve(), args.base_run.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh MIL experiment output")
    output.mkdir(parents=True)
    x, teams, y, groups, shifted, shifted_teams, names, contract = prepare(base, output)
    experiments, oof, columns_by_view = {}, {}, {}
    for view in ("geometry", "typed"):
        columns = [i for i, name in enumerate(names) if view == "typed" or not name.startswith("type_")]
        columns_by_view[view] = columns
        probabilities = np.zeros(len(y)); shifted_oof = {k: np.zeros(len(y)) for k in (-2, 2)}
        folds = {}
        for video in sorted(set(groups)):
            train, test = groups != video, groups == video
            assert not set(groups[train]).intersection(set(groups[test]))
            members = [fit_ranker(x[train][:, :, columns], teams[train], y[train], seed) for seed in contract["seeds"]]
            probabilities[test], _ = predict_rankers(members, x[test][:, :, columns], teams[test])
            for offset in (-2, 2):
                shifted_oof[offset][test], _ = predict_rankers(members, shifted[offset][test][:, :, columns], shifted_teams[offset][test])
            folds[video] = {"train_videos": sorted(set(groups[train])), "test_video": str(video),
                            "train_count": int(train.sum()), "test_count": int(test.sum()),
                            "train_bags_with_competitors": [m["training_usable"] for m in members], **metrics(y[test], probabilities[test])}
            print(f"MIL {view}, held out {video}: "+json.dumps(folds[video]), flush=True)
        experiments[view] = {"pooled": metrics(y, probabilities), "folds": folds,
                             "timing_perturbation": {str(k): metrics(y, v) for k, v in shifted_oof.items()},
                             "supervision": "human team bags, zero person/contact-actor labels"}
        oof[view] = probabilities
    write(output/"comparison.json", {"schema": SCHEMA, "experiments": experiments,
                                     "quality_scope": "team-side model-selection estimates; latent person ranking remains unvalidated"})
    np.savez_compressed(output/"oof_predictions.npz", y=y, groups=groups, **oof)

    baseline_result = json.loads((base/"summary.json").read_bytes())["viewer_result"]
    source = Path(baseline_result)
    bundle_path = next(source.glob("*_player_contact_experiment.json"))
    bundle = json.loads(bundle_path.read_bytes())
    ann_path = next(source.glob("*_annotations.json")); stats_path = next(source.glob("*_stats.json"))
    ann_bytes, stats_bytes = ann_path.read_bytes(), stats_path.read_bytes()
    annotations, stats = json.loads(ann_bytes), json.loads(stats_bytes)
    events, frames, fps = stats["all_events"], annotations["frames"], annotations["fps"]
    features, global_names, actors, indices = extract_events(events, frames, fps, (annotations["frame_width"], annotations["frame_height"]))
    target, target_teams, metadata, target_names = pack_instances(events, frames, fps, features, global_names, actors, indices)
    if target_names != names:
        raise ValueError("JIZ3 MIL feature order mismatch")
    for view, columns in columns_by_view.items():
        members = [fit_ranker(x[:, :, columns], teams, y, seed) for seed in contract["seeds"]]
        scores, shares = predict_rankers(members, target[:, :, columns], target_teams)
        ranked = ranked_actors(metadata, shares)
        method = "mil_"+view
        report = contact_report(events, ranked, indices, scores, fps, annotations.get("player_identity"), method,
                                actor_margin=.10, max_actor_distance=.60, actor_share_min=.35)
        report["method"] = "weak_team_supervised_actor_ranker_v1"
        report["supervision"] = "team labels only; individual author was never human-labelled"
        bundle["reports"][method] = report
        bundle["methods"][method] = {"label": f"Weak actor learner ({view})", "team_lovo": experiments[view]["pooled"], "person_accuracy": None}
        checkpoint = output/"models"/f"mil_{view}.pt"
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save({"schema": SCHEMA, "feature_names": names, "columns": columns, "feature_function_sha256": feature_fingerprint(),
                    "training_manifest_sha256": digest((output/"dataset/manifest.json").read_bytes()),
                    "members": [{"state_dict": m["network"].state_dict(), "mean": torch.from_numpy(m["mean"]),
                                 "scale": torch.from_numpy(m["scale"]), "seed": m["seed"], "epochs": m["epochs"]} for m in members],
                    "final_fit_has_held_out_score": False, "person_labels": 0}, checkpoint)
    best_mil = max(experiments, key=lambda view: experiments[view]["pooled"]["macro_f1"])
    previous_score = bundle["selection"]["team_lovo"]["macro_f1"]
    # A predeclared review tradeoff: prefer learned actor ranking only when
    # its team-side F1 is within 0.02 of the best flat team classifier.
    if experiments[best_mil]["pooled"]["macro_f1"] >= previous_score-.02:
        bundle["selected_method"] = "mil_"+best_mil
        bundle["selection"] = {"basis": "Prefer weak learned actor ranking when team OOF macro F1 is within 0.02 of the strongest flat team model",
                               "team_lovo": experiments[best_mil]["pooled"], "person_accuracy": None,
                               "caveat": "Human review candidate; no person labels exist, so no individual-author accuracy is claimed"}
    bundle["mil_experiment"] = {"manifest": str((output/"dataset/manifest.json").resolve()),
                                "comparison": str((output/"comparison.json").resolve()),
                                "base_run": str(base), "training_person_labels": 0}
    result = output/"viewer_result"
    result.mkdir()
    (result/ann_path.name).write_bytes(ann_bytes); (result/stats_path.name).write_bytes(stats_bytes)
    write(result/bundle_path.name, bundle)
    summary = {"output": str(output), "viewer_result": str(result), "selected_method": bundle["selected_method"],
               "experiments": experiments, "totals": {m: r["totals"] for m, r in bundle["reports"].items()},
               "quality_scope": "team-side held-out folds; individual authors need human review"}
    write(output/"summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
