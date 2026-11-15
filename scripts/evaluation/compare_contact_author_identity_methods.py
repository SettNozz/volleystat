"""Compare contact-author identity methods on saved production analyses.

Cache-only replay of the 2026-11-14 production benchmark: the same fresh
automatic analyses, the same 1,137 human author decisions (585 named near-side
people) and the same target-independent automatic-to-human ID naming. Contact
frames/types are supplied (known-contact control). Nothing here trains on
author labels except the explicit leave-one-video-out (LOVO) refits, whose
held-out source is excluded from every fit.

Stage 1 (``features``) builds frozen runtime feature matrices for each roster
variant and stores them. Stage 2 (``score``) applies model/policy variants and
scores them. No source annotation, cache row, model or dataset is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import pickle
from pathlib import Path
import sys

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

from src.models.inference_cache import (CacheMiss, FrozenOSNet, ModelOutputCache, collect_embeddings,  # noqa: E402
                                        osnet_feature_store, video_frames)
from src.tracking.contact_author_runtime import (MAX_CONTACT_DISTANCE, NewlineCompatibleFeatures,  # noqa: E402
                                                 build_features, crop_requests, photo_roster, track_segments)
from src.tracking.contact_author_person_ranker import chain_posterior, contact_distance, pose_geometry  # noqa: E402
from src.tracking.player_contacts import side  # noqa: E402


def load_scorer(benchmark: Path):
    spec = importlib.util.spec_from_file_location("benchmark_score", benchmark / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


# ---------------------------------------------------------------- rosters

def gallery_roster(annotations, base, per_person=12, min_gap=90):
    """Automatic roster plus clean, strongly associated identity-pass views.

    Uses only the identity pass's own whole-video gallery keys (no contact
    labels). Views are spread in time so one rally does not dominate.
    """
    by_frame = {r["f"]: r for r in annotations["frames"]}
    keys = (annotations.get("player_identity") or {}).get("gallery_keys") or {}
    people = []
    for person in base["players"]:
        pid = person["player_id"]; refs = list(person["references"]); chosen = [r["frame"] for r in refs]
        candidates = []
        for key in keys.get(pid) or []:
            frame, index = map(int, key.split(":"))
            poses = (by_frame.get(frame) or {}).get("poses") or []
            if index >= len(poses):
                continue
            pose = poses[index]
            identity = pose.get("identity") or {}
            if pose.get("player_id") != pid or identity.get("reason") != "associated":
                continue
            candidates.append((identity.get("appearance_distance", 1.), frame, pose["bb"]))
        for _, frame, box in sorted(candidates):
            if len(refs) >= per_person + len(person["references"]):
                break
            if all(abs(frame - f) >= min_gap for f in chosen):
                refs.append({"frame": frame, "bbox": box}); chosen.append(frame)
        people.append({"player_id": pid, "references": refs})
    return {**base, "players": people, "label_source": "model", "gallery": "identity_pass_associated_views"}


def human_roster(data, vid, sha):
    return {"schema": "contact_author_photo_roster_v1", "side": "b", "video_sha256": sha,
            "players": [{"player_id": p["player_id"], "references": [{"frame": r["frame"], "bbox": r["bbox"]}
                                                                     for r in p["references"]]}
                        for p in data["rosters"][vid]["players"]],
            "label_source": "human", "verified": True}


# ---------------------------------------------------------------- features

def embed(annotations, contacts, roster, cache, assets, video, allow_reference_capture):
    _, requests, _ = crop_requests(annotations, contacts, roster)
    store = NewlineCompatibleFeatures(osnet_feature_store(cache, assets, assets / "osnet_msmt17.pth",
                                                          annotations["frame_width"], annotations["frame_height"]))
    missing = [(f, p) for f, players in requests.items() for p in players if store.get(f, p["bb"]) is None]
    pose_missing = [m for m in missing if not m[1]["key"].startswith("ref:")]
    if pose_missing or (missing and not allow_reference_capture):
        raise CacheMiss(f"{len(missing)} missing OSNet crops ({len(pose_missing)} pose crops)")
    encoder = FrozenOSNet(assets)
    vectors = collect_embeddings([{"f": f, "players": p} for f, p in sorted(requests.items())],
                                 lambda wanted: video_frames(video, wanted), encoder, store, batch_size=128)
    return vectors, len(missing)


def stage_features(args, scorer, manifest, data):
    out = args.output / "features"; out.mkdir(parents=True, exist_ok=True)
    assets = args.root / "models/player_reid/osnet_msmt17_v1"
    for source in manifest["sources"]:
        vid = source["video_id"]
        if args.video_id and vid not in args.video_id:
            continue
        target = out / f"{vid}.pkl"
        if target.exists():
            continue
        path = args.benchmark / "pipeline" / vid / f"{vid}_annotations.json"
        annotations = json.loads(path.read_bytes())
        cache = ModelOutputCache(Path(source["video"]), args.root / "data/inference_cache", True,
                                 source_identity={"kind": "video", "sha256": source["video_sha256"]})
        truth = [c for c in data["contacts"] if c["video_id"] == vid]
        contacts = [{"frame": int(c["frame"]), "type": c["type"], "video_id": vid, "event_index": i}
                    for i, c in enumerate(truth)]
        base = photo_roster(annotations, source_sha256=source["video_sha256"])
        rosters = {"auto": base, "auto_gallery": gallery_roster(annotations, base),
                   "human_photos": human_roster(data, vid, source["video_sha256"])}
        result = {"video_id": vid, "contacts": contacts, "truth": truth, "variants": {}}
        for name, roster in rosters.items():
            vectors, captured = embed(annotations, contacts, roster, cache, assets, Path(source["video"]),
                                      allow_reference_capture=name == "human_photos")
            features = build_features(annotations, contacts, roster, vectors, context_events=contacts)
            result["variants"][name] = {"features": features, "roster": roster, "captured_reference_crops": captured}
            print(f"{vid} {name}: {len(contacts)} contacts, {len(roster['players'])} people, "
                  f"{captured} captured reference crops", flush=True)
        result["identity"] = identity_context(annotations, contacts)
        result["on_court"] = ((annotations.get("player_identity") or {}).get("substitutions") or {}).get("on_court")
        result["photo_frames"] = [r["frame"] for p in data["rosters"][vid]["players"] for r in p["references"]]
        result["mapping"], result["naming"] = scorer.map_ids(annotations, data["rosters"][vid], cache)
        cache.db.close()
        with target.open("wb") as stream:
            pickle.dump(result, stream)


def identity_context(annotations, contacts, window=15, track_window=45):
    """Identity-pass view of every observed near pose close to each contact.

    For each contact: candidate poses (frame, index, geometry distance,
    identity-pass player_id) and a time-weighted per-segment ID vote.
    """
    by_frame = {r["f"]: r for r in annotations["frames"]}
    observations, tracks = track_segments(annotations["frames"], annotations["fps"])
    rows = []
    for contact in contacts:
        f0 = contact["frame"]; candidates = []
        for f in range(f0 - window, f0 + window + 1):
            row = by_frame.get(f) or {}
            ball = None if row.get("ball_out_of_frame") else row.get("ball")
            if ball is None:
                continue
            for i, pose in enumerate(row.get("poses") or []):
                g = pose_geometry(pose, ball[:2])
                if g is None:
                    continue
                d = contact_distance(g)
                if d > 1.0:
                    continue
                segment = observations[(f, i)]["segment"]
                votes = Counter()
                for view in tracks[segment]:
                    if abs(view["frame"] - f0) <= track_window and view["pose"].get("player_id"):
                        votes[view["pose"]["player_id"]] += 1 / (1 + abs(view["frame"] - f0) / 15)
                candidates.append({"frame": f, "index": i, "distance": d, "dt": abs(f - f0),
                                   "player_id": pose.get("player_id"), "side": side(pose.get("team")),
                                   "segment": segment, "votes": dict(votes),
                                   "reason": (pose.get("identity") or {}).get("reason")})
        rows.append(candidates)
    return rows


# ---------------------------------------------------------------- models

def load_models(args):
    import joblib
    final = joblib.load(args.root / "models/contact_author/expanded_20261106_v1/model.joblib")
    folds = joblib.load(args.experiment / "authors_all_sides/fold_models.joblib")
    lovo = {}
    for (train, family, view), value in folds.items():
        if family != "hgb" or view != "soft" or len(train) != 9:
            continue
        held = sorted(set(final["manifest"]["source_videos"]) - {str(t) for t in train})
        if len(held) == 1:
            rankers, teams, _, transitions = value
            lovo[held[0]] = {"heads": rankers, "team_models": teams, "transitions": transitions}
    return {"final": {"heads": final["heads"]["all_sides/hgb_soft"], "team_models": final["team_models"],
                      "transitions": final["transitions"]}}, lovo


def model_scores(bundle, features, contacts, vid):
    emission = np.mean([m.predict_proba(features["global_features"])[:, 1] for m in bundle["team_models"]], axis=0)
    posterior = chain_posterior(contacts, emission, bundle["transitions"], {vid})
    scores = [np.mean([m.predict_proba(x)[:, 1] for m in bundle["heads"]], axis=0) for x in features["candidate_features"]]
    return posterior, scores


# ---------------------------------------------------------------- policies

def track_identity(actor, identity_rows, i):
    """Identity-pass ID of the actor pose's local track segment (vote)."""
    if not actor:
        return None
    for c in identity_rows[i]:
        if c["frame"] == actor["frame"] and c["index"] == actor["pose_index"]:
            return max(c["votes"], key=c["votes"].get) if c["votes"] else None
    return None


POLICIES = {
    # name: (substitution on-court mask, guard level, identity source)
    "production": (True, "full", "model"),
    "prod_no_subs": (False, "full", "model"),
    "guards_dist_conflict_no_subs": (False, "conflict", "model"),
    "guards_dist_no_subs": (False, "distance", "model"),
    "raw_hgb_subs": (True, "none", "model"),
    "raw_hgb": (False, "none", "model"),
    "dist_track_override": (False, "distance", "track"),
    "dist_fused": (False, "distance", "fused"),
    "raw_fused": (False, "none", "fused"),
    "geometry_identity_pass": (False, "geometry", "track"),
}


def decide(policy, features, posterior, scores, contacts, identity_rows, on_court, human=False):
    subs, guards, idsource = POLICIES[policy]
    if human:
        subs = False  # substitution intervals live in the automatic namespace
        if guards in {"full", "conflict"}:
            guards = "distance"  # roster support is automatic-namespace evidence
    options = features["options"]; out = []
    for i, s in enumerate(scores):
        evidence = features["evidence"][i]
        if guards == "geometry":
            cands = [c for c in identity_rows[i] if c["distance"] <= MAX_CONTACT_DISTANCE]
            best = min(cands, key=lambda c: c["distance"] + .015 * c["dt"], default=None)
            if posterior[i] < .5:
                out.append("other_team"); continue
            out.append(max(best["votes"], key=best["votes"].get) if best and best["votes"] else None); continue
        eligible = np.asarray(s[:-1], float).copy()
        if subs:
            for k, pid in enumerate(options[:-1]):
                runs = (on_court or {}).get(pid)
                if runs is not None and not any(a <= contacts[i]["frame"] <= b for a, b in runs):
                    eligible[k] = -np.inf
        original = int(np.argmax(eligible)); original_label = options[original] if np.isfinite(eligible[original]) else None
        if guards != "none":
            for k, pid in enumerate(options[:-1]):
                if not np.isfinite(eligible[k]):
                    continue
                actor = evidence.get(pid) or {}
                d = actor.get("distance_body_heights")
                support = actor.get("roster_support") or {}
                if d is None or not np.isfinite(d) or actor.get("ball_out_of_frame") or d > MAX_CONTACT_DISTANCE:
                    eligible[k] = -np.inf
                elif guards in {"full", "conflict"} and support.get("player_id") and support["player_id"] != pid:
                    eligible[k] = -np.inf
        if idsource == "fused":
            for k, pid in enumerate(options[:-1]):
                if not np.isfinite(eligible[k]):
                    continue
                tid = track_identity(evidence.get(pid), identity_rows, i)
                eligible[k] += .3 if tid == pid else -.3 if tid is not None else 0.
        person = int(np.argmax(eligible))
        label = "other_team" if posterior[i] < .5 else (options[person] if np.isfinite(eligible[person]) else None)
        if guards == "full" and label and label != "other_team" and label != original_label:
            support = (evidence.get(label) or {}).get("roster_support") or {}
            if support.get("player_id") != label:
                label = None
        if idsource == "track" and label and label != "other_team":
            label = track_identity(evidence.get(label), identity_rows, i) or label
        out.append(label)
    return out


def label_for(pred, mapping, human_namespace):
    if pred is None:
        return "unresolved"
    if pred == "other_team":
        return "other_team"
    if human_namespace:
        return pred
    return mapping.get(pred, "unmapped:" + pred)


SEEDS = (42, 43, 44)


def training_rows(item, variant):
    """Candidate rows and binary targets in the roster's own namespace.

    Human authors are converted to automatic IDs only through the
    target-independent photo naming; unnamed authors are left out of
    training. Contacts within 60 frames of a human photo are excluded,
    as in the original experiment.
    """
    payload = item["variants"][variant]; options = payload["features"]["options"]
    inverse = {h: a for a, h in item["mapping"].items()} if variant != "human_photos" else None
    xs, ys = [], []
    for c, x in zip(item["truth"], payload["features"]["candidate_features"]):
        if any(abs(c["frame"] - f) <= 60 for f in item["photo_frames"]):
            continue
        if c["author"]["decision"] == "player":
            target = c["author"]["player_id"] if inverse is None else inverse.get(c["author"]["player_id"])
        else:
            target = "other_team"
        if target not in options:
            continue
        xs.append(x); ys.append([int(o == target) for o in options])
    return xs, ys


def fit_rankers(items, heldout, variants):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, Y = [], []
    for vid, item in items.items():
        if vid == heldout:
            continue
        for variant in variants:
            xs, ys = training_rows(item, variant)
            X.extend(xs); Y.extend(ys)
    X = np.concatenate(X); Y = np.concatenate(Y)
    return [HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15, l2_regularization=1.,
                                           min_samples_leaf=10, early_stopping=False, random_state=seed).fit(X, Y)
            for seed in SEEDS]


RETRAIN = {"lovo_retrain_auto": ("auto_gallery",), "lovo_retrain_mix": ("human_photos", "auto_gallery"),
           "lovo_retrain_mix_all": ("human_photos", "auto", "auto_gallery")}
RETRAIN_POLICIES = {"raw_hgb", "production", "guards_dist_no_subs", "raw_fused", "raw_hgb_subs"}


def best_naming(rows):
    """One-to-one automatic->human naming that maximises agreement (optimistic).

    Measures whether each automatic ID consistently stands for one person,
    which is what a user who names P-IDs once in the viewer would see.
    """
    from scipy.optimize import linear_sum_assignment
    correct = 0
    for vid in {r["video_id"] for r in rows}:
        named = [r for r in rows if r["video_id"] == vid and r["truth"] != "other_team"]
        preds = sorted({r["raw"] for r in named if r["raw"] not in {"other_team", "unresolved"}})
        truths = sorted({r["truth"] for r in named})
        if not preds:
            continue
        m = np.zeros((len(preds), len(truths)))
        for r in named:
            if r["raw"] in preds:
                m[preds.index(r["raw"]), truths.index(r["truth"])] += 1
        a, b = linear_sum_assignment(-m)
        correct += int(m[a, b].sum())
    return correct


def stage_score(args, scorer, manifest, data):
    finals, lovo = load_models(args)
    items = {}
    for source in manifest["sources"]:
        path = args.output / "features" / f"{source['video_id']}.pkl"
        if path.exists():
            with path.open("rb") as stream:
                items[source["video_id"]] = pickle.load(stream)
    rows = defaultdict(list)
    for vid, item in items.items():
        models = [("final_fit", finals["final"]), ("lovo", lovo[vid])]
        if not args.skip_retrain:
            for name, variants in RETRAIN.items():
                models.append((name, {**lovo[vid], "heads": fit_rankers(items, vid, variants)}))
            print(f"retrained LOVO rankers without {vid}", flush=True)
        for variant, payload in item["variants"].items():
            features = payload["features"]; human = variant == "human_photos"
            for model_name, bundle in models:
                posterior, scores = model_scores(bundle, features, item["contacts"], vid)
                for policy in POLICIES:
                    if human and POLICIES[policy][2] != "model":
                        continue
                    if model_name.startswith("lovo_retrain") and policy not in RETRAIN_POLICIES:
                        continue
                    preds = decide(policy, features, posterior, scores, item["contacts"], item["identity"],
                                   item["on_court"], human)
                    for c, p in zip(item["truth"], preds):
                        truth = c["author"]["player_id"] if c["author"]["decision"] == "player" else "other_team"
                        rows[(variant, model_name, policy)].append({
                            "video_id": vid, "id": c["id"], "frame": c["frame"], "type": c["type"], "truth": truth,
                            "predicted": label_for(p, item["mapping"], human),
                            "raw": p if p else "unresolved",
                            "photo_guard": all(abs(c["frame"] - f) > 60 for f in item["photo_frames"])})
    summary = []
    (args.output / "rows").mkdir(exist_ok=True)
    for (variant, model_name, policy), method_rows in rows.items():
        s = scorer.author_scores(method_rows)
        g = scorer.author_scores([r for r in method_rows if r["photo_guard"]])
        per_video = {}
        for vid in sorted({r["video_id"] for r in method_rows}):
            v = scorer.author_scores([r for r in method_rows if r["video_id"] == vid])
            per_video[vid] = {"recall": v["exact_person_recall"], "correct": v["correct_named"], "named": v["named_targets"]}
        summary.append({"roster": variant, "model": model_name, "policy": policy,
                        "named": s["named_targets"], "correct": s["correct_named"],
                        "recall": s["exact_person_recall"], "precision": s["exact_person_precision"],
                        "f1": s["person_f1"], "author_or_other": s["author_or_other_team_accuracy"],
                        "other_team_recall": s["other_team_recall"], "errors": s["named_errors"],
                        "unmapped_named": s["unmapped_named_targets"],
                        "best_naming_correct": best_naming(method_rows),
                        "photo_guard_recall": g["exact_person_recall"], "photo_guard_named": g["named_targets"],
                        "per_video": per_video})
        write(args.output / "rows" / f"{variant}__{model_name}__{policy}.json", method_rows)
    summary.sort(key=lambda r: (-r["recall"]))
    write(args.output / "summary.json", summary)
    print(f"{'roster':13} {'model':20} {'policy':28} {'corr':>5} {'rec':>6} {'prec':>6} {'F1':>6} {'acc':>6} {'best':>5}  errors")
    for r in summary:
        print(f"{r['roster']:13} {r['model']:20} {r['policy']:28} {r['correct']:5d} {r['recall']:.3f} "
              f"{r['precision']:.3f} {r['f1']:.3f} {r['author_or_other']:.3f} {r['best_naming_correct']:5d}  {r['errors']}")


def stage_runtime(args, scorer, manifest, data):
    """Score the production runtime itself (final-fit model, default and --hard-guards).

    Known-contact control and end-to-end detected contacts (matched +/-5 frames,
    timestamps paired without author labels), exactly as the 2026-11-14 benchmark.
    The final-fit model has seen all ten sources: this checks implementation
    equivalence and end-to-end behaviour, not held-out accuracy.
    """
    from src.tracking.contact_author_runtime import attribute_contacts
    from src.tracking.contact_moments import contact_moment_groups
    results = {}
    for source in manifest["sources"]:
        vid = source["video_id"]
        if args.video_id and vid not in args.video_id:
            continue
        item_path = args.output / "features" / f"{vid}.pkl"
        with item_path.open("rb") as stream:
            item = pickle.load(stream)
        path = args.benchmark / "pipeline" / vid / f"{vid}_annotations.json"
        annotations = json.loads(path.read_bytes())
        stats = json.loads(path.with_name(f"{vid}_stats.json").read_bytes())
        cache = ModelOutputCache(Path(source["video"]), args.root / "data/inference_cache", True,
                                 source_identity={"kind": "video", "sha256": source["video_sha256"]})
        assets = args.root / "models/player_reid/osnet_msmt17_v1"
        model = args.root / "models/contact_author/expanded_20261106_v1/model.joblib"
        truth = item["truth"]
        for guards in (False, True):
            name = "runtime_hard_guards" if guards else "runtime_default"
            known = attribute_contacts(annotations, [{"frame": c["frame"], "type": c["type"]} for c in truth],
                                       model_path=model, cache=cache, assets=assets, require_cache=True,
                                       use_registry=False, guards=guards)
            proposed = {c["event_index"]: c for c in known["contacts"]}
            known_rows = []
            for i, c in enumerate(truth):
                t = c["author"]["player_id"] if c["author"]["decision"] == "player" else "other_team"
                known_rows.append({"video_id": vid, "id": c["id"], "truth": t,
                                   "predicted": scorer.predicted_label(proposed.get(i), item["mapping"])})
            events = stats["all_events"]
            report = attribute_contacts(annotations, events, model_path=model, cache=cache, assets=assets,
                                        require_cache=True, use_registry=False, guards=guards)
            moments = contact_moment_groups(report["contacts"], events, stats["fps"])
            predictions = [{"frame": int(events[c["event_index"]]["frame"]), "type": c["type"],
                            "predicted": scorer.predicted_label(c, item["mapping"])} for c, _ in moments]
            hits = {t: p for p, t in scorer.pairs(predictions, truth, 5)}
            e2e_rows = []
            for ti, c in enumerate(truth):
                t = c["author"]["player_id"] if c["author"]["decision"] == "player" else "other_team"
                p = predictions[hits[ti]]["predicted"] if ti in hits else "unresolved"
                e2e_rows.append({"video_id": vid, "id": c["id"], "truth": t, "predicted": p, "detected": ti in hits})
            results.setdefault(name, {"known": [], "e2e": []})
            results[name]["known"].extend(known_rows); results[name]["e2e"].extend(e2e_rows)
            print(f"{vid} {name}: known {sum(r['truth'] == r['predicted'] for r in known_rows if r['truth'] != 'other_team')}"
                  f" e2e@5 {sum(r['truth'] == r['predicted'] for r in e2e_rows if r['truth'] != 'other_team')}", flush=True)
        cache.db.close()
    summary = {}
    for name, value in results.items():
        summary[name] = {"known_contact": scorer.author_scores(value["known"]),
                         "end_to_end_at_5": scorer.author_scores(value["e2e"])}
        for part in ("known_contact", "end_to_end_at_5"):
            s = summary[name][part]; s.pop("confusion", None)
            print(f"{name:20} {part:16} {s['correct_named']}/{s['named_targets']} recall {s['exact_person_recall']:.3f} "
                  f"precision {s['exact_person_precision']:.3f} acc {s['author_or_other_team_accuracy']:.3f} {s['named_errors']}")
    write(args.output / ("runtime_summary.json" if not args.video_id else "runtime_summary_partial.json"), summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["features", "score", "runtime"])
    p.add_argument("--root", type=Path, required=True, help="Main checkout holding data/, models/, results/")
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True, help="contact_expanded_training_20261106_v1 directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video-id", nargs="*")
    p.add_argument("--skip-retrain", action="store_true")
    args = p.parse_args()
    scorer = load_scorer(args.benchmark)
    scorer.DEFAULT_ASSETS = args.root / "models/player_reid/osnet_msmt17_v1"
    sys.path.remove(str(args.root)) if str(args.root) in sys.path else None
    manifest = json.loads((args.benchmark / "manifest.json").read_bytes())
    data = json.loads((args.benchmark / "ground_truth.json").read_bytes())
    args.output.mkdir(parents=True, exist_ok=True)
    {"features": stage_features, "score": stage_score, "runtime": stage_runtime}[args.stage](args, scorer, manifest, data)


if __name__ == "__main__":
    main()
