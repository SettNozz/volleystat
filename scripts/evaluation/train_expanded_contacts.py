"""Versioned, source-held-out author and event experiments over frozen human reviews.

Only OSNet crops absent from the shared inference cache may run inference.
Source analyses, annotation databases, previous datasets and production stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import zlib

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

from action_recognition.annotator.contact_author_dataset import AnnotationStore, now, sha, write_new
from src.tracking.contact_author_features import extract_contact
from src.tracking.contact_author_person_ranker import (
    chain_posterior, contact_distance, count_transitions, person_candidates, pose_geometry,
)
from src.tracking.contact_author_person_evaluation import metrics as author_metrics
from src.tracking.contact_author_reid_benchmark import sparse_images
from src.tracking.player_contacts import side

SCHEMA = "expanded_contacts_experiment_v1"
OLD = {"Khe4P8QcJkg_5_4k", "j0gRhgFCgIs_4_4k", "j3BO9GgUFBI_3_4k"}
TCURU = "TcUru_ou4CQ_4k"
TYPES = ("serve", "reception", "pass", "set", "dig", "attack", "block", "free_ball", "other")
SEEDS = (42, 43, 44)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def snapshot(projects, output):
    if output.exists():
        raise ValueError("Choose a fresh versioned output")
    stores = [AnnotationStore(p) for p in projects]
    snapshots = [s.export() for s in stores]
    merged = {"schema": "human_contact_author_dataset_v1", "created_at": now(),
              "videos": {}, "rosters": {}, "contacts": [], "queue": [], "decisions": {},
              "event_reviews": {}, "source_projects": [], "sources": []}
    for store, data in zip(stores, snapshots):
        if data["counts"]["unlabelled"]:
            raise ValueError("Complete all source decisions before this run")
        if set(merged["videos"]) & set(data["videos"]):
            raise ValueError("Source video appears in multiple projects")
        for key in ("videos", "rosters", "decisions", "event_reviews"):
            merged[key].update(data.get(key, {}))
        for key in ("contacts", "queue", "sources"):
            merged[key].extend(data[key])
        merged["source_projects"].append({"path": str(store.directory.resolve()),
            "project_sha256": store.fingerprint, "revision": data["revision"],
            "counts": data["counts"]})
    for video, roster in merged["rosters"].items():
        if not roster["confirmed"] or not roster["players"]:
            raise ValueError(f"Unconfirmed/empty roster: {video}")
    if len(merged["rosters"][TCURU]["players"]) != 5:
        raise ValueError("User-confirmed TcUru exception requires five roster people")
    merged["roster_policy"] = {"standard_max_on_court": 6, "exceptions": {TCURU: 5},
        "roster_can_exceed_on_court_cap_for_substitutes": True,
        "provenance": "explicit user instruction; confirmed photo rosters; no missing sixth person fabricated"}
    contract = {"schema": SCHEMA, "created_at": now(),
        "manifest": str((output/"manifest.json").resolve()),
        "features": str((output/"features").resolve()), "source_videos": sorted(merged["videos"]),
        "source_projects": merged["source_projects"],
        "author_contacts": len(merged["contacts"]),
        "author_counts": dict(Counter(c["author"]["decision"] for c in merged["contacts"])),
        "explicit_event_type_reviews": len(merged["event_reviews"]),
        "event_label_policy": "old human event labels plus explicit new event_reviews only; unchanged model types excluded",
        "timing_policy": "new candidate timing is model-derived and remains unverified",
        "feature_schema": "OSNet confirmed photo gallery + sampled observed pose geometry + frozen global side v1; events resolution_normalized_v2",
        "split_policy": "complete source_video; fixed budgets and seeds; no test-source fitting/early stopping",
        "missing_model_stage": "OSNet reference/local-pose crops only, reuse shared cache before inference; no YOLO/SAM/ball/spotter rerun",
        "roster_policy": merged["roster_policy"], "production_promoted": False,
        "related_commit": "9b0b581: prefetch does not change training; same OSNet batch128 optimisation applied explicitly"}
    emit(contract)
    output.mkdir(parents=True)
    write_new(output/"human_labels.json", merged)
    for i, store in enumerate(stores):
        with store.connect() as source, sqlite3.connect(output/f"annotation_snapshot_{i}.sqlite3") as target:
            source.backup(target)
    contract["snapshot_sha256"] = sha(output/"human_labels.json")
    contract["code_sha256"] = sha(__file__)
    contract["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
    write_new(output/"manifest.json", contract)


def load_snapshot(run):
    manifest = json.loads((run/"manifest.json").read_bytes())
    if sha(run/"human_labels.json") != manifest["snapshot_sha256"]:
        raise ValueError("Frozen human labels changed")
    return manifest, json.loads((run/"human_labels.json").read_bytes())


def photo_distance(vectors, galleries, vector):
    return {pid: 1-float(np.max(np.stack([vectors[k] for k in keys]) @ vector))
            for pid, keys in galleries.items()}


class NewlineCompatibleFeatures:
    """Read an audited LF/CRLF-only legacy backend without relabelling cache rows.

    Exact runtime, weights and preprocessing must still match. New values go
    exclusively to current stage. Original stage and import provenance survive.
    """
    def __init__(self, current):
        self.current = current; self.cache = current.cache; self.legacy = []
        manifests = {}
        for key, body in self.cache.db.execute("SELECT item,body FROM outputs WHERE stage='stage_manifests'"):
            value = json.loads(zlib.decompress(body))
            if value["name"] == "osnet_embeddings": manifests[key] = value["spec"]
        expected = manifests[current.stage]
        encoder_path = ROOT/"src/models/player_reid_encoder.py"
        current_code = encoder_path.read_bytes().replace(b"\r\n", b"\n")
        for _, body in self.cache.db.execute("SELECT item,body FROM outputs WHERE stage='imports'"):
            imported = json.loads(zlib.decompress(body)); folder = Path(imported["source"])
            original = folder/"encoder_source.py"
            if not original.exists() or original.read_bytes().replace(b"\r\n", b"\n") != current_code:
                continue
            for stage, spec in manifests.items():
                if stage == current.stage or spec.get("backend") != sha(original): continue
                if {k: v for k, v in spec.items() if k != "backend"} != {k: v for k, v in expected.items() if k != "backend"}: continue
                self.legacy.append({"stage": stage, "original_encoder": str(original), "encoder_sha256": sha(original),
                    "equivalence": "byte-identical after LF/CRLF normalization; runtime/weights/settings identical",
                    "import": imported})
        self.reused = 0

    def get(self, frame, box):
        from src.models.inference_cache import payload_array
        value = self.current.get(frame, box)
        if value is not None: return value
        for legacy in self.legacy:
            raw = self.cache.get(legacy["stage"], self.current.key(frame, box))
            if raw is not None:
                self.reused += 1
                return payload_array(raw)
        return None

    def put(self, frame, box, value):
        self.current.put(frame, box, value)


def soft_features(by_frame, contact, people, vectors, galleries, height, all_sides=False):
    """Pool each person's appearance/geometry without hard identity assignment.

    Author labels are not arguments. Identity photo references are setup only.
    Missing ball measurements create no proximity evidence.
    """
    observations = defaultdict(list)
    for f, row in by_frame.items():
        if abs(f-contact["frame"]) > 15:
            continue
        ball = None if row.get("ball_out_of_frame") else row.get("ball")
        for i, pose in enumerate(row.get("poses") or []):
            geometry = pose_geometry(pose, ball[:2]) if ball is not None else None
            distances = photo_distance(vectors, galleries, vectors[f"{f}:{i}"]) if all_sides or side(pose.get("team")) == "b" else {}
            for pid in (people if distances else ["other_team"]):
                d = distances.get(pid, 0.)
                observations[pid].append((abs(f-contact["frame"]), d,
                    contact_distance(geometry) if geometry else None, pose["bb"][3]/height,
                    (pose["bb"][3]-pose["bb"][1])/height,
                    geometry["hand_up"] if geometry else None))
            if all_sides and not pose.get("player_id"):
                observations["other_team"].append((abs(f-contact["frame"]), 0.,
                    contact_distance(geometry) if geometry else None, pose["bb"][3]/height,
                    (pose["bb"][3]-pose["bb"][1])/height, geometry["hand_up"] if geometry else None))
    result = {}
    for pid in [*people, "other_team"]:
        values = observations[pid]; x = {}
        for window in (3, 6, 15):
            subset = [v for v in values if v[0] <= window]
            measured = [v for v in subset if v[2] is not None]
            x[f"soft_app{window}"] = min((v[1] for v in subset), default=1.)
            x[f"soft_joint{window}"] = min((v[2]+2*v[1]+.015*v[0] for v in measured), default=9.)
            for threshold in (.25, .4, .6):
                valid = [v[2] for v in measured if v[1] < threshold]
                x[f"soft_d{window}_{threshold}"] = min(valid, default=9.)
                x[f"soft_n{window}_{threshold}"] = len(valid)
            best = min(measured, key=lambda v: v[2]+2*v[1]+.015*v[0]) if measured else None
            for n, col in (("depth", 3), ("height", 4), ("hands", 5)):
                x[f"soft_{n}{window}"] = best[col] if best and best[col] is not None else 9.
        result[pid] = x
    return result


def refine_appearance(ann, needed, vectors, galleries):
    """Unlabelled clean-track pooling and conservative gallery expansion.

    Uses observed boxes and frozen embeddings only. No contact-author target,
    interpolated pose, new inference or relabelled source observation.
    """
    from src.tracking.contact_author_benchmark import track_segments, unit
    from src.tracking.player_identity import overlap
    observations, _ = track_segments(ann["frames"], ann["fps"])
    by_frame = {r["f"]: r for r in ann["frames"]}; clean = defaultdict(list)
    for f in sorted(needed):
        poses = by_frame[f].get("poses") or []
        for i, p in enumerate(poses):
            key = f"{f}:{i}"
            if key not in vectors: continue
            occlusion = max((overlap(p["bb"], q["bb"]) for j, q in enumerate(poses) if j != i), default=0.)
            kp = p.get("kp") or []
            if occlusion < .3 and len(kp) == 17 and min(kp[j][2] for j in (5, 6, 11, 12)) >= .5:
                clean[observations[f, i]["segment"]].append((f, key))
    refined = dict(vectors); pooled_count = 0
    for f in sorted(needed):
        for i, _ in enumerate(by_frame[f].get("poses") or []):
            key = f"{f}:{i}"
            if key not in vectors: continue
            views = sorted((v for v in clean[observations[f, i]["segment"]] if abs(v[0]-f) <= 45), key=lambda v: abs(v[0]-f))[:8]
            if views:
                pooled = unit(np.mean([vectors[k] for _, k in views], axis=0))
                refined[key] = unit(.35*vectors[key]+.65*pooled); pooled_count += 1
    expanded = {pid: list(keys) for pid, keys in galleries.items()}; options = []
    for views in clean.values():
        for _, key in views:
            distances = sorted((d, pid) for pid, d in photo_distance(vectors, galleries, vectors[key]).items())
            if len(distances) > 1 and distances[0][0] < .25 and distances[1][0]-distances[0][0] > .08:
                options.append((distances[1][0]-distances[0][0], distances[0][1], key))
    for _, pid, key in sorted(options, reverse=True):
        if len(expanded[pid]) >= len(galleries[pid])+8: continue
        if max(float(vectors[key]@vectors[k]) for k in expanded[pid]) > .985: continue
        expanded[pid].append(key)
    return refined, expanded, {"pooled_observations": pooled_count,
        "original_gallery_sizes": {p: len(k) for p, k in galleries.items()},
        "expanded_gallery_sizes": {p: len(k) for p, k in expanded.items()},
        "labels_used": False, "original_embeddings_changed": False,
        "method": "same split-at-gap/teleport raw-track segment; up to8 clean observed views within45 frames; .35 local+.65 mean; conservative unlabelled gallery expansion"}


def prepare(run, shared, require_cache=False, refine=False, all_sides=False):
    from src.models.inference_cache import ModelOutputCache, FrozenOSNet, collect_embeddings, osnet_feature_store
    manifest, data = load_snapshot(run)
    emit({**manifest, "operation": "feature preparation", "require_cache": require_cache})
    feature_dir = run/("features_all_sides" if all_sides else "features_refined" if refine else "features"); feature_dir.mkdir(exist_ok=True)
    assets = shared/"models/player_reid/osnet_msmt17_v1"
    encoder = FrozenOSNet(assets, device="cuda", threads=4)
    for video, meta in sorted(data["videos"].items()):
        feature_path = feature_dir/f"{video}.npz"
        case_path = feature_dir/f"{video}.json"
        if feature_path.exists() and case_path.exists():
            saved = json.loads(case_path.read_bytes())
            if saved["features_sha256"] != sha(feature_path):
                raise ValueError("Prepared features changed")
            print(f"Reuse sealed features: {video}", flush=True); continue
        source = Path(meta["annotations_source"]["path"])
        if sha(source) != meta["annotations_source"]["sha256"]:
            raise ValueError("Source analysis changed: "+video)
        stat = Path(meta["video_source"]["path"]).stat()
        if stat.st_size != meta["video_source"]["size"] or stat.st_mtime_ns != meta["video_source"]["mtime_ns"]:
            raise ValueError("Original video changed: "+video)
        ann = json.loads(source.read_bytes()); by_frame = {r["f"]: r for r in ann["frames"]}
        contacts = sorted([c for c in data["contacts"] if c["video_id"] == video], key=lambda c: c["frame"])
        roster = data["rosters"][video]["players"]; people = [p["player_id"] for p in roster]
        # Sampling is relative to each contact, always includes exact candidate frame.
        needed = {f for c in contacts for f in range(c["frame"]-15, c["frame"]+16, 3) if f in by_frame}
        requests = defaultdict(list); galleries = {}
        for f in sorted(needed):
            for i, pose in enumerate(by_frame[f].get("poses") or []):
                if all_sides or side(pose.get("team")) == "b":
                    requests[f].append({"key": f"{f}:{i}", "bb": pose["bb"]})
        for p in roster:
            galleries[p["player_id"]] = []
            for i, reference in enumerate(p["references"]):
                key = f"ref:{p['player_id']}:{i}"
                requests[reference["frame"]].append({"key": key, "bb": reference["bbox"]})
                galleries[p["player_id"]].append(key)
        cache = ModelOutputCache(meta["video_source"]["path"], shared/"data/inference_cache", require_cache,
            source_identity={"kind": "video", "sha256": meta["video_source"]["sha256"]})
        store = NewlineCompatibleFeatures(osnet_feature_store(cache, assets, assets/"osnet_msmt17.pth", meta["width"], meta["height"]))
        frames = [{"f": f, "players": players} for f, players in sorted(requests.items())]
        missing = sum(store.get(f, p["bb"]) is None for f, players in requests.items() for p in players)
        emit({"video": video, "observed_frames": len(needed), "requested_crops": sum(len(v) for v in requests.values()),
              "missing_OSNet_crops": missing, "other_model_inference": False})
        vectors = collect_embeddings(frames,
            lambda needed: sparse_images(meta["video_source"]["path"], needed, (meta["width"], meta["height"])),
            encoder, store, batch_size=128,
            progress=lambda done, total: print(f"{video} missing OSNet frames {done}/{total}", flush=True) if done % 100 == 0 or done == total else None)
        refinement = None
        if refine:
            vectors, galleries, refinement = refine_appearance(ann, needed, vectors, galleries)
        sampled = {}
        for f in sorted(needed):
            row = deepcopy(by_frame[f]); near = []; costs = []
            for i, p in enumerate(row.get("poses") or []):
                p.pop("player_id", None); p["id"] = None
                if all_sides or side(p.get("team")) == "b":
                    distance = photo_distance(vectors, galleries, vectors[f"{f}:{i}"])
                    near.append(i); costs.append([distance[pid] for pid in people])
            if near:
                rr, cc = linear_sum_assignment(np.concatenate([np.array(costs), np.full((len(near), len(near)), .65)], axis=1))
                for r, c in zip(rr, cc):
                    if c < len(people) and costs[r][c] < .65:
                        row["poses"][near[r]]["player_id"] = people[c]
            if all_sides:
                for p in row.get("poses") or []:
                    p["source_team"] = p.get("team")
                    p["team"] = "b" if p.get("player_id") else "a"
                    p["feature_team_source"] = "derived_photo_gallery_assignment_not_source_team_edit"
            if row.get("ball_out_of_frame"):
                row["ball"] = None
            sampled[f] = row
        cases = []; hard_rows = []; soft_rows = []; global_rows = []; event_rows = []; event_sequences = []
        from action_recognition.rally_spotting.prepare import build_continuous_features
        _, continuous = build_continuous_features(ann)
        times = sorted(c["frame"] for c in data["queue"] if c["video_id"] == video)
        for contact in contacts:
            f = contact["frame"]
            local = {k: v for k, v in sampled.items() if abs(k-f) <= 15 and (k-f) % 3 == 0}
            hard = person_candidates(local, contact, people, meta["height"], vote_share=1.1)
            soft = soft_features(local, contact, people, vectors, galleries, meta["height"], all_sides=all_sides)
            gx, gn, _ = extract_contact({"frame": f, "type": contact["type"]}, by_frame, meta["fps"],
                (meta["width"], meta["height"]), min((abs(t-f) for t in times if t != f), default=100))
            options = [*people, "other_team"]; truth = contact["author"]["player_id"] or "other_team"
            away = all(abs(f-r["frame"]) > 60 for p in roster for r in p["references"])
            cases.append({"contact": contact, "options": options, "target": options.index(truth), "strict": bool(away),
                "timing_verified": contact["timing_verified"]})
            hard_rows.append([list(hard[k].values()) for k in options])
            soft_rows.append([list(hard[k].values())+list(soft[k].values()) for k in options]); global_rows.append(gx)
        # Only explicit type reviews enter new event targets, regardless of author decision.
        for c in (c for c in data["queue"] if c["video_id"] == video):
            d = data["decisions"].get(c["id"], {})
            if d.get("decision") == "not_contact":
                continue
            if c.get("event_label_source") != "human":
                continue
            if video not in OLD and c["id"] not in data["event_reviews"]:
                raise ValueError("Unchanged model type cannot become a human event label")
            raw = c["type"]
            if raw not in TYPES:
                continue
            f = c["frame"]; sequence = continuous[max(0, f-119):f+1]
            # Continuous features add observed/new flags before the eight saved pose channels.
            sequence = np.concatenate([sequence[:, :10], sequence[:, 12:20]], axis=1)
            sequence = np.pad(sequence, ((120-len(sequence), 0), (0, 0)))
            sequence[:, 7] = np.linspace(0, 1, 120)
            event_sequences.append(sequence)
            event_rows.append({"id": c["id"], "video_id": video, "frame": f, "type": raw,
                "timing_verified": c["timing_verified"], "label_source": "human",
                "type_provenance": "explicit_type_review" if video not in OLD else "human_gold",
                "source_prediction": (c.get("original_event") or {}).get("type")})
        # Ragged person counts stay explicit, never padded into a fictitious player.
        np.savez_compressed(feature_path, hard=np.array(hard_rows, np.float32), soft=np.array(soft_rows, np.float32),
            global_features=np.array(global_rows, np.float32), event_sequences=np.array(event_sequences, np.float32))
        write_new(case_path, {"cases": cases, "event_cases": event_rows, "features_sha256": sha(feature_path),
            "hard_columns": list(next(iter(hard.values()))),
            "soft_columns": list(next(iter(hard.values())))+list(next(iter(soft.values()))), "global_columns": gn,
            "source_sha256": meta["annotations_source"]["sha256"], "osnet_cache": cache.summary(),
            "newline_compatible_cache_reuse": {"stages": store.legacy, "lookups": store.reused},
            "encoder": encoder.provenance, "sampling": "contact-relative -15..15 stride3; no temporal interpolation",
            "appearance_refinement": refinement,
            "all_sides_compete": all_sides,
            "identity": "per-frame Hungarian assignment from confirmed photo OSNet galleries; raw/source player IDs ignored",
            "five_person_exception": video == TCURU})
        cache.db.close()
        print(f"Features sealed {video}: {len(cases)} authors, {len(event_rows)} human event types", flush=True)


def read_features(run, refined=False, all_sides=False):
    _, data = load_snapshot(run)
    cases = []; hard = []; soft = []; global_rows = []; event_cases = []; event_x = []
    feature_dir = run/("features_all_sides" if all_sides else "features_refined" if refined else "features")
    for video in sorted(data["videos"]):
        meta = json.loads((feature_dir/f"{video}.json").read_bytes())
        path = feature_dir/f"{video}.npz"
        if sha(path) != meta["features_sha256"]:
            raise ValueError("Feature hash mismatch")
        with np.load(path) as a:
            cases.extend(meta["cases"]); hard.extend(a["hard"]); soft.extend(a["soft"]); global_rows.extend(a["global_features"])
            event_cases.extend(meta["event_cases"]); event_x.extend(a["event_sequences"])
    return cases, hard, soft, np.array(global_rows), event_cases, np.array(event_x)


def training_sources(groups, heldout, expanded):
    return set(groups)-{heldout} if expanded else OLD-{heldout}


def eligible_event(data, contact):
    """Type confirmation and author confirmation are independent scopes."""
    decision = data["decisions"].get(contact["id"], {})
    if decision.get("decision") == "not_contact": return False
    if contact.get("event_label_source") != "human": return False
    if contact["video_id"] not in OLD and contact["id"] not in data["event_reviews"]: return False
    return contact["type"] in TYPES


def event_metrics(y, p, types=None):
    from action_recognition.event_clips.config import EVENT_TYPES
    types = list(types or EVENT_TYPES)
    if not len(y): return {"count": 0}
    report = classification_report(y, p, labels=range(len(types)), target_names=types,
        output_dict=True, zero_division=0)
    present = [name for i, name in enumerate(types) if np.any(y == i)]
    return {"count": len(y), "accuracy": float(np.mean(y == p)),
        "macro_f1": report["macro avg"]["f1-score"],
        "present_class_macro_f1": float(np.mean([report[k]["f1-score"] for k in present])),
        "per_class": {k: report[k] for k in types},
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(types))).tolist()}


def pooled_event_features(x):
    values = []
    for width in (15, 30, 60, 120):
        view = x[:, -width:]
        values.extend([view.mean(axis=1), view.std(axis=1), view.min(axis=1), view.max(axis=1), view[:, -1]-view[:, 0]])
    return np.concatenate(values, axis=1)


def training_channel_mask(x, train):
    """A held-out value cannot activate a feature absent during fitting."""
    return x[train].std(axis=(0, 1)) >= 1e-6


def normalise_event(x, mean, std, active):
    return np.where(np.asarray(active, bool), (x-mean)/std, 0.).astype(np.float32)


def events(run, shared):
    import torch
    from torch import nn
    from action_recognition.data.event_clip_dataset import EventClipDataset
    from action_recognition.event_clips.config import EVENT_TYPES, collapse_labels
    from action_recognition.rally_spotting.prepare import build_continuous_features
    from action_recognition.models.sequence_event_model import build_model, save_checkpoint
    from scripts.models.train_contact_spotter import enforce_determinism
    manifest, data = load_snapshot(run)
    output = run/"events"; output.mkdir()
    original = shared/"data/event_clips_three_full_gold_causal_120f_v1"
    original_manifest = json.loads((original/"manifest.json").read_bytes())
    schema = json.loads((original/"features/feature_schema.json").read_bytes())
    if original_manifest["label_source"] != "human" or original_manifest["feature_schema"] != "resolution_normalized_v2":
        raise ValueError("Expected authoritative active human causal dataset")
    ds = EventClipDataset(original_manifest["clips"], str(original/"features"), sequence_len=120)
    xs = [ds._prepare_sequence(features) for _, features in ds.items]
    rows = [{"id": c["clip_id"], "video_id": c["source_stem"],
        "type": collapse_labels(c["event_types"], c.get("fsm_event_type")), "raw_types": c.get("full_event_types"),
        "timing_verified": False, "timing_source": "original_human_annotation_causal_window",
        "label_source": "human", "type_provenance": "active_human_gold",
        "feature_source": str(original/"features"/(c["clip_id"]+".npz"))} for c, _ in ds.items]
    approved = [c for c in data["queue"] if c["video_id"] not in OLD and eligible_event(data, c)]
    allowed = [c for c in approved if collapse_labels([c["type"]]) is not None]
    preflight = {**manifest, "operation": "event training", "original_manifest": str(original/"manifest.json"),
        "original_manifest_sha256": sha(original/"manifest.json"), "original_features": str(original/"features"),
        "original_feature_sidecar": schema, "features": str(output/"dataset.npz"),
        "original_clips": len(original_manifest["clips"]), "original_core_clips": len(rows),
        "explicit_new_type_reviews": len(data["event_reviews"]), "eligible_new_core_types": len(allowed),
        "new_type_counts": dict(Counter(c["type"] for c in allowed)),
        "split_policy": "leave one full source_video out; source-specific normalization from training only",
        "task": "causal clip typing given contact proposal; no dense spotting score or missed-contact recall",
        "selection": "fixed 50 epochs, no test early stopping; three-seed probability ensemble",
        "methods": ["pooled ExtraTrees", "production-architecture multiscale transformer"],
        "seeds": SEEDS, "timing": "new frames are model proposals, explicitly unverified; no dense frame targets created"}
    emit(preflight); write_new(output/"preflight.json", preflight)
    for video in sorted({c["video_id"] for c in allowed}):
        meta = data["videos"][video]; source = Path(meta["annotations_source"]["path"])
        if sha(source) != meta["annotations_source"]["sha256"]: raise ValueError("Source analysis changed")
        ann = json.loads(source.read_bytes()); _, features = build_continuous_features(ann)
        for c in (c for c in allowed if c["video_id"] == video):
            f = c["frame"]; seq = features[max(0, f-119):f+1]
            seq = np.concatenate([seq[:, :10], seq[:, 12:20]], axis=1)
            seq = np.pad(seq, ((120-len(seq), 0), (0, 0)))
            seq[:, 7] = np.linspace(0, 1, 120); xs.append(seq)
            rows.append({"id": c["id"], "video_id": video, "type": collapse_labels([c["type"]]),
                "raw_types": [c["type"]], "frame": f, "timing_verified": False,
                "timing_source": c["timing_source"], "label_source": "human", "type_provenance": "explicit_type_review",
                "original_prediction": c["original_event"]["type"], "feature_source": str(source),
                "feature_source_sha256": meta["annotations_source"]["sha256"]})
    X = np.array(xs, np.float32); y = np.array([EVENT_TYPES.index(c["type"]) for c in rows]); groups = np.array([c["video_id"] for c in rows])
    if set(groups[np.arange(len(rows)) < len(ds)]) != OLD: raise ValueError("Unexpected original sources")
    write_new(output/"dataset_manifest.json", {"schema": "expanded_human_event_types_v1", "clips": rows,
        "feature_schema": "resolution_normalized_v2", "label_source": "human", "split_unit": "source_video",
        "counts": dict(Counter(groups)), "class_counts": dict(Counter(c["type"] for c in rows)),
        "new_unchanged_model_types_promoted": 0})
    np.savez_compressed(output/"dataset.npz", features=X, targets=y, groups=groups)
    print(f"Event features sealed: {len(ds)} original + {len(rows)-len(ds)} explicit reviews", flush=True)
    enforce_determinism(); torch.set_num_threads(4); device = torch.device("cuda")
    predictions = {f"{scope}_{family}": np.full(len(y), -1, int) for scope in ("old3", "expanded") for family in ("trees", "multiscale")}
    cache = {}; audit = []; histories = []; pooled = pooled_event_features(X)
    for heldout in sorted(set(groups)):
        test = groups == heldout
        for expanded in (False, True):
            sources = training_sources(groups, heldout, expanded)
            train = np.isin(groups, list(sources)); scope = "expanded" if expanded else "old3"
            cachekey = tuple(sorted(sources))
            if heldout in sources: raise ValueError("Event test-source leakage")
            if cachekey not in cache:
                mean = X[train].mean(axis=(0, 1)); std = X[train].std(axis=(0, 1)); std[std < 1e-6] = 1
                norm = (X-mean)/std; probabilities = {"trees": [], "multiscale": []}
                for seed in SEEDS:
                    trees = ExtraTreesClassifier(500, min_samples_leaf=2, max_features=.5, class_weight="balanced", n_jobs=4, random_state=seed)
                    trees.fit(pooled[train], y[train]); probabilities["trees"].append(trees.predict_proba(pooled))
                    torch.manual_seed(seed); np.random.seed(seed)
                    model = build_model("multiscale", input_dim=18, short_sequence_len=60, sequence_len=120).to(device)
                    counts = np.bincount(y[train], minlength=4); weights = len(y[train])/np.maximum(counts*4, 1)
                    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=.05)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
                    generator = np.random.default_rng(seed); tr = np.flatnonzero(train)
                    x_device = torch.from_numpy(norm).to(device); y_device = torch.from_numpy(y).to(device)
                    losses = []
                    for epoch in range(50):
                        model.train(); loss_total = []
                        order = generator.permutation(tr)
                        for start in range(0, len(order), 32):
                            idx = order[start:start+32]; logits = model(x_device[idx]); loss = criterion(logits, y_device[idx])
                            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step(); loss_total.append(float(loss.detach()))
                        scheduler.step(); losses.append(float(np.mean(loss_total)))
                    model.eval()
                    with torch.no_grad():
                        probs = torch.cat([model(x_device[i:i+64]).softmax(-1).cpu() for i in range(0, len(y), 64)]).numpy()
                    probabilities["multiscale"].append(probs)
                    folder = output/"models"/("__".join(cachekey)); folder.mkdir(parents=True, exist_ok=True)
                    joblib.dump(trees, folder/f"trees_seed{seed}.joblib", compress=3)
                    save_checkpoint(folder/f"multiscale_seed{seed}.pt", model, "multiscale", 18,
                        extra={"feature_schema": "resolution_normalized_v2", "feature_mean": mean.tolist(), "feature_std": std.tolist(),
                            "feature_active_mask": (X[train].std(axis=(0, 1)) >= 1e-6).tolist(),
                            "sequence_len": 120, "short_sequence_len": 60, "training_sources": list(cachekey),
                            "seed": seed, "fixed_epochs": 50, "production_promoted": False})
                    histories.append({"training_sources": list(cachekey), "seed": seed, "loss": losses})
                    print(f"Event fit heldout={heldout} scope={scope} seed={seed} loss={losses[-1]:.4f}", flush=True)
                cache[cachekey] = {family: np.mean(values, axis=0).argmax(axis=1) for family, values in probabilities.items()}
                audit.append({"heldout": heldout, "training_sources": list(cachekey), "train_count": int(train.sum()), "test_used_for_epoch_selection": False})
            for family in ("trees", "multiscale"):
                predictions[f"{scope}_{family}"][test] = cache[cachekey][family][test]
        emit({"event_heldout": heldout, "results": {n: event_metrics(y[test], p[test]) for n, p in predictions.items()}})
        write_new(output/"training_history_partial.json" if not (output/"training_history_partial.json").exists() else output/f"history_after_{heldout}.json", histories)
    scores = {name: {"all": event_metrics(y, p), "old3": event_metrics(y[np.isin(groups, list(OLD))], p[np.isin(groups, list(OLD))]),
        "new_explicit_corrections_only": event_metrics(y[~np.isin(groups, list(OLD))], p[~np.isin(groups, list(OLD))]),
        "by_video": {v: event_metrics(y[groups == v], p[groups == v]) for v in sorted(set(groups))}}
        for name, p in predictions.items()}
    write_new(output/"predictions.json", [{**c, "truth_core": EVENT_TYPES[y[i]],
        "predictions": {n: EVENT_TYPES[p[i]] for n, p in predictions.items()}} for i, c in enumerate(rows)])
    write_new(output/"training_history.json", histories)
    write_new(output/"report.json", {"schema": SCHEMA, "results": scores, "fit_audit": audit, "seeds": SEEDS,
        "new_subset_warning": "Explicit corrections are deliberately hard/selected examples; not an unbiased score on all new contacts",
        "spotting_metrics": None, "spotting_reason": "No exhaustive new contact/phase/timing labels; no false-negative or background ground truth",
        "production_promoted": False})


def raw_event_indices(rows):
    types = ("serve", "reception", "set", "dig", "attack", "block")
    return [i for i, c in enumerate(rows) if len(c.get("raw_types") or []) == 1 and c["raw_types"][0] in types]


def events_raw(run):
    """Six raw types: preserve set/dig and block/attack rather than collapsing."""
    import torch
    from torch import nn
    from scripts.models.train_contact_spotter import enforce_determinism
    types = ("serve", "reception", "set", "dig", "attack", "block")
    manifest, _ = load_snapshot(run)
    source = json.loads((run/"events/dataset_manifest.json").read_bytes())
    chosen = raw_event_indices(source["clips"])
    rows = [source["clips"][i] for i in chosen]
    with np.load(run/"events/dataset.npz") as a: X = a["features"][chosen]
    y = np.array([types.index(c["raw_types"][0]) for c in rows]); groups = np.array([c["video_id"] for c in rows])
    output = run/"events_raw"; output.mkdir()
    emit({**manifest, "stage": "six-class typing", "feature_path": str(run/"events/dataset.npz"),
        "feature_schema": "resolution_normalized_v2", "source_videos": sorted(set(groups)),
        "counts": dict(Counter(groups)), "raw_type_counts": dict(Counter(types[i] for i in y)),
        "label_provenance": "active human full_event_types plus explicitly reviewed new raw types only",
        "excluded": "ambiguous multiple raw types and plain pass (cannot infer set vs dig)",
        "split_policy": "whole-source held out; train-only normalization; fixed 35 epochs; seeds42/43/44",
        "task": "causal known-contact classification; not continuous spotting"})
    write_new(output/"dataset_manifest.json", {"types": types, "clips": rows, "feature_indices": chosen,
        "source_dataset_sha256": sha(run/"events/dataset.npz")})
    enforce_determinism(); torch.set_num_threads(4); device = torch.device("cuda")

    class ClipTCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []; channels = 18
            for dilation in (1, 2, 4, 8):
                layers.extend([nn.Conv1d(channels, 48, 5, padding=2*dilation, dilation=dilation), nn.GELU(), nn.Dropout(.15)])
                channels = 48
            self.body = nn.Sequential(*layers); self.head = nn.Linear(96, 6)

        def forward(self, x):
            z = self.body(x.transpose(1, 2))
            return self.head(torch.cat([z.mean(-1), z[:, :, -20:].mean(-1)], dim=1))

    predictions = {f"{scope}_{family}": np.full(len(y), -1, int) for scope in ("old3", "expanded") for family in ("trees", "tcn")}
    cache = {}; audit = []; histories = []; pooled = pooled_event_features(X)
    for heldout in sorted(set(groups)):
        test = groups == heldout
        for expanded in (False, True):
            sources = training_sources(groups, heldout, expanded); key = tuple(sorted(sources))
            train = np.isin(groups, list(sources)); scope = "expanded" if expanded else "old3"
            if heldout in sources: raise ValueError("Raw-event test-source leakage")
            if key not in cache:
                mean = X[train].mean(axis=(0, 1)); std = X[train].std(axis=(0, 1)); std[std < 1e-6] = 1
                norm = (X-mean)/std; probs = {"trees": [], "tcn": []}
                folder = output/"models"/(__import__("hashlib").sha256("|".join(key).encode()).hexdigest()[:16]); folder.mkdir(parents=True)
                for seed in SEEDS:
                    tree = ExtraTreesClassifier(500, min_samples_leaf=2, max_features=.5, class_weight="balanced", n_jobs=4, random_state=seed)
                    tree.fit(pooled[train], y[train]); probs["trees"].append(tree.predict_proba(pooled))
                    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = ClipTCN().to(device)
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                    counts = np.bincount(y[train], minlength=6); weights = len(y[train])/np.maximum(counts*6, 1)
                    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device, dtype=torch.float32), label_smoothing=.05)
                    xd = torch.from_numpy(norm).to(device); yd = torch.from_numpy(y).to(device); losses = []
                    for epoch in range(35):
                        model.train(); order = rng.permutation(np.flatnonzero(train)); values = []
                        for start in range(0, len(order), 64):
                            idx = order[start:start+64]; loss = loss_fn(model(xd[idx]), yd[idx]); opt.zero_grad(set_to_none=True)
                            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); values.append(float(loss.detach()))
                        losses.append(float(np.mean(values)))
                    model.eval()
                    with torch.no_grad(): probs["tcn"].append(torch.cat([model(xd[i:i+64]).softmax(-1).cpu() for i in range(0, len(y), 64)]).numpy())
                    joblib.dump(tree, folder/f"trees_seed{seed}.joblib", compress=3)
                    torch.save({"schema": "causal_raw6_clip_tcn_v1", "state_dict": model.state_dict(), "types": types,
                        "mean": mean.tolist(), "std": std.tolist(), "training_sources": list(key), "epochs": 35,
                        "seed": seed, "production_promoted": False}, folder/f"tcn_seed{seed}.pt")
                    histories.append({"sources": list(key), "seed": seed, "loss": losses})
                cache[key] = {family: np.mean(p, axis=0).argmax(1) for family, p in probs.items()}
                audit.append({"heldout": heldout, "training_sources": list(key), "train_count": int(train.sum()), "test_used_for_epoch_selection": False})
            for family in ("trees", "tcn"): predictions[f"{scope}_{family}"][test] = cache[key][family][test]
        emit({"raw_event_heldout": heldout, "metrics": {n: event_metrics(y[test], p[test], types) for n, p in predictions.items()}})
        write_new(output/f"history_after_{heldout}.json", histories)
    scores = {n: {"all": event_metrics(y, p, types),
        "old3": event_metrics(y[np.isin(groups, list(OLD))], p[np.isin(groups, list(OLD))], types),
        "new_explicit_corrections_only": event_metrics(y[~np.isin(groups, list(OLD))], p[~np.isin(groups, list(OLD))], types),
        "by_video": {v: event_metrics(y[groups == v], p[groups == v], types) for v in sorted(set(groups))}}
        for n, p in predictions.items()}
    write_new(output/"predictions.json", [{**c, "truth": types[y[i]], "predictions": {n: types[p[i]] for n, p in predictions.items()}} for i, c in enumerate(rows)])
    write_new(output/"training_history.json", histories)
    write_new(output/"report.json", {"schema": SCHEMA, "types": types, "results": scores, "fit_audit": audit,
        "task": "known-contact raw six-class typing; new test subset contains corrections only", "production_promoted": False})


def score_authors(cases, predictions, mask):
    indices = np.flatnonzero(mask)
    return author_metrics(cases, {int(i): predictions[i] for i in indices}, indices)


def reference_side_conflicts(roster, by_frame):
    """Human near-side photo setup versus weak saved pose-side assignments."""
    from src.tracking.contact_author_benchmark import iou
    conflicts = []
    for person in roster["players"]:
        for reference in person["references"]:
            poses = by_frame.get(reference["frame"], {}).get("poses") or []
            ranked = sorted(((iou(reference["bbox"], p["bb"]), i) for i, p in enumerate(poses)), reverse=True)
            if not ranked or ranked[0][0] < .35 or len(ranked) > 1 and ranked[0][0]-ranked[1][0] < .1: continue
            score, index = ranked[0]
            if side(poses[index].get("team")) == "a":
                conflicts.append({"person": person["player_id"], "reference_frame": reference["frame"],
                    "source_pose_index": index, "iou": score, "source_team": poses[index]["team"]})
    return conflicts


def vote_labels(labels):
    """Fixed equal vote, ties resolve to the first (hard HGB) method."""
    counts = Counter(labels); maximum = max(counts.values())
    return next(label for label in labels if counts[label] == maximum)


def authors_routed(run):
    _, data = load_snapshot(run); cases, _, _, _, _, _ = read_features(run, refined=True)
    output = run/"authors_routed"; output.mkdir()
    parent = {stage: json.loads((run/stage/"predictions.json").read_bytes()) for stage in ("authors_refined", "authors_all_sides")}
    for values in parent.values():
        if [r["id"] for r in values] != [c["contact"]["id"] for c in cases]: raise ValueError("Mismatched routing parents")
    routing = {}
    for video, meta in data["videos"].items():
        source = Path(meta["annotations_source"]["path"])
        if sha(source) != meta["annotations_source"]["sha256"]: raise ValueError("Routing source changed")
        ann = json.loads(source.read_bytes())
        conflicts = reference_side_conflicts(data["rosters"][video], {r["f"]: r for r in ann["frames"]})
        routing[video] = {"stage": "authors_all_sides" if conflicts else "authors_refined", "conflicts": conflicts,
            "contact_author_labels_used": False}
    predictions = {name: [] for name in ("old3_routed_hgb", "expanded_routed_hgb", "expanded_routed_vote", "expanded_all_sides_vote", "expanded_refined_vote")}
    for i, case in enumerate(cases):
        stage = routing[case["contact"]["video_id"]]["stage"]
        selected = parent[stage][i]["predictions"]
        predictions["old3_routed_hgb"].append(selected["old3_hgb"])
        predictions["expanded_routed_hgb"].append(selected["expanded_hgb"])
        vote = lambda row: vote_labels([row[n] for n in ("expanded_hgb", "expanded_trees", "expanded_soft_hgb")])
        predictions["expanded_routed_vote"].append(vote(selected))
        predictions["expanded_all_sides_vote"].append(vote(parent["authors_all_sides"][i]["predictions"]))
        predictions["expanded_refined_vote"].append(vote(parent["authors_refined"][i]["predictions"]))
    groups = np.array([c["contact"]["video_id"] for c in cases]); away = np.array([c["strict"] for c in cases])
    results = {}
    for name, p in predictions.items():
        results[name] = {"all": score_authors(cases, p, np.ones(len(cases), bool)),
            "away_from_references": score_authors(cases, p, away),
            "new7_away": score_authors(cases, p, away & ~np.isin(groups, list(OLD))),
            "old3_away": score_authors(cases, p, away & np.isin(groups, list(OLD))),
            "by_video": {v: score_authors(cases, p, away & (groups == v)) for v in sorted(set(groups))},
            "by_type": {t: score_authors(cases, p, away & np.array([c["contact"]["type"] == t for c in cases])) for t in sorted({c["contact"]["type"] for c in cases})}}
    write_new(output/"predictions.json", [{"id": c["contact"]["id"], "video_id": groups[i], "truth": c["options"][c["target"]],
        "away_from_references": bool(away[i]), "predictions": {n: p[i] for n, p in predictions.items()}} for i, c in enumerate(cases)])
    write_new(output/"report.json", {"schema": SCHEMA, "results": results, "routing": routing,
        "parent_predictions": {stage: sha(run/stage/"predictions.json") for stage in parent},
        "selection": "fixed setup-conflict route and equal three-model vote; no fitted router, no test-contact labels",
        "production_promoted": False})
    emit({"routing": routing, "results": {n: m["away_from_references"] for n, m in results.items()}})


def authors_soft_control(run):
    """Matched old-three-source control for the winning all-side soft HGB."""
    cases, _, soft, _, _, _ = read_features(run, refined=True, all_sides=True)
    parent = json.loads((run/"authors_all_sides/predictions.json").read_bytes())
    if [r["id"] for r in parent] != [c["contact"]["id"] for c in cases]: raise ValueError("Soft control parent differs")
    output = run/"authors_soft_control"; output.mkdir()
    groups = np.array([c["contact"]["video_id"] for c in cases]); away = np.array([c["strict"] for c in cases])
    row_groups = np.concatenate([[groups[i]]*len(c["options"]) for i, c in enumerate(cases)])
    row_away = np.concatenate([[away[i]]*len(c["options"]) for i, c in enumerate(cases)])
    y = np.concatenate([[int(j == c["target"]) for j in range(len(c["options"]))] for c in cases]); X = np.concatenate(soft)
    emit({"stage": "matched old3 soft HGB control", "manifest": str((run/"manifest.json").resolve()),
        "features": str((run/"features_all_sides").resolve()), "source_videos": sorted(set(groups)),
        "label_provenance": "frozen human snapshots421/862; only old3 fit targets",
        "feature_schema": "all-side refined hard+soft person v1", "contacts": len(cases),
        "split_policy": "complete source held out; old3 only; photo-reference guard; fixed300 iterations, seeds42/43/44"})
    cache = {}; predictions = [None]*len(cases); audit = []
    for heldout in sorted(set(groups)):
        sources = tuple(sorted(OLD-{heldout})); train = np.isin(row_groups, list(sources)) & row_away
        if sources not in cache:
            cache[sources] = [HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15,
                l2_regularization=1., min_samples_leaf=10, early_stopping=False, random_state=seed).fit(X[train], y[train]) for seed in SEEDS]
            audit.append({"heldout": heldout, "training_sources": list(sources), "test_labels_used": False})
        for i in np.flatnonzero(groups == heldout):
            score = np.mean([m.predict_proba(soft[i])[:, 1] for m in cache[sources]], axis=0)
            person = max((j for j, p in enumerate(cases[i]["options"]) if p != "other_team"), key=lambda j: score[j])
            predictions[i] = "other_team" if parent[i]["predictions"]["old3_hgb"] == "other_team" else cases[i]["options"][person]
    p = predictions
    result = {"all": score_authors(cases, p, np.ones(len(cases), bool)), "away_from_references": score_authors(cases, p, away),
        "new7_away": score_authors(cases, p, away & ~np.isin(groups, list(OLD))),
        "old3_away": score_authors(cases, p, away & np.isin(groups, list(OLD))),
        "by_video": {v: score_authors(cases, p, away & (groups == v)) for v in sorted(set(groups))},
        "by_type": {t: score_authors(cases, p, away & np.array([c["contact"]["type"] == t for c in cases])) for t in sorted({c["contact"]["type"] for c in cases})}}
    write_new(output/"predictions.json", [{"id": c["contact"]["id"], "video_id": groups[i], "truth": c["options"][c["target"]],
        "away_from_references": bool(away[i]), "predictions": {"old3_soft_hgb": p[i]}} for i, c in enumerate(cases)])
    write_new(output/"report.json", {"schema": SCHEMA, "results": {"old3_soft_hgb": result}, "fit_audit": audit,
        "matched_to": "authors_all_sides/expanded_soft_hgb: identical soft features/ranker budget/seeds; old3 team fits and transitions",
        "production_promoted": False})
    joblib.dump(cache, output/"fold_models.joblib", compress=3)
    emit(result)


def authors_listwise(run):
    """Shared scorer learns the correct person among alternatives on near labels.

    Contrast with pointwise binary ranking; team decisions stay matched to the
    frozen baseline fold predictions. No author labels enter the side decoder.
    """
    from src.tracking.contact_author_evaluation import fit_rankers, ranker_scores
    _, data = load_snapshot(run)
    cases, _, soft, _, _, _ = read_features(run, refined=True)
    base = json.loads((run/"authors/predictions.json").read_bytes())
    if [r["id"] for r in base] != [c["contact"]["id"] for c in cases]: raise ValueError("Mismatched author controls")
    output = run/"authors_listwise"; output.mkdir()
    groups = np.array([c["contact"]["video_id"] for c in cases]); away = np.array([c["strict"] for c in cases])
    near = np.array([c["options"][c["target"]] != "other_team" for c in cases]); y = np.zeros(len(cases), int)
    maximum = max(len(c["options"])-1 for c in cases)
    x = np.zeros((len(cases), maximum, soft[0].shape[-1]), np.float32); classes = np.full(x.shape[:2], -1, int)
    for i, (case, values) in enumerate(zip(cases, soft)):
        count = len(case["options"])-1
        if case["options"][-1] != "other_team": raise ValueError("Unexpected candidate order")
        x[i, :count] = values[:count]; classes[i, :count] = np.arange(count)+1
        y[i] = case["target"]+1 if near[i] else 0
    emit({"stage": "listwise author contrast", "feature_schema": "refined soft person geometry v1",
        "features": str(run/"features_refined"), "contacts": len(cases), "source_videos": sorted(set(groups)),
        "label_provenance": "human near authors only for person scorer; frozen source-held-out team decisions",
        "split_policy": "complete held-out source; train-only normalization; fixed140 epochs; seeds42/43/44",
        "heads": ["shared linear", "shared MLP48"], "numeric_person_ids_as_features": False})
    predictions = {f"{scope}_{family}": [None]*len(cases) for scope in ("old3", "expanded") for family in ("linear", "mlp")}
    fitted = {}; audit = []
    for heldout in sorted(set(groups)):
        test = groups == heldout
        for expanded in (False, True):
            sources = training_sources(groups, heldout, expanded); train = np.isin(groups, list(sources)) & away & near
            scope = "expanded" if expanded else "old3"; team_key = "expanded_hgb" if expanded else "old3_hgb"
            for family, hidden in (("linear", 0), ("mlp", 48)):
                key = (tuple(sorted(sources)), family)
                if key not in fitted:
                    fitted[key] = fit_rankers(x, classes, y, train, hidden=hidden, epochs=140, seeds=SEEDS)
                    audit.append({"heldout": heldout, "method": scope+"_"+family, "training_sources": sorted(sources),
                        "near_training_count": int(train.sum()), "test_labels_used": False})
                probabilities, _ = ranker_scores(fitted[key], x[test], classes[test], maximum+1)
                for j, i in enumerate(np.flatnonzero(test)):
                    person = int(np.argmax(probabilities[j, 1:len(cases[i]["options"])]))
                    predictions[scope+"_"+family][i] = "other_team" if base[i]["predictions"][team_key] == "other_team" else cases[i]["options"][person]
        emit({"listwise_heldout": heldout, "metrics": {n: score_authors(cases, p, away & test) for n, p in predictions.items()}})
    results = {}
    for name, p in predictions.items():
        results[name] = {"all": score_authors(cases, p, np.ones(len(cases), bool)),
            "away_from_references": score_authors(cases, p, away),
            "new7_away": score_authors(cases, p, away & ~np.isin(groups, list(OLD))),
            "old3_away": score_authors(cases, p, away & np.isin(groups, list(OLD))),
            "by_video": {v: score_authors(cases, p, away & (groups == v)) for v in sorted(set(groups))},
            "by_type": {t: score_authors(cases, p, away & np.array([c["contact"]["type"] == t for c in cases])) for t in sorted({c["contact"]["type"] for c in cases})}}
    write_new(output/"predictions.json", [{"id": c["contact"]["id"], "video_id": groups[i], "truth": c["options"][c["target"]],
        "away_from_references": bool(away[i]), "predictions": {n: p[i] for n, p in predictions.items()}} for i, c in enumerate(cases)])
    write_new(output/"report.json", {"results": results, "fit_audit": audit, "schema": SCHEMA,
        "team_predictions_sha256": sha(run/"authors/predictions.json"), "production_promoted": False})
    joblib.dump(fitted, output/"fold_models.joblib", compress=3)


def authors(run, refined=False, all_sides=False):
    manifest, data = load_snapshot(run)
    cases, hard, soft, gx, _, _ = read_features(run, refined, all_sides)
    output = run/("authors_all_sides" if all_sides else "authors_refined" if refined else "authors"); output.mkdir()
    contacts = [c["contact"] for c in cases]; groups = np.array([c["video_id"] for c in contacts])
    near = np.array([int(c["options"][c["target"]] != "other_team") for c in cases])
    away = np.array([c["strict"] for c in cases]); videos = sorted(set(groups))
    emit({**manifest, "stage": "author training", "author_cases": len(cases), "away_from_roster_references": int(away.sum()),
        "seeds": SEEDS, "methods": ["geometry", "old3_hgb", "expanded_hgb", "expanded_trees", "expanded_soft_hgb"],
        "metric_scope": "given saved candidate time/type, human authors; new timing is not human gold",
        "selection": "fixed methods; comparison is development/model selection, no untouched final test", "refined_appearance": refined})
    predictions = {name: [None]*len(cases) for name in ("geometry", "old3_hgb", "expanded_hgb", "expanded_trees", "expanded_soft_hgb")}
    audit = []; models = {}; team_cache = {}
    row_groups = np.concatenate([[groups[i]]*len(c["options"]) for i, c in enumerate(cases)])
    row_y = np.concatenate([[int(j == c["target"]) for j in range(len(c["options"]))] for c in cases])
    row_away = np.concatenate([[away[i]]*len(c["options"]) for i, c in enumerate(cases)])
    for video in videos:
        test = groups == video; test_indices = np.flatnonzero(test)
        for name in ("old3_hgb", "expanded_hgb", "expanded_trees", "expanded_soft_hgb"):
            sources = training_sources(groups, video, name != "old3_hgb")
            fit_key = (tuple(sorted(sources)), "trees" if name == "expanded_trees" else "hgb", "soft" if "soft" in name else "hard")
            train = np.isin(groups, list(sources)) & away
            if video in sources:
                raise ValueError("Test source leakage")
            if fit_key not in models:
                tx = np.concatenate(soft if "soft" in name else hard)
                row_train = np.isin(row_groups, list(sources)) & row_away
                fitted_ranker = []
                team_key = tuple(sorted(sources))
                if team_key not in team_cache:
                    emissions = []; fitted_team = []
                    for seed in SEEDS:
                        team = ExtraTreesClassifier(400, min_samples_leaf=2, max_features=.3, random_state=seed, n_jobs=4).fit(gx[train], near[train])
                        emissions.append(team.predict_proba(gx)[:, 1]); fitted_team.append(team)
                    transitions = count_transitions([c for i, c in enumerate(contacts) if train[i]], near[train], sources)
                    team_cache[team_key] = (fitted_team, np.mean(emissions, axis=0), transitions)
                for seed in SEEDS:
                    if name == "expanded_trees":
                        ranker = ExtraTreesClassifier(500, min_samples_leaf=2, max_features=.4, random_state=seed, n_jobs=4)
                    else:
                        ranker = HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15,
                            l2_regularization=1., min_samples_leaf=10, early_stopping=False, random_state=seed)
                    fitted_ranker.append(ranker.fit(tx[row_train], row_y[row_train]))
                fitted_team, emission, transitions = team_cache[team_key]
                models[fit_key] = (fitted_ranker, fitted_team, emission, transitions)
                audit.append({"heldout": video, "method": name, "training_sources": sorted(sources),
                    "training_contacts": int(train.sum()), "test_labels_used": False})
            rankers, teams, emission, transitions = models[fit_key]
            posterior = chain_posterior(contacts, emission, transitions, {video})
            for i in test_indices:
                x = soft[i] if "soft" in name else hard[i]
                scores = np.mean([m.predict_proba(x)[:, 1] for m in rankers], axis=0)
                person = max((j for j, pid in enumerate(cases[i]["options"]) if pid != "other_team"), key=lambda j: scores[j])
                predictions[name][i] = "other_team" if posterior[i] < .5 else cases[i]["options"][person]
        for i in test_indices:
            # hard columns begin with d1,n1,d3,n3,... . Geometry baseline includes both sides.
            best = int(np.argmin(np.asarray(hard[i])[:, 4]))
            predictions["geometry"][i] = cases[i]["options"][best]
        partial = {name: score_authors(cases, p, test & away) for name, p in predictions.items()}
        emit({"heldout": video, "metrics": partial})
        write_new(output/f"{video}_metrics.json", partial)
    scores = {}
    for name, p in predictions.items():
        scores[name] = {"all": score_authors(cases, p, np.ones(len(cases), bool)),
            "away_from_references": score_authors(cases, p, away),
            "new7_away": score_authors(cases, p, away & ~np.isin(groups, list(OLD))),
            "old3_away": score_authors(cases, p, away & np.isin(groups, list(OLD))),
            "by_video": {v: score_authors(cases, p, away & (groups == v)) for v in videos},
            "by_person": {f"{v}:{pid}": score_authors(cases, p, away & (groups == v) &
                np.array([c["options"][c["target"]] == pid for c in cases]))
                for v in videos for pid in [p["player_id"] for p in data["rosters"][v]["players"]]},
            "by_type": {t: score_authors(cases, p, away & np.array([c["type"] == t for c in contacts])) for t in sorted({c["type"] for c in contacts})}}
    write_new(output/"predictions.json", [{"id": c["contact"]["id"], "video_id": groups[i],
        "truth": c["options"][c["target"]], "away_from_references": bool(away[i]),
        "predictions": {n: p[i] for n, p in predictions.items()}} for i, c in enumerate(cases)])
    write_new(output/"report.json", {"schema": SCHEMA, "created_at": now(), "manifest_sha256": sha(run/"manifest.json"),
        "results": scores, "fit_audit": audit, "seeds": SEEDS, "production_promoted": False})
    # These fold models are historical experimental artifacts; no registry updates.
    joblib.dump(models, output/"fold_models.joblib", compress=3)


def report(run):
    """Keep a compact checked-in history plus complete local numerical artifacts."""
    import shutil
    import sklearn
    import torch
    manifest, data = load_snapshot(run)
    names = ["authors", "events", "events_raw", "events_validated", "events_raw_validated"]
    for feature_dir, stage in (("features_refined", "authors_refined"), ("features_all_sides", "authors_all_sides")):
        if (run/feature_dir).exists(): names.append(stage)
    if (run/"authors_listwise").exists(): names.append("authors_listwise")
    if (run/"authors_routed").exists(): names.append("authors_routed")
    if (run/"authors_soft_control").exists(): names.append("authors_soft_control")
    reports = {name: json.loads((run/name/"report.json").read_bytes()) for name in names}
    preservation = {}
    for video, meta in data["videos"].items():
        source = meta["annotations_source"]; actual = sha(source["path"])
        if actual != source["sha256"]: raise ValueError("Source annotations changed: "+video)
        stat = Path(meta["video_source"]["path"]).stat()
        if stat.st_size != meta["video_source"]["size"] or stat.st_mtime_ns != meta["video_source"]["mtime_ns"]:
            raise ValueError("Original video changed: "+video)
        preservation[video] = {"path": source["path"], "sha256": actual, "unchanged": True,
            "video": meta["video_source"], "video_stat_unchanged": True}
    feature_provenance = {}
    for feature_dir in ("features", "features_refined", "features_all_sides"):
        for path in sorted((run/feature_dir).glob("*.json")):
            saved = json.loads(path.read_bytes())
            feature_provenance[feature_dir+"/"+path.stem] = {k: saved.get(k) for k in ("features_sha256", "source_sha256", "osnet_cache", "newline_compatible_cache_reuse", "sampling", "identity", "appearance_refinement", "all_sides_compete")}
    archive = run/"code"; archive.mkdir()
    shutil.copy2(__file__, archive/Path(__file__).name)
    for path in ("scripts/evaluation/rescore_expanded_events.py", "src/tracking/contact_author_features.py", "src/tracking/contact_author_person_ranker.py",
                 "src/tracking/contact_author_evaluation.py", "src/tracking/contact_author_reid_benchmark.py",
                 "src/tracking/contact_author_benchmark.py", "action_recognition/annotator/contact_author_dataset.py",
                 "action_recognition/models/sequence_event_model.py", "src/models/player_reid_encoder.py"):
        dest = archive/path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/path, dest)
    source_proof = {"annotations": preservation, "feature_provenance": feature_provenance,
        "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "training_script_sha256": sha(__file__), "source_databases_overwritten": False,
        "related_commit": manifest["related_commit"]}
    write_new(run/"source_preservation.json", source_proof)
    bundle = {"schema": SCHEMA, "created_at": now(), "manifest": manifest, "results": reports,
        "artifacts": str(run.resolve()), "training_script_sha256": sha(__file__),
        "source_preservation_sha256": sha(run/"source_preservation.json"), "production_promoted": False}
    if (run/"final_author/artifact.json").exists():
        final_meta = json.loads((run/"final_author/artifact.json").read_bytes())
        if sha(run/"final_author/model.joblib") != final_meta["model_sha256"]: raise ValueError("Final experimental model changed")
        bundle["final_author"] = {"manifest": json.loads((run/"final_author/manifest.json").read_bytes()), "artifact": final_meta}
    write_new(run/"comparison.json", bundle)
    a = dict(reports["authors"]["results"]); e = reports["events_validated"]["results"]; raw = reports["events_raw_validated"]["results"]
    for key, prefix in (("authors_refined", "refined_"), ("authors_all_sides", "all_sides_"), ("authors_listwise", "listwise_"), ("authors_routed", "routed_"), ("authors_soft_control", "soft_control_")):
        if key in reports: a.update({prefix+n: m for n, m in reports[key]["results"].items()})
    best_author = max((n for n in a if "expanded" in n), key=lambda n: a[n]["away_from_references"]["near_author_recall"])
    control = "routed_old3_routed_hgb" if best_author.startswith("routed_") else "listwise_old3_mlp" if best_author.startswith("listwise_") else "all_sides_old3_hgb" if best_author.startswith("all_sides_") else "refined_old3_hgb" if best_author.startswith("refined_") else "old3_hgb"
    if best_author == "all_sides_expanded_soft_hgb" and "soft_control_old3_soft_hgb" in a: control = "soft_control_old3_soft_hgb"
    pct = lambda value: f"{100*value:.2f}%"
    lines = ["# Expanded contact experiments — 2026-11-06", "",
        f"Artifacts: `{run.resolve()}`", "",
        "Frozen annotation revisions 421 and 862. Ten complete source videos. "
        "1,137 human author labels (585 near-person, 552 other-team). "
        "Event types: 410 active human core clips + 28 explicit type reviews = 438. "
        "Unchanged model types are not human targets. TcUru has five roster people.", "",
        "## Author attribution", "",
        "Primary view excludes contacts within 60 frames of any roster photo. "
        "Offline candidate timing/type is supplied; new source timing remains model-derived. "
        "All controls share local OSNet photo assignment and saved input features. "
        "The old-three-source HGB control is not a reproduction of the historic full-video-identity Claude result.", "",
        "| Method | N | Near correct / N | Near recall | Near precision | Team | Exact author or other |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in a.items():
        m = result["away_from_references"]
        lines.append(f"| {name} | {m['count']} | {m['near_correct']}/{m['near_count']} | {pct(m['near_author_recall'])} | {pct(m['near_assignment_precision'])} | {pct(m['team_accuracy'])} | {pct(m['author_accuracy'])} |")
    lines += ["", "New seven-video subset (same cases for both fits):", "",
        "| Method | N | Near correct / N | Near recall | Exact author or other |", "|---|---:|---:|---:|---:|"]
    for name in dict.fromkeys(("old3_hgb", control, best_author)):
        m = a[name]["new7_away"]
        lines.append(f"| {name} | {m['count']} | {m['near_correct']}/{m['near_count']} | {pct(m['near_author_recall'])} | {pct(m['author_accuracy'])} |")
    lines += ["", f"Best observed expanded method: `{best_author}`. This is selection on development folds, not an untouched final test.", "",
        "| Held-out source | Near N | Old-three near recall | Expanded near recall | Expanded team |",
        "|---|---:|---:|---:|---:|"]
    for v, m in a[best_author]["by_video"].items():
        before = a[control]["by_video"][v]
        lines.append(f"| {v} | {m['near_count']} | {pct(before['near_author_recall'])} | {pct(m['near_author_recall'])} | {pct(m['team_accuracy'])} |")
    lines += ["", "| Contact type | Near correct / N | Near recall |", "|---|---:|---:|"]
    for t, m in a[best_author]["by_type"].items():
        lines.append(f"| {t} | {m['near_correct']}/{m['near_count']} | {pct(m['near_author_recall'])} |")
    for heading, values in (("Four-class causal event typing", e), ("Six-class causal event typing", raw)):
        lines += ["", "## "+heading, "",
            "Old three videos are the main matched typing evaluation. New types are explicit corrections only, "
            "a deliberately selected hard subset; their accuracy is not average new-video accuracy. "
            "Core collapse groups set/dig as pass and block as attack; the six-class view preserves those distinctions.", "",
            "| Method | Old-3 N | Old-3 accuracy | Old-3 macro F1 | New corrections N | New corrections accuracy |",
            "|---|---:|---:|---:|---:|---:|"]
        for name, result in values.items():
            gold = result["old3"]; new = result["new_explicit_corrections_only"]
            lines.append(f"| {name} | {gold['count']} | {pct(gold['accuracy'])} | {pct(gold['macro_f1'])} | {new['count']} | {pct(new['accuracy'])} |")
        best = max((n for n in values if n.startswith("expanded")), key=lambda n: values[n]["old3"]["macro_f1"])
        lines += ["", f"Per-class old-three-source metrics for `{best}`:", "",
            "| Type | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
        for t, m in values[best]["old3"]["per_class"].items():
            lines.append(f"| {t} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['f1-score'])} | {int(m['support'])} |")
    lines += ["", "## Limits and promotion", "",
        "Main event scores apply the production inactive-channel mask computed from training sources only. "
        "Old active clip features have zero pose/player-distance channels; feeding live values into those "
        "untrained channels gave an artificial failure on new sources. Original unmasked runs are retained "
        "as diagnostics, not the matched main control. Rescoring changes no model weights. "
        "The new four-class correction subset is 18/28 attack (64.29% constant-attack accuracy), "
        "so a comparable model score is not evidence of useful discrimination.", "",
        "Variants preserve the original side-gated feature experiment, clean-track/gallery refinement, "
        "and an all-side candidate experiment. Human near-side reference photos sometimes match saved "
        "poses marked as the opposite side (e.g. LdOgh P7/P8/P12). All-side variants compare both sides "
        "against the confirmed photo roster and use derived feature-only associations; no saved pose/team "
        "is edited. Additional inference is limited to missing OSNet crops of those additional candidates.", "",
        "No dense spotting score is available from the new review: it does not label missed contacts, "
        "background or exact contact timing. Training-source normalization, transitions and model fits "
        "exclude the full held-out source. Epoch budgets are fixed (multiscale 50, raw TCN 35), "
        "with seeds 42/43/44. No held-out early stopping or contact-level random split is used.", "",
        "All checkpoints remain experimental. Production models, registry and source annotations are unchanged. "
        "A final fit on every source would have no held-out score. The result does not validate unseen "
        "camera placements beyond the reviewed source-held-out comparisons.", "",
        "Commit 9b0b581 is accounted for by explicit OSNet batch128 and cache reuse; raw inference prefetch "
        "does not affect these saved-input training experiments. JIZ3 LF/CRLF-only legacy cache reuse "
        "preserves original stage provenance after exact normalized-code/runtime/weight/settings checks.", ""]
    if "final_author" in bundle:
        candidate = bundle["final_author"]["manifest"]
        lines += ["## Final experimental author fit", "",
            f"`final_author/model.joblib`: {candidate['selected_stage']}/{candidate['selected_method']}, "
            f"{candidate['training_contacts']} contacts from all ten sources, three seeds. "
            "This all-source fit has no held-out score; the fold reports above evaluate the method. "
            "It is not promoted or loaded by production.", ""]
    report_text = "\n".join(lines)
    (run/"REPORT.md").write_text(report_text, encoding="utf-8")
    history = ROOT/"docs/experiments"; history.mkdir(exist_ok=True)
    write_new(history/(run.name+".json"), bundle)
    with (history/(run.name+".md")).open("x", encoding="utf-8") as stream: stream.write(report_text)
    with (history/"contact_training_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"created_at": bundle["created_at"], "run": run.name, "artifacts": str(run.resolve()),
            "snapshot_sha256": manifest["snapshot_sha256"], "report": run.name+".md", "best_observed_author": best_author,
            "production_promoted": False}, ensure_ascii=False)+"\n")
    emit({"best_observed_author": best_author, "authors": a[best_author]["away_from_references"],
        "events": {n: m["old3"] for n, m in e.items()}, "events_raw": {n: m["old3"] for n, m in raw.items()}})


def final_author(run):
    """Final experimental all-source author fit; carries no held-out score."""
    methods = []
    for stage, refined, all_sides in (("authors", False, False), ("authors_refined", True, False), ("authors_all_sides", True, True)):
        saved = json.loads((run/stage/"report.json").read_bytes())
        for name, result in saved["results"].items():
            if name.startswith("expanded"):
                methods.append((result["away_from_references"]["near_author_recall"], stage, name, refined, all_sides))
    if (run/"authors_routed/report.json").exists():
        saved = json.loads((run/"authors_routed/report.json").read_bytes())
        for name, result in saved["results"].items():
            if name.startswith("expanded"):
                methods.append((result["away_from_references"]["near_author_recall"], "authors_routed", name, True, False))
    _, stage, method, refined, all_sides = max(methods, key=lambda item: item[0])
    manifest, snapshot = load_snapshot(run); cases, _, _, gx, _, _ = read_features(run, refined, all_sides)
    output = run/"final_author"; output.mkdir()
    away = np.array([c["strict"] for c in cases]); contacts = [c["contact"] for c in cases]
    near = np.array([int(c["options"][c["target"]] != "other_team") for c in cases])
    row_train = np.concatenate([[away[i]]*len(c["options"]) for i, c in enumerate(cases)])
    y = np.concatenate([[int(j == c["target"]) for j in range(len(c["options"]))] for c in cases])
    if stage == "authors_routed":
        views = ["refined", "all_sides"] if "routed" in method else ["all_sides" if "all_sides" in method else "refined"]
        families = ["hgb_hard", "trees_hard", "hgb_soft"] if method.endswith("vote") else ["hgb_hard"]
    else:
        views = ["all_sides" if all_sides else "refined" if refined else "original"]
        families = ["trees_hard" if method == "expanded_trees" else "hgb_soft" if "soft" in method else "hgb_hard"]
    metadata = {}
    for view in views:
        feature_dir = run/("features" if view == "original" else "features_"+view)
        feature_meta = json.loads(next(iter(sorted(feature_dir.glob("*.json")))).read_bytes())
        metadata[view] = {"path": str(feature_dir.resolve()), "hard_columns": feature_meta["hard_columns"],
            "soft_columns": feature_meta["soft_columns"], "global_columns": feature_meta["global_columns"]}
    contract = {"schema": "contact_author_expanded_candidate_v2", "created_at": now(),
        "manifest": str((run/"manifest.json").resolve()), "snapshot_sha256": manifest["snapshot_sha256"],
        "feature_views": metadata, "ranker_families": families,
        "source_videos": sorted(snapshot["videos"]), "label_provenance": "frozen human author snapshots421/862",
        "training_contacts": int(away.sum()), "training_near": int((away & (near == 1)).sum()),
        "feature_schema": "person candidate hard/soft v1 with explicit view/head contracts",
        "router": "confirmed near-side photo conflicts with saved pose side; unique IoU>=.35 margin>=.1; no author targets" if stage == "authors_routed" and "routed" in method else None,
        "head_combination": "equal label vote; ties prefer hgb_hard" if method.endswith("vote") else "single family",
        "selected_stage": stage, "selected_method": method, "seeds": SEEDS,
        "selection": "best observed fixed pointwise or setup-routed/vote variant on development source-held-out folds",
        "heldout_score": None, "evaluation_reference": str((run/stage/"report.json").resolve()),
        "roster_policy": snapshot["roster_policy"], "production_promoted": False}
    emit(contract); write_new(output/"manifest.json", contract)
    heads = {}; teams = []
    for view in views:
        other_cases, hard, soft, other_gx, _, _ = read_features(run, refined=view != "original", all_sides=view == "all_sides")
        if [c["contact"]["id"] for c in other_cases] != [c["contact"]["id"] for c in cases] or not np.array_equal(gx, other_gx):
            raise ValueError("Final source feature views differ in their case/global contract")
        for family in families:
            x = np.concatenate(soft if family.endswith("soft") else hard); rankers = []
            for seed in SEEDS:
                if family.startswith("trees"):
                    ranker = ExtraTreesClassifier(500, min_samples_leaf=2, max_features=.4, random_state=seed, n_jobs=4)
                else:
                    ranker = HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15,
                        l2_regularization=1., min_samples_leaf=10, early_stopping=False, random_state=seed)
                rankers.append(ranker.fit(x[row_train], y[row_train]))
            heads[view+"/"+family] = rankers
            print(f"Final author head {view}/{family} fitted", flush=True)
    for seed in SEEDS:
        teams.append(ExtraTreesClassifier(400, min_samples_leaf=2, max_features=.3, random_state=seed, n_jobs=4).fit(gx[away], near[away]))
    transitions = count_transitions([c for i, c in enumerate(contacts) if away[i]], near[away], set(snapshot["videos"]))
    joblib.dump({"manifest": contract, "heads": heads, "team_models": teams, "transitions": transitions}, output/"model.joblib", compress=3)
    write_new(output/"artifact.json", {"model_sha256": sha(output/"model.joblib"), "heldout_score": None, "production_promoted": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("snapshot", "prepare", "prepare-refined", "prepare-all-sides", "authors", "authors-refined", "authors-all-sides", "authors-listwise", "authors-routed", "authors-soft-control", "final-author", "events", "events-raw", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--projects", type=Path, nargs="+")
    parser.add_argument("--shared", type=Path, default=Path("C:/Users/illya/Documents/volleyball_analitics/volleystat"))
    parser.add_argument("--require-model-cache", action="store_true")
    args = parser.parse_args()
    if args.stage == "snapshot": snapshot(args.projects, args.run)
    elif args.stage == "prepare": prepare(args.run, args.shared, args.require_model_cache)
    elif args.stage == "prepare-refined": prepare(args.run, args.shared, True, refine=True)
    elif args.stage == "prepare-all-sides": prepare(args.run, args.shared, args.require_model_cache, refine=True, all_sides=True)
    elif args.stage == "authors": authors(args.run)
    elif args.stage == "authors-refined": authors(args.run, refined=True)
    elif args.stage == "authors-all-sides": authors(args.run, refined=True, all_sides=True)
    elif args.stage == "authors-listwise": authors_listwise(args.run)
    elif args.stage == "authors-routed": authors_routed(args.run)
    elif args.stage == "authors-soft-control": authors_soft_control(args.run)
    elif args.stage == "events": events(args.run, args.shared)
    elif args.stage == "events-raw": events_raw(args.run)
    elif args.stage == "report": report(args.run)
    elif args.stage == "final-author": final_author(args.run)


if __name__ == "__main__":
    main()
