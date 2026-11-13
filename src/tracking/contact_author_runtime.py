"""Offline contact-author runtime for the promoted all-side soft HGB.

Frozen photo setup and saved observed poses compete on both sides. No event,
identity, source pose, or human author label is rewritten by attribution.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import zlib

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.models.inference_cache import file_digest as sha
from src.tracking.player_contacts import side
from src.tracking.contact_author_person_ranker import person_candidates, pose_geometry, contact_distance

ROOT = Path(__file__).resolve().parents[2]
METHOD = "osnet_temporal_all_sides_soft_hgb_v1"
MODEL_SHA256 = "4fd55ffaea4a7280a1a87bb358c303fcfeb7a364bed92324389fe018dbf96091"
MAX_CONTACT_DISTANCE = .35


def observed_roster_support(actor, contact_frame, observations, tracks):
    """Require repeated, strong IDs on the actor's continuous source track.

    Unknown IDs, occluded-only appearance, gaps and teleports cannot establish
    a roster constraint. Raw track numbers alone never identify a person.
    """
    observation = observations[(actor["frame"], actor["pose_index"])]
    pose = observation["pose"]
    pid = pose.get("player_id")

    def strong(p):
        identity = p.get("identity") or {}
        return (identity.get("label_source") == "human" and identity.get("verified") is True
                or identity.get("reason") == "associated" and identity.get("appearance_distance", 1.) <= .25)

    if not pid or pose.get("id") is None or side(pose.get("team")) != "b" or not strong(pose):
        return None
    views = [v for v in tracks[observation["segment"]] if abs(v["frame"] - contact_frame) <= 15]
    matching = [v for v in views if v["pose"].get("player_id") == pid and side(v["pose"].get("team")) == "b"]
    strong_frames = {v["frame"] for v in matching if strong(v["pose"])}
    if len(strong_frames) < 3 or len(matching) < .9 * len(views):
        return None
    return {"player_id": pid, "strong_frames": len(strong_frames), "observations": len(views),
            "agreement": len(matching) / len(views), "method": "continuous_observed_roster_v1"}


def photo_roster(annotations, explicit=None, source_sha256=None):
    """Identity setup only; never reads contact-author decisions or model targets."""
    identity = annotations.get("player_identity") or {}
    if explicit is not None:
        roster = json.loads(Path(explicit).read_bytes()) if isinstance(explicit, (str, Path)) else deepcopy(explicit)
        if (roster.get("schema") != "contact_author_photo_roster_v1" or not isinstance(roster.get("video_sha256"), str)
                or roster.get("video_sha256") != source_sha256):
            raise ValueError("Contact photo roster belongs to another source or schema")
    else:
        if side(identity.get("side")) != "b":
            raise ValueError("Contact author model requires a near-side roster")
        by_frame = {r["f"]: r for r in annotations["frames"]}; seed = identity.get("seed") or {}
        players = []
        def reference(key, pid):
            frame, index = map(int, key.split(":")); pose = by_frame[frame]["poses"][index]
            players.append({"player_id": pid, "references": [{"frame": frame, "bbox": pose["bb"]}]})
        for key, pid in (seed.get("assignments") or {}).items(): reference(key, pid)
        for addition in seed.get("introductions") or []: reference(addition["key"], addition["player_id"])
        # Substitutes found from lineup changes (src/tracking/player_substitutions.py).
        for person in (identity.get("substitutions") or {}).get("players") or []:
            reference(person["reference_key"], person["player_id"])
        if set(identity.get("gallery_keys") or {})- {p["player_id"] for p in players}:
            raise ValueError("Saved roster is missing identity photo references")
        roster = {"schema": "contact_author_photo_roster_v1", "side": "b", "players": players,
            "label_source": seed.get("label_source", "model"), "verified": bool(seed.get("verified")),
            "scope": "identity_references_only", "video_sha256": source_sha256,
            "max_on_court": identity.get("max_on_court", 6)}
    people = roster.get("players") or []
    if roster.get("side") != "b" or not people: raise ValueError("A near-side photo roster is required")
    ids = [p.get("player_id") for p in people]
    if any(not isinstance(p, str) or not p or p in {"other_team", "near_unknown"} for p in ids) or len(set(ids)) != len(ids):
        raise ValueError("Photo roster IDs must be distinct")
    by_frame = {r["f"]: r for r in annotations["frames"]}
    for person in people:
        if not person.get("references"): raise ValueError("Every roster person requires a photo")
        for ref in person["references"]:
            bb = ref.get("bbox") or []
            if ref.get("frame") not in by_frame or len(bb) != 4 or not np.isfinite(bb).all() or bb[2] <= bb[0] or bb[3] <= bb[1]:
                raise ValueError("Invalid contact identity photo reference")
    return roster


def crop_requests(annotations, contacts, roster):
    by_frame = {r["f"]: r for r in annotations["frames"]}
    needed = {f for c in contacts for f in range(c["frame"]-15, c["frame"]+16, 3) if f in by_frame}
    requests = defaultdict(list); galleries = {}
    for f in sorted(needed):
        for i, pose in enumerate(by_frame[f].get("poses") or []):
            requests[f].append({"key": f"{f}:{i}", "bb": pose["bb"]})
    for person in roster["players"]:
        pid = person["player_id"]; galleries[pid] = []
        for i, ref in enumerate(person["references"]):
            key = f"ref:{pid}:{i}"; galleries[pid].append(key)
            requests[ref["frame"]].append({"key": key, "bb": ref["bbox"]})
    return needed, requests, galleries


def build_features(annotations, contacts, roster, vectors, context_events=None):
    """Exact winning feature path; no contact targets or source edits."""
    from src.tracking.contact_author_features import extract_contact
    by_frame = {r["f"]: r for r in annotations["frames"]}
    needed, _, galleries = crop_requests(annotations, contacts, roster)
    vectors, galleries, refinement = refine_appearance(annotations, needed, vectors, galleries)
    observations, tracks = track_segments(annotations["frames"], annotations["fps"])
    people = [p["player_id"] for p in roster["players"]]; sampled = {}
    for f in sorted(needed):
        row = deepcopy(by_frame[f]); costs = []
        for i, p in enumerate(row.get("poses") or []):
            p.pop("player_id", None); p["id"] = None
            distance = photo_distance(vectors, galleries, vectors[f"{f}:{i}"])
            costs.append([distance[pid] for pid in people])
        if costs:
            rr, cc = linear_sum_assignment(np.concatenate([np.array(costs), np.full((len(costs), len(costs)), .65)], axis=1))
            for r, c in zip(rr, cc):
                if c < len(people) and costs[r][c] < .65: row["poses"][r]["player_id"] = people[c]
        for p in row.get("poses") or []:
            p["source_team"] = p.get("team"); p["team"] = "b" if p.get("player_id") else "a"
        if row.get("ball_out_of_frame"): row["ball"] = None
        sampled[f] = row
    times = sorted(c["frame"] for c in (context_events if context_events is not None else contacts))
    matrices = []; globals_ = []; evidence = []; columns = None; global_columns = None
    for contact in contacts:
        f = contact["frame"]; local = {k: v for k, v in sampled.items() if abs(k-f) <= 15 and (k-f) % 3 == 0}
        actors = {}
        hard = person_candidates(local, contact, people, annotations["frame_height"], vote_share=1.1, evidence=actors)
        soft = soft_features(local, contact, people, vectors, galleries, annotations["frame_height"], all_sides=True)
        names = list(next(iter(hard.values())))+list(next(iter(soft.values())))
        gx, gn, _ = extract_contact({"frame": f, "type": contact["type"]}, by_frame, annotations["fps"],
            (annotations["frame_width"], annotations["frame_height"]), min((abs(t-f) for t in times if t != f), default=100))
        if columns is not None and (names != columns or gn != global_columns): raise ValueError("Feature order drift")
        columns = names; global_columns = gn
        matrices.append(np.array([list(hard[k].values())+list(soft[k].values()) for k in [*people, "other_team"]], np.float32))
        globals_.append(gx)
        # Display observed support for the soft person scorer, even when the
        # one-to-one hard assignment rejected that person. This changes only
        # review evidence, never the trained numeric inputs or source poses.
        support = {}
        for number, row in local.items():
            ball = None if row.get("ball_out_of_frame") else row.get("ball")
            if ball is None: continue
            for index, p in enumerate(row.get("poses") or []):
                g = pose_geometry(p, ball[:2])
                if g is None: continue
                for pid, distance in photo_distance(vectors, galleries, vectors[f"{number}:{index}"]).items():
                    if distance >= .6: continue
                    cost = contact_distance(g)+2*distance+.015*abs(number-f)
                    if pid not in support or cost < support[pid][0]:
                        support[pid] = (cost, {"frame": number, "pose_index": index, "bbox": p["bb"],
                            "player_id": pid, "distance_body_heights": contact_distance(g),
                            "appearance_distance": distance, "evidence_source": "observed_soft_gallery_support"})
        actors.update({pid: value[1] for pid, value in support.items()})
        for actor in actors.values():
            source_row = by_frame[actor["frame"]]
            original = source_row["poses"][actor["pose_index"]]
            actor["source_player_id"] = original.get("player_id")
            actor["roster_support"] = observed_roster_support(actor, f, observations, tracks)
            actor["raw_track_id"] = original.get("id"); actor["source_side"] = side(original.get("team"))
            actor["ball_source"] = source_row.get("ball_source") or (
                "interpolated_short" if source_row.get("ball_interp") else "saved_trajectory" if source_row.get("ball") is not None else "missing")
            actor["ball_out_of_frame"] = bool(source_row.get("ball_out_of_frame"))
            for key in ("ball_observed", "ball_reconstruction", "ball_confidence"):
                if key in source_row: actor[key] = deepcopy(source_row[key])
        evidence.append(actors)
    return {"candidate_features": matrices, "global_features": np.array(globals_, np.float32),
        "columns": columns, "global_columns": global_columns, "evidence": evidence, "refinement": refinement,
        "options": [*people, "other_team"]}


class ContactAuthorModel:
    def __init__(self, path, expected_sha256=MODEL_SHA256):
        import joblib
        self.path = Path(path); self.sha256 = sha(self.path)
        if self.sha256 != expected_sha256: raise ValueError("Contact author model checksum differs from the promoted version")
        self.bundle = joblib.load(self.path)
        self.manifest = self.bundle["manifest"]
        if (self.manifest.get("schema") != "contact_author_expanded_candidate_v2"
                or self.manifest.get("selected_method") != "expanded_soft_hgb"
                or self.manifest.get("selected_stage") != "authors_all_sides"
                or set(self.bundle["heads"]) != {"all_sides/hgb_soft"}
                or len(self.bundle["heads"]["all_sides/hgb_soft"]) != 3 or len(self.bundle["team_models"]) != 3):
            raise ValueError("Unsupported contact author checkpoint contract")

    def predict_features(self, features, contacts, on_court=None):
        """``on_court`` (``player_id -> [[first, last], ...]``) excludes players
        who were substituted off at the contact frame."""
        from src.tracking.contact_author_person_ranker import chain_posterior
        expected = self.manifest["feature_views"]["all_sides"]
        if features["columns"] != expected["soft_columns"] or features["global_columns"] != expected["global_columns"]:
            raise ValueError("Contact author feature schema differs from checkpoint")
        emission = np.mean([m.predict_proba(features["global_features"])[:, 1] for m in self.bundle["team_models"]], axis=0)
        posterior = chain_posterior(contacts, emission, self.bundle["transitions"], {c["video_id"] for c in contacts})
        predictions = []
        for i, x in enumerate(features["candidate_features"]):
            scores = np.mean([m.predict_proba(x)[:, 1] for m in self.bundle["heads"]["all_sides/hgb_soft"]], axis=0)
            eligible = np.asarray(scores[:-1], dtype=float).copy()
            rejected = {}
            for k, pid in enumerate(features["options"][:-1]):
                runs = (on_court or {}).get(pid)
                if runs is not None and not any(a <= contacts[i]["frame"] <= b for a, b in runs):
                    eligible[k] = -np.inf
                    rejected[pid] = "off_court"
            original_person = int(np.argmax(eligible))
            original_label = features["options"][original_person] if np.isfinite(eligible[original_person]) else None
            for k, pid in enumerate(features["options"][:-1]):
                if pid in rejected:
                    continue
                actor = features["evidence"][i].get(pid) or {}
                distance = actor.get("distance_body_heights")
                support = actor.get("roster_support") or {}
                if distance is None or not np.isfinite(distance) or actor.get("ball_out_of_frame"):
                    eligible[k] = -np.inf
                    rejected[pid] = "no_observed_contact_evidence"
                elif distance > MAX_CONTACT_DISTANCE:
                    eligible[k] = -np.inf
                    rejected[pid] = "too_far_from_observed_ball"
                elif support.get("player_id") and support["player_id"] != pid:
                    eligible[k] = -np.inf
                    rejected[pid] = "observed_roster_conflict"
            person = int(np.argmax(eligible))
            label = "other_team" if posterior[i] < .5 else (
                features["options"][person] if np.isfinite(eligible[person]) else None)
            if label and label != "other_team" and label != original_label:
                support = features["evidence"][i][label].get("roster_support") or {}
                if support.get("player_id") != label:
                    rejected[label] = "replacement_identity_unconfirmed"
                    label = None
            predictions.append({"label": label, "scores": scores.tolist(), "near_side_score": float(posterior[i]),
                "team_emission": float(emission[i]), "actor": features["evidence"][i].get(label),
                "model_player_id": original_label, "eligibility_rejections": rejected})
        return predictions

    def attribute(self, annotations, events, *, roster=None, cache=None, assets=None, device="auto", require_cache=False, images=None):
        from src.models.inference_cache import ModelOutputCache, FrozenOSNet, collect_embeddings, osnet_feature_store, video_frames
        from src.models.player_reid_encoder import DEFAULT_ASSETS
        from src.tracking.player_contacts import CONTACT_TYPES, SCHEMA, summarize
        video = Path(annotations.get("original_video_path") or "")
        owned_cache = cache is None
        if cache is None:
            if not video.is_file(): raise FileNotFoundError("Original video is required for source-bound contact features")
            cache = ModelOutputCache(video, require=require_cache)
        try:
            expected_source = (annotations.get("model_cache") or {}).get("source")
            if expected_source and expected_source != cache.source:
                raise ValueError("Saved analysis and contact cache refer to different sources")
            references = photo_roster(annotations, roster, cache.source.get("sha256"))
            contacts = [{"frame": int(e["frame"]), "type": e["type"], "video_id": annotations.get("video", video.name),
                "event_index": i} for i, e in enumerate(events) if e.get("type") in CONTACT_TYPES]
            if not contacts:
                return summarize({"schema": SCHEMA, "method": METHOD, "side": "b", "contacts": [],
                    "roster_ids": [p["player_id"] for p in references["players"]], "label_source": "derived", "verified": False,
                    "attribution": {"status": "no_contacts", "model_sha256": self.sha256}})
            frame_numbers = {r["f"] for r in annotations["frames"]}
            if any(c["frame"] < min(frame_numbers) or c["frame"] > max(frame_numbers) for c in contacts):
                raise ValueError("Contact is outside the saved analysis")
            needed, requests, _ = crop_requests(annotations, contacts, references)
            assets = Path(assets or DEFAULT_ASSETS)
            encoder = FrozenOSNet(assets, device=device)
            store = NewlineCompatibleFeatures(osnet_feature_store(cache, assets, assets/"osnet_msmt17.pth",
                annotations["frame_width"], annotations["frame_height"]))
            missing = sum(store.get(f, p["bb"]) is None for f, players in requests.items() for p in players)
            print(f"Contact authors: {len(contacts)} contacts; {missing} missing OSNet crops; no detector replay", flush=True)
            if missing and (require_cache or cache.require):
                from src.models.inference_cache import CacheMiss
                raise CacheMiss(f"Contact authors require {missing} missing OSNet crop embeddings")
            vectors = collect_embeddings([{"f": f, "players": p} for f, p in sorted(requests.items())],
                images or (lambda wanted: video_frames(video, wanted)), encoder, store, batch_size=128)
            features = build_features(annotations, contacts, references, vectors, context_events=contacts)
            on_court = ((annotations.get("player_identity") or {}).get("substitutions") or {}).get("on_court")
            proposals = self.predict_features(features, contacts, on_court)
            rows = []
            for contact, proposal in zip(contacts, proposals):
                event = events[contact["event_index"]]; pid = proposal["label"]; actor = proposal.pop("actor")
                if actor: actor.update(player_id=None if pid == "other_team" else pid,
                    side="a" if pid == "other_team" else "b", label_source="derived", verified=False)
                rows.append({"event_index": contact["event_index"], "frame": contact["frame"],
                    "time": contact["frame"]/annotations["fps"], "type": contact["type"], "core_type": event.get("core_type"),
                    "event_source": event.get("source"), "event_confidence": event.get("confidence"),
                    "player_id": None if pid == "other_team" else pid,
                    "status": "other_side" if pid == "other_team" else "assigned" if pid else "unresolved",
                    "reason": METHOD if pid else "no_supported_contact_actor", "label_source": "derived", "verified": False, "score_calibrated": False,
                    "candidates": [actor] if actor else [], "contact_frame": actor["frame"] if actor else contact["frame"],
                    "contact_time": (actor["frame"] if actor else contact["frame"])/annotations["fps"], **proposal})
            return summarize({"schema": SCHEMA, "method": METHOD, "side": "b", "label_source": "derived", "verified": False,
                "roster_ids": features["options"][:-1], "contacts": rows,
                "parameters": {"future_frames": True, "window_frames": 15, "stride": 3,
                    "max_body_distance": MAX_CONTACT_DISTANCE, "roster_constraint": "continuous_observed_roster_v1"},
                "attribution": {"status": "completed", "model_sha256": self.sha256, "model_path": str(self.path.resolve()),
                    "training_sources": self.manifest["source_videos"], "roster_label_source": references.get("label_source", "unknown"),
                    "roster_verified": references.get("verified", False), "missing_osnet_crops": missing,
                    "encoder": encoder.provenance, "refinement": features["refinement"]},
                "limitations": "Offline model proposals; roster setup, event and pose errors propagate. Scores are uncalibrated."})
        finally:
            if owned_cache: cache.db.close()


def attribute_contacts(annotations, events, *, enabled=True, model_path=None, roster=None, cache=None,
                       cache_dir=None, assets=None, device="auto", require_cache=False, use_registry=True, images=None):
    """Shared pipeline/replay selection. Fallbacks are explicit in exported JSON."""
    from src.models.inference_cache import ModelOutputCache, DEFAULT_CACHE
    from src.models.model_registry import resolve_model
    from src.tracking.player_contacts import build_player_contacts
    def fallback(status, error=None):
        result = build_player_contacts(events, annotations["frames"], annotations["fps"], annotations.get("player_identity"))
        result["attribution"] = {"status": status, "requested_method": METHOD, "error": str(error) if error else None}
        return result
    if not enabled: return fallback("disabled")
    if not events: return fallback("no_events")
    if roster is None:
        try: photo_roster(annotations)
        except (ValueError, KeyError, IndexError) as error: return fallback("roster_unavailable", error)
    try:
        resolved = resolve_model("contact_author", override=str(model_path) if model_path else None, use_registry=use_registry)
        model = ContactAuthorModel(resolved.path)
    except FileNotFoundError as error:
        if model_path: raise
        return fallback("model_unavailable", error)
    owned = cache is None
    if owned:
        video = Path(annotations.get("original_video_path") or "")
        if not video.is_file(): return fallback("source_unavailable", "Original source video is unavailable")
        cache = ModelOutputCache(video, cache_dir or DEFAULT_CACHE, require_cache)
    try:
        result = model.attribute(annotations, events, roster=roster, cache=cache, assets=assets,
            device=device, require_cache=require_cache, images=images)
        result["attribution"]["model_resolution"] = {**resolved.as_dict(), "sha256": model.sha256}
        result["attribution"]["model_cache"] = cache.summary()
        from src.models.inference_cache import canonical
        return json.loads(canonical(result))
    except FileNotFoundError as error:
        if assets or model_path: raise
        return fallback("encoder_unavailable", error)
    finally:
        if owned: cache.db.close()

def unit(value):
    value = np.asarray(value,dtype=np.float32)
    return value/max(float(np.linalg.norm(value)),1e-8)

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
