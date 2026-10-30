"""Frozen 2D contact/pose features for an experimental learned team selector.

No human ball coordinates, court-side projection, raw track IDs, player IDs,
video names or absolute match times enter the numeric model input. Candidate
identity is metadata for review only; side labels do not supervise a person.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import hashlib
import inspect
from statistics import median

import numpy as np

from src.tracking.player_contacts import BONES, CONTACT_TYPES, _distance, _point, side, summarize

SCHEMA = "contact_author_side_frozen_v1"
ACTOR_FIELDS = (
    "wrist", "forearm", "upper_arm", "body", "two_wrists", "hand_dx", "hand_dy",
    "shoulder_dx", "shoulder_dy", "hip_dx", "hip_dy", "head_dx", "head_dy",
    "wrist_height", "wrist_span", "arm_extension", "relative_height", "aspect",
    "bbox_dx", "bbox_dy", "valid_wrists", "pose_confidence", "offset_seconds",
    "wrist_speed", "relative_approach", "relative_departure", "relative_cusp",
)


def ball(ann):
    value = ann.get("ball") if ann and not ann.get("ball_out_of_frame") else None
    return value if _point(value) else None


def points(pose):
    return {i: p for i, p in enumerate(pose.get("kp") or [])
            if _point(p) and len(p) == 3 and math.isfinite(p[2]) and p[2] >= .25}


def geometry(pose, point, scale):
    box = pose.get("bb") or []
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[3] <= box[1]:
        return None
    kp = points(pose)
    if not kp:
        return None
    h = box[3]-box[1]
    dist = lambda j: math.dist(point[:2], kp[j][:2]) if j in kp else None
    wrist = [(dist(j), j) for j in (9, 10) if j in kp]
    wrist.sort()
    forearms = [(_distance(point, kp[a], kp[b]), (a, b)) for a, b in ((7, 9), (8, 10)) if a in kp and b in kp]
    arms = [_distance(point, kp[a], kp[b]) for a, b in ((5, 7), (6, 8)) if a in kp and b in kp]
    body_distances = [math.dist(point[:2], p[:2]) for p in kp.values()]
    body_distances += [_distance(point, kp[a], kp[b]) for a, b in BONES if a in kp and b in kp]
    hand = kp[wrist[0][1]][:2] if wrist else None
    mean = lambda ids: [sum(kp[j][k] for j in ids)/len(ids) for k in (0, 1)] if all(j in kp for j in ids) else None
    shoulder, hip = mean((5, 6)), mean((11, 12))
    head = kp.get(0)
    relative = lambda anchor: [(point[k]-anchor[k])/h for k in (0, 1)] if anchor is not None else [None, None]
    values = [
        wrist[0][0]/scale if wrist else None,
        min(d for d, _ in forearms)/scale if forearms else None,
        min(arms)/scale if arms else None,
        min(body_distances)/scale,
        sum(d for d, _ in wrist)/len(wrist)/scale if len(wrist) == 2 else None,
        *relative(hand), *relative(shoulder), *relative(hip), *relative(head),
        (hand[1]-shoulder[1])/h if hand is not None and shoulder is not None else None,
        math.dist(kp[9][:2], kp[10][:2])/h if 9 in kp and 10 in kp else None,
        max((math.dist(kp[a][:2], kp[b][:2])/h for a, b in ((5, 9), (6, 10)) if a in kp and b in kp), default=None),
        h/scale, (box[2]-box[0])/h,
        (point[0]-(box[0]+box[2])*.5)/h, (point[1]-(box[1]+box[3])*.5)/h,
        len(wrist)/2, sum(p[2] for p in kp.values())/len(kp),
    ]
    cost = min([d/scale for d, _ in wrist] + [d/scale for d, _ in forearms]
               + [min(body_distances)/scale+.20])
    chosen = min([(d, "wrist", d) for d, _ in wrist] + [(d, "forearm", d) for d, _ in forearms]
                 + [(min(body_distances)+.20*scale, "body_fallback", min(body_distances))])
    return {"values": values, "cost": cost, "distance_px": chosen[2], "part": chosen[1],
            "height": h, "wrist_joint": wrist[0][1] if wrist else None, "hand": hand}


def _velocity(samples):
    t = np.asarray([r[0] for r in samples], dtype=float)
    xy = np.asarray([r[1][:2] for r in samples], dtype=float)
    t -= t.mean()
    return (t[:, None]*xy).sum(axis=0)/np.square(t).sum() if np.square(t).sum() else np.zeros(2)


def motion(by_frame, frame, scale, fps):
    if ball(by_frame.get(frame)) is None:
        return [None]*8+[0., 0.]
    runs = []
    for direction in (-1, 1):
        samples = []
        for offset in range(1, 7):
            f = frame+direction*offset
            row = by_frame.get(f)
            newer = row if direction > 0 else by_frame.get(f+1)
            if ball(row) is None or newer.get("ball_new") is True:
                break
            samples.append((f, ball(row)))
        runs.append(samples)
    vectors = [_velocity(r) * fps/scale if len(r) >= 3 else None for r in runs]
    values = []
    for vector in vectors:
        values += list(vector) + [float(np.linalg.norm(vector))] if vector is not None else [None]*3
    if all(v is not None for v in vectors):
        before, after = vectors
        values += [float(np.linalg.norm(after-before)), float(np.dot(before, after)/(np.linalg.norm(before)*np.linalg.norm(after)+1e-6))]
    else:
        values += [None, None]
    values += [len(r)/6 for r in runs]
    return values


def _same_pose(ann, pose):
    if ann is None or pose.get("id") is None:
        return None
    options = [p for p in ann.get("poses") or [] if p.get("id") == pose["id"] and side(p.get("team")) == side(pose.get("team"))]
    return options[0] if len(options) == 1 else None


def _temporal_metrics(by_frame, frame, pose, geom, scale, fps):
    values = []
    distances, hands = {}, {}
    for delta in (-2, 2):
        target = frame+delta
        if any(ball(by_frame.get(f)) is None or (f > min(frame, target) and by_frame[f].get("ball_new") is True)
               for f in range(min(frame, target), max(frame, target)+1)):
            continue
        other = _same_pose(by_frame.get(target), pose)
        if other is None:
            continue
        g = geometry(other, ball(by_frame[target]), scale)
        if g is not None:
            distances[delta] = g["cost"]
            hands[delta] = g["hand"]
    speed = math.dist(hands[-2], hands[2])*fps/(4*scale) if hands.get(-2) is not None and hands.get(2) is not None else None
    approach = distances[-2]-geom["cost"] if -2 in distances else None
    departure = distances[2]-geom["cost"] if 2 in distances else None
    cusp = min(approach, departure) if approach is not None and departure is not None else None
    return [speed, approach, departure, cusp]


def _resolve(bucket, best):
    ids = {o["player_id"] for o in bucket if o["player_id"]}
    if len(ids) > 1:
        return None, {"method": "conflicting_local_identities"}
    if best["player_id"]:
        return best["player_id"], {"method": "observed_reid", "frame": best["frame"]}
    if any(w in str(best["identity"].get("reason", "")) for w in ("conflict", "contradiction", "veto")):
        return None, {"method": "reid_conflict_preserved"}
    before = [o for o in bucket if o["frame"] < best["frame"] and o["player_id"]]
    after = [o for o in bucket if o["frame"] > best["frame"] and o["player_id"]]
    if not ids or not before or not after:
        return None, {"method": "requires_two_sided_identity"}
    left, right = max(before, key=lambda o: o["frame"]), min(after, key=lambda o: o["frame"])
    if any(o["identity"].get("appearance_distance") is None or o["identity"]["appearance_distance"] > .4 for o in (left, right)):
        return None, {"method": "weak_identity_anchors"}
    t = (best["frame"]-left["frame"])/(right["frame"]-left["frame"])
    expected = [a*(1-t)+b*t for a, b in zip(left["bbox"], right["bbox"])]
    centre = lambda b: [(b[0]+b[2])*.5, (b[1]+b[3])*.5]
    h = expected[3]-expected[1]
    if h <= 0 or math.dist(centre(expected), centre(best["bbox"]))/h > .2:
        return None, {"method": "identity_motion_conflict"}
    return next(iter(ids)), {"method": "two_sided_identity_proposal", "left_frame": left["frame"], "right_frame": right["frame"], "verified": False}


def extract_contact(event, by_frame, fps, frame_size, neighbour_distance=None):
    frame = int(event["frame"])
    radius = max(1, round(.20*fps))
    if neighbour_distance is not None:
        radius = min(radius, max(0, (neighbour_distance-1)//2))
    buckets = defaultdict(list)
    for f in range(frame-radius, frame+radius+1):
        ann = by_frame.get(f)
        b = ball(ann)
        if b is None or by_frame.get(frame, {}).get("ball_out_of_frame"):
            continue
        if any(t not in by_frame or by_frame[t].get("ball_out_of_frame") or
               (t > min(f, frame) and by_frame[t].get("ball_new") is True)
               for t in range(min(f, frame), max(f, frame)+1)):
            continue
        poses = ann.get("poses") or []
        scale = median([p["bb"][3]-p["bb"][1] for p in poses if len(p.get("bb") or []) == 4 and p["bb"][3] > p["bb"][1]] or [1.])
        for index, pose in enumerate(poses):
            team = side(pose.get("team"))
            g = geometry(pose, b, scale)
            if g is None or team is None:
                continue
            vector = g["values"] + [(f-frame)/fps] + _temporal_metrics(by_frame, f, pose, g, scale, fps)
            assert len(vector) == len(ACTOR_FIELDS)
            score = g["cost"] + .06*((f-frame)/max(radius, 1))**2
            candidate = {"frame": f, "pose_index": index, "raw_track_id": pose.get("id"),
                         "player_id": pose.get("player_id"), "side": team, "score": round(score, 6),
                         "distance_px": round(g["distance_px"], 3), "distance_body_heights": round(g["distance_px"]/g["height"], 5),
                         "contact_part": g["part"], "bbox": pose["bb"], "identity": deepcopy(pose.get("identity") or {}),
                         "ball_source": ann.get("ball_source") or ("interpolated_short" if ann.get("ball_interp") else "detected"),
                         "vector": vector, "scale": scale, "ball": b}
            key = (team, pose["id"]) if pose.get("id") is not None else (team, f, index)
            buckets[key].append(candidate)
    actors = {}
    for key, observations in buckets.items():
        best = deepcopy(min(observations, key=lambda o: o["score"]))
        pid, resolution = _resolve(observations, best)
        best["player_id"], best["identity_resolution"] = pid, resolution
        best["support_frames"] = len(set(o["frame"] for o in observations))
        actor = (best["side"], "player", pid) if pid else key
        if actor not in actors or best["score"] < actors[actor]["score"]:
            actors[actor] = best
    ranked = sorted(actors.values(), key=lambda o: o["score"])
    names, values = [], []
    def add(name, value):
        names.extend((name, name+"__available"))
        valid = value is not None and math.isfinite(value)
        values.extend((float(np.clip(value, -5, 5)) if valid else 0., float(valid)))
    for team in ("a", "b"):
        selected = [o for o in ranked if o["side"] == team]
        for rank in range(3):
            vector = selected[rank]["vector"] if rank < len(selected) else [None]*len(ACTOR_FIELDS)
            for field, value in zip(ACTOR_FIELDS, vector):
                add(f"{team}_actor{rank}_{field}", value)
        add(f"{team}_candidate_count", len(selected))
        add(f"{team}_actor_margin", selected[1]["score"]-selected[0]["score"] if len(selected) >= 2 else None)
    anchor = by_frame.get(frame)
    scale = median([o["scale"] for o in ranked] or [1.])
    for name, value in zip(("vx_before", "vy_before", "speed_before", "vx_after", "vy_after", "speed_after", "velocity_change", "turn_cos", "before_valid", "after_valid"), motion(by_frame, frame, scale, fps)):
        add(name, value)
    raw = event["type"]
    core = "pass" if raw in ("pass", "set", "setting_pass", "dig", "free_ball") else "reception" if raw == "reception_pass" else raw
    for kind in ("serve", "reception", "pass", "attack", "block"):
        add("type_"+kind, float(core == kind))
    b = ball(anchor)
    add("layout_ball_x", b[0]/frame_size[0] if b else None)
    add("layout_ball_y", b[1]/frame_size[1] if b else None)
    public = [{k: v for k, v in actor.items() if k not in ("vector", "scale")} for actor in ranked]
    return np.asarray(values, dtype=np.float32), names, public


def extract_events(events, frames, fps, frame_size):
    by_frame = {a["f"]: a for a in frames}
    times = sorted(set(int(e["frame"]) for e in events if e.get("type") in CONTACT_TYPES))
    vectors, candidates, indices, names = [], [], [], None
    for index, event in enumerate(events):
        if event.get("type") not in CONTACT_TYPES:
            continue
        neighbours = [abs(t-int(event["frame"])) for t in times if t != int(event["frame"])]
        vector, fields, actors = extract_contact(event, by_frame, float(fps), frame_size, min(neighbours) if neighbours else None)
        if names is not None and names != fields:
            raise ValueError("Inconsistent feature order")
        vectors.append(vector); candidates.append(actors); indices.append(index); names = fields
    return np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32), names or [], candidates, indices


def feature_fingerprint():
    functions = (ball, points, geometry, _velocity, motion, _same_pose, _temporal_metrics, _resolve, extract_contact, extract_events)
    contract = SCHEMA + repr(ACTOR_FIELDS) + repr(BONES) + repr(sorted(CONTACT_TYPES))
    contract += "\n".join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(contract.encode()).hexdigest()


def contact_report(events, actors, indices, scores, fps, identity, method_id,
                   actor_margin=.035, side_threshold=.65, max_actor_distance=.50, actor_share_min=None):
    """Learned side + observed candidate ranking. Person labels were NOT learned."""
    target = side((identity or {}).get("side")) or "b"
    roster = set(((identity or {}).get("gallery_keys") or {}).keys())
    roster.update(c["player_id"] for row in actors for c in row if c.get("player_id") and c["side"] == target)
    contacts = []
    for index, candidates, score in zip(indices, actors, scores):
        score = float(score)
        event = events[index]
        proposed_side = "b" if score >= .5 else "a"
        chosen = [c for c in candidates if c["side"] == proposed_side]
        best = chosen[0] if chosen else None
        ranked = chosen + [c for c in candidates if c["side"] != proposed_side]
        margin = chosen[1]["score"]-chosen[0]["score"] if len(chosen) >= 2 else None
        contact = {"event_index": index, "frame": int(event["frame"]), "time": int(event["frame"])/fps,
                   "type": event["type"], "core_type": event.get("core_type"),
                   "event_source": event.get("source"), "event_confidence": event.get("confidence"),
                   "player_id": None, "status": "unresolved", "reason": "ambiguous_team_model",
                   "label_source": "derived", "verified": False,
                   "near_side_score": float(score), "score_calibrated": False,
                   "side_proposal": proposed_side, "side_accepted": score >= side_threshold or score <= 1-side_threshold,
                   "candidates": deepcopy(ranked[:4]), "margin": float(margin) if margin is not None else None}
        if best:
            contact.update(contact_frame=best["frame"], contact_time=best["frame"]/fps,
                           identity_resolution=best["identity_resolution"], uses_estimated_joints=False,
                           proposed_player_id=best["player_id"] if proposed_side == target else None)
            if "actor_share" in best:
                contact["actor_share"] = best["actor_share"]
        if not contact["side_accepted"]:
            pass
        elif best is None:
            contact["reason"] = "no_supported_pose"
        elif best["distance_body_heights"] > max_actor_distance:
            contact["reason"] = "too_far"
        elif proposed_side != target:
            contact.update(status="other_side", reason="learned_other_side")
        elif (margin is not None and margin < actor_margin) or (actor_share_min is not None and best.get("actor_share", 0.) < actor_share_min):
            contact["reason"] = "ambiguous_players"
        elif not best["player_id"]:
            contact["reason"] = "unknown_identity"
        else:
            contact.update(player_id=best["player_id"], status="assigned", reason="learned_side_observed_actor")
        contacts.append(contact)
    return summarize({"schema": "player_contacts_v1", "method": "learned_side_observed_author_v1", "method_id": method_id,
                      "side": target, "roster_ids": sorted(roster, key=lambda p: (int(p[1:]) if p[1:].isdigit() else math.inf, p)),
                      "label_source": "derived", "verified": False, "contacts": contacts,
                      "parameters": {"side_score_threshold": side_threshold, "max_actor_body_distance": max_actor_distance,
                                     "actor_margin": actor_margin, "actor_share_min": actor_share_min,
                                     "estimated_pose_evidence": False},
                      "limitations": "Only team side is learned/validated. Individual author is a geometric/temporal proposal, not labelled accuracy. Source events may be missed or false."})
