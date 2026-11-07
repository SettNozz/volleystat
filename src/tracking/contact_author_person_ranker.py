"""Person-level contact-author ranking over full-video roster identities.

A contact's candidates are roster people and ``other_team``. Each roster person
collects the saved near-side poses that carry its production ``player_id`` (or a
local raw-track vote for that ID) inside a short window around the known contact
frame, so a player hidden at the exact contact frame still competes. The other
team is one aggregated candidate. A team posterior from a rally-chain HMM decides
near versus far; the ranker then chooses the near person.

Ball and pose coordinates are read, never rewritten. Missing balls stay missing:
geometry is only measured on frames where the saved annotation has a ball. Human
author labels are used only as training targets of other source videos.
"""
from collections import Counter, defaultdict
import math

import numpy as np

NEAR = ("b", "team_b")
TYPES = ("serve", "reception", "block", "attack", "set", "dig", "free_ball", "other")
WINDOWS = (1, 3, 6, 10, 15)
MISSING = 9.
NO_FEATURE = ("d_tnear", "dt_tnear", "bw", "bf", "bu", "bb", "binside", "bhand", "bdx", "bdy", "bdt", "bh", "foot")


def base_type(event_type):
    raw = set(str(event_type).split(" / "))
    return next((t for t in TYPES if t in raw), "other")


def roster_seed(snapshot, video, annotations):
    """Identity roster v2 from a confirmed human photo roster; contact authors unused."""
    meta = snapshot["videos"][video]; players = snapshot["rosters"][video]["players"]
    start = players[0]["references"][0]["frame"]
    initial = [p for p in players if p["references"][0]["frame"] == start]
    by_frame = {f["f"]: f for f in annotations["frames"]}
    near = [p for p in by_frame[start].get("poses") or [] if p.get("team") in NEAR]
    if len(near) != len(initial):
        raise ValueError(f"{video}: roster frame has {len(near)} near poses for {len(initial)} people")
    introductions = []
    for person in players:
        if person in initial:
            continue
        ref = person["references"][0]; poses = by_frame[ref["frame"]].get("poses") or []
        ranked = sorted(((iou(ref["bbox"], p["bb"]), i) for i, p in enumerate(poses)), reverse=True)
        if not ranked or ranked[0][0] < .4 or len(ranked) > 1 and ranked[0][0]-ranked[1][0] < .1:
            raise ValueError(f"{video}: {person['player_id']} reference has no unique detector box")
        overlap, index = ranked[0]; pose = poses[index]
        track = [f["f"] for f in annotations["frames"] for p in f.get("poses") or []
                 if pose.get("id") is not None and p.get("id") == pose["id"] and p.get("team") in NEAR]
        introductions.append({
            "player_id": person["player_id"],
            "anchor": {"frame": ref["frame"], "bb": pose["bb"],
                       "provenance": {"label_source": "model", "verified": False, "pose_index": index,
                                      "matched_from": f"human photo-roster reference, IoU {overlap:.3f}"}},
            "available_from_frame": min(track, default=ref["frame"]),
            "availability_provenance": {"label_source": "derived", "verified": False,
                                        "method": "first selected observation of anchor raw track"},
            "identity_confirmation": {"label_source": "human", "verified": True, "scope": "new_person",
                                      "note": f"confirmed new person in contact-author roster revision {snapshot['revision']}"}})
    return {"schema": "player_identity_roster_v2", "video_sha256": meta["video_source"]["sha256"],
            "video_name": meta["video_source"]["path"].replace("\\", "/").rsplit("/", 1)[-1],
            "side": "b", "max_on_court": len(initial),
            "initial": {"frame": start, "label_source": "human", "verified": True,
                        "players": [{"player_id": p["player_id"], "bb": p["references"][0]["bbox"]} for p in initial]},
            "introductions": introductions,
            "provenance": {"source": "contact-author photo roster; human identity confirmation",
                           "human_labels_revision": snapshot["revision"], "contact_author_targets_used": False}}


def iou(a, b):
    area = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    return area/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area, 1.)


def _segment(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b)); ab = b-a
    t = np.clip(np.dot(p-a, ab)/max(np.dot(ab, ab), 1e-6), 0, 1)
    return float(np.linalg.norm(p-(a+t*ab)))


def pose_geometry(pose, ball):
    """Body-height-normalized distances from an observed ball to hands, arms and body."""
    kp = {j: (k[0], k[1]) for j, k in enumerate(pose.get("kp") or []) if len(k) > 2 and k[2] >= .3}
    box = pose["bb"]; h = max(box[3]-box[1], 1.)
    if not kp:
        return None
    wrists = [math.dist(ball, kp[j]) for j in (9, 10) if j in kp]
    forearms = [_segment(ball, kp[a], kp[b]) for a, b in ((7, 9), (8, 10)) if a in kp and b in kp]
    uppers = [_segment(ball, kp[a], kp[b]) for a, b in ((5, 7), (6, 8)) if a in kp and b in kp]
    shoulders = [kp[j][1] for j in (5, 6) if j in kp]; hands = [kp[j][1] for j in (9, 10) if j in kp]
    return {"w": min(wrists)/h if wrists else MISSING, "f": min(forearms)/h if forearms else MISSING,
            "u": min(uppers)/h if uppers else MISSING, "b": min(math.dist(ball, q) for q in kp.values())/h,
            "inside": float(box[0] <= ball[0] <= box[2] and box[1]-.3*h <= ball[1] <= box[3]),
            "hand_up": (min(hands)-np.mean(shoulders))/h if shoulders and hands else 0.,
            "dx": (ball[0]-(box[0]+box[2])/2)/h, "dy": (ball[1]-box[1])/h, "h": h}


def contact_distance(g):
    return min(g["w"], g["f"], g["u"]+.05, g["b"]+.2)


def track_votes(by_frame, frame, span=150):
    """Time-weighted production IDs per raw near track; raw IDs are not person truth."""
    votes = defaultdict(Counter)
    for f in range(frame-span, frame+span+1):
        for pose in (by_frame.get(f) or {}).get("poses") or []:
            if pose.get("player_id") and pose.get("id") is not None and pose.get("team") in NEAR:
                votes[pose["id"]][pose["player_id"]] += 1/(1+abs(f-frame)/30)
    return votes


def person_candidates(by_frame, contact, people, height, window=15, vote_share=.6, evidence=None):
    """Feature dict per roster person plus ``other_team`` for one known contact.

    ``evidence``, when given, receives each candidate's saved pose that the
    features measured closest to the ball (or nearest in time without a ball).
    """
    frame = contact["frame"]; votes = track_votes(by_frame, frame); observations = defaultdict(list)
    ball_frames = []
    for f in range(frame-window, frame+window+1):
        row = by_frame.get(f) or {}
        ball = row.get("ball")[:2] if row.get("ball") is not None else None
        if ball is not None:
            ball_frames.append(abs(f-frame))
        for index, pose in enumerate(row.get("poses") or []):
            pose = {**pose, "_frame": f, "_index": index}
            if pose.get("team") in NEAR:
                key, source = pose.get("player_id"), "pid"
                if key is None and pose.get("id") in votes:
                    top, count = votes[pose["id"]].most_common(1)[0]
                    if count/sum(votes[pose["id"]].values()) >= vote_share:
                        key, source = top, "vote"
                key = key or "near_unknown"
            else:
                key, source = "other_team", "far"
            observations[key].append((f-frame, pose_geometry(pose, ball) if ball is not None else None, pose, source))

    def summary(key):
        rows = observations.get(key, []); x = {}
        for w in WINDOWS:
            distances = [contact_distance(g) for dt, g, *_ in rows if g is not None and abs(dt) <= w]
            x[f"d{w}"] = min(distances, default=MISSING); x[f"n{w}"] = sum(abs(dt) <= w for dt, *_ in rows)
        measured = [(abs(dt), contact_distance(g), g, dt, p) for dt, g, p, _ in rows if g is not None]
        if measured:
            closest = min(measured, key=lambda t: (t[0], t[1])); best = min(measured, key=lambda t: t[1]+.02*t[0])
            g = best[2]
            x.update(d_tnear=closest[1], dt_tnear=closest[0], bw=g["w"], bf=g["f"], bu=g["u"], bb=g["b"],
                     binside=g["inside"], bhand=g["hand_up"], bdx=g["dx"], bdy=g["dy"], bdt=best[3],
                     bh=g["h"]/height, foot=best[4]["bb"][3]/height)
        else:
            x.update({k: -1. if k == "binside" else MISSING for k in NO_FEATURE})
        x["src_pid"] = float(np.mean([s == "pid" for *_, s in rows])) if rows else 0.
        x["present0"] = float(any(dt == 0 for dt, *_ in rows))
        if evidence is not None and rows:
            p = best[4] if measured else min(rows, key=lambda r: abs(r[0]))[2]
            evidence[key] = {"frame": p["_frame"], "pose_index": p["_index"], "raw_track_id": p.get("id"),
                             "bbox": p["bb"], "player_id": p.get("player_id"),
                             "distance_body_heights": best[1] if measured else None}
        return x

    rows = {k: summary(k) for k in [*people, "other_team", "near_unknown"]}
    visible = sorted((k for k in people if rows[k]["foot"] < MISSING), key=lambda k: -rows[k]["foot"])
    missing = sum(rows[k]["n3"] == 0 for k in people); kind = base_type(contact["type"]); out = {}
    for key in [*people, "other_team"]:
        x = dict(rows[key]); x["is_other"] = float(key == "other_team")
        for w in (3, 6, 15):
            others = [rows[j][f"d{w}"] for j in rows if j != key]
            x[f"gap{w}"] = x[f"d{w}"]-min(others); x[f"rank{w}"] = float(sum(o < x[f"d{w}"] for o in others))
        x.update(unk_d3=rows["near_unknown"]["d3"], unk_d6=rows["near_unknown"]["d6"],
                 unk_n3=rows["near_unknown"]["n3"], nmiss=missing,
                 missing=float(key != "other_team" and rows[key]["n3"] == 0),
                 # Larger image y is nearer the fixed camera: back row. Rank 0 = deepest back.
                 depth_rank=float(visible.index(key)) if key in visible else -1., nvis=len(visible),
                 ball_dt=min(ball_frames, default=99))
        x.update({f"t_{t}": float(kind == t) for t in TYPES})
        out[key] = x
    return out


def rally_chains(contacts, gap=150):
    """Contact index chains per source; reset at serves, long gaps and source changes."""
    order = sorted(range(len(contacts)), key=lambda i: (contacts[i]["video_id"], contacts[i]["frame"]))
    chains, current = [], []
    for i in order:
        c = contacts[i]
        if current and (c["video_id"] != contacts[current[-1]]["video_id"] or base_type(c["type"]) == "serve"
                        or c["frame"]-contacts[current[-1]]["frame"] > gap):
            chains.append(current); current = []
        current.append(i)
    return chains+[current] if current else chains


def count_transitions(contacts, near, train, alpha=1.):
    """P(same team | previous type, type), counted from training sources only."""
    counts = defaultdict(lambda: [alpha, alpha])
    for chain in rally_chains(contacts):
        if contacts[chain[0]]["video_id"] not in train:
            continue
        for a, b in zip(chain, chain[1:]):
            counts[base_type(contacts[a]["type"]), base_type(contacts[b]["type"])][int(near[a] == near[b])] += 1
    return {k: v[1]/sum(v) for k, v in counts.items()}


def chain_posterior(contacts, p_near, transitions, sources):
    """Two-state forward-backward over each rally chain of the given sources."""
    posterior = np.asarray(p_near, float).copy()
    for chain in rally_chains(contacts):
        if contacts[chain[0]]["video_id"] not in sources:
            continue
        e = np.array([[1-p_near[i], p_near[i]] for i in chain]); steps = []
        for a, b in zip(chain, chain[1:]):
            same = transitions.get((base_type(contacts[a]["type"]), base_type(contacts[b]["type"])), .5)
            steps.append(np.array([[same, 1-same], [1-same, same]]))
        forward = np.zeros_like(e); forward[0] = e[0]/e[0].sum()
        for i in range(1, len(chain)):
            forward[i] = forward[i-1]@steps[i-1]*e[i]; forward[i] /= forward[i].sum()
        backward = np.ones_like(e)
        for i in range(len(chain)-2, -1, -1):
            backward[i] = steps[i]@(e[i+1]*backward[i+1]); backward[i] /= backward[i].sum()
        joint = forward*backward
        for i, value in zip(chain, joint[:, 1]/joint.sum(axis=1)):
            posterior[i] = value
    return posterior


def decide(person_scores, p_near, threshold=.5):
    """``other_team`` below the team threshold, else the best-scoring near person."""
    near = {k: v for k, v in person_scores.items() if k != "other_team"}
    return "other_team" if p_near < threshold or not near else max(near, key=near.get)
