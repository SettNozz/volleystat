"""Offline volleyball hypotheses over frozen observations, never detector labels.

The quadratic fit is an image-space approximation, not a calibrated 3D flight.
Sequence priors learn only team transitions; they do not verify a person's ID.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
import json
from pathlib import Path

import numpy as np

from src.tracking.contact_author_features import ball, geometry
from src.tracking.player_contacts import side

SCHEMA = "volleyball_contact_lab_v1"
EVIDENCE_NAMES = ("ballistic", "interaction", "joint")


def load_lab(path, source, stats):
    bundle = json.loads(Path(path).read_bytes())
    if bundle.get("schema") != SCHEMA or bundle.get("source") != source:
        raise ValueError("Contact lab source changed; rebuild into a fresh output")
    methods = bundle.get("methods") or {}
    if bundle.get("selected_method") not in methods:
        raise ValueError("Missing selected contact experiment")
    original = stats.get("all_events") or []
    for method in methods.values():
        for event in method["events"]:
            if event.get("label_source") != "derived" or event.get("verified") is not False:
                raise ValueError("Experimental proposals cannot claim human verification")
            if type(event.get("frame")) is not int or event["frame"] < 0:
                raise ValueError("Invalid contact frame")
            if event.get("player_id") is not None and event["player_id"] not in bundle["roster_ids"]:
                raise ValueError("Unknown roster identity")
            index = event.get("source_event_index")
            if index is not None:
                if type(index) is not int or not 0 <= index < len(original):
                    raise ValueError("Invalid source event index")
                old = original[index]
                if event.get("original_frame") != old["frame"] or event["type"] != old["type"]:
                    raise ValueError("Source type/frame provenance was changed")
    return bundle


def observed_ball(frame):
    if not frame or frame.get("ball_interp") or frame.get("ball_reconstructed"):
        return None
    source = str(frame.get("ball_source", "detected"))
    if "reconstruct" in source or "interpolat" in source:
        return None
    return ball(frame)


def dense_evidence(frames, length=None, radius=6):
    """Piecewise flight and moving-limb proximity, with explicit support masks.

    Gaps, off-camera frames and new tracks invalidate flight fits. No hidden
    pose/ball is filled. Pose distances before/after use the SAME raw track.
    """
    if radius < 3:
        raise ValueError("At least three samples per flight branch required")
    length = length or max(int(a["f"]) for a in frames)+1
    xy = np.full((length, 2), np.nan)
    scale = np.ones(length)
    restarts = np.zeros(length, dtype=bool)
    actors = [dict() for _ in range(length)]
    for row in frames:
        f = int(row["f"])
        if not 0 <= f < length:
            continue
        restarts[f] = bool(row.get("ball_new"))
        b = observed_ball(row)
        if b is None:
            continue
        xy[f] = b[:2]
        poses = row.get("poses") or []
        heights = [p["bb"][3]-p["bb"][1] for p in poses
                   if len(p.get("bb") or []) == 4 and p["bb"][3] > p["bb"][1]]
        scale[f] = np.median(heights) if heights else 200.
        for i, pose in enumerate(poses):
            team = side(pose.get("team"))
            g = geometry(pose, b, scale[f]) if team else None
            if g is None:
                continue
            # Unknown raw IDs cannot establish temporal continuity.
            key = (team, pose["id"]) if pose.get("id") is not None else (team, f, i)
            actors[f][key] = {"cost": g["cost"], "hand": g["hand"], "pose_index": i}

    output = np.zeros((length, 3), dtype=np.float32)
    support = np.zeros(length, dtype=bool)
    before = np.arange(-radius, 1, dtype=float)
    after = np.arange(0, radius+1, dtype=float)
    design = lambda t: np.column_stack((np.ones(len(t)), t, t*t))
    left, right = design(before), design(after)
    whole = design(np.arange(-radius, radius+1, dtype=float))
    inv_left, inv_right, inv_whole = map(np.linalg.pinv, (left, right, whole))
    for f in range(radius, length-radius):
        values = xy[f-radius:f+radius+1]
        if not np.isfinite(values).all() or restarts[f-radius+1:f+radius+1].any():
            continue
        values = (values-xy[f])/scale[f]
        lcoef, rcoef = inv_left@values[:radius+1], inv_right@values[radius:]
        error_one = np.square(values-whole@(inv_whole@values)).sum()
        error_two = (np.square(values[:radius+1]-left@lcoef).sum()
                     + np.square(values[radius:]-right@rcoef).sum())
        improvement = max(0., 1-error_two/max(error_one, 1e-8))
        impulse = np.linalg.norm(rcoef[1]-lcoef[1])
        continuity = np.linalg.norm(rcoef[0]-lcoef[0])
        speed = max(np.linalg.norm(lcoef[1]), np.linalg.norm(rcoef[1]))
        output[f, 0] = ((1-np.exp(-impulse/.06))*improvement
                        * np.exp(-continuity/.25)*(1-np.exp(-speed/.02)))
        support[f] = True

    for f in range(3, length-3):
        if not np.isfinite(xy[f-3:f+4]).all() or restarts[f-2:f+4].any():
            continue
        best = 0.
        for key, actor in actors[f].items():
            previous, following = actors[f-3].get(key), actors[f+3].get(key)
            if previous is None or following is None:
                continue
            approach = min(previous["cost"]-actor["cost"], following["cost"]-actor["cost"])
            # Relative hand/ball segment tests fast between-frame approaches.
            swept = actor["cost"]
            if previous["hand"] is not None and following["hand"] is not None:
                a = xy[f-3]-previous["hand"]
                b = xy[f+3]-following["hand"]
                delta = b-a
                t = np.clip(-np.dot(a, delta)/(np.dot(delta, delta)+1e-9), 0., 1.)
                swept = min(swept, np.linalg.norm(a+t*delta)/scale[f])
            closeness = np.exp(-swept/.22)
            pulse = np.clip(approach/.10, 0., 1.)
            # A 2D crossing alone is weak; a distance minimum strengthens it.
            best = max(best, closeness*(.20+.80*pulse))
        output[f, 1] = best
    output[:, 2] = np.sqrt(output[:, 0]*output[:, 1])
    return output, support


def peaks(scores, threshold, radius=5):
    """Score-ordered NMS, linear suppression storage instead of all-pairs."""
    values = np.asarray(scores)
    if values.ndim != 1 or radius < 0 or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional scores and nonnegative radius required")
    candidates = np.flatnonzero((values >= threshold) & (values > 0))
    blocked = np.zeros(len(values), dtype=bool)
    selected = []
    for f in candidates[np.argsort(-values[candidates], kind="stable")]:
        if not blocked[f]:
            selected.append(int(f))
            blocked[max(0, f-radius):min(len(values), f+radius+1)] = True
    return sorted(selected)


def consensus(source_frames, new_frames, radius=5):
    """Recall control: preserve source events, add only nonoverlapping peaks."""
    kept = sorted(set(int(f) for f in source_frames))
    for f in sorted(new_frames):
        if all(abs(f-old) > radius for old in kept):
            kept.append(int(f))
    return sorted(kept)


def kind(event):
    value = event.get("type", event.get("contact_type", "contact"))
    return "pass" if value in ("set", "dig", "free_ball", "setting_pass", "followup_contact") else value


def transition_key(previous, current, fps):
    gap = (int(current["frame"])-int(previous["frame"]))/fps
    return f"{kind(previous)}>{kind(current)}:{'fast' if gap < .5 else 'normal'}"


def resets(previous, current, fps):
    return (previous is None or current.get("team_label") is None
            or previous.get("team_label") is None
            or kind(current) == "serve"
            or (int(current["frame"])-int(previous["frame"]))/fps > 4.)


def learn_transitions(sequences, fps):
    counts = defaultdict(Counter)
    for sequence in sequences:
        previous = None
        for event in sorted(sequence, key=lambda e: e["frame"]):
            if not resets(previous, event, fps):
                same = previous["team_label"] == event["team_label"]
                counts[transition_key(previous, event, fps)][bool(same)] += 1
            previous = event
    return {key: {"same": (values[True]+2)/(sum(values.values())+4),
                  "count": sum(values.values())} for key, values in counts.items()}


def sequence_posteriors(events, scores, priors, fps, strength=.7):
    """Forward/backward team inference. Unknowns/serves/long gaps reset.

    This is a soft learned sequence prior, not a rule declaring faults or
    forcing three hits. Block contacts are represented separately.
    """
    scores = np.asarray(scores, dtype=float)
    if len(events) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("One finite side score per event required")
    if not len(events):
        return scores.copy()
    # team_label is only an availability separator; its value is NEVER read.
    emission = np.column_stack((1-np.clip(scores, 1e-6, 1-1e-6), np.clip(scores, 1e-6, 1-1e-6)))
    transitions = []
    for previous, current in zip(events, events[1:]):
        if resets(previous, current, fps):
            transitions.append(np.full((2, 2), .5))
        else:
            same = priors.get(transition_key(previous, current, fps), {}).get("same", .5)
            matrix = np.array([[same, 1-same], [1-same, same]])**strength
            transitions.append(matrix/matrix.sum(axis=1, keepdims=True))
    forward = emission.copy()
    for i in range(1, len(events)):
        forward[i] *= forward[i-1]@transitions[i-1]
        forward[i] /= forward[i].sum()
    backward = np.ones_like(forward)
    for i in range(len(events)-2, -1, -1):
        backward[i] = transitions[i]@(emission[i+1]*backward[i+1])
        backward[i] /= backward[i].sum()
    posterior = forward*backward
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior[:, 1]


def joint_rank(candidates, evidence):
    """Re-rank observed actor/time proposals with independent flight evidence."""
    ranked = deepcopy(candidates)
    for actor in ranked:
        f = actor["frame"]
        actor["proximity_score"] = actor["score"]
        actor["flight_evidence"] = float(evidence[f, 0])
        actor["interaction_evidence"] = float(evidence[f, 1])
        actor["score"] = float(actor["score"] + .18*(1-evidence[f, 0]) + .10*(1-evidence[f, 1]))
    return sorted(ranked, key=lambda c: c["score"])
