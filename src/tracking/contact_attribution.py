"""Offline contact-author experiments over frozen observations, never labels.

These are review proposals. Image-plane overlap does not prove a touch or its
side. Source event frames/types, poses, ReID assignments and ball stay intact.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
from statistics import median

from src.tracking.player_contacts import (
    BONES, CONTACT_TYPES, SCHEMA, _distance, _point, build_player_contacts, side, summarize,
)

METHODS = {
    "body_baseline": {"label": "Body baseline", "window": .10},
    "hands": {"label": "Hands + track deduplication", "window": .12},
    "trajectory": {"label": "Hands + ball motion", "window": .22, "motion": True},
    "temporal": {"label": "Hands + motion + temporal identity", "window": .22, "motion": True, "temporal": True},
    "anatomical": {"label": "Action-specific anatomy", "window": .22, "motion": True, "temporal": True, "typed": True},
    "estimated": {"label": "Temporal + estimated joints", "window": .22, "motion": True, "temporal": True, "estimates": True},
}


def _ball(ann):
    value = ann.get("ball") if ann and not ann.get("ball_out_of_frame") else None
    return value if _point(value) else None


def _geometry(pose, ball, kind, typed=False, estimate=None):
    box = pose.get("bb") or []
    if len(box) != 4 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box) or box[3] <= box[1]:
        return None
    points = {i: p[:2] for i, p in enumerate(pose.get("kp") or [])
              if _point(p) and len(p) >= 3 and p[2] >= .25}
    estimated = set()
    if estimate is not None:
        estimated = set(estimate["estimated_joints"])
        for i, point in enumerate(estimate["keypoints"]):
            if point is not None and (i in estimated or i not in points):
                points[i] = point
    if not points:
        return None
    options = []
    for i, point in points.items():
        penalty = 0. if i in (9, 10) else .08 if i in (7, 8) else .20
        options.append((math.dist(ball[:2], point), penalty, f"joint_{i}", i in estimated))
    for a, b in BONES:
        if a not in points or b not in points:
            continue
        penalty = 0. if (a, b) in ((7, 9), (8, 10)) else .10 if (a, b) in ((5, 7), (6, 8)) else .20
        options.append((_distance(ball, points[a], points[b]), penalty, f"segment_{a}_{b}", bool(estimated.intersection((a, b)))))
    height = box[3] - box[1]
    if typed and kind in ("attack", "block", "serve"):
        options = [(d, p + (.12 if part not in ("joint_9", "joint_10") else 0.), part, est)
                   for d, p, part, est in options]
    if typed and kind in ("set", "setting_pass") and 9 in points and 10 in points:
        midpoint = [(a+b)*.5 for a, b in zip(points[9], points[10])]
        options.append((math.dist(ball[:2], midpoint), -.03, "two_hand_midpoint", bool(estimated.intersection((9, 10)))))
    distance, penalty, part, uses_estimates = min(options, key=lambda row: row[0]/height + row[1])
    return distance, height, penalty, part, uses_estimates


def _velocity(samples):
    centre = sum(f for f, _ in samples)/len(samples)
    denominator = sum((f-centre)**2 for f, _ in samples)
    return [sum((f-centre)*p[j] for f, p in samples)/denominator for j in (0, 1)] if denominator else [0., 0.]


def ball_turn(by_frame, frame):
    """Observed local velocity change; stop at missing/off-camera/new-track rows."""
    anchor = by_frame.get(frame)
    if _ball(anchor) is None:
        return None
    runs = []
    for direction in (-1, 1):
        samples = []
        for d in range(1, 6):
            f = frame + direction*d
            row = by_frame.get(f)
            newer = row if direction > 0 else by_frame.get(f+1)
            if _ball(row) is None or newer.get("ball_new") is True:
                break
            samples.append((f, _ball(row)))
        if len(samples) < 3:
            return None
        runs.append(_velocity(samples))
    before, after = runs
    change = math.dist(before, after)/(math.hypot(*before)+math.hypot(*after)+.1)
    return min(1., max(0., change))


def _identity_for_track(observations, best, temporal):
    identities = {o["player_id"] for o in observations if o["player_id"]}
    if len(identities) > 1:
        return None, {"method": "conflicting_local_identities", "ids": sorted(identities)}
    if best["player_id"]:
        return best["player_id"], {"method": "observed_reid", "frame": best["frame"]}
    if not temporal or not identities or best["raw_track_id"] is None:
        return None, {"method": "unknown"}
    # A rejected conflict/appearance contradiction is not repaired by a vote.
    reason = str(best["identity"].get("reason", ""))
    if any(word in reason for word in ("conflict", "contradiction", "veto")):
        return None, {"method": "reid_conflict_preserved"}
    before = [o for o in observations if o["frame"] < best["frame"] and o["player_id"]]
    after = [o for o in observations if o["frame"] > best["frame"] and o["player_id"]]
    if not before or not after:
        return None, {"method": "requires_two_sided_identity"}
    left, right = max(before, key=lambda o: o["frame"]), min(after, key=lambda o: o["frame"])
    if any(o["identity"].get("appearance_distance", 1.) is None or
           o["identity"].get("appearance_distance", 1.) > .40 for o in (left, right)):
        return None, {"method": "weak_identity_anchors"}
    fraction = (best["frame"]-left["frame"])/(right["frame"]-left["frame"])
    expected = [a*(1-fraction)+b*fraction for a, b in zip(left["bbox"], right["bbox"])]
    centre = lambda b: [(b[0]+b[2])*.5, (b[1]+b[3])*.5]
    height = expected[3]-expected[1]
    if (height <= 0 or math.dist(centre(expected), centre(best["bbox"]))/height > .20
            or not .7 <= (best["bbox"][3]-best["bbox"][1])/height <= 1.4):
        return None, {"method": "identity_motion_conflict"}
    return next(iter(identities)), {"method": "two_sided_identity_proposal", "left_frame": left["frame"],
                                   "right_frame": right["frame"], "verified": False}


def build_contact_attribution(events, frames, fps, identity=None, method="temporal"):
    if method not in METHODS:
        raise ValueError("Unknown contact attribution method")
    if method == "body_baseline":
        report = build_player_contacts(events, frames, fps, identity)
        report["method_id"] = method
        for c in report["contacts"]:
            c["side_proposal"] = c["candidates"][0]["side"] if c["candidates"] else None
            c["side_accepted"] = c["status"] != "unresolved" or c["reason"] == "unknown_identity"
        return report
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    config = METHODS[method]
    identity = identity or {}
    target = side(identity.get("side")) or "b"
    by_frame = {a["f"]: a for a in frames}
    if len(by_frame) != len(frames):
        raise ValueError("Duplicate source frames")
    roster = set((identity.get("gallery_keys") or {}).keys())
    roster.update(p.get("player_id") for a in frames for p in a.get("poses") or []
                  if side(p.get("team")) == target and p.get("player_id"))
    roster_ids = sorted(roster, key=lambda pid: (int(pid[1:]) if pid[1:].isdigit() else math.inf, pid))
    event_frames = sorted(set(int(e["frame"]) for e in events if e.get("type") in CONTACT_TYPES))
    turns = {}
    contacts = []
    for index, event in enumerate(events):
        if event.get("type") not in CONTACT_TYPES:
            continue
        frame = int(event["frame"])
        radius = max(1, round(config["window"]*fps))
        neighbours = [abs(f-frame) for f in event_frames if f != frame]
        if neighbours:
            radius = min(radius, max(0, (min(neighbours)-1)//2))
        buckets = defaultdict(list)
        for f in range(frame-radius, frame+radius+1):
            ann = by_frame.get(f)
            ball = _ball(ann)
            if ball is None:
                continue
            if by_frame.get(frame, {}).get("ball_out_of_frame"):
                continue
            if any(t not in by_frame or by_frame[t].get("ball_out_of_frame")
                   for t in range(min(f, frame), max(f, frame)+1)):
                continue
            # Retiming uses another actual observation, not invented ball
            # coordinates. Never cross an explicit source-track restart.
            if any(by_frame.get(t, {}).get("ball_new") is True
                   for t in range(min(f, frame)+1, max(f, frame)+1)):
                continue
            poses = ann.get("poses") or []
            heights = [p["bb"][3]-p["bb"][1] for p in poses if len(p.get("bb") or []) == 4 and p["bb"][3] > p["bb"][1]]
            scale = median(heights) if heights else 1.
            extras = {e["observed_pose_index"]: e for e in ann.get("pose_estimates") or []
                      if e.get("source") == "temporal_pose_completion_v1" and e.get("observed_pose_index") is not None} if config.get("estimates") else {}
            entries = [(i, p, None) for i, p in enumerate(poses)]
            if config.get("estimates"):
                entries += [(i, poses[i], e) for i, e in extras.items() if 0 <= i < len(poses)]
                entries += [(None, {"bb": e["bb"], "kp": [], "player_id": e["player_id"], "team": e["team"], "id": None}, e)
                            for e in ann.get("pose_estimates") or [] if e.get("source") == "temporal_pose_completion_v1" and e.get("observed_pose_index") is None]
            for pose_index, pose, estimate in entries:
                geometry = _geometry(pose, ball, event["type"], config.get("typed", False), estimate)
                if geometry is None or side(pose.get("team")) is None:
                    continue
                distance, height, penalty, part, uses_estimates = geometry
                if f not in turns:
                    turns[f] = ball_turn(by_frame, f)
                strength = turns[f]
                score = distance/scale + penalty + .06*((f-frame)/max(radius, 1))**2
                if config.get("motion"):
                    score += .10*(1-strength) if strength is not None else .06
                if uses_estimates:
                    score += .10
                evidence = {
                    "frame": f, "pose_index": pose_index, "raw_track_id": pose.get("id"),
                    "player_id": pose.get("player_id"), "side": side(pose.get("team")),
                    "bbox": pose["bb"], "distance_px": round(distance, 3),
                    "distance_body_heights": round(distance/height, 5), "score": round(score, 5),
                    "contact_part": part, "ball_source": ann.get("ball_source") or "detected",
                    "identity": deepcopy(pose.get("identity") or {}), "ball_turn": strength,
                    "uses_estimated_joints": uses_estimates,
                }
                key = (evidence["side"], "raw", pose["id"]) if pose.get("id") is not None else (
                    evidence["side"], "roster", pose.get("player_id")) if pose.get("player_id") else (evidence["side"], f, pose_index)
                buckets[key].append(evidence)
        actors = {}
        for key, observations in buckets.items():
            best = deepcopy(min(observations, key=lambda o: o["score"]))
            pid, resolution = _identity_for_track(observations, best, config.get("temporal", False))
            best["player_id"] = pid
            best["identity_resolution"] = resolution
            best["support_frames"] = len(set(o["frame"] for o in observations))
            actor = (best["side"], "roster", pid) if pid else key
            if actor not in actors or best["score"] < actors[actor]["score"]:
                actors[actor] = best
        ranked = sorted(actors.values(), key=lambda o: o["score"])
        contact = {
            "event_index": index, "frame": frame, "time": frame/fps, "type": event["type"],
            "core_type": event.get("core_type"), "event_source": event.get("source"),
            "event_confidence": event.get("confidence"), "player_id": None, "status": "unresolved",
            "reason": "no_ball_or_pose", "label_source": "derived", "verified": False,
            "candidates": ranked[:3], "side_proposal": ranked[0]["side"] if ranked else None,
        }
        if ranked:
            best = ranked[0]
            margin = ranked[1]["score"]-best["score"] if len(ranked) > 1 else None
            opposite = next((o for o in ranked if o["side"] != best["side"]), None)
            side_margin = opposite["score"]-best["score"] if opposite else None
            contact.update(contact_frame=best["frame"], contact_time=best["frame"]/fps,
                           margin=round(margin, 5) if margin is not None else None,
                           side_margin=round(side_margin, 5) if side_margin is not None else None,
                           identity_resolution=best["identity_resolution"], uses_estimated_joints=best["uses_estimated_joints"])
            contact["side_accepted"] = best["distance_body_heights"] <= .50 and (side_margin is None or side_margin >= .04)
            if best["distance_body_heights"] > .50:
                contact["reason"] = "too_far"
            elif side_margin is not None and side_margin < .04:
                contact["reason"] = "ambiguous_sides"
            elif best["side"] != target:
                contact.update(status="other_side", reason="other_side_candidate")
            elif margin is not None and margin < .04:
                contact["reason"] = "ambiguous_players"
            elif best["player_id"] is None:
                contact["reason"] = "unknown_identity"
            else:
                contact.update(status="assigned", player_id=best["player_id"], reason="temporal_hand_proximity")
        contacts.append(contact)
    return summarize({"schema": SCHEMA, "method": "frozen_contact_attribution_experiment_v1", "method_id": method,
                      "side": target, "roster_ids": roster_ids, "label_source": "derived", "verified": False,
                      "parameters": dict(config), "contacts": contacts,
                      "limitations": "Unverified contact-author proposals; team-only development labels do not verify individual people or JIZ3 accuracy."})
