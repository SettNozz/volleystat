"""Derived contact-to-roster proposals from saved ball/pose observations.

Image-plane proximity is evidence, not verified touch or team ground truth.
Both sides and unidentified poses compete; ambiguous evidence stays unknown.
No detector, ReID model, interpolation, or court-side projection runs here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import math

SCHEMA = "player_contacts_v1"
CONTACT_TYPES = {
    "serve", "reception", "reception_pass", "dig", "pass", "set",
    "setting_pass", "attack", "block", "free_ball", "followup_contact",
}
BONES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def side(value):
    return {"team_a": "a", "team_b": "b", "a": "a", "b": "b"}.get(value)


def category(kind):
    if kind in {"reception", "reception_pass"}:
        return "reception"
    if kind in {"dig", "pass", "set", "setting_pass", "free_ball", "followup_contact"}:
        return "dig_pass"
    return kind if kind in {"serve", "attack", "block"} else "other"


def _point(value):
    return (isinstance(value, (list, tuple)) and len(value) >= 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in value[:2]))


def _distance(ball, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    t = max(0., min(1., ((ball[0]-a[0])*dx + (ball[1]-a[1])*dy) / length)) if length else 0.
    return math.hypot(ball[0]-a[0]-t*dx, ball[1]-a[1]-t*dy)


def _evidence(ann, pose, event_frame, pose_index, fps):
    ball, box = ann.get("ball"), pose.get("bb")
    if ann.get("ball_out_of_frame") or not _point(ball):
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in box
    ) or box[3] <= box[1]:
        return None
    points = {i: p for i, p in enumerate(pose.get("kp") or [])
              if _point(p) and len(p) >= 3 and isinstance(p[2], (int, float))
              and math.isfinite(p[2]) and p[2] >= .3}
    if not points:
        return None
    distances = [math.dist(ball[:2], p[:2]) for p in points.values()]
    distances += [_distance(ball, points[a], points[b]) for a, b in BONES
                  if a in points and b in points]
    distance = min(distances)
    normalized = distance / (box[3] - box[1])
    return {
        "frame": int(ann["f"]), "pose_index": pose_index,
        "raw_track_id": pose.get("id"), "player_id": pose.get("player_id"),
        "side": side(pose.get("team")), "distance_px": round(distance, 3),
        "distance_body_heights": round(normalized, 5),
        "score": round(normalized + .3 * abs(int(ann["f"])-event_frame) / fps, 5),
        "ball_source": ann.get("ball_source") or "detected",
        "identity": deepcopy(pose.get("identity") or {}),
    }


def summarize(report):
    """Recount after proposals or human corrections; never count unknowns as touches."""
    roster = []
    for player_id in report["roster_ids"]:
        contacts = [c for c in report["contacts"] if c.get("player_id") == player_id]
        counts = Counter(category(c["type"]) for c in contacts)
        roster.append({"player_id": player_id, "total": len(contacts),
                       "counts": {k: counts[k] for k in ("reception", "dig_pass", "attack", "block", "serve", "other")},
                       "raw_counts": dict(Counter(c["type"] for c in contacts)),
                       "reviewed": sum(c.get("verified") is True for c in contacts)})
    report["players"] = roster
    report["totals"] = dict(Counter(c["status"] for c in report["contacts"]))
    return report


def build_player_contacts(events, frames, fps, identity=None):
    """Propose one person per saved contact. Inputs are never modified.

    Search +/-0.10 s, using ball and skeleton from the SAME saved frame. No
    search through absent/off-camera ball, annotation gaps or track restarts.
    Distances use observed body height; thresholds are engineering defaults,
    not calibrated probabilities or measured attribution accuracy.
    """
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    identity = identity or {}
    target = side(identity.get("side")) or "b"
    roster = set((identity.get("gallery_keys") or {}).keys())
    roster.update((identity.get("seed") or {}).get("assignments", {}).values())
    roster.update(p.get("player_id") for a in frames for p in a.get("poses") or []
                  if side(p.get("team")) == target and p.get("player_id"))
    roster.discard(None)
    roster = sorted(roster, key=lambda p: (int(p[1:]) if p[1:].isdigit() else math.inf, p))
    by_frame = {int(a["f"]): a for a in frames}
    radius = max(1, round(fps * .10))
    contacts = []
    for index, event in enumerate(events):
        if event.get("type") not in CONTACT_TYPES:
            continue
        frame = int(event["frame"])
        contact = {
            "event_index": index, "frame": frame,
            "time": frame / fps, "type": event["type"],
            "core_type": event.get("core_type"), "event_source": event.get("source"),
            "event_confidence": event.get("confidence"),
            "player_id": None, "status": "unresolved", "reason": "no_ball_or_pose",
            "label_source": "derived", "verified": False, "candidates": [],
        }
        # Anchor at the actual contact frame. Missing contact-time ball cannot
        # be replaced by a ball on the other side of a gap or outside camera.
        anchor = by_frame.get(frame)
        candidates = {}
        if anchor and not anchor.get("ball_out_of_frame") and _point(anchor.get("ball")):
            reachable = [anchor]
            for direction in (-1, 1):
                for offset in range(1, radius + 1):
                    ann = by_frame.get(frame + direction * offset)
                    previous = by_frame.get(frame + direction * (offset-1))
                    newer = ann if direction > 0 else previous
                    if (not ann or ann.get("ball_out_of_frame") or not _point(ann.get("ball"))
                            or newer.get("ball_new") is True):
                        break
                    reachable.append(ann)
            for ann in reachable:
                for pose_index, pose in enumerate(ann.get("poses") or []):
                    evidence = _evidence(ann, pose, frame, pose_index, fps)
                    if evidence is None:
                        continue
                    # Unidentified and opposite-side detections still compete.
                    person = evidence["player_id"] or (
                        ("track", evidence["raw_track_id"]) if evidence["raw_track_id"] is not None
                        else ("pose", int(ann["f"]), pose_index))
                    key = (evidence["side"], person)
                    if key not in candidates or evidence["score"] < candidates[key]["score"]:
                        candidates[key] = evidence
        ranked = sorted(candidates.values(), key=lambda c: c["score"])
        contact["candidates"] = ranked[:3]
        if ranked:
            best = ranked[0]
            margin = ranked[1]["score"] - best["score"] if len(ranked) > 1 else None
            contact["margin"] = round(margin, 5) if margin is not None else None
            if best["distance_body_heights"] > .35:
                contact["reason"] = "too_far"
            elif margin is not None and margin < .10:
                contact["reason"] = "ambiguous_players"
            elif best["side"] is None:
                contact["reason"] = "unknown_side"
            elif best["side"] != target:
                contact.update(status="other_side", reason="other_side_candidate")
            elif not best["player_id"]:
                contact["reason"] = "unknown_identity"
            else:
                contact.update(player_id=best["player_id"], status="assigned", reason="body_proximity")
        contacts.append(contact)
    return summarize({
        "schema": SCHEMA, "method": "saved_ball_skeleton_proximity_v1", "side": target,
        "label_source": "derived", "verified": False, "roster_ids": roster,
        "parameters": {"window_seconds": .10, "max_body_distance": .35, "min_margin": .10,
                       "keypoint_confidence": .3},
        "limitations": "Unverified image-plane proposals; event detection/type and identity errors propagate. Review ambiguous contacts.",
        "contacts": contacts,
    })


def source_fingerprint(stats_bytes, annotations_bytes):
    return {"stats_sha256": hashlib.sha256(stats_bytes).hexdigest(),
            "annotations_sha256": hashlib.sha256(annotations_bytes).hexdigest()}


def apply_reviews(report, reviews):
    result = deepcopy(report)
    indexed = {c["event_index"]: c for c in result["contacts"]}
    for review in reviews:
        contact = indexed.get(review["event_index"])
        if contact is None:
            raise ValueError("Review refers to an unknown contact")
        if review.get("reset"):
            contact.clear()
            contact.update(deepcopy(next(c for c in report["contacts"]
                                         if c["event_index"] == review["event_index"])))
            continue
        player_id = review.get("player_id")
        if player_id is not None and player_id not in result["roster_ids"]:
            raise ValueError("Player is not in this result's roster")
        if player_id and review.get("other_side"):
            raise ValueError("Choose a roster player or the other side")
        contact.update(player_id=player_id, status="assigned" if player_id else
                       ("other_side" if review.get("other_side") else "unresolved"),
                       reason="human_review", label_source="human", verified=bool(player_id or review.get("other_side")))
    return summarize(result)
