"""Developer review proposals and explicit, non-destructive human corrections.

This module never reads/writes a dataset, runs inference or generates geometry.
Event indexes stay stable so existing source-bound author/point reviews keep
their meaning. Excluded events remain in the event array with an explicit flag.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import math
import re
import uuid

from src.tracking.player_contacts import CONTACT_TYPES, category, side, summarize


REVIEW_TYPES = tuple(sorted(CONTACT_TYPES | {"serve_fault"}))
_ACTIONS = {"add", "update", "remove", "reset", "dismiss"}
_FIELDS = {"frame", "type", "side", "player_id", "other_side", "detection_verified", "type_verified"}


def _integer(value):
    return type(value) is int and value >= 0


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _core(kind):
    return {"serve_fault": "serve", "serve": "serve", "reception": "reception",
            "reception_pass": "reception", "attack": "attack", "block": "attack",
            "set": "pass", "pass": "pass", "setting_pass": "pass", "dig": "pass"}.get(kind)


def _clear_frame_evidence(contact):
    contact["candidates"] = []
    for key in ("contact_frame", "contact_time", "proposed_player_id", "near_side_score", "actor_share", "margin"):
        contact.pop(key, None)


def validate_contact_edit(body, stats, report, total_frames):
    """Validate one edit against the current projected view before persistence.

Schema: ``{action, event_index?, review_id?, changes?: {...}, item_ref?}``.
Add also accepts frame/type/side/player_id at the top level. Verification axes
are independent: author-only corrections cannot verify event detection/type.
"""
    if not isinstance(body, dict) or not isinstance(body.get("action"), str) or body["action"] not in _ACTIONS:
        raise ValueError("Unknown contact review action")
    action = body["action"]
    result = {"action": action}
    events = stats.get("all_events") or []
    if action == "dismiss":
        ref = body.get("item_ref")
        if not isinstance(ref, str) or not ref or len(ref) > 180:
            raise ValueError("A queue item reference is required")
        if ref.startswith("event:"):
            try:
                index = int(ref.split(":", 1)[1])
            except ValueError as error:
                raise ValueError("Unknown contact review target") from error
            if not 0 <= index < len(events):
                raise ValueError("Unknown contact review target")
        elif ref.startswith("point:"):
            namespace, separator, index_text = ref[len("point:"):].rpartition(":")
            score_key = "service_attempts_fsm" if "service_attempts_fsm" in stats else ("service_attempts" if stats.get("service_attempts") else "rallies")
            if namespace != score_key or not separator or not index_text.isdigit() or int(index_text) >= len(stats.get(score_key) or []):
                raise ValueError("Unknown point review target")
        else:
            raise ValueError("Unknown queue item reference")
        result["item_ref"] = ref
        return result
    if action != "add":
        index = body.get("event_index")
        if not _integer(index) or index >= len(events):
            raise ValueError("Unknown contact review target")
        if events[index].get("type") not in REVIEW_TYPES:
            raise ValueError("Review target is not a contact")
        result["event_index"] = index
        if action in {"remove", "reset"}:
            if action == "reset" and "axes" in body:
                if body["axes"] != ["author"]:
                    raise ValueError("Only the author axis can be reset independently")
                result["axes"] = ["author"]
            return result
    changes = body.get("changes", {})
    if not isinstance(changes, dict):
        raise ValueError("Contact changes must be an object")
    changes = dict(changes)
    if action == "add":
        changes.update({key: body[key] for key in _FIELDS if key in body})
        if not {"frame", "type"} <= changes.keys():
            raise ValueError("A new contact needs a frame and type")
        review_id = body.get("review_id") or uuid.uuid4().hex
        if not isinstance(review_id, str) or not re.fullmatch(r"[A-Za-z0-9_:-]{1,128}", review_id):
            raise ValueError("Invalid contact review ID")
        previous = next((event for event in events if event.get("review_id") == review_id), None)
        if previous and any(previous.get(key) != changes[key] for key in ("frame", "type")):
            raise ValueError("Contact review ID already exists with different timing or type")
        result["review_id"] = review_id
    if not changes or set(changes) - _FIELDS:
        raise ValueError("Unknown or empty contact changes")
    if "frame" in changes:
        frame = changes["frame"]
        if not _integer(frame) or (_integer(total_frames) and total_frames > 0 and frame >= total_frames):
            raise ValueError("Contact frame is outside the video")
    if "type" in changes and changes["type"] not in REVIEW_TYPES:
        raise ValueError("Unknown contact type")
    if "side" in changes:
        value = changes["side"]
        if value is not None and (not isinstance(value, str) or side(value) is None):
            raise ValueError("Contact side must be a, b or null")
        changes["side"] = side(value)
    if "player_id" in changes:
        player = changes["player_id"]
        if player is not None and (not isinstance(player, str) or player not in report.get("roster_ids", [])):
            raise ValueError("Player is not in this result's roster")
    for key in ("other_side", "detection_verified", "type_verified"):
        if key in changes and type(changes[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if changes.get("player_id") and changes.get("other_side"):
        raise ValueError("Choose a roster player or the other side")
    result["changes"] = changes
    return result


def apply_contact_edits(stats, report, edits, *, author_proposals=None):
    """Project append-only edits; original arrays, geometry and labels survive."""
    effective, authors = deepcopy(stats), deepcopy(report)
    events = effective.setdefault("all_events", [])
    source_events = deepcopy(events)
    source_contacts = {c["event_index"]: deepcopy(c) for c in authors.get("contacts") or []}
    proposal_contacts = {c["event_index"]: deepcopy(c) for c in (author_proposals or report).get("contacts") or []}
    by_index = {c["event_index"]: c for c in authors.get("contacts") or []}
    dismissed = set((effective.get("developer_review") or {}).get("dismissed_refs") or [])
    fps = stats.get("fps") if _finite(stats.get("fps")) and stats["fps"] > 0 else 30
    for edit in edits:
        action = edit["action"]
        if action == "dismiss":
            dismissed.add(edit["item_ref"])
            continue
        if action == "add":
            if any(event.get("review_id") == edit["review_id"] for event in events):
                continue
            index = len(events)
            events.append({"frame": edit["changes"]["frame"], "type": edit["changes"]["type"],
                           "source": "human_review_added", "label_source": "human", "verified": False,
                           "review_added": True, "review_id": edit["review_id"], "review_axes": {}})
        else:
            index = edit["event_index"]
            if not _integer(index) or index >= len(events):
                raise ValueError("Unknown contact review target")
        event = events[index]
        if action == "reset":
            if edit.get("axes") == ["author"]:
                baseline = deepcopy(proposal_contacts.get(index) or {})
                baseline.update(event_index=index, frame=event["frame"], time=event["frame"] / fps,
                                type=event["type"], core_type=event.get("core_type"))
                baseline.setdefault("player_id", None)
                baseline.setdefault("status", "unresolved")
                baseline.setdefault("label_source", "derived")
                baseline.setdefault("verified", False)
                if index >= len(source_events) or event["frame"] != source_events[index].get("frame"):
                    baseline.update(player_id=None, status="unresolved", label_source="derived", verified=False,
                                    candidates=[], reason="timing_changed_author_unverified")
                    _clear_frame_evidence(baseline)
                event.setdefault("review_axes", {}).pop("author", None)
                baseline["review_axes"] = deepcopy(event["review_axes"])
                by_index[index] = baseline
                continue
            dismissed.discard(f"event:{index}")
            if index < len(source_events):
                events[index] = deepcopy(source_events[index])
                if index in source_contacts:
                    by_index[index] = deepcopy(source_contacts[index])
                else:
                    by_index.pop(index, None)
            else:
                event["review_excluded"] = True
                by_index.pop(index, None)
            continue
        if action == "remove":
            event.setdefault("review_original", deepcopy(event))
            event["review_excluded"] = True
            event["verified"] = False
            event.setdefault("review_axes", {})["detection"] = {"label_source": "human", "verified": True, "value": False}
            by_index.pop(index, None)
            dismissed.discard(f"event:{index}")
            continue
        changes = edit["changes"]
        event.setdefault("review_original", deepcopy(event))
        if event.get("label_source") == "human" and event.get("verified") is True and not event.get("review_axes"):
            event["review_axes"] = {axis: {"label_source": "human", "verified": True} for axis in ("detection", "type")}
        axes = event.setdefault("review_axes", {})
        if action == "add" or not event.get("review_excluded") or changes.get("detection_verified") is True:
            event["review_excluded"] = False
        if "frame" in changes:
            event["frame"] = changes["frame"]
            event["time"] = changes["frame"] / fps
            axes["timing"] = {"label_source": "human", "verified": True}
        if "type" in changes:
            event["type"] = changes["type"]
            event["core_type"] = _core(changes["type"])
            axes["type"] = {"label_source": "human", "verified": True}
        if "side" in changes:
            value = f"team_{changes['side']}" if changes["side"] else None
            event["court_side"], event["team"] = value, value
            axes["side"] = {"label_source": "human", "verified": value is not None}
        for field, axis in (("detection_verified", "detection"), ("type_verified", "type")):
            if field in changes:
                axes[axis] = {"label_source": "human", "verified": changes[field]}
                if axis == "detection" and changes[field] is True:
                    axes[axis]["value"] = True
        if action == "add":
            # Inserting an explicitly selected type verifies existence and type.
            axes.setdefault("detection", {"label_source": "human", "verified": True, "value": True})
        if "detection" in axes or "type" in axes:
            event["verified"] = bool(not event.get("review_excluded") and axes.get("detection", {}).get("verified")
                                     and axes.get("detection", {}).get("value") is not False and axes.get("type", {}).get("verified"))
            if event["verified"]:
                event["label_source"] = "human"
        contact = by_index.get(index)
        if contact is None:
            contact = {"event_index": index, "player_id": None, "status": "unresolved",
                       "label_source": "derived", "verified": False, "candidates": [], "reason": "human_event_unassigned"}
            by_index[index] = contact
        contact.update(frame=event["frame"], time=event["frame"] / fps,
                       type=event["type"], core_type=event.get("core_type"))
        # Previous frame-specific automatic author evidence cannot support a moved contact.
        if "frame" in changes:
            _clear_frame_evidence(contact)
            if "player_id" not in changes and "other_side" not in changes:
                contact.update(player_id=None, status="unresolved", label_source="derived", verified=False,
                               reason="timing_changed_author_unverified")
        if "player_id" in changes or "other_side" in changes:
            player = changes.get("player_id")
            opposite = changes.get("other_side", False)
            contact.update(player_id=player, status="assigned" if player else ("other_side" if opposite else "unresolved"),
                           label_source="human", verified=bool(player or opposite), reason="human_review")
            axes["author"] = {"label_source": "human", "verified": bool(player or opposite)}
        contact["review_axes"] = deepcopy(axes)
        dismissed.discard(f"event:{index}")
    authors["contacts"] = [by_index[index] for index in sorted(by_index) if not events[index].get("review_excluded")]
    authors = summarize(authors)
    effective["developer_review"] = {"schema": "developer_review_v1", "dismissed_refs": sorted(dismissed), "edit_count": len(edits)}
    return effective, authors


def build_review_queue(stats, report, reconciliation, annotations=None):
    """Rank saved evidence by review impact; priority is not a probability."""
    events = stats.get("all_events") or []
    authors = {c["event_index"]: c for c in report.get("contacts") or []}
    reconciled = {c["event_index"]: c for c in reconciliation.get("contacts") or []}
    dismissed = set((stats.get("developer_review") or {}).get("dismissed_refs") or [])
    fps = stats.get("fps") if _finite(stats.get("fps")) and stats["fps"] > 0 else 30
    neighbors = defaultdict(list)
    for index, event in enumerate(events):
        if event.get("type") in REVIEW_TYPES and not event.get("review_excluded") and _integer(event.get("frame")):
            neighbors[event["frame"]].append(index)
    observed_frames = {row.get("f"): row for row in (annotations or {}).get("frames") or []}
    items = []
    for index, event in enumerate(events):
        if event.get("type") not in REVIEW_TYPES or not _integer(event.get("frame")):
            continue
        frame, ref = event["frame"], f"event:{index}"
        contact, context = authors.get(index, {}), reconciled.get(index, {})
        reasons, priority = [], 10
        if context.get("issues"):
            reasons.extend(context["issues"])
            priority = 100
        if contact.get("status") not in {"assigned", "other_side"}:
            reasons.append("unresolved_author")
            priority += 35
        if len(neighbors[frame]) > 1:
            reasons.append("same_frame_contacts")
            priority += 40
        confidence = event.get("confidence")
        if _finite(confidence) and confidence < .5 and not event.get("verified"):
            reasons.append("low_event_confidence")
            priority += 20
        observation = observed_frames.get(frame)
        if observation is not None and (observation.get("ball_out_of_frame") or observation.get("ball") is None):
            reasons.append("contact_ball_unobserved")
            priority += 10
        if context.get("outcome") == "unknown" and category(event["type"]) in {"attack", "serve", "reception"}:
            reasons.append("unresolved_outcome")
            priority += 15
        if event["type"] in {"attack", "serve_fault"}:
            priority += 10
        axes = event.get("review_axes") or {}
        reviewed = bool(axes.get("detection", {}).get("verified") and axes.get("detection", {}).get("value") is not False
                        and axes.get("type", {}).get("verified"))
        status = "removed" if event.get("review_excluded") else ("dismissed" if ref in dismissed else ("reviewed" if reviewed else "pending"))
        if status in {"removed", "dismissed", "reviewed"}:
            priority = 0
        if not reasons:
            reasons.append("unreviewed_contact" if not reviewed else "human_reviewed_event")
        items.append({"ref": ref, "kind": "contact", "event_index": index, "frame": frame,
                      "time": frame / fps, "type": event["type"], "side": side(event.get("court_side") or event.get("team")),
                      "player_id": contact.get("player_id"), "author_status": contact.get("status", "unresolved"),
                      "author_verified": contact.get("label_source") == "human" and contact.get("verified") is True,
                      "event_verified": reviewed, "confidence": confidence if _finite(confidence) else None,
                      "event_source": event.get("source"), "reasons": reasons, "priority": priority, "status": status,
                      "outcome": context.get("outcome", "unknown"), "review_axes": deepcopy(axes),
                      "review_added": event.get("review_added") is True, "original": deepcopy(event.get("review_original"))})
    for point in reconciliation.get("points") or []:
        if point.get("status") == "confirmed" and not point.get("issues"):
            continue
        frame = point.get("start_frame")
        ref = f"point:{point['attempt_ref']}"
        items.append({"ref": ref, "kind": "point", "attempt_ref": point["attempt_ref"],
                      "frame": frame if _integer(frame) else 0, "time": frame / fps if _integer(frame) else 0,
                      "type": "point", "side": point.get("winner_side"), "priority": 120 if point.get("issues") else 80,
                      "status": "dismissed" if ref in dismissed else "pending",
                      "reasons": list(point.get("issues") or ["point_winner_unresolved"]),
                      "winner_side": point.get("winner_side"), "event_verified": False})
    items.sort(key=lambda row: (row["status"] != "pending", -row["priority"], row["frame"], row["ref"]))
    counts = Counter(item["status"] for item in items)
    return {"items": items, "summary": {"total": len(items), **dict(counts)},
            "contact_types": list(REVIEW_TYPES), "roster_ids": list(report.get("roster_ids") or []),
            "limitations": "Priority ranks review impact, not model confidence. Review axes remain independent; no dataset promotion occurs."}
