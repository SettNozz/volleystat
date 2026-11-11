"""Read-only contact/rally/point joins; source IDs are never joined by ordinal.

Point reviews confirm winners only, never contact detection, type or outcome.
"""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import math

from src.tracking.player_contacts import CONTACT_TYPES, category, side

KINDS = {"attack", "serve", "reception", "dig_pass", "block"}
OUTCOMES = {
    "attack": {"point", "continued", "out", "net", "lost", "unknown"},
    "serve": {"in_play", "ace", "error", "unknown"},
    "reception": {"controlled", "overpass", "error", "unknown"},
    "dig_pass": {"not_scored"}, "block": {"not_scored"},
}


def _frame(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _attempts(stats, key):
    result = []
    for index, item in enumerate(stats.get(key) or []):
        lc = item.get("lifecycle") or {}
        start = lc.get("serve_frame")
        if not _frame(start):
            start = item.get("start_frame")
        end = item.get("end_frame")
        result.append({
            "ref": f"{key}:{index}", "source": key, "rally_id": item.get("rally_id"),
            "set_id": item.get("set_id"), "lifecycle": lc,
            "start_frame": start if _frame(start) else None,
            "end_frame": end if _frame(end) and (not _frame(start) or end >= start) else None,
            "serve_frame": lc.get("serve_frame") if _frame(lc.get("serve_frame")) else None,
        })
    return result


def reconcile_statistics(stats, report, point_reviews=()):
    """Return explainable outcomes without mutating saved analysis or reviews."""
    events = stats.get("all_events") or []
    current_key = "service_attempts" if stats.get("service_attempts") else "rallies"
    score_key = "service_attempts_fsm" if "service_attempts_fsm" in stats else current_key
    current, scored = _attempts(stats, current_key), _attempts(stats, score_key)
    ledger = defaultdict(list)
    for point in (stats.get("match_analytics") or {}).get("points") or []:
        if point.get("rally_id") is not None:
            ledger[point["rally_id"]].append(point)
    id_counts = Counter(a["rally_id"] for a in scored if a["rally_id"] is not None)
    refs = {a["ref"] for a in scored}
    reviews = {}
    for review in point_reviews:
        ref = review.get("attempt_ref")
        if ref not in refs:
            raise ValueError("Point review refers to an unknown attempt")
        if review.get("reset"):
            reviews.pop(ref, None)
        else:
            if review.get("winner_side") not in ("a", "b", None):
                raise ValueError("Point winner must be a, b or null")
            reviews[ref] = review
    points = []
    for attempt in scored:
        candidates = ledger[attempt["rally_id"]] if attempt["rally_id"] is not None else []
        issues = []
        if len(candidates) > 1 or id_counts[attempt["rally_id"]] > 1:
            issues.append("ambiguous_point_link")
        original = candidates[0] if len(candidates) == 1 else {}
        winner = side(original.get("winner_court_side")) if original.get("status") == "confirmed" else None
        if original.get("status") == "confirmed" and not winner:
            issues.append("point_side_missing")
        if (original.get("set_id") is not None and attempt["set_id"] is not None
                and original["set_id"] != attempt["set_id"]):
            issues.append("point_set_mismatch")
        review = reviews.get(attempt["ref"])
        if review is not None:
            winner = review.get("winner_side")
        elif issues:
            winner = None
        points.append({
            "attempt_ref": attempt["ref"], "rally_id": attempt["rally_id"], "set_id": attempt["set_id"],
            "start_frame": attempt["start_frame"], "end_frame": attempt["end_frame"], "winner_side": winner,
            "status": "confirmed" if winner else ("conflict" if issues and review is None else "unresolved"),
            "source": "human_review" if review is not None else "match_analytics",
            "evidence": "human_point_review" if review is not None else original.get("evidence"),
            "reviewed": review is not None, "verified": review is not None and winner is not None,
            "issues": issues, "original": dict(original),
        })
    points_by_ref = {p["attempt_ref"]: p for p in points}
    indexed = [(i, e) for i, e in enumerate(events)
               if e.get("type") in CONTACT_TYPES | {"serve_fault"} and _frame(e.get("frame"))]
    ordered = sorted(indexed, key=lambda pair: pair[1]["frame"])
    frames = [event["frame"] for _, event in ordered]

    def memberships(attempts):
        result = defaultdict(list)
        for attempt in attempts:
            start, end = attempt["start_frame"], attempt["end_frame"]
            if start is None or end is None:
                continue
            for pos in range(bisect_left(frames, start), bisect_right(frames, end)):
                result[ordered[pos][0]].append(attempt)
        return result

    current_members, score_members = memberships(current), memberships(scored)
    by_serve, current_serves = defaultdict(list), defaultdict(list)
    for attempts, result in ((scored, by_serve), (current, current_serves)):
        for attempt in attempts:
            if attempt["serve_frame"] is not None:
                result[attempt["serve_frame"]].append(attempt)
    authors = {}
    for contact in report["contacts"]:
        authors.setdefault(contact["event_index"], contact)
    target = side(report.get("side"))
    contacts = []
    for index, event in indexed:
        kind = "serve" if event["type"] == "serve_fault" else category(event["type"])
        if kind not in KINDS:
            kind = category(event.get("core_type"))
        if kind not in KINDS:
            continue
        frame, author = event["frame"], authors.get(index, {})
        assigned = author.get("status") == "assigned" and author.get("player_id") in report["roster_ids"]
        event_side = side(event.get("court_side") or event.get("team"))
        own = current_serves[frame] if kind == "serve" else current_members[index]
        matched = by_serve[frame] if kind == "serve" else score_members[index]
        issues = []
        if len(own) > 1:
            issues.append("overlapping_event_attempts")
        if len(matched) > 1:
            issues.append("overlapping_score_attempts")
        attempt = matched[0] if len(matched) == 1 else None
        if attempt and len(own) == 1 and own[0]["serve_frame"] is not None and attempt["serve_frame"] is not None:
            if own[0]["serve_frame"] != attempt["serve_frame"]:
                issues.append("serve_anchor_mismatch")
        if event_side and target and assigned and event_side != target:
            issues.append("contact_side_mismatch")
        point = points_by_ref.get(attempt["ref"]) if attempt else None
        outcome, outcome_source = "unknown", None
        reason = issues[0] if issues else "no_score_attempt"
        lc = attempt["lifecycle"] if attempt else {}
        if assigned and target and not issues and event_side in (None, target):
            if kind == "serve":
                if event["type"] == "serve_fault":
                    outcome, outcome_source = "error", "explicit_serve_fault"
                else:
                    lifecycle = lc if attempt else (own[0]["lifecycle"] if len(own) == 1 else {})
                    if side(lifecycle.get("serving_team")) in (None, target):
                        outcome = {"in_play": "in_play", "ace": "ace", "fault": "error"}.get(lifecycle.get("serve_outcome"), "unknown")
                        outcome_source = attempt["source"] if attempt else (own[0]["source"] if len(own) == 1 else None)
                if point and point["status"] == "confirmed":
                    if ((outcome == "ace" and point["winner_side"] != target)
                            or (outcome == "error" and point["winner_side"] == target)):
                        issues.append("serve_outcome_point_mismatch")
            elif kind == "reception" and attempt:
                if lc.get("reception_frame") == frame and side(lc.get("receiving_team")) in (None, target):
                    outcome = {"controlled": "controlled", "overpass": "overpass", "error": "error"}.get(lc.get("reception_quality"), "unknown")
                    outcome_source = attempt["source"]
                reason = "reception_not_matched" if lc.get("reception_frame") != frame else "reception_quality_unknown"
            elif kind == "attack" and attempt and attempt["end_frame"] is not None:
                last = bisect_right(frames, attempt["end_frame"]) - 1
                terminal = last >= 0 and ordered[last][0] == index
                duplicate = last > 0 and frames[last - 1] == frame and terminal
                if duplicate or (last >= 0 and frames[last] == frame and not terminal):
                    issues.append("multiple_terminal_contacts")
                elif not terminal:
                    outcome, outcome_source = "continued", attempt["source"]
                elif point and point["status"] == "confirmed":
                    if point["winner_side"] == target and lc.get("end_reason") in {"attack_out", "attack_into_net"}:
                        issues.append("attack_fault_point_mismatch")
                    else:
                        outcome = "point" if point["winner_side"] == target else {"attack_out": "out", "attack_into_net": "net"}.get(lc.get("end_reason"), "lost")
                        outcome_source = "terminal_contact_and_point"
                reason = "point_conflict" if point and point["status"] == "conflict" else "point_unresolved"
        if issues:
            outcome, outcome_source, reason = "unknown", None, issues[0]
        if outcome != "unknown":
            reason = "derived_outcome"
        if kind in {"dig_pass", "block"}:
            outcome, outcome_source, reason = "not_scored", None, "outcome_not_supported"
        video_frame = author.get("contact_frame") if author.get("label_source") != "human" else None
        if not _frame(video_frame):
            video_frame = frame
        contacts.append({
            "event_index": index, "frame": frame, "video_frame": int(video_frame),
            "kind": kind, "type": event["type"], "event_source": event.get("source") or "saved_event",
            "player_id": author.get("player_id") if assigned else None,
            "author_status": "assigned" if assigned else author.get("status", "unresolved"),
            "author_source": ("human" if author.get("label_source") == "human" and author.get("verified") is True else "automatic") if assigned else "unresolved",
            "event_verified": event.get("label_source") == "human" and event.get("verified") is True,
            "outcome": outcome, "outcome_source": outcome_source, "outcome_verified": False,
            "reason": reason, "issues": issues, "attempt_ref": attempt["ref"] if attempt else None,
            "event_attempt_refs": [a["ref"] for a in own],
            "point_source": point["source"] if point else None, "point_reviewed": point["verified"] if point else False,
            "start_frame": attempt["start_frame"] if attempt else None,
            "end_frame": attempt["end_frame"] if attempt else None, "event_side": event_side,
        })
    contact_conflicts = [c for c in contacts if c["issues"] and c["player_id"] is not None
                        and c["kind"] in {"attack", "serve", "reception"}]
    return {"schema": "statistics_reconciliation_v1", "contacts": contacts, "points": points,
            "event_attempt_source": current_key, "score_attempt_source": score_key,
            "contact_conflicts": contact_conflicts,
            "conflicts": len(contact_conflicts) + sum(p["status"] == "conflict" for p in points),
            "point_summary": dict(Counter(p["status"] for p in points))}


def statistic_evidence(summary, player, kind, metric):
    """Select exactly the contacts counted by a displayed metric."""
    if kind not in KINDS:
        raise ValueError("Unknown statistics category")
    allowed = OUTCOMES[kind] | {"total"}
    if kind == "attack":
        allowed |= {"errors"}
    if kind in {"serve", "reception"}:
        allowed |= {"rate"}
    if metric not in allowed:
        raise ValueError("Unknown statistics metric")
    if player not in {*summary["players"], "all", "unassigned"}:
        raise ValueError("Unknown roster player")
    selected = []
    for contact in summary["reconciliation"]["contacts"]:
        if contact["kind"] != kind:
            continue
        if player == "unassigned":
            if contact["author_status"] == "other_side" or contact["player_id"] is not None or contact["event_side"] not in (None, summary["side"]):
                continue
        elif player == "all":
            if contact["player_id"] is None:
                continue
        elif contact["player_id"] != player:
            continue
        outcome = contact["outcome"]
        if metric == "errors" and outcome not in {"out", "net"}:
            continue
        if metric == "rate" and outcome == "unknown":
            continue
        if metric not in {"total", "errors", "rate"} and outcome != metric:
            continue
        selected.append(contact)
    numerator = "error" if kind == "serve" else "controlled"
    return {"contacts": selected, "total": len(selected),
            "human_authors": sum(c["author_source"] == "human" for c in selected),
            "automatic_authors": sum(c["author_source"] == "automatic" for c in selected),
            "unassigned_authors": sum(c["player_id"] is None for c in selected),
            "unknown_outcomes": sum(c["outcome"] == "unknown" for c in selected),
            "numerator": sum(c["outcome"] == numerator for c in selected) if metric == "rate" else None}
