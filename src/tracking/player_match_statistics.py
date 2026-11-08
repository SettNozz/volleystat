"""Derive per-player outcomes from saved contacts and the team point ledger."""

from bisect import bisect_left, bisect_right
from collections import Counter

from src.tracking.player_contacts import CONTACT_TYPES, category, side


ATTACK_ERRORS = {"attack_out": "out", "attack_into_net": "net"}
RECEPTION_QUALITY = {"controlled": "controlled", "overpass": "overpass", "error": "error"}
SERVE_OUTCOMES = {"in_play": "in_play", "ace": "ace", "fault": "error"}


def build_player_match_statistics(stats, report):
    players = {
        player_id: {kind: Counter() for kind in ("attack", "reception", "serve")}
        for player_id in report["roster_ids"]
    }
    events = stats.get("all_events") or []
    point_by_rally = {
        point["rally_id"]: point for point in
        (stats.get("match_analytics") or {}).get("points", [])
        if point.get("rally_id") and point.get("status") == "confirmed"
    }
    attempts = stats.get("service_attempts_fsm") or stats.get("service_attempts") or stats.get("rallies") or []
    serves = {}
    for records in (stats.get("service_attempts") or [], attempts):
        for attempt in records:
            lifecycle = attempt.get("lifecycle") or {}
            if lifecycle.get("serve_frame") is not None:
                serves[lifecycle["serve_frame"]] = lifecycle
    contact_frames = sorted(
        event["frame"] for event in events
        if event.get("type") in CONTACT_TYPES or event.get("type") == "serve_fault"
    )
    # Dense spotter exports retain the FSM attempts used by the ledger.
    # Never join their independent rally IDs to the ledger by ordinal.
    outcomes = {}
    for attempt in attempts:
        lifecycle = attempt.get("lifecycle") or {}
        start, end = attempt.get("start_frame"), attempt.get("end_frame")
        if start is None or end is None:
            continue
        start = lifecycle.get("serve_frame") if lifecycle.get("serve_frame") is not None else start
        first = bisect_left(contact_frames, start)
        last = bisect_right(contact_frames, end)
        point = point_by_rally.get(attempt.get("rally_id"), {})
        for index in range(first, last):
            frame = contact_frames[index]
            terminal = index == last - 1 and (index == first or contact_frames[index - 1] != frame)
            # Overlapping attempt windows are ambiguous.
            outcomes[frame] = None if frame in outcomes else (lifecycle, point, terminal)
    target_side = side(report.get("side"))
    unassigned = Counter()
    for contact in report["contacts"]:
        event = events[contact["event_index"]]
        kind = "serve" if event.get("type") == "serve_fault" else category(event.get("type"))
        if kind not in {"attack", "reception", "serve"}:
            kind = category(event.get("core_type"))
        if kind not in {"attack", "reception", "serve"}:
            continue
        player_id = contact.get("player_id")
        event_side = side(event.get("court_side") or event.get("team"))
        if contact.get("status") != "assigned" or player_id not in players:
            if contact.get("status") != "other_side" and event_side in (None, target_side):
                unassigned[kind] += 1
            continue
        counts = players[player_id][kind]
        counts["total"] += 1
        outcome = "unknown"
        context = outcomes.get(event["frame"])
        if kind == "serve":
            lifecycle = serves.get(event["frame"], {})
            serving_side = side(lifecycle.get("serving_team"))
            if target_side and event_side in (None, target_side) and serving_side in (None, target_side):
                outcome = ("error" if event.get("type") == "serve_fault"
                           else SERVE_OUTCOMES.get(lifecycle.get("serve_outcome"), "unknown"))
        elif context and target_side and event_side in (None, target_side):
            lifecycle, point, terminal = context
            winner = side(point.get("winner_court_side"))
            if kind == "attack":
                if not terminal:
                    outcome = "continued"
                elif winner == target_side:
                    outcome = "point"
                elif winner:
                    outcome = ATTACK_ERRORS.get(lifecycle.get("end_reason"), "lost")
            elif lifecycle.get("reception_frame") == event["frame"]:
                receiving_side = side(lifecycle.get("receiving_team"))
                if receiving_side in (None, target_side):
                    outcome = RECEPTION_QUALITY.get(lifecycle.get("reception_quality"), "unknown")
        counts[outcome] += 1
    return {"players": players, "side": target_side, "unassigned": unassigned}
