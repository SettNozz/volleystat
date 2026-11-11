"""Derive player totals and their evidence from one reconciled contact view."""

from collections import Counter

from src.tracking.player_contacts import side
from src.tracking.statistics_reconciliation import reconcile_statistics


def build_player_match_statistics(stats, report, point_reviews=()):
    players = {player: {kind: Counter() for kind in ("attack", "reception", "serve")}
               for player in report["roster_ids"]}
    coverage = {player: {kind: Counter() for kind in ("attack", "reception", "serve")}
                for player in players}
    reconciled = reconcile_statistics(stats, report, point_reviews)
    target = side(report.get("side"))
    unassigned = Counter()
    touch_counts = {player: Counter() for player in players}
    for contact in reconciled["contacts"]:
        kind, player = contact["kind"], contact["player_id"]
        if player is not None:
            touch_counts[player][kind] += 1
        if kind not in {"attack", "reception", "serve"}:
            continue
        if player is None:
            if contact["author_status"] != "other_side" and contact["event_side"] in (None, target):
                unassigned[kind] += 1
            continue
        counts = players[player][kind]
        counts["total"] += 1
        counts[contact["outcome"]] += 1
        evidence = coverage[player][kind]
        evidence["total"] += 1
        evidence[contact["author_source"]] += 1
        evidence["unknown" if contact["outcome"] == "unknown" else "classified"] += 1
    return {"players": players, "side": target, "unassigned": unassigned,
            "touch_counts": touch_counts, "coverage": coverage, "reconciliation": reconciled}
