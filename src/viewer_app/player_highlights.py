"""Per-player highlight moments from saved contacts; no inference.

A moment is one attributed contact. Clip windows are cut in the browser
around ``time`` so the user can change lead-in and follow-through.
Unresolved contacts never become a player's moment; model suggestions for
the player are returned only on request and stay marked as uncertain.
"""

import math

from src.tracking.player_contacts import category

HIGHLIGHT_CATEGORIES = {
    "attack": "Attacks", "reception": "Receptions", "serve": "Serves",
    "dig_pass": "Digs, passes and sets", "block": "Blocks",
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def player_highlights(report, events, fps, player_id, *, extra_contacts=(), include_suggestions=False):
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if player_id not in report["roster_ids"]:
        raise ValueError("Unknown roster player")
    moments = []
    seen = set()
    for contact in list(report["contacts"]) + list(extra_contacts):
        assigned = contact.get("status") == "assigned" and contact.get("player_id") == player_id
        suggested = (include_suggestions and not assigned and contact.get("label_source") != "human"
                     and contact.get("proposed_player_id") == player_id)
        if not (assigned or suggested) or contact["event_index"] in seen:
            continue
        seen.add(contact["event_index"])
        event = events[contact["event_index"]] if 0 <= contact["event_index"] < len(events) else {}
        kind = event.get("type") or contact.get("type")
        group = "serve" if kind == "serve_fault" else category(kind)
        if group == "other":
            group = category(event.get("core_type") or contact.get("core_type"))
        derived = contact.get("label_source") != "human"
        frame = contact.get("contact_frame") if derived and _finite(contact.get("contact_frame")) else contact["frame"]
        moments.append({
            "event_index": contact["event_index"], "frame": int(frame), "time": int(frame) / fps,
            "type": kind, "category": group,
            "label_source": contact.get("label_source"), "verified": contact.get("verified") is True,
            "uncertain": not assigned,
        })
    moments.sort(key=lambda moment: (moment["frame"], moment["event_index"]))
    return moments
