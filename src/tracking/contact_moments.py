"""Group repeated model proposals supported by the same observed contact.

Source events and human decisions remain intact. Consumers share these groups
so highlight cards, player totals and exported evidence count the same moments.
"""

import math

from src.tracking.player_contacts import CONTACT_TYPES


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def contact_moment_groups(contacts, events, fps):
    """Return ``(representative, source_event_indices)`` pairs without edits.

    Merge only adjacent, same-type proposals for the same person, at most
    0.30 seconds apart, whose observed support is the same pose or the same
    raw track within 0.10 seconds. Human reviews and intervening contacts
    prevent a merge. Compare every member with the first, avoiding chains.
    """
    rate = float(fps) if _finite(fps) and fps > 0 else 0.
    event_gap, support_gap = .30 * rate, .10 * rate
    by_index = {}
    for contact in contacts:
        by_index.setdefault(contact["event_index"], contact)
    ordered = sorted(by_index.values(), key=lambda c: (events[c["event_index"]].get("frame", 0), c["event_index"]))
    sequence = sorted((e.get("frame", 0), i) for i, e in enumerate(events)
                      if e.get("type") in CONTACT_TYPES | {"serve_fault"})
    positions = {index: pos for pos, (_, index) in enumerate(sequence)}

    def same_touch(first, second):
        if first.get("label_source") == "human" or second.get("label_source") == "human":
            return False
        identity = lambda c: (c.get("status"), c.get("player_id") or c.get("proposed_player_id"))
        if not identity(first)[1] or identity(first) != identity(second):
            return False
        left, right = events[first["event_index"]], events[second["event_index"]]
        if any(e.get("label_source") == "human" or e.get("verified") is True for e in (left, right)):
            return False
        if left.get("type") != right.get("type") or abs(right.get("frame", 0) - left.get("frame", 0)) > event_gap:
            return False
        a = (first.get("candidates") or [{}])[0]
        b = (second.get("candidates") or [{}])[0]
        if a.get("ball_out_of_frame") or b.get("ball_out_of_frame"):
            return False
        if not all(_finite(c.get("frame")) and _finite(c.get("distance_body_heights")) for c in (a, b)):
            return False
        if a["frame"] == b["frame"] and type(a.get("pose_index")) is int and a["pose_index"] == b.get("pose_index"):
            return True
        return (a.get("raw_track_id") is not None and a["raw_track_id"] == b.get("raw_track_id")
                and 0 < abs(a["frame"] - b["frame"]) <= support_gap
                and _continuous_boxes(a.get("bbox"), b.get("bbox")))

    groups = []
    for contact in ordered:
        if (groups and positions.get(contact["event_index"], -1) == positions.get(groups[-1][-1]["event_index"], -3) + 1
                and same_touch(groups[-1][0], contact)):
            groups[-1].append(contact)
        else:
            groups.append([contact])
    result = []
    for group in groups:
        def strength(c):
            confidence = events[c["event_index"]].get("confidence")
            return confidence if _finite(confidence) else -1.
        result.append((max(group, key=strength), [c["event_index"] for c in group]))
    return result


def _continuous_boxes(a, b):
    if not all(isinstance(box, (tuple, list)) and len(box) == 4 and all(_finite(v) for v in box)
               and box[2] > box[0] and box[3] > box[1] for box in (a, b)):
        return False
    distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    return distance < .5 * min(a[3] - a[1], b[3] - b[1])
