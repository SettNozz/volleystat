"""Conservative tactical summaries from saved contacts, geometry and point evidence.

Court geometry supplies approximate attack lanes only. Teams, setters and
rotations always require explicit saved or human-confirmed evidence.
"""

from collections import Counter
import math

from src.tracking.player_contacts import CONTACT_TYPES, side
from src.tracking.statistics_reconciliation import reconcile_statistics

SET_TYPES = {"set", "setting_pass", "pass"}
RECEPTION_TYPES = {"reception", "reception_pass"}
QUALITY = {"controlled", "overpass", "error", "negative", "positive", "perfect"}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _active(item):
    return not item.get("review_excluded") and not item.get("excluded_from_match_stats")


def _court_side(event):
    explicit = {value for value in (side(event.get("court_side")), side(event.get("team"))) if value}
    return next(iter(explicit)) if len(explicit) == 1 else None


def _context(frame, config):
    mapping = {side(key): value for key, value in (config.get("team_identity_map") or {}).items()
               if side(key) and isinstance(value, str) and value}
    segments = [segment for segment in config.get("segments") or []
                if _number(segment.get("start_frame")) and _number(segment.get("end_frame"))
                and segment["start_frame"] <= frame <= segment["end_frame"]]
    for court_side in ("a", "b"):
        overrides = {value for segment in segments
                     for key, value in (segment.get("team_identity_map") or {}).items()
                     if side(key) == court_side and isinstance(value, str) and value}
        if overrides:
            mapping[court_side] = next(iter(overrides)) if len(overrides) == 1 else None
    sets = {segment["set_id"] for segment in segments if segment.get("set_id") is not None}
    return mapping, next(iter(sets)) if len(sets) == 1 else None


def _team(event, frame, config):
    mapping, _ = _context(frame, config)
    court_side = _court_side(event)
    if court_side in mapping:
        return mapping[court_side]
    return event.get("stable_team_id") or None


def _rotation(frame, team, config):
    if not team:
        return None
    rows = [row for row in config.get("rotations") or [] if row.get("team_id") == team
            and _number(row.get("start_frame")) and _number(row.get("end_frame"))
            and row["start_frame"] <= frame <= row["end_frame"]
            and type(row.get("rotation")) is int and 1 <= row["rotation"] <= 6]
    return rows[0]["rotation"] if len(rows) == 1 else None


def _windows(stats, key):
    rows = []
    for index, item in enumerate(stats.get(key) or []):
        if not _active(item):
            continue
        lc = item.get("lifecycle") or {}
        start = lc.get("serve_frame")
        if not _number(start):
            start = item.get("start_frame")
        end = item.get("end_frame")
        if _number(start) and _number(end) and 0 <= start <= end:
            rows.append((index, start, end, item))
    return rows


def _point(ann):
    ball = ann.get("ball")
    return (isinstance(ball, (list, tuple)) and len(ball) >= 2
            and all(_number(v) for v in ball[:2]) and not ann.get("ball_out_of_frame"))


def _ball_path(frames, start, end):
    derived = 0
    for frame in range(start, end + 1):
        ann = frames.get(frame)
        if not ann or not _point(ann):
            return None, "ball_gap_or_off_camera"
        if frame > start and ann.get("ball_new"):
            return None, "ball_track_restart"
        derived += bool(ann.get("ball_interp") or ann.get("ball_source") not in (None, "detected"))
    return derived, None


def _zone(ann, annotations):
    left = annotations.get("net_floor_left") or [annotations.get("net_x_left")]
    right = annotations.get("net_floor_right") or [annotations.get("net_x_right")]
    if not left or not right or not _number(left[0]) or not _number(right[0]) or left[0] == right[0]:
        return "unknown"
    fraction = (ann["ball"][0] - min(left[0], right[0])) / abs(right[0] - left[0])
    if not 0 <= fraction <= 1:
        return "unknown"
    return "left" if fraction < 1 / 3 else "right" if fraction > 2 / 3 else "center"


def _reception_quality(ordered, position, attempt, pair_side):
    if position < 1 or not attempt or not pair_side:
        return "unknown"
    reception = ordered[position - 1][1]
    if reception.get("type") not in RECEPTION_TYPES or _court_side(reception) != pair_side:
        return "unknown"
    # Only the serve-reception possession is conditioned on serve quality.
    lc = attempt.get("lifecycle") or {}
    if reception["frame"] != lc.get("reception_frame") or side(lc.get("receiving_team")) not in (None, pair_side):
        return "unknown"
    human_quality = (reception.get("reception_quality") if reception.get("reception_quality_verified") is True
                     and reception.get("reception_quality_label_source") == "human" else None)
    value = human_quality or lc.get("reception_quality")
    return value if value in QUALITY else "unknown"


def _rotation_rows(stats, reconciliation, config):
    key = reconciliation["score_attempt_source"]
    windows = _windows(stats, key)
    by_index = {row[0]: row for row in windows}
    rows = []
    for point in reconciliation["points"]:
        index = int(point["attempt_ref"].rsplit(":", 1)[1])
        attempt = (stats.get(key) or [])[index]
        if not _active(attempt) or not _number(point.get("start_frame")):
            continue
        frame = point["start_frame"]
        window = by_index.get(index)
        issue = "invalid_score_window" if window is None else None
        if window is not None and any(other[0] != index and (
                other[1] == window[1] or max(other[1], window[1]) < min(other[2], window[2]))
                for other in windows):
            issue = "overlapping_score_windows"
        mapping, set_id = _context(frame, config)
        lc = attempt.get("lifecycle") or {}
        serving_side = side(lc.get("serving_team"))
        receiving_side = side(lc.get("receiving_team"))
        # Saved stable IDs are usable when no explicit context map overrides them.
        for court_side, field in ((serving_side, "serving_team_id"), (receiving_side, "receiving_team_id")):
            if court_side and court_side not in mapping and attempt.get(field):
                mapping[court_side] = attempt[field]
        original = point.get("original") or {}
        original_side = side(original.get("winner_court_side"))
        if original_side and original_side not in mapping and original.get("winner_team_id"):
            mapping[original_side] = original["winner_team_id"]
        serving = mapping.get(serving_side)
        winner = mapping.get(point.get("winner_side")) if point["status"] == "confirmed" else None
        for team in sorted({value for value in mapping.values() if value}):
            role = "serve" if serving == team else "receive" if serving and team != serving else "unknown"
            rows.append({"attempt_ref": point["attempt_ref"], "set_id": set_id or attempt.get("set_id"),
                         "team_id": team, "rotation": _rotation(frame, team, config), "role": role,
                         "won": winner == team if winner and issue is None else None, "winner_team_id": winner,
                         "serving_team_id": serving, "start_frame": frame, "end_frame": point.get("end_frame"),
                         "point_source": point["source"], "point_verified": point["verified"], "evidence_issue": issue})
    return rows


def summarize_rotation_rows(rows):
    """Keep unknown winners and roles outside both classified denominators."""
    groups = {}
    for row in rows:
        key = (row["team_id"], row["set_id"], row["rotation"])
        group = groups.setdefault(key, {"team_id": key[0], "set_id": key[1], "rotation": key[2],
            "rallies": 0, "unknown_roles": 0,
            "side_out": {"attempts": 0, "classified": 0, "wins": 0, "unknown": 0},
            "break_point": {"attempts": 0, "classified": 0, "wins": 0, "unknown": 0}})
        group["rallies"] += 1
        if row["role"] == "unknown":
            group["unknown_roles"] += 1
            continue
        metric = group["side_out" if row["role"] == "receive" else "break_point"]
        metric["attempts"] += 1
        if row["won"] is None:
            metric["unknown"] += 1
        else:
            metric["classified"] += 1
            metric["wins"] += row["won"]
    for group in groups.values():
        for key in ("side_out", "break_point"):
            metric = group[key]
            metric["rate"] = metric["wins"] / metric["classified"] if metric["classified"] else None
    return list(groups.values())


def build_tactics(stats, report, annotations, context=None, point_reviews=()):
    """Build tactical evidence without inference, source writes or side guesses."""
    config = context or {}
    if context is None:
        config = {"team_identity_map": (stats.get("match_analytics") or {}).get("team_identity_map") or {}}
    events = stats.get("all_events") or []
    ordered = sorted(((i, event) for i, event in enumerate(events)
                      if _active(event) and event.get("type") in CONTACT_TYPES
                      and type(event.get("frame")) is int and event["frame"] >= 0), key=lambda pair: pair[1]["frame"])
    current_key = "service_attempts" if stats.get("service_attempts") else "rallies"
    windows = _windows(stats, current_key)
    reconciliation = reconcile_statistics(stats, report, point_reviews)
    reconciled = {row["event_index"]: row for row in reconciliation["contacts"]}
    score_key = reconciliation["score_attempt_source"]
    frames = {ann["f"]: ann for ann in annotations.get("frames") or [] if type(ann.get("f")) is int}
    authors = {contact["event_index"]: contact for contact in report.get("contacts") or []}
    pairs, rejected = [], Counter()
    candidates = 0
    for position in range(len(ordered) - 1):
        set_index, setting = ordered[position]
        attack_index, attack = ordered[position + 1]
        if setting.get("type") not in SET_TYPES or attack.get("type") != "attack":
            continue
        candidates += 1
        start, end = setting["frame"], attack["frame"]
        if any(segment.get("match_phase") == "non_game"
               and _number(segment.get("start_frame")) and _number(segment.get("end_frame"))
               and segment["start_frame"] <= end and segment["end_frame"] >= start
               for segment in config.get("segments") or []):
            rejected["non_game_interval"] += 1
            continue
        matching = [row for row in windows if row[1] <= start < end <= row[2]]
        if len(matching) != 1:
            rejected["ambiguous_or_missing_rally"] += 1
            continue
        set_side, attack_side = _court_side(setting), _court_side(attack)
        if any(len({value for value in (side(event.get("court_side")), side(event.get("team"))) if value}) > 1
               for event in (setting, attack)):
            rejected["conflicting_contact_sides"] += 1
            continue
        if set_side and attack_side and set_side != attack_side:
            rejected["opposing_contact_sides"] += 1
            continue
        set_team, attack_team = _team(setting, start, config), _team(attack, end, config)
        _, start_set = _context(start, config)
        _, end_set = _context(end, config)
        if (set_team and attack_team and set_team != attack_team) or (start_set and end_set and start_set != end_set):
            rejected["match_context_boundary"] += 1
            continue
        derived, problem = _ball_path(frames, start, end)
        if problem:
            rejected[problem] += 1
            continue
        court_side = set_side if set_side == attack_side else None
        team = set_team if set_team and set_team == attack_team else None
        contact = authors.get(set_index, {})
        setter = contact.get("player_id") if (contact.get("status") == "assigned"
            and contact.get("frame") == start and contact.get("type") == setting["type"]
            and contact.get("player_id") in report.get("roster_ids", [])) else None
        setter_reason = "assigned_contact" if setter else "unidentified_setter"
        roster_team = next((row.get("team_id") for row in config.get("roster") or []
                            if row.get("player_id") == setter and row.get("confirmed") is True), None)
        if setter and team and roster_team and roster_team != team:
            setter, setter_reason = None, "confirmed_roster_team_conflict"
        context_row = reconciled.get(set_index, {})
        attack_row = reconciled.get(attack_index, {})
        score_ref = context_row.get("attempt_ref")
        score_attempt = None
        if (score_ref and score_ref == attack_row.get("attempt_ref")
                and not context_row.get("issues") and not attack_row.get("issues")):
            score_attempt = (stats.get(score_key) or [])[int(score_ref.rsplit(":", 1)[1])]
        rally_index, rally_start, rally_end, rally = matching[0]
        pairs.append({"set_event_index": set_index, "attack_event_index": attack_index,
            "start_frame": start, "end_frame": end, "rally_start_frame": rally_start, "rally_end_frame": rally_end,
            "rally_ref": f"{current_key}:{rally_index}", "score_attempt_ref": score_ref if score_attempt else None,
            "setter": setter, "setter_reason": setter_reason, "setter_source": ("human" if contact.get("label_source") == "human"
                and contact.get("verified") is True else "automatic") if setter else "unknown",
            "court_side": court_side, "team_id": team, "set_id": start_set or rally.get("set_id"),
            "rotation": _rotation(rally_start, team, config), "zone": _zone(frames[end], annotations),
            "reception_quality": _reception_quality(ordered, position, score_attempt, court_side),
            "reception_quality_source": score_ref if score_attempt else None,
            "ball_provenance": "derived" if derived else "observed", "derived_ball_frames": derived,
            "attack_ball_source": frames[end].get("ball_source") or "detected",
            "attack_outcome": attack_row.get("outcome", "unknown") if score_attempt else "unknown"})
    rotation_rows = _rotation_rows(stats, reconciliation, config)
    distribution = Counter((p["setter"], p["team_id"], p["zone"], p["reception_quality"]) for p in pairs)
    return {"schema": "tactics_v1", "pairs": pairs, "rotation_rallies": rotation_rows,
            "distribution": [{"setter": key[0], "team_id": key[1], "zone": key[2], "reception_quality": key[3], "count": value}
                             for key, value in distribution.items()],
            "rotations": summarize_rotation_rows(rotation_rows),
            "summary": {"candidate_pairs": candidates, "accepted_pairs": len(pairs), "rejected_pairs": sum(rejected.values()),
                        "rejection_reasons": dict(rejected), "unidentified_setters": sum(p["setter"] is None for p in pairs),
                        "unknown_teams": sum(p["team_id"] is None for p in pairs), "unknown_zones": sum(p["zone"] == "unknown" for p in pairs),
                        "unknown_reception_quality": sum(p["reception_quality"] == "unknown" for p in pairs),
                        "points_without_team_scope": len({p["attempt_ref"] for p in reconciliation["points"]}
                            - {row["attempt_ref"] for row in rotation_rows})},
            "options": {"teams": sorted({p["team_id"] for p in pairs if p["team_id"]} | {r["team_id"] for r in rotation_rows}),
                        "sets": sorted({str(p["set_id"]) for p in pairs if p["set_id"] is not None} | {str(r["set_id"]) for r in rotation_rows if r["set_id"] is not None}),
                        "setters": sorted({p["setter"] for p in pairs if p["setter"]})}}
