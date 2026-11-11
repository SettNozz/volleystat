"""Source-bound viewer context and developer edits; original results stay intact."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from src.tracking.match_analytics import DEFAULT_TEAM_IDENTITY_MAP, MATCH_PHASES, enrich_match_analytics
from src.tracking.player_contacts import apply_reviews, side, source_fingerprint, summarize
from src.viewer_app.player_contacts_io import _load_player_contact_inputs
from src.viewer_app.statistics_reviews import StatisticsSourceChanged, load_point_reviews, review_path


def workspace_path(stats_path):
    return stats_path.with_name(stats_path.name.removesuffix("_stats.json") + "_workspace_reviews.jsonl")


def author_review_path(stats_path):
    return stats_path.with_name(stats_path.name.removesuffix("_stats.json") + "_player_contact_reviews.jsonl")


def workspace_revision(stats_path, source):
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode())
    for path in (workspace_path(stats_path), author_review_path(stats_path), review_path(stats_path)):
        digest.update(path.name.encode())
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_history(stats_path, source):
    history = []
    path = workspace_path(stats_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("schema") != "match_workspace_v1" or record.get("source") != source:
                raise StatisticsSourceChanged("Analysis changed; workspace reviews belong to another source")
            if record.get("kind") not in {"context", "contact"}:
                raise ValueError("Unknown workspace record")
            history.append(record)
    return history


def guard_source(job, source):
    with Path(job["stats_path"]).open("rb") as stream:
        stats_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with Path(job["annotations_path"]).open("rb") as stream:
        ann_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if {"stats_sha256": stats_sha, "annotations_sha256": ann_sha} != source:
        raise StatisticsSourceChanged("Analysis changed; reload this workspace")


def append_record(job, source, revision, kind, payload):
    """Caller holds the viewer write lock and validates the full projection first."""
    path = Path(job["stats_path"])
    guard_source(job, source)
    if workspace_revision(path, source) != revision:
        raise StatisticsSourceChanged("Workspace changed; reload before saving")
    record = {"schema": "match_workspace_v1", "source": source, "kind": kind,
              "saved_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
    with workspace_path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def total_frames(stats, annotations=None):
    candidates = [stats.get("total_frames"), (annotations or {}).get("total_frames")]
    for event in stats.get("all_events") or []:
        if type(event.get("frame")) is int:
            candidates.append(event["frame"] + 1)
    for key in ("service_attempts", "service_attempts_fsm", "rallies"):
        for item in stats.get(key) or []:
            if type(item.get("end_frame")) is int:
                candidates.append(item["end_frame"] + 1)
    for frame in (annotations or {}).get("frames") or []:
        if type(frame.get("f")) is int:
            candidates.append(frame["f"] + 1)
    return max((v for v in candidates if type(v) is int and v > 0), default=1)


def default_context(stats, report):
    saved = stats.get("match_config") or {}
    mapping = dict((stats.get("match_analytics") or {}).get("team_identity_map") or DEFAULT_TEAM_IDENTITY_MAP)
    mapping.update(saved.get("team_identity_map") or {})
    return {"schema": "viewer_match_context_v1", "team_identity_map": mapping,
            "teams": {team: {"name": team.replace("_", " ").title()} for team in dict.fromkeys(mapping.values())},
            "segments": deepcopy(saved.get("segments") or stats.get("match_segments") or []),
            "initial_score": {team: 0 for team in mapping.values()}, "rotations": [],
            "roster": [{"player_id": player, "team_id": None, "confirmed": False} for player in report["roster_ids"]]}


def validate_context(config, count, roster_ids):
    if not isinstance(config, dict):
        raise ValueError("Match context must be an object")
    # Also reject non-finite numbers hidden in optional metadata.
    json.dumps(config, allow_nan=False)
    result = deepcopy(config)
    teams = result.get("teams")
    if not isinstance(teams, dict) or len(teams) != 2:
        raise ValueError("Define two stable teams")
    for team, profile in teams.items():
        if not isinstance(team, str) or not re.fullmatch(r"[\w.-]{1,64}", team):
            raise ValueError("Team ID must contain letters, numbers, dots, underscores or hyphens")
        if not isinstance(profile, dict) or not isinstance(profile.get("name"), str) or not profile["name"].strip() or len(profile["name"]) > 64:
            raise ValueError("Each team needs a name of 1–64 characters")
        profile["name"] = profile["name"].strip()

    def mapping(value):
        if not isinstance(value, dict) or set(value) != {"team_a", "team_b"} or set(value.values()) != set(teams):
            raise ValueError("Map each court side to a different stable team")

    def bounds(item):
        start, end = item.get("start_frame"), item.get("end_frame")
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < count:
            raise ValueError("Interval bounds must be whole frames within this video")

    mapping(result.get("team_identity_map"))
    segments = result.setdefault("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Segments must be a list")
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Invalid match segment")
        bounds(segment)
        if "match_phase" in segment and segment["match_phase"] not in MATCH_PHASES:
            raise ValueError("Unknown match phase")
        if "set_id" in segment and (not isinstance(segment["set_id"], str) or not segment["set_id"].strip() or len(segment["set_id"]) > 64):
            raise ValueError("Set ID must be text of 1–64 characters")
        if "team_identity_map" in segment:
            mapping(segment["team_identity_map"])
    segments.sort(key=lambda item: item["start_frame"])
    if any(left["end_frame"] >= right["start_frame"] for left, right in zip(segments, segments[1:])):
        raise ValueError("Match segments must not overlap; split the interval first")
    scores = result.setdefault("initial_score", {team: 0 for team in teams})
    if not isinstance(scores, dict) or set(scores) != set(teams) or any(type(value) is not int or not 0 <= value <= 999 for value in scores.values()):
        raise ValueError("Initial score needs a nonnegative whole number for each team")
    roster = result.setdefault("roster", [])
    if not isinstance(roster, list):
        raise ValueError("Roster must be a list")
    seen = set()
    for player in roster:
        if not isinstance(player, dict) or player.get("player_id") not in roster_ids or player["player_id"] in seen:
            raise ValueError("Roster entries must refer to distinct saved players")
        seen.add(player["player_id"])
        if type(player.get("confirmed")) is not bool or player.get("team_id") not in (None, *teams):
            raise ValueError("Invalid roster confirmation")
        if player["confirmed"] and player.get("team_id") is None:
            raise ValueError("Confirm a team for the player first")
    rotations = result.setdefault("rotations", [])
    if not isinstance(rotations, list):
        raise ValueError("Rotations must be a list")
    for rotation in rotations:
        if not isinstance(rotation, dict):
            raise ValueError("Invalid rotation")
        bounds(rotation)
        if rotation.get("team_id") not in teams or type(rotation.get("rotation")) is not int or not 1 <= rotation["rotation"] <= 6:
            raise ValueError("A confirmed rotation needs a team and a number from 1 to 6")
    rotations.sort(key=lambda item: (item["team_id"], item["start_frame"]))
    for left, right in zip(rotations, rotations[1:]):
        if left["team_id"] == right["team_id"] and left["end_frame"] >= right["start_frame"]:
            raise ValueError("Rotations for the same team must not overlap")
    result["schema"] = "viewer_match_context_v1"
    return result


def context_at(config, frame):
    segment = next((item for item in config.get("segments", []) if item["start_frame"] <= frame <= item["end_frame"]), {})
    mapping = dict(config["team_identity_map"])
    mapping.update(segment.get("team_identity_map") or {})
    return segment, mapping


def project_context(stats, config):
    """Recompute context from saved FSM attempts only, keeping source indexes."""
    result = deepcopy(stats)
    for event in result.get("all_events") or []:
        segment, mapping = context_at(config, event["frame"])
        event["excluded_from_match_stats"] = segment.get("match_phase") == "non_game"
        event["set_id"] = segment.get("set_id", event.get("set_id"))
        court_side = side(event.get("court_side")) or side(event.get("team"))
        event["stable_team_id"] = mapping.get(f"team_{court_side}") if court_side else None
        if segment.get("match_phase"):
            event["match_phase"] = segment["match_phase"]
    score_key = "service_attempts_fsm" if "service_attempts_fsm" in result else ("service_attempts" if result.get("service_attempts") else "rallies")
    active = []
    for key in ("service_attempts_fsm", "service_attempts", "rallies", "rallies_fsm"):
        for index, attempt in enumerate(result.get(key) or []):
            lc = attempt.get("lifecycle") or {}
            start = lc.get("serve_frame") if lc.get("serve_frame") is not None else attempt.get("start_frame", -1)
            segment, mapping = context_at(config, start)
            attempt["excluded_from_match_stats"] = segment.get("match_phase") == "non_game"
            attempt["set_id"] = segment.get("set_id", attempt.get("set_id"))
            attempt["team_identity_map"] = mapping
            if key == score_key and not attempt["excluded_from_match_stats"]:
                # A local index survives enrich_match_analytics's chronological sort.
                copy = deepcopy(attempt)
                copy["_workspace_source_index"] = index
                active.append(copy)
    analytics = enrich_match_analytics(active, config["team_identity_map"], config.get("segments", []))
    for attempt in active:
        index = attempt.pop("_workspace_source_index")
        result[score_key][index] = attempt
    scores = config.get("initial_score") or {}
    first_set = next((point.get("set_id") for point in analytics["points"]), None)
    for point in analytics["points"]:
        point["score_after"] = {team: point.get("score_after", {}).get(team, 0) + scores.get(team, 0) for team in config["teams"]}
        if point.get("set_id") == first_set:
            point["set_score_after"] = {team: point.get("set_score_after", {}).get(team, 0) + scores.get(team, 0) for team in config["teams"]}
    analytics["score"] = {team: analytics["score"].get(team, 0) + scores.get(team, 0) for team in config["teams"]}
    set_key = str(first_set or "unassigned")
    existing = analytics["score_by_set"].get(set_key, {})
    analytics["score_by_set"][set_key] = {team: existing.get(team, 0) + scores.get(team, 0) for team in config["teams"]}
    analytics["context_source"] = "human_viewer_match_context"
    result["match_analytics_original"] = result.get("match_analytics")
    result["match_analytics"] = analytics
    lifecycle = result.get("rally_lifecycle")
    if isinstance(lifecycle, dict):
        lifecycle["service_attempts"] = sum(not a.get("excluded_from_match_stats") for a in result.get("service_attempts", []))
        lifecycle["playable_rallies"] = sum(not a.get("excluded_from_match_stats") for a in result.get("rallies", []))
    result["match_context"] = config
    return result


def read_workspace(job, *, method=None, include_annotations=False):
    """Load saved predictions, legacy reviews, and the current workspace projection."""
    base_report, source, bundle, references, faults = _load_player_contact_inputs(job)
    if bundle:
        chosen = method or bundle["selected_method"]
        if chosen not in bundle["reports"]:
            raise ValueError("Unknown experimental contact method")
        base_report = bundle["reports"][chosen]
    else:
        chosen = base_report["method"] if base_report.get("attribution", {}).get("status") == "completed" else "body_baseline"
        if method and method != chosen:
            raise ValueError("This result has no contact-method experiment")
    stats_path = Path(job["stats_path"])
    stats_bytes = stats_path.read_bytes()
    guard_source(job, source)
    if hashlib.sha256(stats_bytes).hexdigest() != source["stats_sha256"]:
        raise StatisticsSourceChanged("Analysis changed; reload this workspace")
    raw = json.loads(stats_bytes)
    report = deepcopy(base_report)
    seen = {c["event_index"] for c in report["contacts"]}
    report["contacts"].extend(deepcopy(c) for c in faults if c["event_index"] not in seen)
    reviews = []
    path = author_review_path(stats_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("source") != source:
                raise StatisticsSourceChanged("Analysis changed; saved contact reviews belong to another source")
            reviews.append(record["review"])
    legacy_report = apply_reviews(report, reviews)
    history = load_history(stats_path, source)
    annotation = json.loads(Path(job["annotations_path"]).read_bytes()) if include_annotations else None
    count = total_frames(raw, annotation)
    config = default_context(raw, report)
    context_saved = False
    edits = []
    for record in history:
        if record["kind"] == "context":
            config = record["payload"]
            context_saved = True
        else:
            edits.append(record["payload"])
    if context_saved:
        config = validate_context(config, count, report["roster_ids"])
    effective = deepcopy(raw)
    if edits:
        from src.viewer_app.developer_review import apply_contact_edits
        effective, effective_report = apply_contact_edits(effective, legacy_report, edits)
    else:
        effective_report = summarize(legacy_report)
    if context_saved:
        effective = project_context(effective, config)
    if edits or context_saved:
        active_events = [event for event in effective.get("all_events") or []
                         if not event.get("review_excluded") and not event.get("excluded_from_match_stats")]
        for key in ("rallies", "service_attempts"):
            for attempt in effective.get(key) or []:
                start, end = attempt.get("start_frame"), attempt.get("end_frame")
                if type(start) is int and type(end) is int:
                    attempt["events_original"] = attempt.get("events")
                    attempt["events"] = sorted((deepcopy(event) for event in active_events if start <= event["frame"] <= end), key=lambda event: event["frame"])
    for contact in effective_report["contacts"]:
        index = contact["event_index"]
        if 0 <= index < len(effective.get("all_events") or []):
            event = effective["all_events"][index]
            contact["excluded_from_match_stats"] = bool(event.get("excluded_from_match_stats"))
    summarize(effective_report)
    return {"stats": effective, "raw_stats": raw, "report": effective_report,
            "raw_report": report, "legacy_report": legacy_report, "author_reviews": reviews,
            "source": source, "revision": workspace_revision(stats_path, source), "history": history,
            "config": config, "context_saved": context_saved, "edits": edits, "annotations": annotation,
            "total_frames": count, "fps": float(raw.get("fps", 30)), "bundle": bundle,
            "method_id": chosen, "references": references,
            "point_reviews": load_point_reviews(stats_path, source)}


def score_projection(state):
    """Reviewed score for display, keyed by stable team and original score slot."""
    if not state["context_saved"]:
        return None
    from src.tracking.player_match_statistics import build_player_match_statistics
    summary = build_player_match_statistics(state["stats"], state["report"], state["point_reviews"])
    points = []
    for point in summary["reconciliation"]["points"]:
        start = point["start_frame"]
        _, mapping = context_at(state["config"], start if start is not None else -1)
        winner = mapping.get(f"team_{point['winner_side']}") if point["winner_side"] else None
        frame = point["end_frame"] if point["reviewed"] else point["original"].get("evidence_frame")
        if type(frame) is not int:
            frame = point["end_frame"]
        points.append({"frame": frame, "start_frame": start, "set_id": point["set_id"],
                       "winner_team_id": winner, "status": point["status"]})
    first_set = next((point["set_id"] for point in points), None)
    if not points:
        first_set = next((segment.get("set_id") for segment in state["config"]["segments"] if segment.get("set_id")), None)
    return {"config": state["config"], "points": points, "first_set": first_set}
