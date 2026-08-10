"""Contextual match analytics built on top of sparse FSM events.

The contact detector answers *when* the ball was touched.  This module adds
the longer-lived context that cannot be decided from an isolated clip:

* stable team identities (court sides may change between sets),
* rally / possession / touch identifiers,
* physical-action and contextual-role weak labels,
* match phases, and
* an event-sourced point ledger confirmed by the next serve.

All enrichment is additive.  Existing result JSON and v1-v3 clip manifests
remain readable because their legacy fields are never renamed or removed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional


MATCH_PHASES = (
    "unknown",
    "non_game",
    "dead_ball",
    "serve_setup",
    "serve_flight",
    "rally_live",
    "rally_end",
)

PHYSICAL_ACTIONS = (
    "unknown",
    "serve",
    "forearm_pass",
    "overhead_set",
    "attack_hit",
    "block_touch",
    "other_touch",
)

CONTEXT_ROLES = (
    "unknown",
    "serve",
    "serve_reception",
    "defensive_dig",
    "set",
    "attack",
    "block",
    "free_ball",
    "other",
)

RECEPTION_QUALITIES = (
    "not_applicable",
    "unknown",
    "no_touch_ace",
    "error",
    "overpass",
    "negative",
    "positive",
    "perfect",
    "controlled",
)

DEFAULT_TEAM_IDENTITY_MAP = {
    "team_a": "opponent_team",  # far side
    "team_b": "observed_team",  # near / camera side
}

_EVENT_CONTEXT = {
    "serve": ("serve", "serve"),
    "serve_fault": ("serve", "serve"),
    "reception": ("forearm_pass", "serve_reception"),
    "dig": ("forearm_pass", "defensive_dig"),
    "pass": ("overhead_set", "set"),
    "attack": ("attack_hit", "attack"),
    "block": ("block_touch", "block"),
}


def opposite_court_side(side: Optional[str]) -> Optional[str]:
    if side == "team_a":
        return "team_b"
    if side == "team_b":
        return "team_a"
    return None


def match_phase_from_fsm(state: Optional[str]) -> str:
    """Map a legacy FSM state to the canonical coarse match phase."""
    return {
        "serve_prep": "dead_ball",
        "serve_ready": "serve_setup",
        "serving": "serve_flight",
        "rally_active": "rally_live",
        "rally_end": "rally_end",
        "timeout": "non_game",
        "set_end": "non_game",
        "match_end": "non_game",
    }.get(str(state or ""), "unknown")


def manual_match_phase_at_frame(
    frame_number: int,
    manual_segments: Iterable[Mapping[str, Any]],
) -> Optional[str]:
    """Return the explicit phase override at a frame, if one exists."""
    for segment in manual_segments:
        phase = segment.get("match_phase")
        if (
            phase is not None
            and int(segment.get("start_frame", -1)) <= frame_number
            <= int(segment.get("end_frame", -1))
        ):
            if phase not in MATCH_PHASES:
                raise ValueError(f"unknown match phase {phase!r}")
            return str(phase)
    return None


def filter_non_game_records(
    records: Iterable[MutableMapping[str, Any]],
    manual_segments: Iterable[Mapping[str, Any]],
    frame_field: str = "frame",
) -> List[MutableMapping[str, Any]]:
    """Keep records outside explicit ``non_game`` intervals."""
    segments = list(manual_segments)
    kept: List[MutableMapping[str, Any]] = []
    for record in records:
        lifecycle = record.get("lifecycle") or {}
        raw_frame = record.get(frame_field)
        if raw_frame is None:
            raw_frame = lifecycle.get("serve_frame", record.get("start_frame", -1))
        phase = manual_match_phase_at_frame(int(raw_frame), segments)
        if phase == "non_game":
            record["excluded_from_match_stats"] = True
            continue
        record["excluded_from_match_stats"] = False
        kept.append(record)
    return kept


def annotate_match_phases(
    frames: Iterable[MutableMapping[str, Any]],
    manual_segments: Iterable[Mapping[str, Any]] = (),
) -> None:
    """Add ``match_phase`` to frame annotations.

    Manual segments win over the FSM mapping.  They make it possible to mark a
    whole warm-up, referee delay, timeout, or between-set interval with two
    boundary clicks before a phase classifier is available.
    """
    segments = list(manual_segments)
    for segment in segments:
        phase = segment.get("match_phase")
        if phase is None:
            continue
        if phase not in MATCH_PHASES:
            raise ValueError(f"unknown match phase {phase!r}")
    for frame in frames:
        frame_number = int(frame.get("f", 0))
        phase = match_phase_from_fsm(frame.get("state"))
        override = manual_match_phase_at_frame(frame_number, segments)
        if override is not None:
            phase = override
        frame["match_phase"] = phase


def annotate_score_timeline(
    frames: Iterable[MutableMapping[str, Any]],
    match_analytics: Mapping[str, Any],
) -> None:
    """Attach the delayed, confirmed score to every frame annotation."""
    base_score = {
        str(team_id): 0 for team_id in (match_analytics.get("score") or {})
    }
    confirmations = sorted(
        (
            int(point["evidence_frame"]),
            point,
        )
        for point in (match_analytics.get("points") or [])
        if point.get("status") == "confirmed"
        and point.get("evidence_frame") is not None
    )
    score = dict(base_score)
    confirmation_index = 0
    last_point_id: Optional[str] = None
    for frame in frames:
        frame_number = int(frame.get("f", 0))
        while (
            confirmation_index < len(confirmations)
            and confirmations[confirmation_index][0] <= frame_number
        ):
            _, point = confirmations[confirmation_index]
            winner = point.get("winner_team_id")
            if winner:
                score[winner] = score.get(winner, 0) + 1
                last_point_id = point.get("point_id")
            confirmation_index += 1
        frame["score"] = dict(score)
        frame["last_confirmed_point_id"] = last_point_id


def _stable_team_id(
    court_side: Optional[str],
    identity_map: Mapping[str, str],
) -> Optional[str]:
    if court_side is None:
        return None
    return identity_map.get(court_side, court_side)


def _event_context(event_type: Optional[str]) -> tuple[str, str]:
    return _EVENT_CONTEXT.get(str(event_type or ""), ("other_touch", "other"))


def _confirmed_serve(attempt: Mapping[str, Any]) -> bool:
    lifecycle = attempt.get("lifecycle") or {}
    return bool(
        (
            lifecycle.get("serve_confirmed")
            or lifecycle.get("status") == "serve_fault"
        )
        and lifecycle.get("serve_frame") is not None
        and lifecycle.get("serving_team") in ("team_a", "team_b")
    )


def _assign_event_context(
    attempt: MutableMapping[str, Any],
    rally_id: str,
    identity_map: Mapping[str, str],
) -> None:
    possession_number = 0
    possession_team: Optional[str] = None
    touch_index = 0

    for event_number, event in enumerate(
        sorted(attempt.get("events") or [], key=lambda item: item.get("frame", -1)),
        start=1,
    ):
        court_side = event.get("team")
        event_type = event.get("type")
        physical_action, context_role = _event_context(event_type)

        if event_type in ("serve", "serve_fault"):
            possession_number = 0
            possession_team = court_side
            touch_index = 0
        elif event_type == "block":
            if court_side != possession_team:
                possession_number += 1
            possession_team = court_side
            touch_index = 0  # a block does not consume a team contact
        else:
            if court_side != possession_team:
                possession_number += 1
                possession_team = court_side
                touch_index = 0
            touch_index += 1

        event.update(
            event_id=f"{rally_id}_event_{event_number:03d}",
            rally_id=rally_id,
            possession_id=(
                f"{rally_id}_possession_{possession_number:03d}"
                if possession_number else None
            ),
            touch_index=touch_index,
            court_side=court_side,
            stable_team_id=_stable_team_id(court_side, identity_map),
            physical_action=physical_action,
            context_role=context_role,
            match_phase=(
                "serve_flight"
                if event_type in ("serve", "serve_fault")
                else "rally_live"
            ),
            context_label_source="weak_fsm",
            context_label_confidence=(
                0.98 if context_role in ("serve", "serve_reception") else 0.75
            ),
        )


def _point_record(
    attempt: Mapping[str, Any],
    point_number: int,
    winner_side: Optional[str],
    winner_team_id: Optional[str],
    evidence: str,
    evidence_frame: Optional[int],
    confidence: float,
) -> Dict[str, Any]:
    return {
        "point_id": f"point_{point_number:04d}",
        "rally_id": attempt["rally_id"],
        "set_id": attempt.get("set_id"),
        "winner_team_id": winner_team_id,
        "winner_court_side": winner_side,
        "evidence": evidence,
        "evidence_frame": evidence_frame,
        "confidence": confidence,
        "status": "confirmed" if winner_team_id else "unresolved",
    }


def _attempt_frame(attempt: Mapping[str, Any]) -> int:
    lifecycle = attempt.get("lifecycle") or {}
    frame = lifecycle.get("serve_frame")
    return int(frame if frame is not None else attempt.get("start_frame", -1))


def _segment_at_frame(
    frame: int, segments: Iterable[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    return next(
        (
            segment for segment in segments
            if int(segment.get("start_frame", -1)) <= frame
            <= int(segment.get("end_frame", -1))
        ),
        None,
    )


def enrich_match_analytics(
    service_attempts: List[MutableMapping[str, Any]],
    team_identity_map: Optional[Mapping[str, str]] = None,
    match_segments: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Enrich service attempts and return the point-ledger summary.

    Point ``n`` is normally confirmed when service attempt ``n + 1`` begins:
    the team serving next won the preceding rally under rally scoring.  The
    final attempt therefore remains unresolved unless it is an explicit serve
    fault, which awards the point to the receiver immediately.
    """
    identity_map = dict(DEFAULT_TEAM_IDENTITY_MAP)
    if team_identity_map:
        identity_map.update(team_identity_map)
    segments = list(match_segments)

    attempts = sorted(
        service_attempts,
        key=lambda item: (
            (item.get("lifecycle") or {}).get("serve_frame")
            if (item.get("lifecycle") or {}).get("serve_frame") is not None
            else item.get("start_frame", -1)
        ),
    )
    service_attempts[:] = attempts
    attempt_maps: List[Dict[str, str]] = []
    for index, attempt in enumerate(attempts, start=1):
        rally_id = str(attempt.get("rally_id") or f"rally_{index:04d}")
        attempt["rally_id"] = rally_id
        attempt["service_attempt_id"] = f"service_attempt_{index:04d}"
        lifecycle = attempt.get("lifecycle") or {}
        segment = _segment_at_frame(_attempt_frame(attempt), segments)
        local_identity_map = dict(identity_map)
        if segment and isinstance(segment.get("team_identity_map"), Mapping):
            local_identity_map.update(segment["team_identity_map"])
        attempt_maps.append(local_identity_map)
        if attempt.get("set_id") is None and segment is not None:
            attempt["set_id"] = segment.get("set_id")
        serving_side = lifecycle.get("serving_team")
        receiving_side = lifecycle.get("receiving_team") or opposite_court_side(
            serving_side
        )
        attempt["serving_court_side"] = serving_side
        attempt["receiving_court_side"] = receiving_side
        attempt["serving_team_id"] = _stable_team_id(
            serving_side, local_identity_map
        )
        attempt["receiving_team_id"] = _stable_team_id(
            receiving_side, local_identity_map
        )
        attempt["match_phase"] = "rally_live"
        _assign_event_context(attempt, rally_id, local_identity_map)

    ledger: List[Dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        next_attempt = attempts[index + 1] if index + 1 < len(attempts) else None
        lifecycle = attempt.get("lifecycle") or {}
        same_set = not (
            next_attempt is not None
            and attempt.get("set_id") is not None
            and next_attempt.get("set_id") is not None
            and attempt.get("set_id") != next_attempt.get("set_id")
        )
        if next_attempt is not None and same_set and _confirmed_serve(next_attempt):
            next_lifecycle = next_attempt["lifecycle"]
            point = _point_record(
                attempt,
                index + 1,
                next_lifecycle.get("serving_team"),
                next_attempt.get("serving_team_id"),
                "next_confirmed_serve",
                int(next_lifecycle["serve_frame"]),
                0.98,
            )
        elif lifecycle.get("status") == "serve_fault":
            point = _point_record(
                attempt,
                index + 1,
                opposite_court_side(lifecycle.get("serving_team")),
                _stable_team_id(
                    opposite_court_side(lifecycle.get("serving_team")),
                    attempt_maps[index],
                ),
                "serve_fault",
                attempt.get("end_frame", lifecycle.get("serve_frame")),
                0.95,
            )
        else:
            point = _point_record(
                attempt,
                index + 1,
                None,
                None,
                "set_boundary" if next_attempt is not None and not same_set
                else "awaiting_next_serve",
                None,
                0.0,
            )
        ledger.append(point)
        attempt["point"] = deepcopy(point)

    known_team_ids = list(identity_map.values())
    for attempt in attempts:
        known_team_ids.extend(
            team_id for team_id in (
                attempt.get("serving_team_id"), attempt.get("receiving_team_id")
            ) if team_id
        )
    score = {team_id: 0 for team_id in dict.fromkeys(known_team_ids)}
    score_by_set: Dict[str, Dict[str, int]] = {}
    for point in ledger:
        winner = point.get("winner_team_id")
        if winner:
            score[winner] = score.get(winner, 0) + 1
            set_key = str(point.get("set_id") or "unassigned")
            set_score = score_by_set.setdefault(
                set_key, {team_id: 0 for team_id in score}
            )
            set_score[winner] = set_score.get(winner, 0) + 1
        point["score_after"] = dict(score)
        point["set_score_after"] = dict(
            score_by_set.get(str(point.get("set_id") or "unassigned"), {})
        )

    return {
        "schema_version": 1,
        "team_identity_map": identity_map,
        "score": score,
        "score_by_set": score_by_set,
        "confirmed_points": sum(p["status"] == "confirmed" for p in ledger),
        "unresolved_points": sum(p["status"] == "unresolved" for p in ledger),
        "points": ledger,
    }
