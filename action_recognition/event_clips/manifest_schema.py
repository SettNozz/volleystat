"""Versioned, backwards-compatible event-clip manifest schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.tracking.match_analytics import (
    CONTEXT_ROLES,
    DEFAULT_TEAM_IDENTITY_MAP,
    MATCH_PHASES,
    PHYSICAL_ACTIONS,
    RECEPTION_QUALITIES,
)


MANIFEST_SCHEMA_VERSION = 2

_FSM_CONTEXT = {
    "serve": ("serve", "serve", 0),
    "serve_fault": ("serve", "serve", 0),
    "reception": ("forearm_pass", "serve_reception", 1),
    "dig": ("forearm_pass", "defensive_dig", 1),
    "pass": ("overhead_set", "set", 2),
    "attack": ("attack_hit", "attack", 3),
    "block": ("block_touch", "block", 0),
}


def manifest_taxonomies() -> dict[str, list[str]]:
    return {
        "match_phases": list(MATCH_PHASES),
        "physical_actions": list(PHYSICAL_ACTIONS),
        "context_roles": list(CONTEXT_ROLES),
        "reception_qualities": list(RECEPTION_QUALITIES),
    }


def upgrade_clip(clip: Mapping[str, Any]) -> dict[str, Any]:
    """Return a v2 clip while preserving every legacy/custom field."""
    upgraded = deepcopy(dict(clip))
    fsm_type = str(upgraded.get("fsm_event_type") or "")
    physical, role, inferred_touch = _FSM_CONTEXT.get(
        fsm_type, ("unknown", "unknown", None)
    )
    court_side = upgraded.get("court_side") or upgraded.get("team")
    stable_team_id = upgraded.get("stable_team_id")
    if stable_team_id is None:
        stable_team_id = DEFAULT_TEAM_IDENTITY_MAP.get(court_side)

    upgraded.setdefault("rally_id", None)
    upgraded.setdefault("possession_id", None)
    if upgraded.get("touch_index") is None:
        upgraded["touch_index"] = inferred_touch
    if upgraded.get("court_side") is None:
        upgraded["court_side"] = court_side
    if upgraded.get("stable_team_id") is None:
        upgraded["stable_team_id"] = stable_team_id
    if upgraded.get("match_phase") is None:
        upgraded["match_phase"] = (
            "serve_flight" if fsm_type in ("serve", "serve_fault")
            else "rally_live" if fsm_type else "unknown"
        )
    if upgraded.get("physical_action") is None:
        upgraded["physical_action"] = physical
    if upgraded.get("context_role") is None:
        upgraded["context_role"] = role
    upgraded.setdefault("is_service_reception", role == "serve_reception")
    if upgraded.get("serve_outcome") is None:
        upgraded["serve_outcome"] = "unknown"
    if upgraded.get("reception_quality") is None:
        upgraded["reception_quality"] = (
            "unknown" if upgraded["context_role"] == "serve_reception"
            else "not_applicable"
        )
    upgraded.setdefault("rally_context_start_frame", upgraded.get("start_frame"))
    upgraded.setdefault("rally_context_end_frame", upgraded.get("end_frame"))
    upgraded.setdefault("point_winner_team_id", None)
    upgraded.setdefault("point_status", "unresolved")
    # A human legacy action label does not imply that the newly introduced
    # phase/role/quality axes were reviewed by a human. Keep inferred context
    # explicitly weak until it is accepted in the contextual annotator.
    if upgraded.get("context_label_source") is None:
        upgraded["context_label_source"] = "weak_fsm" if fsm_type else "unlabeled"
    upgraded.setdefault("context_label_confidence", None)
    return upgraded


def upgrade_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade an in-memory v1/v2 manifest without discarding metadata."""
    upgraded = deepcopy(dict(manifest))
    upgraded["schema_version"] = MANIFEST_SCHEMA_VERSION
    upgraded.setdefault("taxonomies", manifest_taxonomies())
    upgraded["clips"] = [upgrade_clip(clip) for clip in manifest.get("clips", [])]
    return upgraded


def validate_context_value(field: str, value: str) -> str:
    allowed = {
        "match_phase": MATCH_PHASES,
        "physical_action": PHYSICAL_ACTIONS,
        "context_role": CONTEXT_ROLES,
        "reception_quality": RECEPTION_QUALITIES,
    }
    if field not in allowed:
        raise ValueError(f"unknown contextual field {field!r}")
    if value not in allowed[field]:
        raise ValueError(
            f"unknown {field} {value!r}; expected one of {allowed[field]}"
        )
    return value
