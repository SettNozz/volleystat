"""Canonical class order for hierarchical model heads."""

from __future__ import annotations

from typing import Final

from src.tracking.match_analytics import (
    CONTEXT_ROLES,
    MATCH_PHASES,
    PHYSICAL_ACTIONS,
    RECEPTION_QUALITIES,
)

TEAM_SIDES: Final[tuple[str, ...]] = ("unknown", "team_a", "team_b")

MATCH_PHASE_TO_INDEX = {name: index for index, name in enumerate(MATCH_PHASES)}
PHYSICAL_ACTION_TO_INDEX = {
    name: index for index, name in enumerate(PHYSICAL_ACTIONS)
}
CONTEXT_ROLE_TO_INDEX = {name: index for index, name in enumerate(CONTEXT_ROLES)}
RECEPTION_QUALITY_TO_INDEX = {
    name: index for index, name in enumerate(RECEPTION_QUALITIES)
}
TEAM_SIDE_TO_INDEX = {name: index for index, name in enumerate(TEAM_SIDES)}

