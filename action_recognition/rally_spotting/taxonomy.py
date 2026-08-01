"""Canonical event taxonomy and label provenance."""

from __future__ import annotations

from typing import Final

EVENT_TYPES: Final[tuple[str, ...]] = (
    "serve_contact",
    "reception_contact",
    "followup_contact",
    "rally_end",
)
EVENT_TO_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(EVENT_TYPES)
}
NUM_CLASSES: Final[int] = len(EVENT_TYPES)

HUMAN_LABEL_SOURCE: Final[str] = "human"
WEAK_LABEL_SOURCE: Final[str] = "weak_fsm"
LABEL_SOURCES: Final[tuple[str, ...]] = (
    HUMAN_LABEL_SOURCE,
    WEAK_LABEL_SOURCE,
)


def validate_event_type(value: str) -> str:
    """Validate and return an event type."""
    if value not in EVENT_TO_INDEX:
        raise ValueError(
            f"Unknown rally event {value!r}; expected one of {EVENT_TYPES}"
        )
    return value


def normalize_label_source(value: str) -> str:
    """Normalize supported provenance aliases to their canonical value."""
    aliases = {
        "human": HUMAN_LABEL_SOURCE,
        "weak": WEAK_LABEL_SOURCE,
        "weak_fsm": WEAK_LABEL_SOURCE,
        "fsm_rule": WEAK_LABEL_SOURCE,
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unknown label_source {value!r}; expected human or weak_fsm"
        ) from exc
