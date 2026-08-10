"""Dataset/model contract for hierarchical event training."""

from __future__ import annotations

from typing import Any, Mapping

from action_recognition.event_clips.manifest_schema import upgrade_clip
from .taxonomy import (
    CONTEXT_ROLE_TO_INDEX,
    MATCH_PHASE_TO_INDEX,
    PHYSICAL_ACTION_TO_INDEX,
    RECEPTION_QUALITY_TO_INDEX,
    TEAM_SIDE_TO_INDEX,
)


MODEL_OUTPUT_KEYS = (
    "phase_logits",
    "contact_logits",
    "team_logits",
    "physical_action_logits",
    "context_role_logits",
    "reception_quality_logits",
)


def encode_clip_context(clip: Mapping[str, Any]) -> dict[str, Any]:
    """Encode one v1/v2 clip into head targets and supervision masks."""
    item = upgrade_clip(clip)
    is_contact = item.get("contact_frame") is not None
    is_reception = item["context_role"] == "serve_reception"
    context_human = item.get("context_label_source") == "human"
    return {
        "phase_target": MATCH_PHASE_TO_INDEX[item["match_phase"]],
        "contact_target": float(is_contact),
        "team_target": TEAM_SIDE_TO_INDEX.get(item.get("court_side"), 0),
        "physical_action_target": PHYSICAL_ACTION_TO_INDEX[item["physical_action"]],
        "context_role_target": CONTEXT_ROLE_TO_INDEX[item["context_role"]],
        "reception_quality_target": RECEPTION_QUALITY_TO_INDEX[
            item["reception_quality"]
        ],
        "phase_mask": item["match_phase"] != "unknown",
        "contact_mask": True,
        "team_mask": is_contact and item.get("court_side") in ("team_a", "team_b"),
        "physical_action_mask": is_contact and item["physical_action"] != "unknown",
        "context_role_mask": is_contact and item["context_role"] != "unknown",
        "reception_quality_mask": is_reception,
        "human_context": context_human,
        "label_weight": 1.0 if context_human else 0.5,
    }


def validate_model_outputs(outputs: Mapping[str, Any]) -> None:
    missing = [key for key in MODEL_OUTPUT_KEYS if key not in outputs]
    if missing:
        raise ValueError(f"hierarchical model output is missing {missing}")

