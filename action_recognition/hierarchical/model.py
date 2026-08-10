"""Multi-head temporal model for phase, contacts, technique, and context.

The model intentionally does not predict the score.  Point winners are
event-sourced from the next confirmed server by ``match_analytics``.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contract import validate_model_outputs
from .taxonomy import (
    CONTEXT_ROLES,
    MATCH_PHASES,
    PHYSICAL_ACTIONS,
    RECEPTION_QUALITIES,
    TEAM_SIDES,
)


class DilatedTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        encoded = self.net(inputs.transpose(1, 2)).transpose(1, 2)
        return self.norm(residual + encoded)


class HierarchicalEventModel(nn.Module):
    """Return frame-aligned logits for six independently supervised heads."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {
            "feature_dim": int(feature_dim),
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "architecture": "hierarchical_event_tcn_v1",
        }
        self.input_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.local_encoder = nn.Sequential(
            *[
                DilatedTemporalBlock(hidden_dim, dilation, dropout)
                for dilation in (1, 2, 4)
            ]
        )
        self.long_encoder = nn.Sequential(
            *[
                DilatedTemporalBlock(hidden_dim, dilation, dropout)
                for dilation in (8, 16, 32)
            ]
        )
        self.phase_head = nn.Linear(hidden_dim, len(MATCH_PHASES))
        self.contact_head = nn.Linear(hidden_dim, 1)
        self.team_head = nn.Linear(hidden_dim, len(TEAM_SIDES))
        self.physical_action_head = nn.Linear(hidden_dim, len(PHYSICAL_ACTIONS))
        self.context_role_head = nn.Linear(hidden_dim, len(CONTEXT_ROLES))
        self.reception_quality_head = nn.Linear(
            hidden_dim, len(RECEPTION_QUALITIES)
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape (batch, frames, channels)")
        local = self.local_encoder(self.input_projection(features))
        long_context = self.long_encoder(local)
        return {
            "phase_logits": self.phase_head(long_context),
            "contact_logits": self.contact_head(local).squeeze(-1),
            "team_logits": self.team_head(local),
            "physical_action_logits": self.physical_action_head(local),
            "context_role_logits": self.context_role_head(local),
            "reception_quality_logits": self.reception_quality_head(local),
        }


def _masked_cross_entropy(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    label_weight: Optional[Tensor] = None,
) -> Tensor:
    selected = mask.bool() & (target >= 0)
    if not torch.any(selected):
        return logits.sum() * 0.0
    values = F.cross_entropy(
        logits[selected], target[selected].long(), reduction="none"
    )
    if label_weight is None:
        return values.mean()
    selected_weights = label_weight[selected].float()
    return (values * selected_weights).sum() / selected_weights.sum().clamp_min(1e-6)


def hierarchical_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Optional[Mapping[str, Tensor]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute masked losses so weak/missing labels do not become negatives."""
    validate_model_outputs(outputs)
    masks = dict(masks or {})
    weights = dict(weights or {})
    contact_target = targets["contact_target"].float()
    label_weight = targets.get(
        "label_weight", torch.ones_like(contact_target, dtype=torch.float32)
    ).float()
    contact_mask = masks.get(
        "contact_mask", torch.ones_like(contact_target, dtype=torch.bool)
    ).bool()
    if torch.any(contact_mask):
        contact_values = F.binary_cross_entropy_with_logits(
            outputs["contact_logits"][contact_mask],
            contact_target[contact_mask],
            reduction="none",
        )
        selected_weights = label_weight[contact_mask]
        contact_loss = (
            (contact_values * selected_weights).sum()
            / selected_weights.sum().clamp_min(1e-6)
        )
    else:
        contact_loss = outputs["contact_logits"].sum() * 0.0

    contact_frames = contact_target > 0.5
    losses = {
        "phase": _masked_cross_entropy(
            outputs["phase_logits"],
            targets["phase_target"],
            masks.get("phase_mask", torch.ones_like(contact_frames)),
            label_weight,
        ),
        "contact": contact_loss,
        "team": _masked_cross_entropy(
            outputs["team_logits"],
            targets["team_target"],
            masks.get("team_mask", contact_frames),
            label_weight,
        ),
        "physical_action": _masked_cross_entropy(
            outputs["physical_action_logits"],
            targets["physical_action_target"],
            masks.get("physical_action_mask", contact_frames),
            label_weight,
        ),
        "context_role": _masked_cross_entropy(
            outputs["context_role_logits"],
            targets["context_role_target"],
            masks.get("context_role_mask", contact_frames),
            label_weight,
        ),
        "reception_quality": _masked_cross_entropy(
            outputs["reception_quality_logits"],
            targets["reception_quality_target"],
            masks.get(
                "reception_quality_mask",
                torch.zeros_like(contact_frames, dtype=torch.bool),
            ),
            label_weight,
        ),
    }
    total = sum(weights.get(name, 1.0) * value for name, value in losses.items())
    return total, losses
