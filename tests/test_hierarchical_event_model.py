import torch

from action_recognition.hierarchical.model import (
    HierarchicalEventModel,
    hierarchical_loss,
)
from action_recognition.hierarchical.taxonomy import (
    CONTEXT_ROLES,
    MATCH_PHASES,
    PHYSICAL_ACTIONS,
    RECEPTION_QUALITIES,
    TEAM_SIDES,
)


def test_hierarchical_model_shapes_and_masked_loss():
    batch, frames, features = 2, 64, 22
    model = HierarchicalEventModel(features, hidden_dim=32, dropout=0.0)
    outputs = model(torch.randn(batch, frames, features))

    assert outputs["phase_logits"].shape == (batch, frames, len(MATCH_PHASES))
    assert outputs["contact_logits"].shape == (batch, frames)
    assert outputs["team_logits"].shape == (batch, frames, len(TEAM_SIDES))
    assert outputs["physical_action_logits"].shape[-1] == len(PHYSICAL_ACTIONS)
    assert outputs["context_role_logits"].shape[-1] == len(CONTEXT_ROLES)
    assert outputs["reception_quality_logits"].shape[-1] == len(
        RECEPTION_QUALITIES
    )

    contact = torch.zeros(batch, frames)
    contact[:, 10] = 1.0
    targets = {
        "phase_target": torch.zeros(batch, frames, dtype=torch.long),
        "contact_target": contact,
        "team_target": torch.zeros(batch, frames, dtype=torch.long),
        "physical_action_target": torch.zeros(batch, frames, dtype=torch.long),
        "context_role_target": torch.zeros(batch, frames, dtype=torch.long),
        "reception_quality_target": torch.zeros(batch, frames, dtype=torch.long),
    }
    total, losses = hierarchical_loss(outputs, targets)
    assert torch.isfinite(total)
    assert set(losses) == {
        "phase",
        "contact",
        "team",
        "physical_action",
        "context_role",
        "reception_quality",
    }
    total.backward()

