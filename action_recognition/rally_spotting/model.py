"""Compact temporal convolutional model for per-frame lifecycle logits."""

from __future__ import annotations

import torch
from torch import nn


class DilatedResidualBlock(nn.Module):
    """Two same-length dilated convolutions with a residual connection."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels, channels, kernel_size=3,
                padding=padding, dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels, channels, kernel_size=3,
                padding=padding, dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class RallySpottingTCN(nn.Module):
    """Small TCN returning logits shaped ``(batch, frames, classes)``."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout = float(dropout)
        self.input_projection = nn.Conv1d(self.input_dim, self.hidden_dim, 1)
        self.blocks = nn.Sequential(*[
            DilatedResidualBlock(self.hidden_dim, dilation, self.dropout)
            for dilation in self.dilations
        ])
        self.head = nn.Conv1d(self.hidden_dim, self.num_classes, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected (batch, frames, {self.input_dim}), "
                f"got {tuple(values.shape)}"
            )
        hidden = self.input_projection(values.transpose(1, 2))
        return self.head(self.blocks(hidden)).transpose(1, 2)

    def config(self) -> dict:
        """Serializable architecture config stored in checkpoints."""
        return {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "dilations": list(self.dilations),
            "dropout": self.dropout,
        }
