"""
Sequence models for multi-label event classification.

SequenceEventLSTM        — LayerNorm + BiLSTM over (T, F) features
SequenceEventTransformer — input projection + TransformerEncoder + attn pool
DualStreamProjectedLSTM  — separate LN+projection per modality
                           (trajectory+pose 18→48, CNN 576→96) → concat →
                           BiLSTM (the architecture that won ablations v4/v5)

Checkpoints are saved as dicts:
    {"model_type", "input_dim", "state_dict", "event_types", ...}
so inference can rebuild the right class without guessing.
"""

import torch
import torch.nn as nn

from action_recognition.event_clips.config import (
    EVENT_TYPES, LEGACY_EVENT_TYPES, NUM_CLASSES, NUM_FEATURES, VIDEO_FEATURE_DIM,
)


class SequenceEventLSTM(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, hidden: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x):                       # x: (B, T, F)
        out, _ = self.lstm(self.norm(x))
        return self.head(out[:, -1])            # last step (contact at end)


class SequenceEventTransformer(nn.Module):
    def __init__(self, input_dim: int = NUM_FEATURES, d_model: int = 96,
                 nhead: int = 4, num_layers: int = 3, dropout: float = 0.2,
                 num_classes: int = NUM_CLASSES, max_len: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):                       # x: (B, T, F)
        h = self.proj(self.norm(x)) + self.pos[:, :x.shape[1]]
        h = self.encoder(h)
        w = torch.softmax(self.attn_pool(h), dim=1)   # (B, T, 1)
        return self.head((h * w).sum(dim=1))


class _TemporalTransformerBranch(nn.Module):
    """Transformer encoder returning one pooled temporal representation."""

    def __init__(self, input_dim: int, d_model: int = 96, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.2,
                 max_len: int = 256, contact_center_pool: bool = False):
        super().__init__()
        self.contact_center_pool = bool(contact_center_pool)
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attn_pool = nn.Linear(d_model, 1)


    def forward(self, x):
        if x.shape[1] > self.pos.shape[1]:
            raise ValueError(
                f"sequence has {x.shape[1]} frames; branch supports "
                f"at most {self.pos.shape[1]}"
            )
        h = self.proj(self.norm(x)) + self.pos[:, :x.shape[1]]
        h = self.encoder(h)
        weights = torch.softmax(self.attn_pool(h), dim=1)
        if self.contact_center_pool:
            return h[:, x.shape[1] // 2]
        return (h * weights).sum(dim=1)


class MultiScaleEventTransformer(nn.Module):
    """Long serve/reception branch plus recent pass/attack branch."""

    def __init__(self, input_dim: int = NUM_FEATURES,
                 short_sequence_len: int = 60, d_model: int = 96,
                 dropout: float = 0.2, num_classes: int = NUM_CLASSES,
                 max_len: int = 256, short_window_centered: bool = False,
                 contact_center_pool: bool = False):
        super().__init__()
        if num_classes != 4 or EVENT_TYPES != ["serve", "reception", "pass", "attack"]:
            raise ValueError("multi-scale model requires the four-class core taxonomy")
        if short_sequence_len <= 0:
            raise ValueError("short_sequence_len must be positive")
        self.short_sequence_len = int(short_sequence_len)
        self.short_window_centered = bool(short_window_centered)
        self.contact_center_pool = bool(contact_center_pool)
        if self.contact_center_pool and not self.short_window_centered:
            raise ValueError('Contact-centre pooling requires centred offline windows')
        self.long_branch = _TemporalTransformerBranch(
            input_dim, d_model=d_model, dropout=dropout, max_len=max_len,
            contact_center_pool=self.contact_center_pool,
        )
        self.short_branch = _TemporalTransformerBranch(
            input_dim, d_model=d_model, dropout=dropout,
            max_len=self.short_sequence_len, contact_center_pool=self.contact_center_pool,
        )
        self.long_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))
        self.short_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, x):
        if x.shape[1] < self.short_sequence_len:
            raise ValueError(
                f"multi-scale input needs at least {self.short_sequence_len} frames"
            )
        long_logits = self.long_head(self.long_branch(x))
        if self.short_window_centered:
            start = (x.shape[1] - self.short_sequence_len) // 2
            short_context = x[:, start:start + self.short_sequence_len]
        else:
            short_context = x[:, -self.short_sequence_len:]
        short_logits = self.short_head(self.short_branch(short_context))
        return torch.cat((long_logits, short_logits), dim=1)


class DualStreamProjectedLSTM(nn.Module):
    """Trajectory+pose stream and CNN stream, each normalized+projected."""

    def __init__(self, traj_dim: int = NUM_FEATURES, traj_proj: int = 48,
                 video_dim: int = VIDEO_FEATURE_DIM, video_proj: int = 96,
                 hidden: int = 128, dropout: float = 0.3,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.traj_norm = nn.LayerNorm(traj_dim)
        self.traj_proj = nn.Linear(traj_dim, traj_proj)
        self.video_norm = nn.LayerNorm(video_dim)
        self.video_proj = nn.Linear(video_dim, video_proj)
        fused = traj_proj + video_proj
        self.fuse = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused, 128), nn.GELU())
        self.lstm = nn.LSTM(128, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, num_classes))

    def forward(self, x, v):                    # x: (B, T, F)  v: (B, Tv, D)
        t = self.traj_proj(self.traj_norm(x))                       # (B, T, tp)
        vv = self.video_proj(self.video_norm(v))                    # (B, Tv, vp)
        # Upsample video stream along time to match trajectory length
        vv = torch.nn.functional.interpolate(
            vv.transpose(1, 2), size=t.shape[1], mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        h = self.fuse(torch.cat([t, vv], dim=-1))
        out, _ = self.lstm(h)
        return self.head(out[:, -1])


def build_model(model_type: str, input_dim: int = NUM_FEATURES,
                num_classes: int = NUM_CLASSES,
                short_sequence_len: int = 60,
                sequence_len: int = 60,
                short_window_centered: bool = False,
                contact_center_pool: bool = False) -> nn.Module:
    if model_type == "lstm":
        return SequenceEventLSTM(input_dim=input_dim, num_classes=num_classes)
    if model_type == "transformer":
        return SequenceEventTransformer(input_dim=input_dim, num_classes=num_classes)
    if model_type == "multiscale":
        return MultiScaleEventTransformer(
            input_dim=input_dim,
            short_sequence_len=short_sequence_len,
            num_classes=num_classes,
            max_len=max(256, sequence_len),
            short_window_centered=short_window_centered,
            contact_center_pool=contact_center_pool,
        )
    if model_type == "dual":
        return DualStreamProjectedLSTM(traj_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model type: {model_type}")


def save_checkpoint(path, model, model_type: str, input_dim: int, extra=None,
                    event_types=None):
    checkpoint_event_types = list(event_types or EVENT_TYPES)
    payload = {
        "model_type": model_type,
        "input_dim": input_dim,
        "event_types": checkpoint_event_types,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    checkpoint_event_types = list(ckpt.get("event_types") or LEGACY_EVENT_TYPES)
    model = build_model(
        ckpt["model_type"],
        ckpt.get("input_dim", NUM_FEATURES),
        num_classes=len(checkpoint_event_types),
        short_sequence_len=int(ckpt.get("short_sequence_len", 60)),
        sequence_len=int(ckpt.get("sequence_len", 60)),
        short_window_centered=bool(ckpt.get("short_window_centered", False)),
        contact_center_pool=bool(ckpt.get("contact_center_pool", False)),
    )
    ckpt["event_types"] = checkpoint_event_types
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
