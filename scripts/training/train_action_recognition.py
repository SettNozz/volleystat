"""
Fusion Dual-Stream LSTM for volleyball action recognition.

Inputs per clip:
  - Trajectory features: (N_frames x 10) resampled to num_frames
    [ball_x, ball_y, ball_speed, ball_dir, ball_net_dist,
     player_dist, team_side, frame_norm, ball_vy, ball_above_net]
  - Video features: (16 x 576) from MobileNetV3-small CNN

Architecture:
  traj_stream: Linear(10->32) -> LSTM -> last hidden
  cnn_stream:  Linear(576->96) -> LSTM -> last hidden
  classifier:  Linear(32+96 -> num_classes)

Dataset: data/event_clips_merged_v5
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = "data/event_clips_merged_v5"
DEFAULT_CHECKPOINT_DIR = "models/action_recognition"
DEFAULT_CHECKPOINT_NAME = "best_event_sequence.pt"

CLASSES = ["serve", "reception", "pass", "attack", "block", "other"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

TRAJ_DIM = 10
CNN_DIM = 576
TRAJ_PROJ = 32
CNN_PROJ = 96
LSTM_HIDDEN = 128
NUM_FRAMES = 16
NUM_CLASSES = len(CLASSES)


# ── Model ─────────────────────────────────────────────────────────────────────

class FusionDualStreamLSTM(nn.Module):
    def __init__(self, traj_dim=TRAJ_DIM, cnn_dim=CNN_DIM,
                 traj_proj=TRAJ_PROJ, cnn_proj=CNN_PROJ,
                 lstm_hidden=LSTM_HIDDEN, num_classes=NUM_CLASSES,
                 dropout=0.3):
        super().__init__()

        # Trajectory stream
        self.traj_proj = nn.Sequential(
            nn.Linear(traj_dim, traj_proj),
            nn.LayerNorm(traj_proj),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.traj_lstm = nn.LSTM(traj_proj, lstm_hidden, batch_first=True,
                                 num_layers=2, dropout=dropout)

        # CNN feature stream
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, cnn_proj),
            nn.LayerNorm(cnn_proj),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cnn_lstm = nn.LSTM(cnn_proj, lstm_hidden, batch_first=True,
                                num_layers=2, dropout=dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, traj, cnn):
        # traj: (B, T, 10)  cnn: (B, 16, 576)
        t = self.traj_proj(traj)
        _, (ht, _) = self.traj_lstm(t)
        traj_feat = ht[-1]  # (B, lstm_hidden)

        c = self.cnn_proj(cnn)
        _, (hc, _) = self.cnn_lstm(c)
        cnn_feat = hc[-1]  # (B, lstm_hidden)

        merged = torch.cat([traj_feat, cnn_feat], dim=1)
        return self.classifier(merged)


# ── Dataset ───────────────────────────────────────────────────────────────────

def resample_sequence(arr: np.ndarray, target: int) -> np.ndarray:
    """Resample (N, D) array to (target, D) via linear interpolation."""
    n = len(arr)
    if n == target:
        return arr
    idx = np.linspace(0, n - 1, target)
    left = np.floor(idx).astype(int)
    right = np.minimum(left + 1, n - 1)
    alpha = idx - left
    return (arr[left] * (1 - alpha[:, None]) + arr[right] * alpha[:, None]).astype(np.float32)


class EventClipDataset(Dataset):
    def __init__(self, clips, data_dir, num_frames=NUM_FRAMES, augment=False):
        self.clips = clips
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.augment = augment

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        clip_id = clip["clip_id"]
        label_str = clip.get("primary_event_type", "other")
        label = CLASS2IDX.get(label_str, CLASS2IDX["other"])

        # Load trajectory features
        traj_path = self.data_dir / "features" / f"{clip_id}.parquet"
        traj_df = pd.read_parquet(traj_path)
        traj = traj_df.values.astype(np.float32)
        traj = np.nan_to_num(traj, 0.0)
        traj = resample_sequence(traj, self.num_frames)

        # Load video features
        vid_path = self.data_dir / "video_features" / f"{clip_id}.parquet"
        vid_df = pd.read_parquet(vid_path)
        cnn = vid_df.values.astype(np.float32)
        cnn = np.nan_to_num(cnn, 0.0)
        if len(cnn) != self.num_frames:
            cnn = resample_sequence(cnn, self.num_frames)

        if self.augment:
            # Gaussian noise on trajectory
            traj += np.random.normal(0, 0.02, traj.shape).astype(np.float32)
            # Random temporal flip
            if random.random() < 0.3:
                traj = traj[::-1].copy()
                cnn = cnn[::-1].copy()

        return (
            torch.from_numpy(traj),
            torch.from_numpy(cnn),
            torch.tensor(label, dtype=torch.long),
        )


# ── Training utilities ────────────────────────────────────────────────────────

def make_weighted_sampler(clips):
    labels = [CLASS2IDX.get(c.get("primary_event_type", "other"), 5) for c in clips]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights))


def compute_metrics(preds, targets, classes=CLASSES):
    from sklearn.metrics import f1_score, precision_score, recall_score
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)
    per_class_f1 = f1_score(targets, preds, average=None,
                             labels=list(range(len(classes))), zero_division=0)
    metrics = {"macro_f1": macro_f1, "micro_f1": micro_f1}
    for i, cls in enumerate(classes):
        metrics[f"f1_{cls}"] = per_class_f1[i]
    return metrics


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for traj, cnn, labels in loader:
            traj, cnn, labels = traj.to(device), cnn.to(device), labels.to(device)
            logits = model(traj, cnn)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_targets)
    return avg_loss, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--checkpoint_name", default=DEFAULT_CHECKPOINT_NAME)
    p.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--early_stop_patience", type=int, default=25)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--no_mlflow", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifest
    manifest_path = Path(args.data_dir) / "manifest.json"
    manifest = json.load(open(manifest_path))
    clips = manifest["clips"]

    # Filter clips that have both feature files
    feat_dir = Path(args.data_dir) / "features"
    vid_dir = Path(args.data_dir) / "video_features"
    clips = [
        c for c in clips
        if (feat_dir / f"{c['clip_id']}.parquet").exists()
        and (vid_dir / f"{c['clip_id']}.parquet").exists()
    ]
    print(f"Valid clips: {len(clips)}")
    labels_all = [c.get("primary_event_type", "other") for c in clips]
    print(f"Distribution: {Counter(labels_all)}")

    # Stratified split
    indices = list(range(len(clips)))
    label_ids = [CLASS2IDX.get(l, 5) for l in labels_all]

    train_idx, tmp_idx = train_test_split(
        indices, test_size=args.val_size + args.test_size,
        stratify=label_ids, random_state=args.seed
    )
    tmp_labels = [label_ids[i] for i in tmp_idx]
    val_ratio = args.val_size / (args.val_size + args.test_size)
    # Disable stratify if any class has <2 members in tmp split
    tmp_counts = Counter(tmp_labels)
    tmp_stratify = tmp_labels if min(tmp_counts.values()) >= 2 else None
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=1 - val_ratio,
        stratify=tmp_stratify, random_state=args.seed
    )

    train_clips = [clips[i] for i in train_idx]
    val_clips = [clips[i] for i in val_idx]
    test_clips = [clips[i] for i in test_idx]
    print(f"Split: train={len(train_clips)} val={len(val_clips)} test={len(test_clips)}")

    # Datasets and loaders
    train_ds = EventClipDataset(train_clips, args.data_dir, args.num_frames, augment=True)
    val_ds = EventClipDataset(val_clips, args.data_dir, args.num_frames)
    test_ds = EventClipDataset(test_clips, args.data_dir, args.num_frames)

    sampler = make_weighted_sampler(train_clips)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Model
    model = FusionDualStreamLSTM(
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )
    criterion = nn.CrossEntropyLoss()

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / args.checkpoint_name

    # MLflow
    run_name = f"fusion_dual_lstm_v{args.lstm_hidden}h"
    use_mlflow = not args.no_mlflow and MLFLOW_AVAILABLE
    if use_mlflow:
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(vars(args))

    best_macro_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss = 0.0
        for traj, cnn, labels in train_loader:
            traj, cnn, labels = traj.to(device), cnn.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(traj, cnn)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_loader.dataset)

        val_loss, val_metrics = evaluate(model, val_loader, device, criterion)
        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)

        if use_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} macro_f1={macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [BEST] Saved best model (macro_f1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}")
                break

    # Test evaluation
    print("\n=== Test Evaluation ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_metrics = evaluate(model, test_loader, device, criterion)
    print(f"macro_f1: {test_metrics['macro_f1']:.4f}  micro_f1: {test_metrics['micro_f1']:.4f}")
    for cls in CLASSES:
        print(f"  {cls:12s}: f1={test_metrics[f'f1_{cls}']:.4f}")

    if use_mlflow:
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        # Save class mapping
        mapping_path = ckpt_dir / "event_class_mapping.json"
        json.dump({str(i): c for i, c in enumerate(CLASSES)},
                  open(mapping_path, "w"), indent=2)
        mlflow.log_artifact(str(ckpt_path))
        mlflow.log_artifact(str(mapping_path))
        mlflow.end_run()

    print(f"\nCheckpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
