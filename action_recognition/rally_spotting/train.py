"""Train the frame-level rally lifecycle TCN.

The split unit is always a complete ``source_video``. No windows from the same
video can occur in different splits. By default only human labels are accepted;
using rule-derived labels requires the explicit ``--allow-weak-labels`` flag.

CLI::

    python -m action_recognition.rally_spotting.train \
      --data-dir data/rally_4k_v1 \
      --checkpoint-dir action_recognition/models/rally_spotting_v1 \
      --epochs 40 --window 256 --stride 128

For an intentional weak-label bootstrap run, append ``--allow-weak-labels``.
Reported validation metrics are precision/recall/F1 at ±5, ±10 and ±15 source
frames after per-class NMS.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from action_recognition.rally_spotting.metrics import spotting_metrics
from action_recognition.rally_spotting.model import RallySpottingTCN
from action_recognition.rally_spotting.taxonomy import (
    EVENT_TYPES,
    WEAK_LABEL_SOURCE,
)


def split_records_by_video(
    records: Sequence[dict[str, Any]],
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split complete source-video groups."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >=0 and sum to <1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("source_key") or record["source_video"])
        groups.setdefault(key, []).append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    count = len(keys)
    test_count = int(round(count * test_fraction))
    val_count = int(round(count * val_fraction))
    if test_fraction > 0 and count >= 3:
        test_count = max(1, test_count)
    if val_fraction > 0 and count - test_count >= 2:
        val_count = max(1, val_count)
    if test_count + val_count >= count:
        overflow = test_count + val_count - count + 1
        val_count = max(0, val_count - overflow)
    test_keys = set(keys[:test_count])
    val_keys = set(keys[test_count:test_count + val_count])
    return {
        "train": [
            record for key in keys if key not in test_keys | val_keys
            for record in groups[key]
        ],
        "val": [
            record for key in keys if key in val_keys
            for record in groups[key]
        ],
        "test": [
            record for key in keys if key in test_keys
            for record in groups[key]
        ],
    }


def validate_label_policy(
    records: Sequence[dict[str, Any]],
    allow_weak_labels: bool = False,
) -> None:
    """Reject accidental training on pseudo-labels."""
    weak = [
        record.get("id", record.get("source_video", "?"))
        for record in records
        if record.get("label_source") == WEAK_LABEL_SOURCE
    ]
    if weak and not allow_weak_labels:
        preview = ", ".join(map(str, weak[:3]))
        raise ValueError(
            f"dataset contains {len(weak)} weak_fsm video(s) ({preview}); "
            "training on weak labels is disabled. Curate human labels or pass "
            "--allow-weak-labels intentionally."
        )


def gaussian_targets(targets: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Turn hard timestamp labels into bounded Gaussian supervision."""
    if sigma <= 0:
        return targets.astype(np.float32, copy=True)
    output = np.zeros_like(targets, dtype=np.float32)
    radius = max(1, int(math.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2).astype(np.float32)
    for class_index in range(targets.shape[1]):
        for frame in np.flatnonzero(targets[:, class_index] > 0.5):
            left = max(0, frame - radius)
            right = min(len(targets), frame + radius + 1)
            kernel_left = left - (frame - radius)
            kernel_right = kernel_left + right - left
            output[left:right, class_index] = np.maximum(
                output[left:right, class_index],
                kernel[kernel_left:kernel_right],
            )
    return output


class WindowDataset(Dataset):
    """Windows created only from already assigned video records."""

    def __init__(
        self,
        data_dir: Path,
        records: Sequence[dict[str, Any]],
        window: int,
        stride: int,
        label_sigma: float = 2.0,
    ) -> None:
        if window <= 0 or stride <= 0:
            raise ValueError("window and stride must be positive")
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for record in records:
            with np.load(data_dir / record["npz"]) as arrays:
                features = arrays["features"].astype(np.float32)
                targets = gaussian_targets(arrays["targets"], label_sigma)
            starts = list(range(0, max(1, len(features) - window + 1), stride))
            last_start = max(0, len(features) - window)
            if not starts or starts[-1] != last_start:
                starts.append(last_start)
            for start in starts:
                end = min(len(features), start + window)
                x = np.zeros((window, features.shape[1]), dtype=np.float32)
                y = np.zeros((window, targets.shape[1]), dtype=np.float32)
                x[:end - start] = features[start:end]
                y[:end - start] = targets[start:end]
                self.items.append((x, y))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, targets = self.items[index]
        return torch.from_numpy(features), torch.from_numpy(targets)


@torch.no_grad()
def predict_full_video(
    model: nn.Module,
    features: np.ndarray,
    window: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    """Overlap-average logits for a complete source timeline."""
    model.eval()
    total = len(features)
    sums = np.zeros((total, len(EVENT_TYPES)), dtype=np.float32)
    counts = np.zeros((total, 1), dtype=np.float32)
    starts = list(range(0, max(1, total - window + 1), stride))
    last_start = max(0, total - window)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        end = min(total, start + window)
        padded = np.zeros((window, features.shape[1]), dtype=np.float32)
        padded[:end - start] = features[start:end]
        tensor = torch.from_numpy(padded[None]).to(device)
        probabilities = torch.sigmoid(model(tensor))[0, :end - start].cpu().numpy()
        sums[start:end] += probabilities
        counts[start:end] += 1.0
    return sums / np.maximum(counts, 1.0)


def evaluate_records(
    model: nn.Module,
    data_dir: Path,
    records: Sequence[dict[str, Any]],
    window: int,
    stride: int,
    device: torch.device,
) -> dict[str, dict]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in records:
        with np.load(data_dir / record["npz"]) as arrays:
            features = arrays["features"].astype(np.float32)
            hard_targets = arrays["targets"].astype(np.float32)
        predictions.append(
            predict_full_video(model, features, window, stride, device)
        )
        targets.append(hard_targets)
    return spotting_metrics(predictions, targets, EVENT_TYPES)


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train and save the checkpoint selected by F1@10 on the declared split."""
    data_dir = args.data_dir.resolve()
    with (data_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if tuple(manifest.get("event_types", ())) != EVENT_TYPES:
        raise ValueError("dataset event taxonomy does not match this trainer")
    records = manifest.get("records") or []
    validate_label_policy(records, args.allow_weak_labels)
    explicit_val_keys = set(args.val_source_keys or [])
    if explicit_val_keys:
        splits = {
            "train": [r for r in records if str(r.get("source_key")) not in explicit_val_keys],
            "val": [r for r in records if str(r.get("source_key")) in explicit_val_keys],
            "test": [],
        }
        missing = explicit_val_keys - {str(r.get("source_key")) for r in records}
        if missing:
            raise ValueError(f"unknown --val-source-keys: {sorted(missing)}")
    else:
        splits = split_records_by_video(
            records, args.val_fraction, args.test_fraction, args.seed,
        )
    if not splits["train"]:
        raise ValueError("training split is empty")
    if args.val_fraction > 0 and not splits["val"]:
        source_count = len({
            str(record.get("source_key") or record["source_video"])
            for record in records
        })
        raise ValueError(
            "validation split is empty; provide at least two independent "
            f"source videos (got {source_count}) or set --val-fraction 0 "
            "for an explicit training-only bootstrap run"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    train_data = WindowDataset(
        data_dir, splits["train"], args.window, args.stride, args.label_sigma,
    )
    loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    input_dim = len(manifest["feature_names"])
    model = RallySpottingTCN(
        input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
    ).to(device)

    event_totals = Counter()
    total_frames = 0
    for record in splits["train"]:
        total_frames += int(record["num_frames"])
        event_totals.update(record.get("event_counts") or {})
    positive = np.asarray(
        [event_totals[name] for name in EVENT_TYPES], dtype=np.float32,
    )
    pos_weight = np.clip(
        (total_frames - positive) / np.maximum(positive, 1.0), 1.0, 100.0,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight).to(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        evaluation_records = splits["val"]
        metric_split_name = "validation"
        if not evaluation_records:
            evaluation_records = splits["train"]
            metric_split_name = "training"
        metrics = evaluate_records(
            model, data_dir, evaluation_records,
            args.window, args.stride, device,
        )
        f1 = float(metrics["at_10"]["micro"]["f1"])
        print(
            f"epoch {epoch:03d} loss={np.mean(losses):.5f} "
            f"{metric_split_name}_f1@10={f1:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_metrics = metrics
            patience_left = args.patience
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model.config(),
                "feature_schema": manifest["feature_schema"],
                "feature_names": manifest["feature_names"],
                "event_types": list(EVENT_TYPES),
                "metrics": metrics,
                "split_source_keys": {
                    name: sorted({record["source_key"] for record in split})
                    for name, split in splits.items()
                },
            }, args.checkpoint_dir / "best_rally_spotting.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    summary = {
        "selection_split": (
            "validation" if splits["val"] else "training_bootstrap"
        ),
        "best_selection_f1_at_10": best_f1,
        "best_selection_metrics": best_metrics,
        "label_sources": dict(Counter(
            record["label_source"] for record in records
        )),
        "split_videos": {
            name: sorted({record["source_key"] for record in split})
            for name, split in splits.items()
        },
    }
    with (args.checkpoint_dir / "training_summary.json").open(
        "w", encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-sigma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--val-source-keys", nargs="+", default=None,
        help="Explicit complete source groups to use for validation.",
    )
    parser.add_argument(
        "--allow-weak-labels", action="store_true",
        help="Explicitly allow weak_fsm pseudo-labels in training.",
    )
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
