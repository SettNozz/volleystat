"""
Train the core single-label event classifier (4 classes: serve, reception,
pass, attack) on trajectory + pose features. Pure ``other`` / ``free_ball``
contacts remain in source annotations but are excluded from this classifier.

Works with legacy datasets (10-col Parquet → pose channels zero-filled) and
new pose-aware ones (18-col). Legacy multi-label annotations are collapsed to
one class per clip (see config.collapse_labels). Validation split = whole
source videos (default stem "3") to avoid clip leakage.

Usage:
    python -m action_recognition.training.train_event_sequence \
        --manifest data/event_clips_merged_v5/manifest.json \
        --features-dir data/event_clips_merged_v5/features \
        --model lstm \
        --checkpoint-dir action_recognition/models/checkpoints_lstm_pose_v1 \
        --epochs 50

    # Dual-stream fusion (trajectory+pose + MobileNet clip features):
    ... --model dual --video-features-dir data/event_clips_merged_v5/video_features
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.data.event_clip_dataset import (
    EventClipDataset, FusionClipDataset, load_manifest, split_by_stem,
)
from action_recognition.event_clips.config import EVENT_TYPES, NUM_CLASSES, NUM_FEATURES
from action_recognition.event_clips.feature_schema import (
    FEATURE_SCHEMAS,
    LEGACY_PIXEL_SCHEMA,
    validate_feature_schema,
)
from action_recognition.models.sequence_event_model import build_model, save_checkpoint


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """F1 for every class + macro over classes present in y_true."""
    f1s, present = {}, []
    for k, name in enumerate(EVENT_TYPES):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        f1s[name] = f1
        if (y_true == k).any():
            present.append(f1)
    return f1s, float(np.mean(present)) if present else 0.0


def run_epoch(model, loader, criterion, device, optimizer=None, dual=False):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, preds, trues = 0.0, [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            if dual:
                x, v, y = (b.to(device) for b in batch)
                logits = model(x, v)
            else:
                x, y = (b.to(device) for b in batch)
                logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * y.shape[0]
            preds.append(logits.argmax(dim=1).cpu().numpy())
            trues.append(y.cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return total / len(loader.dataset), preds, trues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--video-features-dir", default=None,
                    help="Enables the fusion dataset (required for --model dual)")
    ap.add_argument(
        "--model", default="lstm",
        choices=["lstm", "transformer", "multiscale", "dual"],
    )
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-stems", nargs="+", default=["3"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05,
                    help="Cross-entropy label smoothing (default: 0.05)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible experiment comparisons")
    ap.add_argument("--sequence-len", type=int, default=60,
                    help="Causal feature frames per example (60=2 s at 30 FPS)")
    ap.add_argument("--short-sequence-len", type=int, default=60,
                    help="Recent-frame branch length for --model multiscale")
    ap.add_argument(
        "--final-fit",
        action="store_true",
        help="Train on every core-action clip for exactly --epochs and save "
             "final_event_sequence.pt (use only after cross-video model selection)",
    )
    ap.add_argument(
        "--feature-schema",
        choices=FEATURE_SCHEMAS,
        default=None,
        help="Override feature semantics. By default read manifest/feature sidecar; "
             "old datasets fall back to legacy_pixels_v1.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dual = args.model == "dual"
    if dual and not args.video_features_dir:
        raise SystemExit("--model dual requires --video-features-dir")
    if args.model == "multiscale" and args.sequence_len <= args.short_sequence_len:
        raise SystemExit(
            "--model multiscale requires --sequence-len greater than "
            "--short-sequence-len"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest_payload = json.load(f)
    sidecar_path = Path(args.features_dir) / "feature_schema.json"
    sidecar_schema = None
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            sidecar_schema = json.load(f).get("feature_schema")
    feature_schema = validate_feature_schema(
        args.feature_schema
        or manifest_payload.get("feature_schema")
        or sidecar_schema
        or LEGACY_PIXEL_SCHEMA
    )
    clip_schemas = {
        c.get("feature_schema")
        for c in manifest_payload.get("clips", [])
        if c.get("feature_schema")
    }
    if clip_schemas and clip_schemas != {feature_schema}:
        raise SystemExit(
            f"Mixed/incompatible clip feature schemas: {sorted(clip_schemas)}; "
            f"training expects {feature_schema}"
        )

    clips = load_manifest(args.manifest)
    if args.final_fit:
        train_clips, val_clips = clips, []
    else:
        train_clips, val_clips = split_by_stem(clips, tuple(args.val_stems))
    print(f"Clips: {len(clips)} labeled | train={len(train_clips)} val={len(val_clips)} "
          f"(val stems: {args.val_stems})")
    print(f"Feature schema: {feature_schema}")

    def make_ds(cl, noise):
        if dual:
            return FusionClipDataset(cl, args.features_dir,
                                     args.video_features_dir, augment_noise=noise,
                                     sequence_len=args.sequence_len)
        return EventClipDataset(
            cl, args.features_dir, augment_noise=noise,
            sequence_len=args.sequence_len,
        )

    train_ds = make_ds(train_clips, args.noise)
    val_ds = None if args.final_fit else make_ds(val_clips, 0.0)
    print(
        "Core-action samples: "
        f"train={len(train_ds)} val={len(val_ds) if val_ds is not None else 0} | "
        f"excluded non-core: train={train_ds.excluded_non_core} "
        f"val={val_ds.excluded_non_core if val_ds is not None else 0}"
    )
    if not train_ds or (not args.final_fit and not val_ds):
        raise SystemExit("Training and validation both need core-action samples")

    # Per-channel z-score from TRAIN stats (saved into the checkpoint so
    # inference applies the same normalization)
    feat_mean, feat_std = train_ds.feature_stats()
    feature_active_mask = train_ds.feature_active_mask()
    train_ds.set_normalization(feat_mean, feat_std)
    if val_ds is not None:
        val_ds.set_normalization(feat_mean, feat_std)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size) if val_ds is not None else None

    # Class weights against imbalance among the four core actions.
    y_train = train_ds.class_indices()
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
    print("class counts:", dict(zip(EVENT_TYPES, counts.astype(int))))
    weights = len(y_train) / np.maximum(counts * NUM_CLASSES, 1.0)
    class_weights = torch.tensor(weights, dtype=torch.float32).clamp(max=10.0).to(device)
    print("class weights:", dict(zip(EVENT_TYPES, [round(float(w), 2) for w in class_weights])))

    model = build_model(
        args.model,
        NUM_FEATURES,
        short_sequence_len=args.short_sequence_len,
        sequence_len=args.sequence_len,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    try:
        import mlflow
        mlflow.set_experiment("volleyball_event_sequence")
        mlflow_run = mlflow.start_run(run_name=f"{args.model}_4cls_{ckpt_dir.name}")
        mlflow.log_params({**vars(args), "num_features": NUM_FEATURES,
                           "num_classes": NUM_CLASSES})
    except Exception:
        mlflow_run = None

    checkpoint_metadata = {
        "taxonomy": "v4_core_4class",
        "feature_schema": feature_schema,
        "seed": args.seed,
        "sequence_len": args.sequence_len,
        "short_sequence_len": args.short_sequence_len,
        "temporal_architecture": (
            "long_serve_reception_short_pass_attack"
            if args.model == "multiscale" else "single_scale"
        ),
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "feature_active_mask": feature_active_mask.tolist(),
    }
    if args.final_fit:
        for epoch in range(args.epochs):
            tr_loss, _, _ = run_epoch(
                model, train_dl, criterion, device, optimizer, dual
            )
            scheduler.step()
            print(f"[{epoch:3d}] train_loss={tr_loss:.4f}")
            if mlflow_run:
                import mlflow
                mlflow.log_metric("train_loss", tr_loss, step=epoch)
        final_path = ckpt_dir / "final_event_sequence.pt"
        save_checkpoint(
            final_path, model, args.model, NUM_FEATURES,
            extra={
                **checkpoint_metadata,
                "epoch": args.epochs - 1,
                "final_fit": True,
                "training_examples": len(train_ds),
                "excluded_non_core": train_ds.excluded_non_core,
            },
        )
        print(f"Final fit on {len(train_ds)} clips -> {final_path}")
        if mlflow_run:
            import mlflow
            mlflow.end_run()
        return

    best_f1, best_epoch = -1.0, -1
    for epoch in range(args.epochs):
        tr_loss, _, _ = run_epoch(model, train_dl, criterion, device, optimizer, dual)
        va_loss, va_pred, va_true = run_epoch(model, val_dl, criterion, device, None, dual)
        scheduler.step()
        f1s, macro = per_class_f1(va_true, va_pred)
        acc = float((va_pred == va_true).mean())
        print(f"[{epoch:3d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"acc={acc:.3f} macro_f1={macro:.4f} | " +
              " ".join(f"{n}={f1s[n]:.2f}" for n in EVENT_TYPES))
        if mlflow_run:
            import mlflow
            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": va_loss,
                                "val_acc": acc, "val_macro_f1": macro}, step=epoch)
        if macro > best_f1:
            best_f1, best_epoch = macro, epoch
            save_checkpoint(ckpt_dir / "best_event_sequence.pt", model,
                            args.model, NUM_FEATURES,
                            extra={"val_macro_f1": macro, "val_acc": acc,
                                   "epoch": epoch, "per_class_f1": f1s,
                                   **checkpoint_metadata})
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping (no improvement for {args.patience} epochs)")
            break

    print(f"Best val macro_f1={best_f1:.4f} @ epoch {best_epoch} -> "
          f"{ckpt_dir / 'best_event_sequence.pt'}")
    if mlflow_run:
        import mlflow
        mlflow.log_metric("best_val_macro_f1", best_f1)
        mlflow.end_run()


if __name__ == "__main__":
    main()
