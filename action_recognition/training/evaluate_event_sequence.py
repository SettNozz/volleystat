"""Evaluate a sequence checkpoint on complete held-out source videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from action_recognition.data.event_clip_dataset import EventClipDataset, load_manifest
from action_recognition.event_clips.config import EVENT_TYPES
from action_recognition.models.sequence_event_model import load_checkpoint
from action_recognition.training.train_event_sequence import per_class_f1, run_epoch


def evaluate(
    manifest_path: Path,
    features_dir: Path,
    checkpoint_path: Path,
    source_stems: set[str],
    batch_size: int = 32,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("model_type") == "dual":
        raise ValueError("dual checkpoints require a video-feature evaluator")
    checkpoint_types = list(checkpoint["event_types"])
    if checkpoint_types != EVENT_TYPES:
        raise ValueError(
            f"expected current taxonomy {EVENT_TYPES}, got {checkpoint_types}"
        )
    clips = [
        clip for clip in load_manifest(str(manifest_path))
        if str(clip.get("source_stem")) in source_stems
    ]
    dataset = EventClipDataset(
        clips,
        str(features_dir),
        sequence_len=int(checkpoint.get("sequence_len", 60)),
    )
    if not dataset:
        raise ValueError(f"no core-action examples for sources {sorted(source_stems)}")
    if checkpoint.get("feature_mean") is not None:
        dataset.set_normalization(
            checkpoint["feature_mean"], checkpoint["feature_std"]
        )
    loader = DataLoader(dataset, batch_size=batch_size)
    model.to(device)
    loss, predicted, truth = run_epoch(
        model, loader, nn.CrossEntropyLoss(), device, optimizer=None
    )
    per_class, macro = per_class_f1(truth, predicted)
    confusion = np.zeros((len(EVENT_TYPES), len(EVENT_TYPES)), dtype=int)
    for actual, guess in zip(truth, predicted):
        confusion[int(actual), int(guess)] += 1
    counts = np.bincount(truth, minlength=len(EVENT_TYPES))
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "source_stems": sorted(source_stems),
        "sequence_len": int(checkpoint.get("sequence_len", 60)),
        "examples": len(dataset),
        "excluded_non_core": dataset.excluded_non_core,
        "loss": float(loss),
        "accuracy": float((predicted == truth).mean()),
        "macro_f1": macro,
        "per_class_f1": per_class,
        "class_counts": dict(zip(EVENT_TYPES, counts.astype(int).tolist())),
        "confusion_matrix": {
            "labels": EVENT_TYPES,
            "rows_actual_columns_predicted": confusion.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-stems", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.manifest, args.features_dir, args.checkpoint,
        set(args.source_stems), args.batch_size,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
