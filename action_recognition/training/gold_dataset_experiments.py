"""Reproducible experiments for a completed full-video gold annotation.

Provides a temporal rally holdout (useful as a learnability check, not a
cross-camera generalization estimate), a grouped reception-quality pilot, and
checkpoint comparison reports.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CONTROLLED_QUALITIES = {"perfect", "positive"}
POOR_QUALITIES = {"negative", "overpass", "error"}


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def build_rally_holdout_manifest(
    source_manifest: Path, output_path: Path, train_through_rally: int = 24,
) -> dict[str, Any]:
    payload = _read(source_manifest.resolve())
    clips = []
    split_counts: Counter[str] = Counter()
    for source in payload.get("clips", []):
        clip = dict(source)
        if not clip.get("rally_id"):
            # Independent legacy sources remain training data in the hybrid
            # experiment. Only the completed video's rallies are partitioned.
            clips.append(clip)
            split_counts["independent_train"] += 1
            continue
        rally_number = int(str(clip["rally_id"]).rsplit("_", 1)[-1])
        split = "gold_train" if rally_number <= train_through_rally else "gold_val"
        clip["original_source_stem"] = clip.get("source_stem")
        clip["source_stem"] = split
        clip["experiment_split"] = split
        clips.append(clip)
        split_counts[split] += 1
    result = {
        **{key: value for key, value in payload.items() if key != "clips"},
        "experiment": {
            "name": "within_video_temporal_rally_holdout",
            "train_rallies": [1, train_through_rally],
            "validation_rallies": [train_through_rally + 1, max(
                int(str(c["rally_id"]).rsplit("_", 1)[-1])
                for c in clips if c.get("rally_id")
            )],
            "warning": (
                "Same camera and match appear in both partitions. This measures "
                "learnability only, not cross-video generalization."
            ),
        },
        "clips": clips,
    }
    _write(output_path, result)
    return {"path": str(output_path), "counts": dict(split_counts)}


def _load_clip_features(features_dir: Path, clip_id: str) -> np.ndarray:
    npz_path = features_dir / f"{clip_id}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as payload:
            return np.asarray(payload["features"], dtype=np.float32)
    parquet_path = features_dir / f"{clip_id}.parquet"
    if parquet_path.exists():
        import pandas as pd
        return pd.read_parquet(parquet_path).to_numpy(dtype=np.float32)
    raise FileNotFoundError(f"no features for {clip_id}")


def _aggregate(sequence: np.ndarray) -> np.ndarray:
    if len(sequence) == 0:
        return np.zeros(sequence.shape[1] * 5, dtype=np.float32)
    return np.concatenate((
        sequence.mean(axis=0), sequence.std(axis=0), sequence.min(axis=0),
        sequence.max(axis=0), sequence[-1] - sequence[0],
    )).astype(np.float32)


def reception_quality_examples(
    manifest_path: Path, features_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    clips = _read(manifest_path.resolve()).get("clips", [])
    pre_features, full_features, labels, groups, metadata = [], [], [], [], []
    for clip in clips:
        if "reception" not in (clip.get("full_event_types") or []):
            continue
        qualities = [
            q for q in (clip.get("reception_qualities") or [])
            if q not in ("not_applicable", "unknown")
        ]
        if len(qualities) != 1:
            continue
        quality = qualities[0]
        if quality in CONTROLLED_QUALITIES:
            target = 1
        elif quality in POOR_QUALITIES:
            target = 0
        else:
            continue
        sequence = _load_clip_features(features_dir.resolve(), str(clip["clip_id"]))
        contact = int(clip["contact_offset"])
        pre = sequence[:contact + 1]
        post = sequence[contact + 1:]
        pre_features.append(_aggregate(pre))
        full_features.append(np.concatenate((_aggregate(pre), _aggregate(post))))
        labels.append(target)
        groups.append(str(clip["rally_id"]))
        metadata.append({
            "clip_id": clip["clip_id"], "rally_id": clip["rally_id"],
            "source_stem": clip.get("original_source_stem") or clip.get("source_stem"),
            "quality": quality, "target": target,
            "pre_frames": len(pre), "post_frames": len(post),
        })
    return (
        {"pre_contact_only": np.stack(pre_features), "pre_plus_post": np.stack(full_features)},
        np.asarray(labels, dtype=np.int64), np.asarray(groups), metadata,
    )


def run_reception_quality_pilot(
    manifest_path: Path, features_dir: Path, output_dir: Path, seed: int = 42,
) -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_sets, y, groups, metadata = reception_quality_examples(
        manifest_path, features_dir,
    )
    folds = list(StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed,
    ).split(feature_sets["pre_contact_only"], y, groups))
    model_factories = {
        # A fixed poor-quality prediction is the transparent 50% balanced
        # baseline for this exactly balanced 15/15 pilot.
        "dummy": lambda: DummyClassifier(strategy="constant", constant=0),
        "logistic_l2": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=3000, random_state=seed,
            ),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed,
        ),
    }
    results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for feature_name, x in feature_sets.items():
        results[feature_name] = {}
        for model_name, factory in model_factories.items():
            probabilities = np.zeros(len(y), dtype=np.float32)
            predictions = np.zeros(len(y), dtype=np.int64)
            for fold_index, (train_index, test_index) in enumerate(folds):
                model = factory()
                model.fit(x[train_index], y[train_index])
                predictions[test_index] = model.predict(x[test_index])
                if hasattr(model, "predict_proba"):
                    probabilities[test_index] = model.predict_proba(x[test_index])[:, 1]
                else:
                    probabilities[test_index] = predictions[test_index]
                for sample_index in test_index:
                    prediction_rows.append({
                        **metadata[int(sample_index)], "feature_set": feature_name,
                        "model": model_name, "fold": fold_index,
                        "predicted": int(predictions[sample_index]),
                        "controlled_probability": float(probabilities[sample_index]),
                    })
            results[feature_name][model_name] = {
                "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
                "f1_controlled": float(f1_score(y, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y, probabilities)),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "binary_reception_quality",
        "classes": {"poor": sorted(POOR_QUALITIES), "controlled": sorted(CONTROLLED_QUALITIES)},
        "num_examples": int(len(y)), "class_counts": {
            "poor": int((y == 0).sum()), "controlled": int((y == 1).sum()),
        },
        "split": "5-fold stratified group CV by rally_id", "seed": seed,
        "results": results,
        "interpretation_warning": "Thirty receptions are enough for a pilot, not a production estimate.",
    }
    _write(output_dir / "reception_quality_results.json", report)
    _write(output_dir / "reception_quality_predictions.json", prediction_rows)
    lines = [
        "# Reception-quality pilot", "",
        f"Examples: {len(y)} (poor={(y == 0).sum()}, controlled={(y == 1).sum()}).", "",
        "| Features | Model | Balanced accuracy | F1 controlled | ROC AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for feature_name, model_results in results.items():
        for model_name, metrics in model_results.items():
            lines.append(
                f"| {feature_name} | {model_name} | {metrics['balanced_accuracy']:.3f} | "
                f"{metrics['f1_controlled']:.3f} | {metrics['roc_auc']:.3f} |"
            )
    lines.extend(["", "Validation is grouped by rally. Treat this as a pilot because n=30."])
    (output_dir / "RECEPTION_QUALITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def run_reception_quality_cross_video(
    manifest_path: Path, features_dir: Path, output_dir: Path, seed: int = 42,
) -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_sets, y, _, metadata = reception_quality_examples(manifest_path, features_dir)
    sources = np.asarray([row["source_stem"] for row in metadata])
    source_names = sorted(set(sources.tolist()))
    if len(source_names) != 2:
        raise ValueError(f"expected exactly two reception sources, got {source_names}")
    factories = {
        "dummy": lambda: DummyClassifier(strategy="constant", constant=0),
        "logistic_l2": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=3000, random_state=seed,
            ),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed,
        ),
    }
    directions: dict[str, Any] = {}
    for validation_source in source_names:
        train = sources != validation_source
        validation = ~train
        direction = f"{next(name for name in source_names if name != validation_source)}_to_{validation_source}"
        directions[direction] = {
            "train_examples": int(train.sum()), "validation_examples": int(validation.sum()),
            "validation_class_counts": {
                "poor": int((y[validation] == 0).sum()),
                "controlled": int((y[validation] == 1).sum()),
            }, "feature_sets": {},
        }
        for feature_name, x in feature_sets.items():
            directions[direction]["feature_sets"][feature_name] = {}
            for model_name, factory in factories.items():
                model = factory()
                model.fit(x[train], y[train])
                predicted = model.predict(x[validation])
                probability = model.predict_proba(x[validation])[:, 1]
                directions[direction]["feature_sets"][feature_name][model_name] = {
                    "balanced_accuracy": float(balanced_accuracy_score(y[validation], predicted)),
                    "f1_controlled": float(f1_score(y[validation], predicted, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y[validation], probability)),
                }
    result = {
        "task": "binary_reception_quality_cross_video",
        "num_examples": int(len(y)), "sources": source_names,
        "directions": directions,
        "warning": "The second video has only four poor receptions; direction-level metrics remain noisy.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "reception_quality_cross_video.json", result)
    return result


def summarize_checkpoints(checkpoints: Iterable[Path], output_path: Path) -> dict[str, Any]:
    import torch

    rows = []
    for path in checkpoints:
        payload = torch.load(path.resolve(), map_location="cpu", weights_only=False)
        extra = payload.get("extra") or payload
        rows.append({
            "checkpoint": str(path.resolve()),
            "model": payload.get("model_type") or payload.get("model_name") or path.parent.name,
            "val_macro_f1": extra.get("val_macro_f1"), "val_accuracy": extra.get("val_acc"),
            "epoch": extra.get("epoch"), "per_class_f1": extra.get("per_class_f1"),
        })
    rows.sort(key=lambda row: float(row.get("val_macro_f1") or -1), reverse=True)
    result = {"experiments": rows, "best_checkpoint": rows[0]["checkpoint"] if rows else None}
    _write(output_path, result)
    return result


def build_spotting_temporal_holdout(
    dataset_dir: Path, event_manifest: Path, output_dir: Path,
    train_through_rally: int = 24,
) -> dict[str, Any]:
    source = _read(dataset_dir.resolve() / "manifest.json")
    if len(source.get("records", [])) != 1:
        raise ValueError("expected one prepared full-video spotting record")
    clips = _read(event_manifest.resolve()).get("clips", [])
    boundary_rally = f"rally_{train_through_rally + 1:04d}"
    boundary = min(int(c["start_frame"]) for c in clips if c.get("rally_id") == boundary_rally)
    record = source["records"][0]
    with np.load(dataset_dir.resolve() / record["npz"], allow_pickle=False) as arrays:
        features = arrays["features"]
        targets = arrays["targets"]
        frames = arrays["frame_numbers"]
    masks = {"gold_train": frames < boundary, "gold_val": frames >= boundary}
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    records = []
    event_types = source["event_types"]
    for key, mask in masks.items():
        relative = Path("videos") / f"{key}.npz"
        np.savez_compressed(
            output_dir / relative, features=features[mask], targets=targets[mask],
            frame_numbers=frames[mask],
        )
        counts = {name: int(targets[mask, i].sum()) for i, name in enumerate(event_types)}
        records.append({
            **record, "id": key, "source_key": key, "npz": relative.as_posix(),
            "num_frames": int(mask.sum()), "first_frame": int(frames[mask][0]),
            "last_frame": int(frames[mask][-1]), "event_counts": counts,
        })
    result = {**source, "records": records, "temporal_holdout": {
        "boundary_frame": boundary, "train_through_rally": train_through_rally,
        "validation_from_rally": train_through_rally + 1,
    }}
    _write(output_dir / "manifest.json", result)
    return result["temporal_holdout"]


def build_experiment_report(
    sequence_summary_path: Path, reception_path: Path,
    spotting_summary_path: Path, output_dir: Path,
) -> dict[str, Any]:
    sequence = _read(sequence_summary_path.resolve())["experiments"]
    reception = _read(reception_path.resolve())
    spotting = _read(spotting_summary_path.resolve())
    groups: dict[str, list[dict[str, Any]]] = {
        "cross_video": [], "clean_temporal": [], "hybrid_temporal": [],
    }
    for row in sequence:
        path = row["checkpoint"]
        if "hybrid" in path:
            groups["hybrid_temporal"].append(row)
        elif "within_gold" in path:
            groups["clean_temporal"].append(row)
        else:
            groups["cross_video"].append(row)
    group_summary = {}
    for name, rows in groups.items():
        values = np.asarray([row["val_macro_f1"] for row in rows], dtype=float)
        group_summary[name] = {
            "runs": len(rows), "mean_macro_f1": float(values.mean()),
            "std_macro_f1": float(values.std()), "best_macro_f1": float(values.max()),
            "best_checkpoint": max(rows, key=lambda row: row["val_macro_f1"])["checkpoint"],
        }
    result = {
        "sequence_classification": group_summary,
        "best_sequence_run": max(sequence, key=lambda row: row["val_macro_f1"]),
        "reception_quality": reception,
        "rally_spotting": spotting,
    }
    _write(output_dir / "experiment_report.json", result)
    best = result["best_sequence_run"]
    rq = reception["results"]
    spot10 = spotting["best_selection_metrics"]["at_10"]
    lines = [
        "# Gold dataset experiment report", "",
        "## Sequence classification", "",
        "| Evaluation | Runs | Mean macro-F1 | Std | Best |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "cross_video": "Train on two old videos → full gold video",
        "clean_temporal": "Clean gold rallies 1–24 → rallies 25–32",
        "hybrid_temporal": "Old videos + clean rallies 1–24 → rallies 25–32",
    }
    for name in ("cross_video", "clean_temporal", "hybrid_temporal"):
        row = group_summary[name]
        lines.append(
            f"| {labels[name]} | {row['runs']} | {row['mean_macro_f1']:.3f} | "
            f"{row['std_macro_f1']:.3f} | {row['best_macro_f1']:.3f} |"
        )
    lines.extend([
        "", f"Best run: **{best['model']} macro-F1 {best['val_macro_f1']:.3f}, "
        f"accuracy {best['val_accuracy']:.3f}**.", "",
        "The temporal splits share one camera and match, so they measure learnability, "
        "not cross-camera generalization.", "",
        "## Reception quality", "",
        f"Grouped CV over {reception['num_examples']} receptions. Random forest balanced "
        f"accuracy improves from {rq['pre_contact_only']['random_forest']['balanced_accuracy']:.3f} "
        f"before contact only to {rq['pre_plus_post']['random_forest']['balanced_accuracy']:.3f} "
        "when post-contact trajectory is included.", "",
        "## Frame-level rally spotting", "",
        f"At ±10 frames: micro-F1 **{spot10['micro']['f1']:.3f}**, precision "
        f"{spot10['micro']['precision']:.3f}, recall {spot10['micro']['recall']:.3f}, "
        f"macro-F1 {spot10['macro_f1']:.3f}.", "",
        "The weakest spotting class is rally_end; contact recall is already high, "
        "but thresholds/NMS need tuning to reduce false positives.",
    ])
    (output_dir / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def merge_spotting_datasets(dataset_dirs: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    manifests = [_read(Path(directory).resolve() / "manifest.json") for directory in dataset_dirs]
    if len(manifests) != 2:
        raise ValueError("two full-video spotting datasets are required")
    reference = manifests[0]
    for manifest in manifests[1:]:
        for field in ("feature_schema", "feature_names", "event_types"):
            if manifest.get(field) != reference.get(field):
                raise ValueError(f"incompatible spotting field {field}")
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for directory, manifest in zip(dataset_dirs, manifests):
        root = Path(directory).resolve()
        for record in manifest.get("records", []):
            source_key = Path(str(record["source_video"])).stem
            destination = videos_dir / f"{source_key}.npz"
            shutil.copy2(root / record["npz"], destination)
            records.append({
                **record, "id": source_key, "source_key": source_key,
                "npz": (Path("videos") / destination.name).as_posix(),
            })
    result = {**reference, "records": records, "dataset_scope": "two_full_gold_videos_only"}
    _write(output_dir / "manifest.json", result)
    return {"records": len(records), "source_keys": [r["source_key"] for r in records]}


def build_two_video_report(
    sequence_summary_path: Path, reception_path: Path,
    spotting_summaries: Iterable[Path], output_dir: Path,
) -> dict[str, Any]:
    rows = _read(sequence_summary_path.resolve())["experiments"]
    reception = _read(reception_path.resolve())
    directions = {
        "Khe_to_j0": [r for r in rows if "khe_to_j0" in r["checkpoint"].lower()],
        "j0_to_Khe": [r for r in rows if "j0_to_khe" in r["checkpoint"].lower()],
    }
    sequence_results = {}
    for direction, direction_rows in directions.items():
        by_model = {}
        for model_name in ("lstm", "transformer"):
            model_rows = [r for r in direction_rows if r["model"] == model_name]
            values = np.asarray([r["val_macro_f1"] for r in model_rows], dtype=float)
            by_model[model_name] = {
                "runs": len(model_rows), "mean_macro_f1": float(values.mean()),
                "std_macro_f1": float(values.std()), "best_macro_f1": float(values.max()),
                "best_run": max(model_rows, key=lambda r: r["val_macro_f1"]),
            }
        sequence_results[direction] = by_model
    transformer_values = [
        row["val_macro_f1"] for row in rows if row["model"] == "transformer"
    ]
    spotting = {}
    for path in spotting_summaries:
        payload = _read(path.resolve())
        train_key = payload["split_videos"]["train"][0]
        val_key = payload["split_videos"]["val"][0]
        spotting[f"{train_key}_to_{val_key}"] = payload
    result = {
        "scope": "only_two_completed_full_video_gold_datasets",
        "sequence": sequence_results,
        "transformer_bidirectional_mean_macro_f1": float(np.mean(transformer_values)),
        "reception_quality": reception,
        "rally_spotting": spotting,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "two_video_experiment_report.json", result)
    lines = [
        "# Two full-video gold experiment report", "",
        "No legacy clips were used. Every validation split is a complete unseen video.", "",
        "## Sequence classification", "",
        "| Direction | Model | Runs | Mean macro-F1 | Std | Best |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for direction, models in sequence_results.items():
        for model, metrics in models.items():
            lines.append(
                f"| {direction} | {model} | {metrics['runs']} | "
                f"{metrics['mean_macro_f1']:.3f} | {metrics['std_macro_f1']:.3f} | "
                f"{metrics['best_macro_f1']:.3f} |"
            )
    lines.extend([
        "", f"Bidirectional Transformer mean macro-F1: **{np.mean(transformer_values):.3f}**.",
        "", "## Reception quality", "",
        "Binary task: controlled (perfect/positive) versus poor (negative/overpass/error).",
    ])
    for direction, payload in reception["directions"].items():
        rf = payload["feature_sets"]["pre_plus_post"]["random_forest"]
        lines.append(
            f"- {direction}: balanced accuracy {rf['balanced_accuracy']:.3f}, "
            f"ROC-AUC {rf['roc_auc']:.3f}."
        )
    lines.extend(["", "## Frame-level spotting", ""])
    for direction, payload in spotting.items():
        metric = payload["best_selection_metrics"]["at_10"]
        lines.append(
            f"- {direction}: micro-F1 {metric['micro']['f1']:.3f}, precision "
            f"{metric['micro']['precision']:.3f}, recall {metric['micro']['recall']:.3f}."
        )
    lines.extend([
        "", "The remaining bottlenecks are reception-quality class imbalance, "
        "spotting false positives, and weak rally_end transfer.",
    ])
    (output_dir / "TWO_VIDEO_EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    holdout = subparsers.add_parser("prepare-rally-holdout")
    holdout.add_argument("--manifest", type=Path, required=True)
    holdout.add_argument("--output", type=Path, required=True)
    holdout.add_argument("--train-through-rally", type=int, default=24)
    quality = subparsers.add_parser("reception-quality")
    quality.add_argument("--manifest", type=Path, required=True)
    quality.add_argument("--features-dir", type=Path, required=True)
    quality.add_argument("--output-dir", type=Path, required=True)
    quality.add_argument("--seed", type=int, default=42)
    quality_cross = subparsers.add_parser("reception-quality-cross-video")
    quality_cross.add_argument("--manifest", type=Path, required=True)
    quality_cross.add_argument("--features-dir", type=Path, required=True)
    quality_cross.add_argument("--output-dir", type=Path, required=True)
    quality_cross.add_argument("--seed", type=int, default=42)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    spotting = subparsers.add_parser("prepare-spotting-holdout")
    spotting.add_argument("--dataset-dir", type=Path, required=True)
    spotting.add_argument("--event-manifest", type=Path, required=True)
    spotting.add_argument("--output-dir", type=Path, required=True)
    spotting.add_argument("--train-through-rally", type=int, default=24)
    report = subparsers.add_parser("report")
    report.add_argument("--sequence-summary", type=Path, required=True)
    report.add_argument("--reception-results", type=Path, required=True)
    report.add_argument("--spotting-summary", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    merge_spotting = subparsers.add_parser("merge-spotting-datasets")
    merge_spotting.add_argument("--dataset-dirs", type=Path, nargs="+", required=True)
    merge_spotting.add_argument("--output-dir", type=Path, required=True)
    two_report = subparsers.add_parser("two-video-report")
    two_report.add_argument("--sequence-summary", type=Path, required=True)
    two_report.add_argument("--reception-results", type=Path, required=True)
    two_report.add_argument("--spotting-summaries", type=Path, nargs="+", required=True)
    two_report.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-rally-holdout":
        result = build_rally_holdout_manifest(args.manifest, args.output, args.train_through_rally)
    elif args.command == "reception-quality":
        result = run_reception_quality_pilot(args.manifest, args.features_dir, args.output_dir, args.seed)
    elif args.command == "reception-quality-cross-video":
        result = run_reception_quality_cross_video(
            args.manifest, args.features_dir, args.output_dir, args.seed,
        )
    elif args.command == "summarize":
        result = summarize_checkpoints(args.checkpoints, args.output)
    elif args.command == "prepare-spotting-holdout":
        result = build_spotting_temporal_holdout(
            args.dataset_dir, args.event_manifest, args.output_dir,
            args.train_through_rally,
        )
    elif args.command == "report":
        result = build_experiment_report(
            args.sequence_summary, args.reception_results,
            args.spotting_summary, args.output_dir,
        )
    elif args.command == "merge-spotting-datasets":
        result = merge_spotting_datasets(args.dataset_dirs, args.output_dir)
    else:
        result = build_two_video_report(
            args.sequence_summary, args.reception_results,
            args.spotting_summaries, args.output_dir,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
