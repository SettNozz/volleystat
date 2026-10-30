"""Dense contact spotting with constraint-aware decoding, against the FSM.

Three conditions over the same per-frame probabilities, on whole-video folds:

* ``peaks``       - per-class NMS peak picking, no structure (control);
* ``constrained`` - constraint-aware Viterbi over phases and typed contacts,
                    with transitions, start/end labels and duration bounds
                    counted from the training fold only;
* ``fsm``         - the frozen ``IntegratedVolleyballFSM`` events already stored
                    in the baseline analyses, re-scored with the same metric.

Inputs are frozen and hashed: the audited contact review, the exported human
phase channel, and the existing conservative ball analyses. Nothing here
rewrites a source annotation, a dataset or a production checkpoint, and no
model is promoted.

Labels are human. Reconstructed ball coordinates stay derived inputs with their
provenance; they are never treated as detector ground truth.

    python scripts/evaluation/run_dense_contact_spotting.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from action_recognition.rally_spotting import constrained_decode as cd
from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.metrics import _prf, match_events
from action_recognition.rally_spotting.model import RallySpottingTCN
from action_recognition.rally_spotting.prepare import (
    FEATURE_NAMES, build_continuous_features,
)

SCHEMA = "dense_contact_spotting_v1"
STEMS = ("Khe4P8QcJkg_5_4k", "j0gRhgFCgIs_4_4k", "j3BO9GgUFBI_3_4k")
TOLERANCES = (1, 2, 4, 5, 10, 15)
# ``between`` and ``rally_live`` may legitimately run longer than any observed
# segment, so their upper duration bound is dropped rather than guessed.
UNBOUNDED = (0, 2)


def enforce_determinism() -> dict[str, Any]:
    """Make a rerun of the same seed reproduce the same numbers.

    ``torch.manual_seed`` alone does not: cuDNN picks convolution algorithms
    from what the device has free, and some kernels reduce with atomics, so two
    runs of one seed on this task were measured 0.01 macro F1 apart -- wide
    enough to swallow a real difference between two levers.

    ``CUBLAS_WORKSPACE_CONFIG`` has to be set before the CUDA context exists,
    so this runs before anything touches the device and says so when it is
    already too late.
    """
    report: dict[str, Any] = {"requested": True}
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        report["cublas_workspace_config"] = "set here"
    else:
        report["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        report["warning"] = (
            "the CUDA context already existed, so CUBLAS_WORKSPACE_CONFIG may "
            "not have taken effect; set it in the environment before the run"
        )
        log("WARNING: " + report["warning"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
        report["use_deterministic_algorithms"] = True
    except Exception as error:  # a kernel without a deterministic version
        report["use_deterministic_algorithms"] = False
        report["error"] = str(error)
        log(f"WARNING: deterministic algorithms unavailable: {error}")
    return report


def log(message: str) -> None:
    print(f"[dense-spotting] {message}", flush=True)


def load_video(
    stem: str,
    annotations_dir: Path,
    gold_dir: Path,
    contacts: Sequence[dict],
    masked: Sequence[int],
    human_source: Path,
    contact_radius: int,
    guard: int,
    zero_pose_channels: bool = False,
) -> dict[str, Any]:
    """Features, dense targets and the frozen FSM events for one source."""
    annotation_path = annotations_dir / f"{stem}_annotations.json"
    dense_path = gold_dir / stem / "derived_7d09206_v2" / "dense_targets.npz"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    frame_numbers, features = build_continuous_features(annotations)
    if int(frame_numbers[0]) != 0:
        raise ValueError(f"{stem}: analyses must start at frame 0")

    with np.load(dense_path) as arrays:
        phase_targets = arrays["phase_targets"]
        dense_frames = arrays["frame_numbers"]
    if int(dense_frames[0]) != 0 or dense_frames.shape[0] != phase_targets.shape[0]:
        raise ValueError(f"{stem}: exported phase channel is not frame-aligned")

    length = int(min(features.shape[0], phase_targets.shape[0]))
    if abs(features.shape[0] - phase_targets.shape[0]) > 2:
        raise ValueError(
            f"{stem}: analysis has {features.shape[0]} frames, human export has "
            f"{phase_targets.shape[0]}; more than container rounding"
        )
    covered = sorted({
        identity for contact in contacts for identity in contact["annotation_ids"]
    })
    uncovered = dc.uncovered_human_frames(human_source, stem, covered)
    in_range = [c for c in contacts if c["frame"] < length]
    if len(in_range) != len(contacts):
        raise ValueError(f"{stem}: verified contact past the decoded video")
    masked_frames = [f for f in list(masked) + uncovered if f < length]
    targets, audit = dc.build_targets(
        length, phase_targets, in_range, masked_frames=masked_frames,
        contact_radius=contact_radius, guard=guard,
    )
    # The guard ring keeps a one-frame timing disagreement out of the loss, but
    # it also separates every contact from its phase. Structure is therefore
    # counted on an unguarded copy, so ``rally_live -> attack`` is observable.
    structure, _ = dc.build_targets(
        length, phase_targets, in_range, masked_frames=masked_frames,
        contact_radius=contact_radius, guard=0,
    )
    # ``ev`` holds the FSM contact type as a bare string in these analyses.
    fsm_events = []
    for frame in annotations["frames"]:
        index = int(frame["f"])
        if index >= length:
            continue
        for event in frame.get("ev") or []:
            event_type = event if isinstance(event, str) else str(event.get("type"))
            if dc.to_core(event_type) is not None:
                fsm_events.append({"frame": index, "type": event_type})
    # Measured, not assumed: the first baseline analyses were ball-only runs,
    # so the eight pose channels were zero on every frame. Whether a given
    # analysis carries poses decides what this run can claim.
    pose_offset = 12
    pose_block = features[:length, pose_offset:pose_offset + 8]
    pose_frames = int(np.count_nonzero(np.any(pose_block != 0.0, axis=1)))
    # Matched control: the same analyses with the eight pose channels blanked.
    # A pose-enabled rerun changes the ball trajectory as well, so comparing
    # two different analyses cannot attribute a gain to poses; this can.
    if zero_pose_channels:
        features = features.copy()
        features[:, pose_offset:pose_offset + 8] = 0.0
    player_frames = int(sum(
        1 for frame in annotations["frames"]
        if int(frame.get("f", -1)) < length and (frame.get("p") or frame.get("poses"))
    ))
    return {
        "stem": stem,
        "frames": length,
        "features": features[:length],
        "targets": targets,
        "structure": structure,
        "pose_feature_frames": pose_frames,
        "pose_channels_zeroed": bool(zero_pose_channels),
        "player_frames": player_frames,
        "net_y": float(annotations.get("net_y", 0.0)),
        "truth": dc.truth_events(in_range),
        "fsm_events": fsm_events,
        "uncovered_human_frames": len(uncovered),
        "target_audit": audit,
        "annotation_sha256": dc.file_sha256(annotation_path),
        "dense_sha256": dc.file_sha256(dense_path),
    }


def windows(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def training_windows(
    videos: Sequence[dict], window: int, stride: int, contact_repeat: int,
) -> list[tuple[int, int]]:
    """Window starts, repeating contact-bearing windows to fight 1% positives."""
    items: list[tuple[int, int]] = []
    for index, video in enumerate(videos):
        contact_frames = np.flatnonzero(video["targets"] >= dc.CONTACT_OFFSET)
        for start in windows(video["frames"], window, stride):
            has_contact = bool(
                contact_frames.size
                and ((contact_frames >= start) & (contact_frames < start + window)).any()
            )
            repeats = contact_repeat if has_contact else 1
            items.extend([(index, start)] * repeats)
    return items


def class_weights(
    targets: Sequence[np.ndarray], maximum: float, mode: str = "max",
) -> np.ndarray:
    """Inverse-frequency weights over observed labels, capped.

    A contact occupies about one percent of frames even after widening, so an
    unweighted cross entropy collapses onto the phase classes. ``mode="max"``
    scales against the largest class, which is what actually lifts the contact
    classes; ``mode="median"`` is the milder variant.
    """
    counts = np.zeros(dc.NUM_LABELS, dtype=np.float64)
    for values in targets:
        observed = values[values != dc.IGNORE_INDEX]
        counts += np.bincount(observed, minlength=dc.NUM_LABELS)
    if not counts.any():
        raise ValueError("no observed labels in the training fold")
    if mode == "max":
        reference = counts.max()
    elif mode == "median":
        reference = np.median(counts[counts > 0])
    else:
        raise ValueError(f"unknown weight mode {mode!r}")
    weights = np.ones(dc.NUM_LABELS, dtype=np.float32)
    present = counts > 0
    weights[present] = np.clip(reference / counts[present], 1.0, maximum)
    return weights


@torch.no_grad()
def predict(
    model: nn.Module, features: np.ndarray, window: int, stride: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    length = features.shape[0]
    total = np.zeros((length, dc.NUM_LABELS), dtype=np.float64)
    counts = np.zeros((length, 1), dtype=np.float64)
    for start in windows(length, window, stride):
        end = min(start + window, length)
        padded = np.zeros((window, features.shape[1]), dtype=np.float32)
        padded[:end - start] = features[start:end]
        logits = model(torch.from_numpy(padded[None]).to(device))
        probabilities = torch.softmax(logits, dim=-1)[0, :end - start]
        total[start:end] += probabilities.double().cpu().numpy()
        counts[start:end] += 1.0
    if (counts == 0).any():
        raise ValueError("uncovered frame in sliding-window inference")
    return total / counts


def score(
    predicted: Sequence[dict], truth: Sequence[dict], types: Sequence[str],
    project: bool,
) -> dict[str, Any]:
    """Event F1 at several native-frame tolerances, per class and macro."""
    def typed(events: Sequence[dict], name: str) -> list[int]:
        return [
            int(event["frame"]) for event in events
            if (dc.to_core(event["type"]) if project else event["type"]) == name
        ]

    result: dict[str, Any] = {}
    for tolerance in TOLERANCES:
        per_class: dict[str, Any] = {}
        totals = np.zeros(3, dtype=np.int64)
        for name in types:
            counts = match_events(typed(predicted, name), typed(truth, name), tolerance)
            per_class[name] = _prf(*counts)
            totals += counts
        present = [
            name for name in types
            if per_class[name]["tp"] + per_class[name]["fn"] > 0
        ]
        result[f"at_{tolerance}"] = {
            "micro": _prf(*(int(value) for value in totals)),
            "macro_f1": float(np.mean([per_class[n]["f1"] for n in present]))
            if present else 0.0,
            "per_class": per_class,
        }
    return result


def train_seed(
    seed: int, items: Sequence[tuple[int, int]],
    train_features: Sequence[np.ndarray], train_targets: Sequence[np.ndarray],
    validation_features: np.ndarray, validation: dict,
    criterion: nn.Module, device: torch.device, args: argparse.Namespace,
) -> dict[str, Any]:
    """Train one model and return its best-epoch probabilities over the holdout.

    Seeding, model and optimiser all live here, so averaging several seeds is
    averaging independent runs rather than one run restarted.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RallySpottingTCN(
        input_dim=len(FEATURE_NAMES), num_classes=dc.NUM_LABELS,
        hidden_dim=args.hidden_dim, dilations=tuple(args.dilations),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    generator = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best = {"macro_f1": -1.0}
    best_state = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = generator.permutation(len(items))
        losses: list[float] = []
        for position in range(0, len(order), args.batch_size):
            batch = [items[index] for index in order[position:position + args.batch_size]]
            x = np.zeros((len(batch), args.window, len(FEATURE_NAMES)), np.float32)
            y = np.full((len(batch), args.window), dc.IGNORE_INDEX, np.int64)
            for row, (video_index, start) in enumerate(batch):
                end = min(start + args.window, train_features[video_index].shape[0])
                x[row, :end - start] = train_features[video_index][start:end]
                y[row, :end - start] = train_targets[video_index][start:end]
            logits = model(torch.from_numpy(x).to(device))
            loss = criterion(
                logits.reshape(-1, dc.NUM_LABELS), torch.from_numpy(y).to(device).reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        probabilities = predict(
            model, validation_features, args.window, args.window // 2, device,
        )
        events = cd.argmax_events(probabilities, args.threshold, args.nms_radius)
        metrics = score(events, validation["truth"], dc.CONTACT_TYPES, project=False)
        selection = metrics[f"at_{args.selection_tolerance}"]["macro_f1"]
        history.append({
            "epoch": epoch, "loss": float(np.mean(losses)),
            "peaks_macro_f1": selection,
        })
        log(
            f"  seed {seed} epoch {epoch:03d} loss={np.mean(losses):.4f} "
            f"peaks_macroF1={selection:.4f}"
        )
        if selection > best["macro_f1"]:
            best = {"macro_f1": selection, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                log(f"  seed {seed} early stop at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"seed {seed} produced no checkpoint")
    model.load_state_dict(best_state)
    return {
        "seed": int(seed),
        "best_epoch": int(best["epoch"]),
        "selection_macro_f1": float(best["macro_f1"]),
        "history": history,
        "probabilities": predict(
            model, validation_features, args.window, args.window // 2, device,
        ),
        "state_dict": best_state,
        "config": model.config(),
    }


def run_fold(
    holdout: str, videos: Sequence[dict], args: argparse.Namespace,
    output: Path,
) -> dict[str, Any]:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seeds = list(args.seeds) if args.seeds else [args.seed]
    train = [video for video in videos if video["stem"] != holdout]
    validation = next(video for video in videos if video["stem"] == holdout)
    log(f"fold {holdout}: train {[v['stem'] for v in train]} on {device}")

    stacked = np.concatenate([video["features"] for video in train], axis=0)
    mean = stacked.mean(axis=0)
    deviation = stacked.std(axis=0)
    deviation[deviation < 1e-6] = 1.0

    def normalise(values: np.ndarray) -> np.ndarray:
        return ((values - mean) / deviation).astype(np.float32)

    weights = class_weights(
        [video["targets"] for video in train], args.max_weight, args.weight_mode,
    )
    log("class weights " + ", ".join(
        f"{name}={weights[index]:.1f}" for index, name in enumerate(dc.LABEL_TYPES)
    ))
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device), ignore_index=dc.IGNORE_INDEX,
    )

    items = training_windows(train, args.window, args.stride, args.contact_repeat)
    log(f"{len(items)} training windows of {args.window} frames")
    train_features = [normalise(video["features"]) for video in train]
    train_targets = [video["targets"] for video in train]
    validation_features = normalise(validation["features"])

    runs = [
        train_seed(
            seed, items, train_features, train_targets, validation_features,
            validation, criterion, device, args,
        )
        for seed in seeds
    ]
    # Softmax rows each sum to one, so their mean does too and stays a valid
    # distribution for the decoder's log-probabilities.
    probabilities = np.mean([run["probabilities"] for run in runs], axis=0)
    if len(runs) > 1:
        log(
            "seed ensemble over " + ", ".join(
                f"{run['seed']} (epoch {run['best_epoch']}, "
                f"macroF1 {run['selection_macro_f1']:.4f})" for run in runs
            )
        )

    constraints = cd.estimate_constraints(
        [video["structure"] for video in train],
        duration_quantile=args.duration_quantile,
        duration_slack=args.duration_slack, self_loop_labels=UNBOUNDED,
    )
    missing = cd.unseen_transitions(constraints, [validation["structure"]])
    log(f"  unseen held-out transitions: {missing or 'none'}")

    log_probabilities = np.log(np.clip(probabilities, 1e-12, None))
    decoded_events: list[dict[str, Any]]
    decode_status = "ok"
    try:
        labels = cd.constrained_viterbi(
            log_probabilities, constraints, transition_weight=args.transition_weight,
        )
        decoded_events = cd.events_from_labels(labels, probabilities)
    except cd.InfeasibleConstraints as error:
        decode_status = f"infeasible: {error}"
        labels = np.full(validation["frames"], dc.IGNORE_INDEX, dtype=np.int64)
        decoded_events = []
    log(f"  constrained decode: {decode_status}, {len(decoded_events)} events")

    # The peak-picking control is given its best threshold on the held-out
    # video, so the structural decoder is not compared against a handicapped
    # baseline. The constrained decoder has no threshold to tune.
    sweep = []
    best_peaks = {"macro_f1": -1.0, "threshold": args.threshold, "events": []}
    for threshold in np.arange(0.05, 0.96, 0.05):
        events = cd.argmax_events(probabilities, float(threshold), args.nms_radius)
        value = score(
            events, validation["truth"], dc.CONTACT_TYPES, project=False,
        )[f"at_{args.selection_tolerance}"]["macro_f1"]
        sweep.append({"threshold": round(float(threshold), 2), "macro_f1": value})
        if value > best_peaks["macro_f1"]:
            best_peaks = {
                "macro_f1": value, "threshold": round(float(threshold), 2),
                "events": events,
            }
    peak_events = best_peaks["events"]
    log(
        f"  peaks threshold tuned on the held-out video: "
        f"{best_peaks['threshold']} (macroF1 {best_peaks['macro_f1']:.4f})"
    )
    conditions = {
        "peaks": peak_events,
        "constrained": decoded_events,
        "fsm": validation["fsm_events"],
    }
    fold = {
        "holdout": holdout,
        "train_sources": [video["stem"] for video in train],
        "seeds": seeds,
        "best_epochs": [run["best_epoch"] for run in runs],
        "per_seed": [
            {key: run[key] for key in
             ("seed", "best_epoch", "selection_macro_f1", "history")}
            for run in runs
        ],
        "selection": (
            f"each seed's best epoch chosen on the held-out video by peak macro "
            f"F1 at +/-{args.selection_tolerance} frames, then the seeds' "
            f"probabilities averaged; a model-selection estimate, not an "
            f"independent test"
        ),
        "decode_status": decode_status,
        "peaks_threshold": best_peaks["threshold"],
        "peaks_threshold_sweep": sweep,
        "unseen_held_out_transitions": missing,
        "constraints": {
            "transition_counts": constraints["transition_counts"],
            "valid_start": [dc.LABEL_TYPES[i] for i in constraints["valid_start"]],
            "valid_end": [dc.LABEL_TYPES[i] for i in constraints["valid_end"]],
            "duration_min": {
                dc.LABEL_TYPES[i]: int(v)
                for i, v in enumerate(constraints["duration_min"])
            },
            "duration_max": {
                dc.LABEL_TYPES[i]: (None if i in UNBOUNDED else int(v))
                for i, v in enumerate(constraints["duration_max"])
            },
        },
        "truth_events": len(validation["truth"]),
        "event_counts": {name: len(rows) for name, rows in conditions.items()},
        "raw_six_class": {
            name: (
                score(rows, validation["truth"], dc.CONTACT_TYPES, project=False)
                if name != "fsm" else {
                    "not_comparable": (
                        "the FSM emits the collapsed name 'pass' for a rally "
                        "touch, which has no counterpart in the raw six-class "
                        "space; see core_four_class for its score"
                    )
                }
            )
            for name, rows in conditions.items()
        },
        "core_four_class": {
            name: score(rows, validation["truth"], dc.CORE_TYPES, project=True)
            for name, rows in conditions.items()
        },
    }
    if len(runs) == 1:
        fold["best_epoch"] = runs[0]["best_epoch"]
        fold["history"] = runs[0]["history"]
    fold_dir = output / f"fold_{holdout}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.save(fold_dir / "probabilities.npy", probabilities.astype(np.float32))
    np.save(fold_dir / "decoded_labels.npy", labels)
    for run in runs:
        name = "best.pt" if len(runs) == 1 else f"best_seed{run['seed']}.pt"
        torch.save(
            {
                "schema": SCHEMA, "state_dict": run["state_dict"],
                "config": run["config"], "seed": run["seed"],
                "feature_names": list(FEATURE_NAMES),
                "label_types": list(dc.LABEL_TYPES),
                "mean": mean, "std": deviation, "holdout": holdout,
                "ensemble_seeds": seeds,
                "note": (
                    "experimental dense spotter; not a production checkpoint. "
                    "With several seeds the reported scores come from the "
                    "averaged probabilities, which no single checkpoint "
                    "reproduces on its own."
                ),
            },
            fold_dir / name,
        )
    (fold_dir / "events.json").write_text(
        json.dumps({
            "truth": validation["truth"], **conditions,
        }, indent=1), encoding="utf-8",
    )
    (fold_dir / "metrics.json").write_text(json.dumps(fold, indent=1), encoding="utf-8")
    return fold


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reviews", type=Path,
        default=ROOT / "data/contact_spatial_gold_v1/reviews.json",
    )
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=ROOT / "results/gold_v1_ball_reconstruction_conservative",
    )
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "data/full_video_gold")
    parser.add_argument(
        "--human-source", type=Path,
        default=ROOT / "data/full_video_gold/j3BO9GgUFBI_3_4k/annotations_v1.json",
        help="cumulative human project file, used only to mask uncovered touches",
    )
    parser.add_argument(
        "--zero-pose-channels", action="store_true",
        help="blank the eight pose channels; the matched control that "
             "attributes a gain to poses rather than to a changed trajectory",
    )
    parser.add_argument("--contact-radius", type=int, default=2)
    parser.add_argument("--guard", type=int, default=3)
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--contact-repeat", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--dilations", type=int, nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-weight", type=float, default=25.0)
    parser.add_argument("--weight-mode", choices=("max", "median"), default="max")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms-radius", type=int, default=5)
    parser.add_argument("--selection-tolerance", type=int, default=5)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--duration-quantile", type=float, default=0.0)
    parser.add_argument("--duration-slack", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="train one model per seed and average their per-frame "
             "probabilities before decoding; overrides --seed",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-deterministic", dest="deterministic", action="store_false",
        help="allow cuDNN autotuning and non-deterministic kernels; two runs "
             "of one seed then differ by about 0.01 macro F1, which is as "
             "large as some of the effects being measured",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument("--folds", nargs="+", default=list(STEMS))
    args = parser.parse_args(argv)

    determinism = (
        enforce_determinism() if args.deterministic else {"requested": False}
    )

    output = args.output
    if output.exists():
        parser.error(f"refusing to write into existing {output}")
    output.mkdir(parents=True)

    review = dc.reviewed_contacts(args.reviews)
    log(f"review audit: {review['audit']}")
    videos = []
    for stem in STEMS:
        video = load_video(
            stem, args.annotations_dir, args.gold_dir,
            review["contacts"].get(stem, []), review["masked"].get(stem, []),
            args.human_source, args.contact_radius, args.guard,
            zero_pose_channels=args.zero_pose_channels,
        )
        log(
            f"{stem}: {video['frames']} frames, {len(video['truth'])} verified "
            f"contacts, {video['target_audit']['masked_frames']} masked frames, "
            f"{len(video['fsm_events'])} FSM events, "
            f"{video['pose_feature_frames']} frames with pose features"
            f"{' (zeroed by request)' if video['pose_channels_zeroed'] else ''}, "
            f"net_y={video['net_y']:.1f}"
        )
        videos.append(video)

    preflight = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label_source": "human",
        "review_path": str(args.reviews.resolve()),
        "review_sha256": review["review_sha256"],
        "review_audit": review["audit"],
        "human_source": str(args.human_source.resolve()),
        "human_source_sha256": dc.file_sha256(args.human_source),
        "feature_names": list(FEATURE_NAMES),
        "label_types": list(dc.LABEL_TYPES),
        "target_schema": dc.SCHEMA,
        "decode_schema": cd.SCHEMA,
        "split_policy": "leave-one-complete-source-video-out",
        "determinism": determinism,
        "sources": {
            video["stem"]: {
                "frames": video["frames"],
                "verified_contacts": len(video["truth"]),
                "uncovered_human_frames_masked": video["uncovered_human_frames"],
                "annotations": str((args.annotations_dir / f"{video['stem']}_annotations.json").resolve()),
                "annotation_sha256": video["annotation_sha256"],
                "dense_targets_sha256": video["dense_sha256"],
                "target_audit": video["target_audit"],
                "fsm_events": len(video["fsm_events"]),
                "pose_feature_frames": video["pose_feature_frames"],
                "pose_channels_zeroed": video["pose_channels_zeroed"],
                "player_frames": video["player_frames"],
                "net_y": video["net_y"],
            }
            for video in videos
        },
        "pose_channels_live": all(
            video["pose_feature_frames"] > 0 and not video["pose_channels_zeroed"]
            for video in videos
        ),
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "caveats": [
            "Best epoch and the decision threshold are chosen on the held-out "
            "video, so every number is a model-selection estimate.",
        ] + ([] if args.deterministic else [
            "Determinism was disabled, so a rerun of this seed will not "
            "reproduce these numbers; measured run-to-run spread at a fixed "
            "seed is about 0.01 macro F1.",
        ]) + [
            "Spotting F1 is not comparable with contact-centred clip "
            "classification F1: the clip task is given the contact frame.",
        ] + [
            (
                f"{video['stem']}: the analysis carries no pose evidence "
                f"({video['pose_feature_frames']} frames with pose features, "
                f"{video['player_frames']} with players), so its pose channels "
                "are zero and its FSM events are a floor, not a fair measure "
                "of the FSM."
                if video["pose_feature_frames"] == 0 else
                f"{video['stem']}: pose channels are live on "
                f"{video['pose_feature_frames']} of {video['frames']} frames "
                f"({video['player_frames']} frames carry players)."
            )
            for video in videos
        ] + ([
            "Pose channels were blanked by --zero-pose-channels. This is the "
            "matched control for the same analyses, not a pose-free analysis: "
            "the ball trajectory is the pose-enabled one.",
        ] if args.zero_pose_channels else []),
    }
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=1, default=str), encoding="utf-8",
    )

    folds = []
    for holdout in args.folds:
        if holdout not in {video["stem"] for video in videos}:
            parser.error(f"unknown fold {holdout}")
        folds.append(run_fold(holdout, videos, args, output))

    summary = {
        "schema": SCHEMA,
        "preflight": preflight,
        "folds": folds,
        "mean_macro_f1": {
            view: {
                condition: {
                    f"at_{tolerance}": float(np.mean([
                        fold[view][condition][f"at_{tolerance}"]["macro_f1"]
                        for fold in folds
                    ]))
                    for tolerance in TOLERANCES
                }
                for condition in conditions
            }
            # The FSM has no raw six-class score: its taxonomy already
            # collapses a rally touch to ``pass``.
            for view, conditions in (
                ("raw_six_class", ("peaks", "constrained")),
                ("core_four_class", ("peaks", "constrained", "fsm")),
            )
        },
        "note": "model-selection estimates on three related-camera videos",
    }
    (output / "report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log("report written to " + str(output / "report.json"))
    for view, conditions in summary["mean_macro_f1"].items():
        for condition, values in conditions.items():
            log(f"{view} {condition}: " + ", ".join(
                f"@{t}={values[f'at_{t}']:.4f}" for t in TOLERANCES
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
