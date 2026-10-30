"""Split a dense spotting run into "where is the contact" and "what type is it".

The typed spotting score of ``run_dense_contact_spotting.py`` mixes two jobs.
This rescores a finished run to separate them, so the two-stage question --
whether a dedicated detector feeding the existing contact-centred clip
classifier would beat one model doing both -- can be answered from artifacts
that already exist. Nothing is trained here.

Three measurements, all from one finished run directory:

* **detection** -- predicted and human events collapsed to a single ``contact``
  class. For peak picking the collapse is also reported after NMS, because a
  per-class control fires several channels on one touch and a real detector
  would have one head.
* **detector proxy** -- peak picking over the summed contact channels of the
  saved probabilities, which is what a binary head would see, with its own
  threshold sweep.
* **typing** -- among contacts the system located (type-agnostic match), how
  often the type is right, plus the confusion between types.

    python scripts/evaluation/score_contact_detection.py --help
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.metrics import _prf, match_events, nms_peaks

SCHEMA = "contact_detection_typing_v1"
TOLERANCES = (1, 2, 4, 5, 10, 15)
THRESHOLDS = tuple(round(0.05 * step, 2) for step in range(1, 20))


def log(message: str) -> None:
    print(f"[detection-score] {message}", flush=True)


def collapse(events: Iterable[dict[str, Any]]) -> list[int]:
    return sorted(int(event["frame"]) for event in events)


def nms_frames(frames: Sequence[int], radius: int) -> list[int]:
    """Keep one frame per cluster, so one touch counts once.

    Without a confidence to rank by, the earliest frame of a cluster is kept;
    the cluster is at most ``radius`` wide, which is inside every tolerance
    this script reports except ``at_1`` and ``at_2``.
    """
    kept: list[int] = []
    for frame in sorted(frames):
        if kept and frame - kept[-1] <= radius:
            continue
        kept.append(frame)
    return kept


def pair_events(
    predicted: Sequence[int], truth: Sequence[int], tolerance: int,
) -> list[tuple[int, int]]:
    """The pairs behind ``match_events``, same earliest-compatible rule."""
    predictions = sorted(int(value) for value in predicted)
    targets = sorted(int(value) for value in truth)
    pairs: list[tuple[int, int]] = []
    pred_index = target_index = 0
    while pred_index < len(predictions) and target_index < len(targets):
        prediction = predictions[pred_index]
        target = targets[target_index]
        if prediction < target - tolerance:
            pred_index += 1
        elif target < prediction - tolerance:
            target_index += 1
        else:
            pairs.append((prediction, target))
            pred_index += 1
            target_index += 1
    return pairs


def detection_scores(
    predicted_frames: Sequence[int], truth_frames: Sequence[int],
) -> dict[str, Any]:
    """Type-agnostic event scores at every tolerance."""
    result = {}
    for tolerance in TOLERANCES:
        counts = match_events(predicted_frames, truth_frames, tolerance)
        result[f"at_{tolerance}"] = _prf(*counts)
    return result


def detector_proxy(
    probabilities: np.ndarray, truth_frames: Sequence[int], radius: int,
    tolerance: int,
) -> dict[str, Any]:
    """Peak picking over the summed contact mass, with a threshold sweep.

    A dedicated binary head is not this -- it would be trained against a
    contact/no-contact target -- but the summed contact channels are the best
    available proxy for one, and they need no training.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    contact_mass = values[:, dc.CONTACT_OFFSET:].sum(axis=1)
    sweep = []
    for threshold in THRESHOLDS:
        frames = nms_peaks(contact_mass, threshold, radius)
        counts = match_events(frames, truth_frames, tolerance)
        sweep.append({"threshold": threshold, **_prf(*counts)})
    best = max(sweep, key=lambda row: row["f1"])
    frames = nms_peaks(contact_mass, best["threshold"], radius)
    return {
        "threshold": best["threshold"],
        "selection": (
            "threshold chosen on the held-out video, the same optimistic "
            "protocol the rest of the experiment uses"
        ),
        "events": len(frames),
        "sweep": sweep,
        "scores": detection_scores(frames, truth_frames),
    }


def typing_scores(
    predicted: Sequence[dict[str, Any]], truth: Sequence[dict[str, Any]],
    tolerance: int,
) -> dict[str, Any]:
    """How often a located contact gets the right type.

    The pairing is type-agnostic, so this isolates the typing job: every pair
    is a contact the system found, whatever label it put on it.
    """
    # Per-class peak picking can fire two channels on one frame. Keep the
    # confident one rather than letting a dict silently drop either.
    predicted_by_frame: dict[int, dict[str, Any]] = {}
    duplicate_frames = 0
    for event in predicted:
        frame = int(event["frame"])
        existing = predicted_by_frame.get(frame)
        if existing is None:
            predicted_by_frame[frame] = event
            continue
        duplicate_frames += 1
        if (event.get("confidence") or 0.0) > (existing.get("confidence") or 0.0):
            predicted_by_frame[frame] = event
    truth_by_frame = {int(event["frame"]): event for event in truth}
    pairs = pair_events(
        list(predicted_by_frame), list(truth_by_frame), tolerance,
    )
    raw_confusion: Counter[tuple[str, str]] = Counter()
    core_confusion: Counter[tuple[str, str]] = Counter()
    raw_correct = core_correct = 0
    raw_comparable = 0
    for prediction_frame, truth_frame in pairs:
        prediction = predicted_by_frame[prediction_frame]
        target = truth_by_frame[truth_frame]
        predicted_type = prediction["type"]
        truth_type = target.get("type")
        predicted_core = dc.to_core(predicted_type)
        truth_core = target.get("core_type") or dc.to_core(truth_type)
        core_confusion[(truth_core, predicted_core)] += 1
        core_correct += int(predicted_core == truth_core)
        if truth_type is not None:
            raw_comparable += 1
            raw_confusion[(truth_type, predicted_type)] += 1
            raw_correct += int(predicted_type == truth_type)

    def per_class(confusion: Counter[tuple[str, str]], names: Sequence[str]):
        rows = {}
        for name in names:
            tp = confusion[(name, name)]
            fn = sum(count for (t, p), count in confusion.items()
                     if t == name and p != name)
            fp = sum(count for (t, p), count in confusion.items()
                     if p == name and t != name)
            if tp + fn == 0 and fp == 0:
                continue
            rows[name] = _prf(tp, fp, fn)
        return rows

    core_rows = per_class(core_confusion, dc.CORE_TYPES)
    raw_rows = per_class(raw_confusion, dc.CONTACT_TYPES)
    return {
        "located_contacts": len(pairs),
        "same_frame_predictions_dropped": duplicate_frames,
        "core_accuracy": core_correct / len(pairs) if pairs else 0.0,
        "core_macro_f1": (
            float(np.mean([row["f1"] for row in core_rows.values()]))
            if core_rows else 0.0
        ),
        "core_per_class": core_rows,
        "core_confusion": {f"{t}->{p}": c for (t, p), c in sorted(core_confusion.items())},
        "raw_comparable": raw_comparable,
        "raw_accuracy": raw_correct / raw_comparable if raw_comparable else 0.0,
        "raw_macro_f1": (
            float(np.mean([row["f1"] for row in raw_rows.values()]))
            if raw_rows else 0.0
        ),
        "raw_per_class": raw_rows,
        "raw_confusion": {f"{t}->{p}": c for (t, p), c in sorted(raw_confusion.items())},
    }


def non_game_mask(gold_dir: Path, stem: str, length: int) -> np.ndarray:
    """Frames the human marked ``non_game`` -- warm-up, breaks, dead time.

    These frames hold real volleyball touches that are deliberately unlabelled
    because they are not match play, so a contact predicted there is scored as
    a false positive while being a correct detection of something nobody
    annotated. Reporting both lets that be told apart from a real error.
    """
    path = gold_dir / stem / "derived_7d09206_v2" / "dense_targets.npz"
    with np.load(path) as arrays:
        phases = arrays["phase_targets"]
    mask = np.zeros(length, dtype=bool)
    shared = min(length, phases.shape[0])
    mask[:shared] = phases[:shared] == dc.EXPORTED_PHASES.index("non_game")
    return mask


def score_fold(
    fold_dir: Path, gold_dir: Path, radius: int, tolerance: int,
) -> dict[str, Any]:
    events = json.loads((fold_dir / "events.json").read_text(encoding="utf-8"))
    metrics = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    probabilities = np.load(fold_dir / "probabilities.npy")
    truth = events["truth"]
    truth_frames = collapse(truth)
    skip = non_game_mask(gold_dir, metrics["holdout"], probabilities.shape[0])

    def in_play(frames: Sequence[int]) -> list[int]:
        return [f for f in frames if not (0 <= f < skip.shape[0] and skip[f])]

    detection: dict[str, Any] = {}
    for condition in ("peaks", "constrained", "fsm"):
        rows = events.get(condition) or []
        frames = collapse(rows)
        detection[condition] = {
            "events": len(frames),
            "scores": detection_scores(frames, truth_frames),
            "in_play_scores": detection_scores(
                in_play(frames), in_play(truth_frames),
            ),
        }
        if condition == "peaks":
            merged = nms_frames(frames, radius)
            detection["peaks_nms"] = {
                "events": len(merged),
                "note": (
                    "per-class peaks merged within the NMS radius, so one touch "
                    "detected on several channels counts once"
                ),
                "scores": detection_scores(merged, truth_frames),
                "in_play_scores": detection_scores(
                    in_play(merged), in_play(truth_frames),
                ),
            }

    return {
        "holdout": metrics["holdout"],
        "truth_events": len(truth_frames),
        "non_game_frames": int(skip.sum()),
        "first_human_contact": truth_frames[0] if truth_frames else None,
        "typed_reference": {
            "core_macro_f1_at_5": (
                metrics.get("core_four_class", {})
                .get("constrained", {}).get("at_5", {}).get("macro_f1")
            ),
        },
        "detection": detection,
        "detector_proxy": detector_proxy(
            probabilities, truth_frames, radius, tolerance,
        ),
        "typing": {
            condition: typing_scores(events[condition], truth, tolerance)
            for condition in ("peaks", "constrained", "fsm")
            if events.get(condition)
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True,
        help="a finished run directory of run_dense_contact_spotting.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "data/full_video_gold")
    parser.add_argument("--nms-radius", type=int, default=5)
    parser.add_argument(
        "--tolerance", type=int, default=5,
        help="tolerance used for the pairing behind the typing numbers",
    )
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"refusing to overwrite existing {args.output}")
    fold_dirs = sorted(path for path in args.run.glob("fold_*") if path.is_dir())
    if not fold_dirs:
        parser.error(f"no fold_* directories under {args.run}")

    folds = []
    for fold_dir in fold_dirs:
        fold = score_fold(fold_dir, args.gold_dir, args.nms_radius, args.tolerance)
        folds.append(fold)
        detection = fold["detection"]
        typing = fold["typing"]
        log(
            f"{fold['holdout']}: {fold['truth_events']} contacts | "
            f"detection F1@{args.tolerance} "
            f"constrained={detection['constrained']['scores'][f'at_{args.tolerance}']['f1']:.4f} "
            f"peaks_nms={detection['peaks_nms']['scores'][f'at_{args.tolerance}']['f1']:.4f} "
            f"proxy={fold['detector_proxy']['scores'][f'at_{args.tolerance}']['f1']:.4f} | "
            f"typing core acc={typing['constrained']['core_accuracy']:.4f} "
            f"raw acc={typing['constrained']['raw_accuracy']:.4f}"
        )

    def mean(path: Sequence[str]) -> float:
        values = []
        for fold in folds:
            node: Any = fold
            for key in path:
                node = node[key]
            values.append(float(node))
        return float(np.mean(values))

    summary = {
        "detection_f1": {
            condition: {
                f"at_{tolerance}": mean(
                    ["detection", condition, "scores", f"at_{tolerance}", "f1"]
                )
                for tolerance in TOLERANCES
            }
            for condition in ("peaks", "peaks_nms", "constrained", "fsm")
        },
        "detection_f1_in_play": {
            condition: {
                f"at_{tolerance}": mean(
                    ["detection", condition, "in_play_scores", f"at_{tolerance}", "f1"]
                )
                for tolerance in TOLERANCES
            }
            for condition in ("peaks", "peaks_nms", "constrained", "fsm")
        },
        "detector_proxy_f1": {
            f"at_{tolerance}": mean(
                ["detector_proxy", "scores", f"at_{tolerance}", "f1"]
            )
            for tolerance in TOLERANCES
        },
        "typing": {
            condition: {
                "core_accuracy": mean(["typing", condition, "core_accuracy"]),
                "core_macro_f1": mean(["typing", condition, "core_macro_f1"]),
                "raw_accuracy": mean(["typing", condition, "raw_accuracy"]),
                "raw_macro_f1": mean(["typing", condition, "raw_macro_f1"]),
            }
            for condition in ("peaks", "constrained", "fsm")
        },
        "typed_core_macro_f1_at_5": mean(
            ["typed_reference", "core_macro_f1_at_5"]
        ),
    }

    payload = {
        "schema": SCHEMA,
        "run": str(args.run.resolve()),
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "caveats": [
            "Rescoring only. No model was trained and no artifact of the run "
            "was modified.",
            "The run's own best epoch and threshold were chosen on the "
            "held-out video, and the detector proxy's threshold is chosen the "
            "same way, so every number is a model-selection estimate.",
            "The detector proxy sums the six contact channels of a typed "
            "model. A head trained against a contact/no-contact target is not "
            "measured here and could do better or worse.",
            "Typing accuracy is conditioned on contacts the system located, "
            "so it is not comparable with a clip classifier's score on human "
            "contact frames without allowing for that difference.",
            "The in-play scores drop frames the human marked non_game. Those "
            "frames hold real touches nobody annotated, so a contact there is "
            "a false positive by scoring but not necessarily a model error. "
            "They are not a claim that non_game detection does not matter.",
        ],
        "summary": summary,
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"written to {args.output}")
    for condition, scores in summary["detection_f1"].items():
        log(f"detection {condition}: " + ", ".join(
            f"@{tolerance}={scores[f'at_{tolerance}']:.4f}"
            for tolerance in TOLERANCES
        ))
    for condition, scores in summary["detection_f1_in_play"].items():
        log(f"detection {condition} excluding non_game: " + ", ".join(
            f"@{tolerance}={scores[f'at_{tolerance}']:.4f}"
            for tolerance in TOLERANCES
        ))
    log("detector proxy: " + ", ".join(
        f"@{tolerance}={summary['detector_proxy_f1'][f'at_{tolerance}']:.4f}"
        for tolerance in TOLERANCES
    ))
    for condition, rows in summary["typing"].items():
        log(
            f"typing {condition}: core acc={rows['core_accuracy']:.4f} "
            f"core macroF1={rows['core_macro_f1']:.4f} "
            f"raw acc={rows['raw_accuracy']:.4f} "
            f"raw macroF1={rows['raw_macro_f1']:.4f}"
        )
    log(f"typed end-to-end core macro F1 @5 for reference: "
        f"{summary['typed_core_macro_f1_at_5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
