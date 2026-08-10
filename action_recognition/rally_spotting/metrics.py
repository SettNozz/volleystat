"""NMS and event-spotting metrics with frame-tolerance matching."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def nms_peaks(
    scores: Sequence[float],
    threshold: float = 0.5,
    radius: int = 5,
) -> list[int]:
    """Return score-descending local peaks, suppressing nearby frames."""
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    candidates = np.flatnonzero(values >= threshold)
    ordered = sorted(candidates.tolist(), key=lambda i: (-values[i], i))
    kept: list[int] = []
    for frame in ordered:
        if all(abs(frame - previous) > radius for previous in kept):
            kept.append(frame)
    return sorted(kept)


def match_events(
    predicted: Iterable[int],
    truth: Iterable[int],
    tolerance: int,
) -> tuple[int, int, int]:
    """Maximum-cardinality one-to-one timestamp matching.

    For sorted 1-D timestamps, pairing the earliest compatible prediction and
    target is optimal. Nearest-neighbour matching is not: a prediction can
    consume a later target and strand an earlier target that no later
    prediction can reach.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    predictions = sorted(int(value) for value in predicted)
    targets = sorted(int(value) for value in truth)
    pred_index = target_index = true_positive = 0
    while pred_index < len(predictions) and target_index < len(targets):
        prediction = predictions[pred_index]
        target = targets[target_index]
        if prediction < target - tolerance:
            pred_index += 1
        elif target < prediction - tolerance:
            target_index += 1
        else:
            true_positive += 1
            pred_index += 1
            target_index += 1
    return (
        true_positive,
        len(predictions) - true_positive,
        len(targets) - true_positive,
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def spotting_metrics(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    event_types: Sequence[str],
    tolerances: Sequence[int] = (5, 10, 15),
    threshold: float = 0.5,
    nms_radius: int = 5,
) -> dict[str, dict]:
    """Evaluate complete videos without matching events across boundaries."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    class_count = len(event_types)
    result: dict[str, dict] = {}
    for tolerance in tolerances:
        per_class: dict[str, dict] = {}
        total_tp = total_fp = total_fn = 0
        for class_index, event_type in enumerate(event_types):
            tp = fp = fn = 0
            for scores, labels in zip(predictions, targets):
                if scores.ndim != 2 or scores.shape[1] != class_count:
                    raise ValueError("prediction arrays must be (frames, classes)")
                if labels.shape != scores.shape:
                    raise ValueError("target shape must equal prediction shape")
                predicted_frames = nms_peaks(
                    scores[:, class_index], threshold, nms_radius,
                )
                true_frames = np.flatnonzero(labels[:, class_index] > 0.5)
                matched = match_events(predicted_frames, true_frames, tolerance)
                tp += matched[0]
                fp += matched[1]
                fn += matched[2]
            per_class[event_type] = _prf(tp, fp, fn)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        result[f"at_{tolerance}"] = {
            "micro": _prf(total_tp, total_fp, total_fn),
            "macro_f1": float(np.mean([
                metrics["f1"] for metrics in per_class.values()
            ])),
            "per_class": per_class,
        }
    return result
