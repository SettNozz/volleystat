"""Run a trained rally-spotting checkpoint on existing analysis results.

Writes a non-destructive viewer-ready copy. Exact NMS peaks are stored in
``rally_spotting_predictions.json``; nearby annotation frames receive a short
``spotting_model`` display label so the prediction remains visible in viewer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action_recognition.rally_spotting.metrics import nms_peaks  # noqa: E402
from action_recognition.rally_spotting.model import RallySpottingTCN  # noqa: E402
from action_recognition.rally_spotting.prepare import (  # noqa: E402
    FEATURE_NAMES,
    build_continuous_features,
)
from action_recognition.rally_spotting.train import predict_full_video  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find(results_dir: Path, suffix: str) -> Path:
    matches = sorted(results_dir.glob(f"*_{suffix}.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one *_{suffix}.json in {results_dir}, got {len(matches)}"
        )
    return matches[0]


def predict_results(
    results_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    threshold: float = 0.5,
    nms_radius: int = 5,
    display_radius: int = 5,
    window: int = 256,
    stride: int = 128,
    device_name: str | None = None,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if nms_radius < 0 or display_radius < 0:
        raise ValueError("radii must be non-negative")

    results_dir = results_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    annotation_path = _find(results_dir, "annotations")
    stats_path = _find(results_dir, "stats")
    annotations = _read(annotation_path)
    stats = _read(stats_path)

    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_features = tuple(checkpoint.get("feature_names") or ())
    if checkpoint_features != tuple(FEATURE_NAMES):
        raise ValueError(
            "checkpoint feature schema mismatch: "
            f"expected {len(FEATURE_NAMES)} canonical channels, got "
            f"{len(checkpoint_features)}"
        )
    event_types = tuple(checkpoint["event_types"])
    model = RallySpottingTCN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    frame_numbers, features = build_continuous_features(annotations)
    probabilities = predict_full_video(
        model, features, window=window, stride=stride, device=device,
    )
    fps = float(annotations["fps"])
    events: list[dict[str, Any]] = []
    for class_index, event_type in enumerate(event_types):
        for row in nms_peaks(
            probabilities[:, class_index], threshold=threshold, radius=nms_radius,
        ):
            frame = int(frame_numbers[row])
            events.append({
                "frame": frame,
                "time": round(frame / fps, 3),
                "type": event_type,
                "score": round(float(probabilities[row, class_index]), 6),
            })
    events.sort(key=lambda event: (event["frame"], -event["score"], event["type"]))

    by_frame = {
        int(frame.get("f", fallback)): frame
        for fallback, frame in enumerate(annotations["frames"])
    }
    for event in events:
        for frame_number in range(
            event["frame"] - display_radius, event["frame"] + display_radius + 1,
        ):
            frame = by_frame.get(frame_number)
            if frame is None:
                continue
            previous = float(frame.get("spotting_confidence", -1.0))
            if event["score"] >= previous:
                frame["spotting_model"] = event["type"]
                frame["spotting_confidence"] = event["score"]
                frame["spotting_peak"] = frame_number == event["frame"]

    report = {
        "checkpoint": str(checkpoint_path),
        "feature_schema": checkpoint.get("feature_schema"),
        "event_types": list(event_types),
        "threshold": threshold,
        "nms_radius": nms_radius,
        "display_radius": display_radius,
        "window": window,
        "stride": stride,
        "num_frames": int(len(frame_numbers)),
        "events": events,
        "counts": {
            event_type: sum(event["type"] == event_type for event in events)
            for event_type in event_types
        },
    }
    annotations["rally_spotting"] = {
        key: value for key, value in report.items() if key != "events"
    }
    stats["rally_spotting_predictions"] = report

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / annotation_path.name).write_text(
        json.dumps(annotations, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / stats_path.name).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (output_dir / "rally_spotting_predictions.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius", type=int, default=5)
    parser.add_argument("--display-radius", type=int, default=5)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    report = predict_results(
        args.results_dir, args.checkpoint, args.output_dir,
        threshold=args.threshold, nms_radius=args.nms_radius,
        display_radius=args.display_radius, window=args.window,
        stride=args.stride, device_name=args.device,
    )
    print(json.dumps({
        "num_frames": report["num_frames"],
        "counts": report["counts"],
        "events": len(report["events"]),
    }, indent=2))


if __name__ == "__main__":
    main()
