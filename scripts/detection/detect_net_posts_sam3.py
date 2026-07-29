"""Detect volleyball-net posts with SAM3 in its isolated Python environment.

The main pipeline runs on Python 3.11 while SAM3 is installed in
``.sam3_env``.  This worker samples several frames, segments both posts by
text prompt, and writes transparent JSON/PNG artifacts for the production
bridge in :mod:`src.detection.net_detector`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TARGET_W, TARGET_H = 1280, 720
FIRST_N_SECONDS = 5.0


def _load_net_priors(calibration: Path, src_w: int) -> tuple[float, float]:
    data = json.loads(calibration.read_text(encoding="utf-8"))
    points = data.get("net_line_points")
    if not points or len(points) != 2:
        raise ValueError("calibration needs two net_line_points")
    calib_w = float(data.get("frame_width") or src_w)
    return tuple(float(p[0]) * TARGET_W / calib_w for p in points)  # type: ignore[return-value]


def _sample_frames(video: str, count: int) -> tuple[list[tuple[int, np.ndarray]], int, int]:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last = max(0, int(FIRST_N_SECONDS * fps) - 1)
    indices = np.linspace(0, last, max(1, count), dtype=int)
    frames: list[tuple[int, np.ndarray]] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, raw = cap.read()
        if ok:
            frames.append((
                int(index),
                cv2.resize(raw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA),
            ))
    cap.release()
    if not frames:
        raise RuntimeError(f"Cannot read frames from video: {video}")
    return frames, src_w, src_h


def _mask_base(mask: np.ndarray) -> tuple[float, float] | None:
    """Robust visible bottom point: median of the lowest 2% of a post mask."""
    ys, xs = np.where(mask.astype(bool))
    if len(xs) < 8:
        return None
    cutoff = np.percentile(ys, 98)
    return float(np.median(xs[ys >= cutoff])), float(np.median(ys[ys >= cutoff]))


def _select_post(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    expected_x: float,
    side: str,
) -> tuple[int, tuple[float, float], float] | None:
    """Pick a tall post near one calibrated end of the net."""
    best: tuple[int, tuple[float, float], float] | None = None
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        base = _mask_base(mask)
        if base is None:
            continue
        x1, y1, x2, y2 = map(float, box)
        width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        x, y = base
        if height < 60 or height / width < 1.35 or not (260 <= y <= 650):
            continue
        signed = x - expected_x
        if side == "left" and signed > 145:
            continue
        if side == "right" and signed < -145:
            continue
        distance = abs(signed)
        if distance > 210:
            continue
        quality = float(score) + 0.002 * height - 0.004 * distance
        if best is None or quality > best[2]:
            best = (i, base, quality)
    return best


def _as_candidates(
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> list[dict]:
    return [
        {
            "index": int(i),
            "box": [float(v) for v in box],
            "score": float(score),
            "base": _mask_base(mask),
        }
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores))
    ]


def _draw_overlay(
    frame: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    selected: dict[str, tuple[int, tuple[float, float], float]],
    expected: tuple[float, float],
    aggregate: tuple[tuple[float, float], tuple[float, float]] | None,
) -> np.ndarray:
    out = frame.copy()
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        tint = np.zeros_like(out)
        tint[:, :, 1] = 220
        active = mask.astype(bool)
        out[active] = cv2.addWeighted(out, 0.45, tint, 0.55, 0)[active]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (80, 200, 80), 1)
        cv2.putText(
            out, f"{i}:{float(score):.2f}", (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 230, 80), 1, cv2.LINE_AA,
        )
    for label, x in zip(("prior L", "prior R"), expected):
        cv2.line(out, (round(x), 250), (round(x), 650), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(
            out, label, (round(x) - 30, 245), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 0, 255), 1, cv2.LINE_AA,
        )
    if len(selected) == 2:
        raw_a, raw_b = selected["left"][1], selected["right"][1]
        cv2.line(out, tuple(map(round, raw_a)), tuple(map(round, raw_b)),
                 (160, 160, 160), 1, cv2.LINE_AA)
    if aggregate is not None:
        a, b = aggregate
        cv2.line(out, tuple(map(round, a)), tuple(map(round, b)),
                 (255, 0, 255), 3, cv2.LINE_AA)
        for label, point in (("A", a), ("B", b)):
            p = tuple(map(round, point))
            cv2.circle(out, p, 8, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                out, label, (p[0] + 10, p[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA,
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", default=str(ROOT / "court_calibration.json"))
    parser.add_argument("--prompt", default="volleyball net post")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "sam3_net_posts"))
    args = parser.parse_args()

    frames, src_w, _ = _sample_frames(args.video, max(1, args.samples))
    expected = _load_net_priors(Path(args.calibration), src_w)

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise SystemExit("Run this with .sam3_env/Scripts/python.exe") from exc

    print(f"Loading SAM3; prompt={args.prompt!r}; samples={len(frames)}; priors={expected}")
    model = build_sam3_image_model(device="cuda")
    processor = Sam3Processor(model, device="cuda", confidence_threshold=0.20)

    sample_results: list[dict] = []
    paired: list[tuple[tuple[float, float], tuple[float, float]]] = []
    representative = None
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for frame_index, frame in frames:
            state = processor.set_image(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            )
            output = processor.set_text_prompt(state=state, prompt=args.prompt)
            masks = output["masks"].squeeze(1).detach().float().cpu().numpy().astype(bool)
            boxes = output["boxes"].detach().float().cpu().numpy()
            scores = output["scores"].detach().float().cpu().numpy()
            left = _select_post(masks, boxes, scores, expected[0], "left")
            right = _select_post(masks, boxes, scores, expected[1], "right")
            selected = {}
            if left:
                selected["left"] = left
            if right:
                selected["right"] = right
            if len(selected) == 2:
                paired.append((selected["left"][1], selected["right"][1]))
                if representative is None:
                    representative = (frame, masks, boxes, scores, selected)
            sample_results.append({
                "frame": frame_index,
                "candidates": _as_candidates(masks, boxes, scores),
                "selected": {
                    side: list(value[1]) for side, value in selected.items()
                },
            })
            print(f"frame={frame_index}: instances={len(scores)} selected={sample_results[-1]['selected']}")

    required = 1 if len(frames) == 1 else 2
    aggregate = None
    if len(paired) >= required:
        left_points = np.asarray([pair[0] for pair in paired], dtype=np.float64)
        right_points = np.asarray([pair[1] for pair in paired], dtype=np.float64)
        aggregate = (
            tuple(np.median(left_points, axis=0).astype(float)),
            tuple(np.median(right_points, axis=0).astype(float)),
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.video).stem
    overlay_path = out_dir / f"{stem}_sam3_posts.png"
    json_path = out_dir / f"{stem}_sam3_posts.json"
    if representative is None:
        fallback = frames[0][1]
        cv2.imwrite(str(overlay_path), fallback)
    else:
        cv2.imwrite(
            str(overlay_path),
            _draw_overlay(*representative, expected, aggregate),
        )
    payload = {
        "resolution": [TARGET_W, TARGET_H],
        "prompt": args.prompt,
        "net_priors": {"left_x": expected[0], "right_x": expected[1]},
        "samples_requested": len(frames),
        "samples_with_both_posts": len(paired),
        "sample_results": sample_results,
        "net_floor_line_720p": (
            {
                "left": aggregate[0],
                "right": aggregate[1],
                "method": "sam3_post_mask_multiframe",
            }
            if aggregate is not None else None
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Selected: {payload['net_floor_line_720p']}")
    print(f"Overlay: {overlay_path}")
    print(f"JSON: {json_path}")
    if aggregate is None:
        raise SystemExit("SAM3 did not yield both posts in enough sampled frames")


if __name__ == "__main__":
    main()
