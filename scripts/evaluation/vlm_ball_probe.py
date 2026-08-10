#!/usr/bin/env python3
"""Can a local VLM locate the ball? Measured against the human GT we already have.

A volleyball is ~65 px across in a 4K frame and ~21 px at 720p. Every VLM
resizes its input, so the question is not "does the model understand volleyball"
but "does the ball survive the resize, and can the model point at it". This
probe answers that with numbers instead of opinion: same GT frames, same
tolerance rule as ``score_ball_candidates.py``, several input widths.

Usage:
    python scripts/evaluation/vlm_ball_probe.py \
        --gt results/ball_eval/6_tiPcRYGcBtc_12090_12987_gt.json \
        --model qwen3-vl:8b-instruct --widths 640 1024 1600 --count 25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.vlm.ollama_client import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OllamaVLMClient,
)

PROMPT = (
    "This is a frame from an indoor volleyball match. Find the volleyball (the ball "
    "being played, not a ball in a bag or on a shelf).\n"
    "Answer with JSON only, no other text: {\"x\": <int 0-1000>, \"y\": <int 0-1000>} "
    "giving the ball centre as a fraction of image width and height times 1000. "
    "If you cannot see the ball, answer {\"x\": null, \"y\": null}."
)


POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": ["number", "null"], "minimum": 0, "maximum": 1000},
        "y": {"type": ["number", "null"], "minimum": 0, "maximum": 1000},
    },
    "required": ["x", "y"],
    "additionalProperties": False,
}


def ask(
    client: OllamaVLMClient, image: np.ndarray, prompt: str = PROMPT
) -> tuple[str, float]:
    response = client.chat_json(
        prompt,
        [image],
        POINT_SCHEMA,
        jpeg_quality=92,
    )
    return json.dumps(response.data), response.seconds


def parse_point(text: str) -> tuple[float, float] | None:
    """Pull {"x": .., "y": ..} out of whatever wrapping the model produced."""
    match = re.search(r"\{[^{}]*\"x\"[^{}]*\}", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if data.get("x") is None or data.get("y") is None:
        return None
    try:
        return float(data["x"]), float(data["y"])
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument(
        "--model", default=os.getenv("OLLAMA_VLM_MODEL", DEFAULT_MODEL)
    )
    ap.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
    )
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--widths", type=int, nargs="+", default=[640, 1024, 1600])
    ap.add_argument("--count", type=int, default=25, help="GT frames to sample")
    ap.add_argument("--tolerance", type=float, default=20.0, help="px at 1280-wide, scaled")
    ap.add_argument("--frames-dir", default=None)
    args = ap.parse_args()

    client = OllamaVLMClient(
        args.ollama_url,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
    )
    status = client.check()
    gt_data = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    video = Path(gt_data["video"])
    width, height = gt_data["resolution"]
    px_scale = width / 1280.0
    tolerance = args.tolerance * px_scale
    frames_dir = (
        Path(args.frames_dir) if args.frames_dir
        else _ROOT / "results" / "ball_eval" / f"frames_{video.stem}"
    )

    labels = {
        int(k): v for k, v in gt_data["labels"].items() if v["state"] == "visible"
    }
    chosen = sorted(labels)[:: max(1, len(labels) // args.count)][: args.count]
    print(f"{video.name} @ {width}x{height}: {len(chosen)} GT frames, "
          f"tolerance {tolerance:.0f} px (ball is ~{65 * px_scale / 3:.0f} px)")
    print(f"Ollama {status['version']} at {status['base_url']}; model {args.model}\n")

    header = f"{'input width':>12} {'hit':>5} {'miss':>5} {'no-answer':>10} {'recall':>7} {'med err px':>11} {'sec/frame':>10}"
    print(header)
    print("-" * len(header))

    for w in args.widths:
        hits = misses = refused = 0
        errors: list[float] = []
        times: list[float] = []
        for frame_idx in chosen:
            path = frames_dir / f"{frame_idx:06d}.jpg"
            if not path.exists():
                continue
            image = cv2.imread(str(path))
            scale = w / image.shape[1]
            small = cv2.resize(image, (w, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            try:
                text, dt = ask(client, small)
            except Exception as exc:  # noqa: BLE001 - a dead server should not lose the run
                print(f"  f{frame_idx}: request failed ({exc})")
                continue
            times.append(dt)
            point = parse_point(text)
            if point is None:
                refused += 1
                continue
            truth = labels[frame_idx]
            got = (point[0] / 1000.0 * width, point[1] / 1000.0 * height)
            err = math.dist(got, (truth["x"], truth["y"]))
            errors.append(err)
            if err <= tolerance:
                hits += 1
            else:
                misses += 1
        answered = hits + misses
        med = sorted(errors)[len(errors) // 2] if errors else float("nan")
        recall = hits / len(chosen) if chosen else 0.0
        avg_t = sum(times) / len(times) if times else float("nan")
        print(f"{w:>12} {hits:>5} {misses:>5} {refused:>10} {recall:>7.3f} {med:>11.0f} {avg_t:>10.1f}")


if __name__ == "__main__":
    main()
