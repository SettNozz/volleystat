#!/usr/bin/env python3
"""Can a local VLM label event clips? Scored against the existing human labels.

The bar to clear is the trained sequence model: macro F1 0.4377 on the same
5-class taxonomy (``checkpoints_lstm_5cls_v2``). A VLM only earns a place in the
labeling loop if it beats that zero-shot *and* is fast enough to run over ~780
clips per hour of footage — otherwise pre-labeling with the trained model stays
the better deal.

Frames are sampled from the end of the clip because a clip spans the *previous*
contact to *this* one: the action being named happens in the last moments.

Usage:
    python scripts/evaluation/vlm_clip_probe.py --model qwen3-vl:8b-instruct --count 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.event_clips.config import EVENT_TYPES, collapse_labels  # noqa: E402
from action_recognition.vlm.ollama_client import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OllamaVLMClient,
)

DEFINITIONS = {
    "serve": "serve: a player behind the end line puts the ball into play over the net",
    "reception": "reception: the receiving team's first touch after a serve, usually a forearm pass",
    "pass": "pass: any other controlled touch — a set for a teammate, or a defensive dig",
    "attack": "attack: a forceful hit sending the ball down over the net, usually jumping; also a block at the net",
    "other": "other: no clear contact, ball out of play, or you cannot tell",
}

PROMPT = """You are watching {n} consecutive frames from an indoor volleyball rally, in order. They cover the moment just before and at a single ball contact by a player.

Name that contact. Choose exactly one:
{options}

Answer with JSON only: {{"action": "{choices}"}}"""


def build_prompt(n: int, classes: list[str]) -> str:
    return PROMPT.format(
        n=n,
        options="\n".join(f"- {DEFINITIONS[c]}" for c in classes),
        choices="|".join(classes),
    )


def build_schema(classes: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": list(classes)}},
        "required": ["action"],
        "additionalProperties": False,
    }


# Kept for callers that import it; the full taxonomy is still the default.
ACTION_SCHEMA = build_schema(EVENT_TYPES)


def clip_frames(path: Path, n: int, width: int, tail_sec: float, fps: float) -> list:
    """Last ``tail_sec`` of the clip, ``n`` frames evenly spaced."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = min(total, max(n, int(tail_sec * fps)))
    start = max(0, total - span)
    want = {start + round(i * (span - 1) / max(n - 1, 1)) for i in range(n)}
    frames, idx = [], 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    while idx < total and len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            scale = width / frame.shape[1]
            frames.append(
                cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            )
        idx += 1
    cap.release()
    return frames



def ball_track(features_dir: Path, clip_id: str) -> list[tuple[float, float]]:
    """Per-frame ball centre in source pixels; (0, 0) where it was not seen."""
    path = features_dir / f"{clip_id}.parquet"
    if not path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_parquet(path)
    except Exception:
        return []
    if "ball_x" not in df:
        return []
    return list(zip(df["ball_x"].tolist(), df["ball_y"].tolist()))


def clip_frames_roi(path: Path, n: int, out_width: int, tail_sec: float, fps: float,
                    track: list[tuple[float, float]], roi: int) -> list:
    """Crop a square around the ball at native resolution, then upscale.

    A vision transformer sees roughly 28 px patches. A 21 px ball at 720p - or a
    13 px one after downscaling the frame to 768 - lands inside a single patch,
    indistinguishable from a scuff on the floor. Cropping tight and scaling up
    spends the same token budget on the part of the frame that carries the
    answer, and the ball ends up spanning several patches.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = min(total, max(n, int(tail_sec * fps)))
    start = max(0, total - span)
    want = [start + round(i * (span - 1) / max(n - 1, 1)) for i in range(n)]
    frames = []
    last_seen = None
    for row in range(min(len(track), total)):
        x, y = track[row]
        if x > 0 or y > 0:
            last_seen = (x, y)
        if row not in want:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, row)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        cx, cy = last_seen if last_seen else (w / 2, h / 2)
        half = roi // 2
        x0 = int(min(max(cx - half, 0), max(w - roi, 0)))
        y0 = int(min(max(cy - half, 0), max(h - roi, 0)))
        crop = frame[y0:y0 + roi, x0:x0 + roi]
        if crop.size == 0:
            continue
        frames.append(cv2.resize(crop, (out_width, out_width), interpolation=cv2.INTER_CUBIC))
    cap.release()
    return frames

def ask(client: OllamaVLMClient, frames: list, classes: list[str]) -> tuple[str, float]:
    response = client.chat_json(
        build_prompt(len(frames), classes),
        frames,
        build_schema(classes),
    )
    return json.dumps(response.data), response.seconds


# The model answers in volleyball English, not in our enum: it says "spike" for
# an attack and "serving" for a serve. Scoring those as unparsed would measure
# our vocabulary, not the model's understanding.
SYNONYMS = {
    "serve": "serve", "serving": "serve", "service": "serve", "jump serve": "serve",
    "reception": "reception", "receive": "reception", "receiving": "reception",
    "bump": "reception", "forearm pass": "reception",
    "pass": "pass", "passing": "pass", "set": "pass", "setting": "pass",
    "dig": "pass", "digging": "pass", "defense": "pass",
    "attack": "attack", "spike": "attack", "spiking": "attack", "hit": "attack",
    "smash": "attack", "block": "attack", "blocking": "attack", "kill": "attack",
    "other": "other", "none": "other", "unclear": "other", "unknown": "other",
}
# Longest first so "jump serve" wins over "serve" and "forearm pass" over "pass".
_SYNONYM_ORDER = sorted(SYNONYMS, key=len, reverse=True)


def normalise(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower()
    if text in SYNONYMS:
        return SYNONYMS[text]
    for key in _SYNONYM_ORDER:
        if re.search(rf"\b{re.escape(key)}\b", text):
            return SYNONYMS[key]
    return None


def parse_action(text: str) -> str | None:
    match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, re.S)
    if match:
        try:
            mapped = normalise(json.loads(match.group(0)).get("action"))
            if mapped:
                return mapped
        except json.JSONDecodeError:
            pass
    # The model often answers correctly but wraps it in prose.
    hits = {SYNONYMS[k] for k in _SYNONYM_ORDER if re.search(rf"\b{re.escape(k)}\b", text, re.I)}
    return hits.pop() if len(hits) == 1 else None


def macro_f1(pairs: list[tuple[str, str]], classes: list[str]) -> tuple[float, dict]:
    per = {}
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        fp = sum(1 for t, p in pairs if t != cls and p == cls)
        fn = sum(1 for t, p in pairs if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[cls] = {"n": tp + fn, "precision": round(precision, 3), "recall": round(recall, 3),
                    "f1": round(f1, 3)}
        f1s.append(f1)
    return sum(f1s) / len(f1s), per


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/event_clips_merged_v5/manifest.json")
    ap.add_argument(
        "--model", default=os.getenv("OLLAMA_VLM_MODEL", DEFAULT_MODEL)
    )
    ap.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
    )
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--tail-sec", type=float, default=1.5)
    ap.add_argument(
        "--classes",
        nargs="+",
        default=list(EVENT_TYPES),
        help="Restrict the taxonomy. 'serve' and 'reception' are defined by where "
             "a touch sits in the rally, not by how it looks, so a single clip "
             "cannot show them and the rules already label them; --classes pass "
             "attack other asks the model only what is actually visible.",
    )
    ap.add_argument("--roi", type=int, default=0,
                    help="Crop this many source pixels around the ball instead of "
                         "sending the whole frame (0 = whole frame)")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    client = OllamaVLMClient(
        args.ollama_url,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
    )
    status = client.check()
    features_dir = (Path(args.features_dir) if args.features_dir
                    else Path(args.manifest).parent / "features")
    classes = [c for c in args.classes if c in EVENT_TYPES]
    if len(classes) != len(args.classes):
        raise SystemExit(f"unknown class in {args.classes}; known: {EVENT_TYPES}")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    usable = [
        c for c in manifest["clips"]
        if c.get("path") and Path(c["path"]).exists() and c.get("event_types")
    ]
    for clip in usable:
        clip["_label"] = collapse_labels(clip["event_types"], clip.get("fsm_event_type"))
    # Clips whose truth is outside the asked-for taxonomy would be unanswerable.
    usable = [c for c in usable if c["_label"] in classes]

    # Balance the sample: with 60 "other" against 17 serves, an unbalanced draw
    # would mostly measure the class prior, not the model.
    by_class = defaultdict(list)
    for clip in usable:
        by_class[clip["_label"]].append(clip)
    rng = random.Random(args.seed)
    per_class = max(1, args.count // len(classes))
    sample = []
    for cls in classes:
        pool = by_class.get(cls, [])
        rng.shuffle(pool)
        sample.extend(pool[:per_class])
    rng.shuffle(sample)

    print(f"{len(usable)} usable clips; sampling {len(sample)} "
          f"({dict(Counter(c['_label'] for c in sample))})")
    print(
        f"Ollama {status['version']} at {status['base_url']}; model {args.model}, "
        f"{args.frames} frames @ {args.width}px, last {args.tail_sec}s\n"
    )

    pairs: list[tuple[str, str]] = []
    unparsed = 0
    times: list[float] = []
    records = []
    for i, clip in enumerate(sample, 1):
        if args.roi:
            track = ball_track(features_dir, clip["clip_id"])
            frames = clip_frames_roi(Path(clip["path"]), args.frames, args.width,
                                     args.tail_sec, clip.get("fps", 30.0), track, args.roi)
        else:
            frames = clip_frames(Path(clip["path"]), args.frames, args.width,
                                 args.tail_sec, clip.get("fps", 30.0))
        if not frames:
            continue
        try:
            text, dt = ask(client, frames, classes)
        except Exception as exc:  # noqa: BLE001
            print(f"  {clip['clip_id']}: request failed ({exc})")
            continue
        times.append(dt)
        pred = parse_action(text)
        if pred is None:
            unparsed += 1
            pred = "other"
        pairs.append((clip["_label"], pred))
        records.append({"clip_id": clip["clip_id"], "truth": clip["_label"],
                        "pred": pred, "seconds": round(dt, 1), "raw": text[:200]})
        if i % 10 == 0:
            acc = sum(1 for t, p in pairs if t == p) / len(pairs)
            print(f"  {i}/{len(sample)}  running accuracy {acc:.3f}  "
                  f"{sum(times)/len(times):.1f}s/clip")

    if not pairs:
        raise SystemExit("no predictions")
    accuracy = sum(1 for t, p in pairs if t == p) / len(pairs)
    mf1, per = macro_f1(pairs, classes)
    print(f"\naccuracy {accuracy:.3f}   macro F1 {mf1:.4f}   "
          f"(trained LSTM baseline 0.4377)")
    print(f"unparsed answers: {unparsed}/{len(pairs)}   {sum(times)/len(times):.1f}s per clip")
    print(f"{'class':>10} {'n':>4} {'prec':>6} {'rec':>6} {'f1':>6}")
    for cls, m in per.items():
        print(f"{cls:>10} {m['n']:>4} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    print(f"predicted distribution: {dict(Counter(p for _, p in pairs))}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"model": args.model, "accuracy": accuracy, "macro_f1": mf1,
                        "per_class": per, "records": records}, indent=1),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
