#!/usr/bin/env python3
"""Pre-label volleyball event clips with an Ollama VLM.

The source manifest is never changed. Results are written incrementally to a
separate manifest and can be resumed after interruption.

Examples:
    python -m action_recognition.vlm.prelabel_events --check
    python -m action_recognition.vlm.prelabel_events \
        --manifest data/event_clips_part2/manifest.json --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2

from action_recognition.event_clips.config import EVENT_TYPES
from action_recognition.vlm.ollama_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OllamaVLMClient,
)

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": EVENT_TYPES + ["unknown"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "contact_visible": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 160},
    },
    "required": ["action", "confidence", "contact_visible", "reason"],
    "additionalProperties": False,
}

PROMPT = """You are given {count} consecutive sampled frames from the END of one indoor
volleyball event clip. The frames are ordered in time. The clip ends at the
player-ball contact that must be classified.

Choose exactly one action:
- serve: behind the end line, starts play and sends the ball over the net
- reception: receiving team's first touch immediately after a serve
- pass: a set, dig, or other controlled non-first touch
- attack: spike, forceful hit, tip, or block at the net
- unknown: no visible/credible contact or too ambiguous to classify

Focus on player pose, court location, ball motion, and temporal change. Do not
invent a contact just because players are near the ball. Return the requested
JSON fields only. Keep reason under 20 words."""


def sample_clip_frames(
    path: Path,
    *,
    count: int = 6,
    width: int = 768,
    tail_seconds: float = 1.5,
    fps_hint: float = 30.0,
) -> list[Any]:
    """Sample ordered frames near the final contact."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0)
    span = min(total, max(count, round(tail_seconds * fps)))
    start = max(0, total - span)
    wanted = {
        start + round(i * max(span - 1, 0) / max(count - 1, 1))
        for i in range(count)
    }
    frames: list[Any] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start
    while index < total and len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            if width > 0 and frame.shape[1] != width:
                scale = width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (width, max(1, round(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
        index += 1
    cap.release()
    return frames


def resolve_clip_path(raw_path: str | None, manifest_path: Path) -> Path | None:
    if not raw_path:
        return None
    direct = Path(raw_path)
    candidates = [direct]
    if not direct.is_absolute():
        candidates.extend([manifest_path.parent / direct, Path.cwd() / direct])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return direct


def save_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def apply_prediction(
    clip: dict[str, Any],
    prediction: dict[str, Any],
    *,
    model: str,
    seconds: float,
) -> None:
    action = prediction.get("action")
    if action not in EVENT_TYPES + ["unknown"]:
        raise ValueError(f"Unknown VLM action: {action!r}")
    confidence = max(0.0, min(1.0, float(prediction.get("confidence", 0.0))))
    record = {
        "action": action,
        "confidence": round(confidence, 3),
        "contact_visible": bool(prediction.get("contact_visible", False)),
        "reason": str(prediction.get("reason", ""))[:160],
        "model": model,
        "seconds": round(float(seconds), 2),
    }
    clip["vlm_prediction"] = record
    clip["primary_event_type"] = action
    clip["event_types"] = [action] if action in EVENT_TYPES else []
    clip["label_source"] = "vlm_prelabel" if action in EVENT_TYPES else "vlm_unknown"
    clip.pop("vlm_error", None)


def merge_resume(source: dict[str, Any], output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        return source
    previous = json.loads(output_path.read_text(encoding="utf-8"))
    by_id = {clip.get("clip_id"): clip for clip in previous.get("clips", [])}
    for clip in source.get("clips", []):
        saved = by_id.get(clip.get("clip_id"))
        if saved and (
            saved.get("vlm_prediction")
            or saved.get("label_source") in {"human", "cleared"}
        ):
            clip.update(saved)
    return source


def should_skip_clip(clip: dict[str, Any], *, force: bool) -> bool:
    """Protect human decisions; optionally replace machine/rule predictions."""
    if clip.get("label_source") in {"human", "cleared"}:
        return True
    return not force and (
        bool(clip.get("vlm_prediction")) or clip.get("label_source") == "fsm_rule"
    )


def run(args: argparse.Namespace) -> None:
    client = OllamaVLMClient(
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        keep_alive=args.keep_alive,
    )
    status = client.check()
    print(
        f"Ollama {status['version']} at {status['base_url']} | "
        f"model {status['model']} ready"
    )
    if args.check:
        return

    manifest_path = Path(args.manifest)
    output_path = (
        Path(args.output)
        if args.output
        else manifest_path.parent / "manifest_vlm_predicted.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.force:
        manifest = merge_resume(manifest, output_path)
    manifest["vlm_prelabel"] = {
        "base_url": client.base_url,
        "model": client.model,
        "frames": args.frames,
        "width": args.width,
        "tail_seconds": args.tail_seconds,
    }

    attempted = predicted = skipped = failed = 0
    clips = manifest.get("clips", [])
    for index, clip in enumerate(clips, 1):
        if should_skip_clip(clip, force=args.force):
            skipped += 1
            continue
        if args.limit is not None and attempted >= args.limit:
            break
        path = resolve_clip_path(clip.get("path"), manifest_path)
        if path is None or not path.exists():
            clip["vlm_error"] = f"clip not found: {path}"
            failed += 1
            save_atomic(output_path, manifest)
            continue

        attempted += 1
        frames = sample_clip_frames(
            path,
            count=args.frames,
            width=args.width,
            tail_seconds=args.tail_seconds,
            fps_hint=float(clip.get("fps", 30.0)),
        )
        if not frames:
            clip["vlm_error"] = "could not decode clip frames"
            failed += 1
            save_atomic(output_path, manifest)
            continue
        try:
            response = client.chat_json(
                PROMPT.format(count=len(frames)), frames, EVENT_SCHEMA
            )
            apply_prediction(
                clip, response.data, model=client.model, seconds=response.seconds
            )
            predicted += 1
            print(
                f"[{index}/{len(clips)}] {clip.get('clip_id')} -> "
                f"{clip['vlm_prediction']['action']} "
                f"({clip['vlm_prediction']['confidence']:.2f}, "
                f"{response.seconds:.1f}s)"
            )
        except Exception as exc:  # keep a long batch resumable
            clip["vlm_error"] = str(exc)[:300]
            failed += 1
            print(f"[{index}/{len(clips)}] {clip.get('clip_id')} FAILED: {exc}")
        save_atomic(output_path, manifest)

    print(
        f"VLM pre-label complete: {predicted} predicted, {skipped} preserved, "
        f"{failed} failed -> {output_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="input event-clips manifest")
    parser.add_argument("--output", help="default: <manifest dir>/manifest_vlm_predicted.json")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
        help="Ollama API root; env: OLLAMA_BASE_URL",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_VLM_MODEL", DEFAULT_MODEL),
        help="vision model tag; env: OLLAMA_VLM_MODEL",
    )
    parser.add_argument("--check", action="store_true", help="only verify server and model")
    parser.add_argument("--limit", type=int, help="maximum new clips this run")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--tail-seconds", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--keep-alive", default="30m")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing VLM/rule pre-labels (human labels remain protected)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.check and not args.manifest:
        parser.error("--manifest is required unless --check is used")
    run(args)


if __name__ == "__main__":
    main()
