"""Extract MobileNetV3-Small features for event clips.

The manifest may reference either a pre-cut clip (``path``) or an interval in
the original recording (``source_video`` + ``start_frame``/``end_frame``).
Supporting the latter is important for 4K labeling batches, where creating a
second encoded video for every candidate is needlessly expensive.

Example::

    python -m action_recognition.event_clips.extract_video_features \
        --manifest data/event_clips_merged_v6/manifest.json \
        --output-dir data/event_clips_merged_v6/video_features \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

from action_recognition.event_clips.config import (
    VIDEO_FEATURE_DIM,
    VIDEO_FEATURE_FRAMES,
)


def _existing_path(raw_path: str | None, manifest_dir: Path) -> Path | None:
    """Resolve absolute, cwd-relative, then manifest-relative video paths."""
    if not raw_path:
        return None
    raw = Path(raw_path).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, manifest_dir / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_clip_source(clip: dict, manifest_dir: Path) -> tuple[Path, int | None, int | None]:
    """Return ``(video_path, start_frame, end_frame)`` for one manifest clip.

    A real pre-cut ``path`` wins. If it is absent (the normal 4K workflow), the
    original recording and frame interval are returned instead.
    """
    clip_path = _existing_path(clip.get("path"), manifest_dir)
    if clip_path is not None:
        return clip_path, None, None

    source_path = _existing_path(clip.get("source_video"), manifest_dir)
    if source_path is None:
        requested = clip.get("source_video") or clip.get("path") or "<missing>"
        raise FileNotFoundError(f"Video source does not exist for {clip.get('clip_id')}: {requested}")

    if clip.get("start_frame") is None or clip.get("end_frame") is None:
        raise ValueError(
            f"{clip.get('clip_id')} uses source_video but has no start_frame/end_frame"
        )
    start, end = int(clip["start_frame"]), int(clip["end_frame"])
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame interval for {clip.get('clip_id')}: {start}..{end}")
    return source_path, start, end


def _sample_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    return np.linspace(start, end, count).round().astype(int).tolist()


def read_sampled_frames(
    video_path: Path,
    start_frame: int | None,
    end_frame: int | None,
    count: int = VIDEO_FEATURE_FRAMES,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
) -> list[np.ndarray]:
    """Decode uniformly sampled RGB frames, sequentially within the interval."""
    cap = capture_factory(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = 0 if start_frame is None else start_frame
        end = total - 1 if end_frame is None else end_frame
        if total > 0:
            end = min(end, total - 1)
        if end < start:
            raise RuntimeError(f"Empty frame interval in {video_path}: {start}..{end}")

        targets = _sample_indices(start, end, count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: list[np.ndarray] = []
        target_pos = 0
        frame_no = start
        last_rgb: np.ndarray | None = None
        while target_pos < len(targets) and frame_no <= end:
            ok, bgr = cap.read()
            if not ok:
                break
            last_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            while target_pos < len(targets) and targets[target_pos] == frame_no:
                frames.append(last_rgb.copy())
                target_pos += 1
            frame_no += 1

        if last_rgb is None:
            raise RuntimeError(f"No frames decoded from {video_path}: {start}..{end}")
        # A corrupt tail should not change tensor shape; duplicate the last good
        # frame and report failures at clip granularity in the caller.
        while len(frames) < count:
            frames.append(last_rgb.copy())
        return frames
    finally:
        cap.release()


def build_encoder(device: torch.device):
    """Return pretrained MobileNet features and its canonical preprocessing."""
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    backbone = mobilenet_v3_small(weights=weights).features.to(device).eval()
    preprocess = weights.transforms()
    return backbone, preprocess


@torch.inference_mode()
def encode_frames(
    frames: Sequence[np.ndarray],
    backbone: torch.nn.Module,
    preprocess,
    device: torch.device,
) -> np.ndarray:
    tensors = [preprocess(torch.from_numpy(frame).permute(2, 0, 1)) for frame in frames]
    batch = torch.stack(tensors).to(device, non_blocking=device.type == "cuda")
    feature_maps = backbone(batch)
    features = torch.nn.functional.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
    result = features.float().cpu().numpy()
    if result.shape != (len(frames), VIDEO_FEATURE_DIM):
        raise RuntimeError(
            f"Unexpected MobileNet feature shape {result.shape}; "
            f"expected ({len(frames)}, {VIDEO_FEATURE_DIM})"
        )
    return result


def feature_columns() -> list[str]:
    # Preserve the column convention used by event_clips_merged_v5.  The
    # dataset consumes values positionally, but matching names makes old/new
    # feature files directly inspectable and mergeable.
    return [f"f_{i:04d}" for i in range(VIDEO_FEATURE_DIM)]


def extract_manifest(
    manifest_path: Path,
    output_dir: Path,
    device_name: str,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips: Iterable[dict] = payload.get("clips", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    backbone, preprocess = build_encoder(device)

    written = skipped = failed = 0
    for clip in clips:
        clip_id = clip["clip_id"]
        destination = output_dir / f"{clip_id}.parquet"
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        try:
            video_path, start, end = resolve_clip_source(clip, manifest_path.parent)
            frames = read_sampled_frames(video_path, start, end)
            values = encode_frames(frames, backbone, preprocess, device)
            pd.DataFrame(values, columns=feature_columns()).to_parquet(
                temporary, index=False
            )
            temporary.replace(destination)
            written += 1
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            failed += 1
            print(f"ERROR {clip_id}: {exc}")
    return written, skipped, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    written, skipped, failed = extract_manifest(
        Path(args.manifest), Path(args.output_dir), args.device, args.overwrite
    )
    print(f"Video features: written={written}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
