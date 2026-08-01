"""Build past-only event windows from complete per-frame annotations.

Unlike the review-oriented clip export, these sequences end exactly on the
contact frame and contain no future frames. This matches live inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from action_recognition.rally_spotting.prepare import build_continuous_features


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SOURCE_STEM=ANNOTATIONS_JSON")
    stem, raw_path = value.split("=", 1)
    if not stem or not raw_path:
        raise argparse.ArgumentTypeError("expected SOURCE_STEM=ANNOTATIONS_JSON")
    return stem, Path(raw_path)


def build_causal_dataset(
    manifest_path: Path | list[Path],
    annotation_paths: dict[str, Path],
    output_dir: Path,
    window_frames: int,
) -> dict[str, Any]:
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    manifest_paths = (
        [manifest_path] if isinstance(manifest_path, Path) else list(manifest_path)
    )
    if not manifest_paths:
        raise ValueError("at least one manifest is required")
    payloads = [_read(path.resolve()) for path in manifest_paths]
    payload = payloads[0]
    source_clips = [
        clip for current in payloads for clip in current.get("clips", [])
    ]
    clip_ids = [str(clip.get("clip_id")) for clip in source_clips]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("input manifests contain duplicate clip_id values")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    timelines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fps_by_source: dict[str, float] = {}
    for stem, path in annotation_paths.items():
        annotations = _read(path.resolve())
        frame_numbers, spotting = build_continuous_features(annotations)
        event_features = np.concatenate(
            (spotting[:, :10], spotting[:, 12:20]), axis=1
        ).astype(np.float32, copy=False)
        timelines[stem] = (frame_numbers, event_features)
        fps_by_source[stem] = float(annotations.get("fps") or 30.0)

    clips = []
    observed_lengths = []
    for source in source_clips:
        clip = dict(source)
        stem = str(clip.get("source_stem"))
        if stem not in timelines:
            raise ValueError(f"missing annotations mapping for {stem}")
        frame_numbers, features = timelines[stem]
        contact = int(clip["contact_frame"])
        matches = np.flatnonzero(frame_numbers == contact)
        if len(matches) != 1:
            raise ValueError(f"contact frame {contact} missing for {clip['clip_id']}")
        end_row = int(matches[0]) + 1
        start_row = max(0, end_row - window_frames)
        sequence = features[start_row:end_row]
        np.savez_compressed(
            features_dir / f"{clip['clip_id']}.npz", features=sequence
        )
        actual_start = int(frame_numbers[start_row])
        clip.update({
            "start_frame": actual_start,
            "end_frame": contact,
            "contact_offset": len(sequence) - 1,
            "post_contact_frames": 0,
            "num_frames": len(sequence),
            "window_policy": "causal_past_only",
        })
        clips.append(clip)
        observed_lengths.append(len(sequence))

    result = {
        **{key: value for key, value in payload.items() if key != "clips"},
        "source_manifests": [str(path.resolve()) for path in manifest_paths],
        "dataset_scope": "completed_full_video_gold_sources_only",
        "window_policy": {
            "type": "causal_past_only",
            "window_frames": window_frames,
            "includes_contact_frame": True,
            "future_frames": 0,
        },
        "fps_by_source": fps_by_source,
        "clips": clips,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (features_dir / "feature_schema.json").write_text(
        json.dumps({
            "feature_schema": result.get("feature_schema", "resolution_normalized_v2"),
            "num_features": 18,
            "sequence_len": window_frames,
            "window_policy": "causal_past_only",
        }, indent=2),
        encoding="utf-8",
    )
    return {
        "clips": len(clips),
        "window_frames": window_frames,
        "min_observed_frames": min(observed_lengths),
        "max_observed_frames": max(observed_lengths),
        "full_windows": sum(length == window_frames for length in observed_lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--annotations", action="append", type=_parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, required=True)
    args = parser.parse_args()
    report = build_causal_dataset(
        args.manifest, dict(args.annotations), args.output_dir, args.window_frames
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
