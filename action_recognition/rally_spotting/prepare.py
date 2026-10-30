"""Prepare a continuous 4K-safe rally-spotting dataset.

The input is a JSON manifest::

    {
      "videos": [{
        "source_video": "data/game_01.mp4",
        "stats_path": "results/game_01_stats.json",
        "annotations_path": "results/game_01_annotations.json",
        "label_source": "weak_fsm"
      }, {
        "source_video": "data/game_02.mp4",
        "stats_path": "results/game_02_stats.json",
        "annotations_path": "results/game_02_annotations.json",
        "label_source": "human",
        "labels_path": "labels/game_02.json"
      }]
    }

Human label files contain either an ``events`` list or a bare list of
``{"frame": int, "type": str}`` objects. Weak labels are exported from
``service_attempts[*].lifecycle`` and are explicitly marked ``weak_fsm``.
The trainer refuses weak labels unless ``--allow-weak-labels`` is supplied.

CLI::

    python -m action_recognition.rally_spotting.prepare \
      --manifest data/rally_4k_sources.json --output-dir data/rally_4k_v1

Each source video becomes one compressed NPZ file. Windows are created only
after splitting by source video at training time, preventing video leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from action_recognition.event_clips.feature_schema import (
    RESOLUTION_NORMALIZED_SCHEMA,
    build_trajectory_features,
)
from src.tracking.pose_feature_extractor import POSE_FEATURE_NAMES
from action_recognition.rally_spotting.taxonomy import (
    EVENT_TO_INDEX,
    EVENT_TYPES,
    HUMAN_LABEL_SOURCE,
    WEAK_LABEL_SOURCE,
    normalize_label_source,
    validate_event_type,
)

FEATURE_NAMES = (
    "ball_x",
    "ball_y",
    "ball_speed",
    "ball_dir",
    "ball_net_dist",
    "player_dist",
    "team_side",
    "frame_norm",
    "ball_vy",
    "ball_above_net",
    "ball_visible",
    "ball_track_start",
) + tuple(POSE_FEATURE_NAMES) + (
    "team_b_back_prior_fraction",
    "serve_back_prior_present",
)
DATASET_VERSION = 1


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _players(frame: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw = frame.get("p")
    if isinstance(raw, list):
        return [
            (float(point[0]), float(point[1]))
            for point in raw
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
    poses = frame.get("poses") or []
    result: list[tuple[float, float]] = []
    for pose in poses:
        bbox = pose.get("bb") if isinstance(pose, dict) else None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            result.append(
                ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))
            )
    return result


def build_continuous_features(
    annotations: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frame_numbers, features)`` for the complete annotation span.

    Missing annotation frames are retained as zero-observation frames so that
    an output row always corresponds to a real source frame. Geometric
    channels use ``resolution_normalized_v2``.
    """
    width = float(annotations.get("frame_width") or 0)
    height = float(annotations.get("frame_height") or 0)
    fps = float(annotations.get("fps") or 0)
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("annotations require positive frame_width, frame_height and fps")
    net_y = float(annotations.get("net_y", height / 2.0))
    frames = annotations.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("annotations.frames must be a non-empty list")

    by_number: dict[int, Mapping[str, Any]] = {}
    for fallback, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        frame_number = int(frame.get("f", fallback))
        if frame_number < 0:
            raise ValueError(f"negative frame number {frame_number}")
        if frame_number in by_number:
            raise ValueError(f"duplicate annotation frame {frame_number}")
        by_number[frame_number] = frame
    if not by_number:
        raise ValueError("annotations.frames contains no frame objects")

    first_frame = min(by_number)
    last_frame = max(by_number)
    frame_numbers = np.arange(first_frame, last_frame + 1, dtype=np.int64)
    features = np.zeros((len(frame_numbers), len(FEATURE_NAMES)), dtype=np.float32)
    previous_ball: Sequence[float] | None = None

    for row, frame_number in enumerate(frame_numbers):
        frame = by_number.get(int(frame_number), {})
        ball = None if frame.get("ball_out_of_frame") else frame.get("ball")
        if not (
            isinstance(ball, (list, tuple))
            and len(ball) >= 2
            and np.isfinite(float(ball[0]))
            and np.isfinite(float(ball[1]))
        ):
            ball = None
        if frame.get("ball_new"):
            previous_ball = None
        frame_norm = (
            0.0 if last_frame == first_frame
            else (int(frame_number) - first_frame) / (last_frame - first_frame)
        )
        trajectory = build_trajectory_features(
            ball=ball,
            previous_ball=previous_ball,
            players=_players(frame),
            net_y=net_y,
            fps=fps,
            frame_norm=frame_norm,
            feature_schema=RESOLUTION_NORMALIZED_SCHEMA,
            frame_size=(width, height),
        )
        features[row, :10] = trajectory
        features[row, 10] = 1.0 if ball is not None else 0.0
        features[row, 11] = 1.0 if frame.get("ball_new") else 0.0
        pose_features = frame.get("pose_f")
        if (
            isinstance(pose_features, (list, tuple))
            and len(pose_features) == len(POSE_FEATURE_NAMES)
        ):
            features[row, 12:12 + len(POSE_FEATURE_NAMES)] = np.asarray(
                pose_features, dtype=np.float32,
            )
        poses = [
            pose for pose in (frame.get("poses") or [])
            if isinstance(pose, dict)
            and pose.get("team") in ("b", "team_b")
        ]
        prior_offset = 12 + len(POSE_FEATURE_NAMES)
        if poses:
            back_priors = [
                pose for pose in poses
                if str(pose.get("team_prior") or "").startswith(
                    ("back_facing_", "serve_back_facing_")
                )
            ]
            features[row, prior_offset] = len(back_priors) / len(poses)
            features[row, prior_offset + 1] = 1.0 if any(
                pose.get("team_prior") == "serve_back_facing_net_override"
                for pose in poses
            ) else 0.0
        if ball is not None:
            previous_ball = ball
        else:
            # Do not measure a future reappearance as one-frame velocity across
            # an occlusion or a detector track restart.
            previous_ball = None

    return frame_numbers, features


def weak_events_from_stats(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract weak lifecycle timestamps without pretending they are truth."""
    events: list[dict[str, Any]] = []
    attempts = stats.get("service_attempts") or stats.get("rallies") or []
    for attempt in attempts:
        lifecycle = attempt.get("lifecycle") if isinstance(attempt, dict) else None
        if not isinstance(lifecycle, dict):
            continue
        mapping = (
            ("serve_frame", "serve_contact"),
            ("reception_frame", "reception_contact"),
            ("followup_frame", "followup_contact"),
        )
        for field, event_type in mapping:
            frame = lifecycle.get(field)
            if frame is not None:
                events.append({"frame": int(frame), "type": event_type})
        end_frame = attempt.get("end_frame")
        if end_frame is None:
            end_frame = lifecycle.get("end_frame")
        if end_frame is not None:
            events.append({"frame": int(end_frame), "type": "rally_end"})
    return events


def load_human_events(path: Path) -> list[dict[str, Any]]:
    """Load and validate human timeline labels."""
    payload = _read_json(path)
    if isinstance(payload, dict):
        source = payload.get("label_source", HUMAN_LABEL_SOURCE)
        if normalize_label_source(source) != HUMAN_LABEL_SOURCE:
            raise ValueError(f"{path}: human labels file has label_source={source!r}")
        payload = payload.get("events")
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected an events list")
    return [dict(event) for event in payload]


def events_to_targets(
    events: Iterable[Mapping[str, Any]],
    frame_numbers: np.ndarray,
) -> tuple[np.ndarray, Counter[str]]:
    """Map sparse absolute-frame events to hard per-frame class targets."""
    targets = np.zeros((len(frame_numbers), len(EVENT_TYPES)), dtype=np.float32)
    first = int(frame_numbers[0])
    last = int(frame_numbers[-1])
    counts: Counter[str] = Counter()
    seen: set[tuple[int, str]] = set()
    for event in events:
        event_type = validate_event_type(str(event.get("type")))
        frame = int(event["frame"])
        key = (frame, event_type)
        if key in seen:
            continue
        seen.add(key)
        if frame < first or frame > last:
            raise ValueError(
                f"{event_type} at frame {frame} is outside annotation span "
                f"[{first}, {last}]"
            )
        targets[frame - first, EVENT_TO_INDEX[event_type]] = 1.0
        counts[event_type] += 1
    return targets, counts


def _record_id(source_video: str) -> str:
    stem = Path(source_video).stem or "video"
    digest = hashlib.sha1(source_video.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def prepare_manifest(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Prepare all videos and write the output dataset manifest."""
    payload = _read_json(manifest_path)
    videos = payload.get("videos") if isinstance(payload, dict) else None
    if not isinstance(videos, list) or not videos:
        raise ValueError("input manifest requires a non-empty videos list")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "videos"
    arrays_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    input_base = manifest_path.parent.resolve()

    for spec in videos:
        if not isinstance(spec, dict):
            raise ValueError("each videos entry must be an object")
        source_video = str(spec["source_video"])
        source = normalize_label_source(str(spec["label_source"]))
        annotations_path = _resolve(str(spec["annotations_path"]), input_base)
        stats_path = _resolve(str(spec["stats_path"]), input_base)
        annotations = _read_json(annotations_path)
        stats = _read_json(stats_path)
        frame_numbers, features = build_continuous_features(annotations)

        if source == HUMAN_LABEL_SOURCE:
            if not spec.get("labels_path"):
                raise ValueError(
                    f"{source_video}: human label_source requires labels_path"
                )
            events = load_human_events(
                _resolve(str(spec["labels_path"]), input_base)
            )
        else:
            events = weak_events_from_stats(stats)
        targets, counts = events_to_targets(events, frame_numbers)

        record_id = _record_id(source_video)
        relative_npz = Path("videos") / f"{record_id}.npz"
        np.savez_compressed(
            output_dir / relative_npz,
            features=features,
            targets=targets,
            frame_numbers=frame_numbers,
        )
        records.append({
            "id": record_id,
            "source_video": source_video,
            "source_key": str(Path(source_video)).replace("\\", "/").lower(),
            "label_source": source,
            "npz": relative_npz.as_posix(),
            "num_frames": int(len(frame_numbers)),
            "first_frame": int(frame_numbers[0]),
            "last_frame": int(frame_numbers[-1]),
            "fps": float(annotations["fps"]),
            "frame_width": int(annotations["frame_width"]),
            "frame_height": int(annotations["frame_height"]),
            "event_counts": dict(counts),
        })

    result = {
        "dataset_version": DATASET_VERSION,
        "feature_schema": RESOLUTION_NORMALIZED_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "event_types": list(EVENT_TYPES),
        "records": records,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_manifest(args.manifest.resolve(), args.output_dir.resolve())
    source_counts = Counter(record["label_source"] for record in result["records"])
    print(
        f"Prepared {len(result['records'])} videos in {args.output_dir} "
        f"(label sources: {dict(source_counts)})"
    )


if __name__ == "__main__":
    main()
