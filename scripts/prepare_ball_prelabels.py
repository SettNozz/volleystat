#!/usr/bin/env python3
"""Run the current ball detector + global trajectory solver over a video queue.

This is deliberately a *pre-label* job. Model predictions are written to fresh
result directories and never overwrite human ball boxes, event labels, or the
source manifest.

Input manifest::

    {
      "videos": [
        {
          "id": "camera_01",
          "path": "C:/videos/camera_01.webm",
          "required_width": 3840,
          "required_height": 2160,
          "segments": [
            {"start_sec": 0, "end_sec": 165, "label": "prematch_negative"}
          ]
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.game_analyzer import GameAnalyzer


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("manifest requires a non-empty 'videos' list")
    return videos


def _video_metadata(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": fps,
            "frames": frames,
            "duration_sec": frames / fps if fps > 0 else 0.0,
        }
    finally:
        cap.release()


def _validate_video(record: dict[str, Any], path: Path) -> dict[str, Any]:
    metadata = _video_metadata(path)
    required_width = int(record.get("required_width") or 0)
    required_height = int(record.get("required_height") or 0)
    if required_width and metadata["width"] != required_width:
        raise ValueError(
            f"{path.name}: expected width {required_width}, got {metadata['width']}"
        )
    if required_height and metadata["height"] != required_height:
        raise ValueError(
            f"{path.name}: expected height {required_height}, got {metadata['height']}"
        )
    return metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare non-destructive ball-coordinate pre-labels",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ball-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "queue_status.json"
    records = _read_manifest(manifest_path)

    queue_state: dict[str, Any] = {
        "status": "initializing",
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "total_videos": len(records),
        "completed_videos": 0,
        "failed_videos": 0,
        "videos": [],
        "updated_at": time.time(),
    }
    _write_json(status_path, queue_state)

    active: dict[str, Any] = {"id": None, "last_write": 0.0}

    def progress(update: dict[str, Any]) -> None:
        now = time.time()
        if now - active["last_write"] < 1.0 and update.get("phase") != "done":
            return
        active["last_write"] = now
        queue_state["status"] = "running"
        queue_state["current_video"] = active["id"]
        queue_state["current_progress"] = update
        queue_state["updated_at"] = now
        _write_json(status_path, queue_state)

    analyzer = GameAnalyzer(
        calibration_path=None,
        output_video=False,
        skip_frames=0,
        ball_trajectory_dp=True,
        ball_batch_size=max(1, int(args.ball_batch_size)),
        event_checkpoint_path=None,
        refresh_net_floor=False,
        enable_pose=False,
        enable_net_detection=False,
        progress_callback=progress,
    )

    review_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        video_id = str(record.get("id") or Path(record["path"]).stem)
        source = Path(record["path"])
        if not source.is_absolute():
            source = (manifest_path.parent / source).resolve()
        active["id"] = video_id
        item: dict[str, Any] = {
            "id": video_id,
            "source_video": str(source),
            "segments": list(record.get("segments") or []),
        }
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            item["video"] = _validate_video(record, source)
            target = output_root / video_id
            annotations = target / f"{source.stem}_annotations.json"
            stats = target / f"{source.stem}_stats.json"
            if args.force or not (annotations.exists() and stats.exists()):
                analyzer.process_video(str(source), output_dir=str(target))
            else:
                item["skipped"] = "existing complete result"
            item.update(
                status="ready_for_eye_check",
                annotations_path=str(annotations),
                stats_path=str(stats),
            )
            queue_state["completed_videos"] += 1
        except Exception as exc:
            item.update(status="error", error=str(exc))
            queue_state["failed_videos"] += 1
        queue_state["videos"].append(item)
        queue_state["current_index"] = index
        queue_state["updated_at"] = time.time()
        _write_json(status_path, queue_state)
        review_records.append(item)

    queue_state["status"] = (
        "done" if queue_state["failed_videos"] == 0 else "done_with_errors"
    )
    queue_state["current_video"] = None
    queue_state["current_progress"] = None
    queue_state["updated_at"] = time.time()
    _write_json(status_path, queue_state)
    _write_json(
        output_root / "review_manifest.json",
        {
            "label_source": "model_prelabel",
            "warning": (
                "Predicted ball coordinates require human eye-check and must "
                "not overwrite manual ground truth."
            ),
            "videos": review_records,
        },
    )
    return 0 if queue_state["failed_videos"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
