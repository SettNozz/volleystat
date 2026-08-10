#!/usr/bin/env python3
"""Import a predicted contact manifest into the full-video review project.

The event classifier operates on contact-centred windows.  Serve/reception
contacts supplied by the rally lifecycle are kept, while later rally contacts
are constrained to pass/attack before choosing the larger model probability.
The raw, unconstrained prediction remains attached to every annotation so a
later human review can measure both model-only and FSM-gated performance.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from action_recognition.annotator.full_video_annotator import EVENT_PRESETS


CORE_EVENTS = ("serve", "reception", "pass", "attack")
IN_RALLY_EVENTS = ("pass", "attack")
UI_EVENT_TYPES = {"serve": "serve", "reception": "reception", "pass": "set", "attack": "attack"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_for_clip(clip: Mapping[str, Any]) -> tuple[str, str]:
    labels = list(clip.get("event_types") or [])
    if clip.get("label_source") == "fsm_rule" and labels:
        label = str(labels[0])
        if label in ("serve", "reception"):
            return label, "fsm_rule"

    probabilities = clip.get("model_probs") or {}
    if all(label in probabilities for label in IN_RALLY_EVENTS):
        label = max(IN_RALLY_EVENTS, key=lambda item: float(probabilities[item]))
        return label, "model_fsm_gated"

    label = str(clip.get("model_prediction") or (labels[0] if labels else ""))
    if label not in CORE_EVENTS:
        raise ValueError(f"clip {clip.get('clip_id')}: no usable prediction")
    return label, "model"


def build_annotations(
    manifest: Mapping[str, Any], video_id: str, fps: float,
) -> list[dict[str, Any]]:
    annotations = []
    seen_frames: set[int] = set()
    for clip in sorted(manifest.get("clips", []), key=lambda item: item["contact_frame"]):
        frame = int(clip["contact_frame"])
        if frame in seen_frames:
            raise ValueError(f"duplicate contact frame {frame}")
        seen_frames.add(frame)

        team = str(clip.get("team") or clip.get("court_side") or "")
        if team not in ("team_a", "team_b"):
            raise ValueError(f"clip {clip.get('clip_id')}: invalid team {team!r}")
        model_event_type, source = _event_for_clip(clip)
        event_type = UI_EVENT_TYPES[model_event_type]
        preset = EVENT_PRESETS[event_type]
        quality = (
            str(clip.get("reception_quality") or "unknown")
            if model_event_type == "reception" else "not_applicable"
        )
        if model_event_type == "reception" and quality == "not_applicable":
            quality = "unknown"
        probabilities = {
            label: float(value)
            for label, value in (clip.get("model_probs") or {}).items()
            if label in CORE_EVENTS
        }
        raw_prediction = clip.get("model_prediction")
        raw_confidence = (
            max(probabilities.values()) if probabilities else None
        )
        gated_denominator = sum(probabilities.get(label, 0.0) for label in IN_RALLY_EVENTS)
        gated_confidence = (
            probabilities.get(model_event_type, 0.0) / gated_denominator
            if source == "model_fsm_gated" and gated_denominator > 0 else raw_confidence
        )

        annotations.append({
            "annotation_id": f"ann_model_{video_id}_{frame}",
            "video_id": video_id,
            "frame": frame,
            "time_seconds": frame / fps,
            "team_side": team,
            "event_type": event_type,
            "match_phase": preset["match_phase"],
            "physical_action": preset["physical_action"],
            "context_role": preset["context_role"],
            "reception_quality": quality,
            "is_service_reception": model_event_type == "reception",
            "label_source": source,
            "model_event_type": model_event_type,
            "raw_model_prediction": raw_prediction,
            "raw_model_confidence": raw_confidence,
            "fsm_gated_confidence": gated_confidence,
            "model_probs": probabilities,
            "fsm_event_type": clip.get("fsm_event_type"),
            "candidate_source": clip.get("candidate_source"),
            "created_at": _now(),
        })
    return annotations


def import_predictions(
    project_path: Path,
    manifest_path: Path,
    source_video: Path,
    video_id: str | None = None,
) -> dict[str, Any]:
    project = _read(project_path)
    manifest = _read(manifest_path)
    clips = list(manifest.get("clips") or [])
    if not clips:
        raise ValueError("predicted manifest has no clips")

    source_video = source_video.resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    video_id = video_id or str(clips[0].get("source_stem") or source_video.stem)
    fps = float(clips[0].get("fps") or 30.0)
    annotations = build_annotations(manifest, video_id, fps)

    project.setdefault("videos", [])
    video = next(
        (item for item in project["videos"] if item.get("video_id") == video_id),
        None,
    )
    if video is None:
        project["videos"].append({
            "video_id": video_id,
            "name": video_id,
            "path": str(source_video),
            "fps": fps,
            "legacy_contact_count": len(annotations),
            "label_source": "model_fsm_gated",
        })
    else:
        video.update(path=str(source_video), fps=fps)

    existing = project.setdefault("annotations", [])
    preserved = [
        annotation for annotation in existing
        if annotation.get("video_id") != video_id
        or annotation.get("label_source") == "human"
    ]
    human_frames = {
        int(annotation["frame"])
        for annotation in preserved
        if annotation.get("video_id") == video_id
        and annotation.get("label_source") == "human"
    }
    annotations = [item for item in annotations if item["frame"] not in human_frames]
    project["annotations"] = sorted(
        preserved + annotations,
        key=lambda item: (str(item.get("video_id")), int(item.get("frame", 0))),
    )
    project["updated_at"] = _now()

    backup = project_path.with_name(
        f"{project_path.stem}_before_{video_id}_model_import_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{project_path.suffix}"
    )
    shutil.copy2(project_path, backup)
    temporary = project_path.with_suffix(project_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    temporary.replace(project_path)

    return {
        "video_id": video_id,
        "annotations": len(annotations),
        "classes": dict(Counter(item["event_type"] for item in annotations)),
        "sources": dict(Counter(item["label_source"] for item in annotations)),
        "backup": str(backup.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--video-id")
    args = parser.parse_args()
    report = import_predictions(
        args.project.resolve(), args.manifest.resolve(), args.source_video,
        args.video_id,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
