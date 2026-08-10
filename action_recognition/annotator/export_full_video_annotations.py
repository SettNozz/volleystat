"""Export a completed full-video annotation into reusable gold datasets.

The source project is treated as immutable.  This module derives rally and
score context, a multi-label clip manifest with post-contact context, exact
frame spotting labels, dense targets, and a non-destructive QC report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EVENT_TYPES = (
    "serve", "reception", "set", "dig", "attack", "block", "free_ball", "other"
)
PHASE_TYPES = ("unlabeled", "non_game", "between_rallies", "serve_flight", "rally_live")
TEAM_IDENTITIES = {"team_a": "opponent_team", "team_b": "observed_team"}
RECEPTION_QUALITIES = {
    "not_applicable", "unknown", "no_touch_ace", "error", "overpass",
    "negative", "positive", "perfect", "controlled",
}
LEGACY_EVENT_MAP = {
    "serve": "serve", "reception": "reception", "set": "pass", "dig": "pass",
    "attack": "attack", "block": "attack", "free_ball": "other", "other": "other",
}
SPOTTING_EVENT_MAP = {
    "serve": "serve_contact", "reception": "reception_contact", "set": "followup_contact",
    "dig": "followup_contact", "attack": "followup_contact", "block": "followup_contact",
    "free_ball": "followup_contact", "other": "followup_contact",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_video_annotations(project: Mapping[str, Any], video_id: str) -> tuple[dict, list[dict], list[dict]]:
    videos = [dict(v) for v in project.get("videos", []) if v.get("video_id") == video_id]
    if len(videos) != 1:
        raise ValueError(f"expected exactly one video metadata record for {video_id!r}")
    events = sorted(
        (dict(e) for e in project.get("annotations", []) if e.get("video_id") == video_id),
        key=lambda e: (int(e["frame"]), str(e.get("annotation_id", ""))),
    )
    segments = sorted(
        (dict(s) for s in project.get("segments", []) if s.get("video_id") == video_id),
        key=lambda s: (int(s["start_frame"]), int(s["end_frame"])),
    )
    if not events:
        raise ValueError(f"video {video_id!r} has no annotations")
    return videos[0], events, segments


def audit_annotations(events: list[dict], segments: list[dict]) -> list[dict]:
    """Return review items without mutating or rejecting the human truth."""
    issues: list[dict] = []
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        by_frame[int(event["frame"])].append(event)
        event_type = str(event.get("event_type"))
        if event_type not in EVENT_TYPES:
            issues.append({"severity": "error", "code": "unknown_event_type", "event": event})
        quality = str(event.get("reception_quality", "unknown"))
        if quality not in RECEPTION_QUALITIES:
            issues.append({"severity": "error", "code": "unknown_reception_quality", "event": event})
        if event_type == "reception" and quality == "not_applicable":
            issues.append({"severity": "warning", "code": "reception_quality_missing", "event": event})
        if event_type != "reception" and quality not in ("not_applicable", "unknown"):
            issues.append({"severity": "warning", "code": "quality_on_non_reception", "event": event})

    for frame, concurrent in sorted(by_frame.items()):
        if len(concurrent) <= 1:
            continue
        types = [str(e.get("event_type")) for e in concurrent]
        code = "duplicate_same_label" if len(types) != len(set(types)) else "concurrent_multi_label_contact"
        issues.append({
            "severity": "warning" if code == "duplicate_same_label" else "info",
            "code": code, "frame": frame,
            "annotation_ids": [e.get("annotation_id") for e in concurrent], "event_types": types,
        })

    unique_frames = sorted(by_frame)
    for left, right in zip(unique_frames, unique_frames[1:]):
        if right - left <= 2:
            issues.append({
                "severity": "review", "code": "nearby_contacts", "frames": [left, right],
                "event_types": [
                    [e.get("event_type") for e in by_frame[left]],
                    [e.get("event_type") for e in by_frame[right]],
                ],
            })

    for index, left in enumerate(segments):
        if int(left["end_frame"]) < int(left["start_frame"]):
            issues.append({"severity": "error", "code": "reversed_phase_segment", "segment": left})
        for right in segments[index + 1:]:
            if int(right["start_frame"]) > int(left["end_frame"]):
                break
            issues.append({
                "severity": "info" if left.get("match_phase") == right.get("match_phase") else "warning",
                "code": "overlapping_phase_segments",
                "segment_ids": [left.get("segment_id"), right.get("segment_id")],
                "overlap": [int(right["start_frame"]), min(int(left["end_frame"]), int(right["end_frame"]))],
            })

    for segment in segments:
        start, end = int(segment["start_frame"]), int(segment["end_frame"])
        inside = [e for e in events if start <= int(e["frame"]) <= end]
        for event in inside:
            distance = min(abs(int(event["frame"]) - start), abs(end - int(event["frame"])))
            issues.append({
                "severity": "info" if distance <= 2 else "review",
                "code": "contact_on_phase_boundary" if distance <= 2 else "contact_inside_no_contact_phase",
                "segment_id": segment.get("segment_id"), "annotation_id": event.get("annotation_id"),
                "frame": int(event["frame"]), "event_type": event.get("event_type"),
            })
    return issues


def deduplicate_exact_events(events: Iterable[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Collapse only identical frame/type/team labels for derived training data.

    The raw gold file remains unchanged. Different labels at one frame remain
    separate so genuine multi-label contacts are preserved.
    """
    kept: list[dict] = []
    duplicate_records: list[dict] = []
    seen: dict[tuple[int, str, str | None], dict] = {}
    for source in events:
        event = dict(source)
        key = (int(event["frame"]), str(event.get("event_type")), event.get("team_side"))
        if key not in seen:
            seen[key] = event
            kept.append(event)
            continue
        duplicate_records.append({
            "kept_annotation_id": seen[key].get("annotation_id"),
            "removed_from_derived_annotation_id": event.get("annotation_id"),
            "frame": key[0], "event_type": key[1], "team_side": key[2],
        })
    return kept, duplicate_records


def _possession_context(rally_id: str, events: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    possession_index = 0
    possession_team: str | None = None
    touch_index = 0
    for event in events:
        item = deepcopy(event)
        team = item.get("team_side")
        event_type = item.get("event_type")
        if team != possession_team:
            possession_index += 1
            possession_team = team
            touch_index = 0
        if event_type == "block":
            assigned_touch = 0
        elif event_type == "serve":
            assigned_touch = 0
        else:
            touch_index += 1
            assigned_touch = touch_index
        item.update({
            "rally_id": rally_id,
            "possession_id": f"{rally_id}_possession_{possession_index:03d}",
            "touch_index": assigned_touch,
            "stable_team_id": TEAM_IDENTITIES.get(team, team),
        })
        enriched.append(item)
    return enriched


def build_rallies(events: list[dict], segments: list[dict], last_frame: int) -> tuple[list[dict], list[dict]]:
    serve_positions = [i for i, event in enumerate(events) if event.get("event_type") == "serve"]
    if not serve_positions:
        raise ValueError("at least one serve is required to derive rallies")
    rallies: list[dict] = []
    enriched_events: list[dict] = []
    score = {team_id: 0 for team_id in TEAM_IDENTITIES.values()}

    for rally_number, event_index in enumerate(serve_positions, start=1):
        next_index = serve_positions[rally_number] if rally_number < len(serve_positions) else len(events)
        rally_events = events[event_index:next_index]
        serve = rally_events[0]
        next_serve = events[next_index] if next_index < len(events) else None
        next_serve_frame = int(next_serve["frame"]) if next_serve else last_frame + 1
        phase_starts = [
            int(s["start_frame"]) for s in segments
            if s.get("match_phase") == "between_rallies"
            and int(serve["frame"]) < int(s["start_frame"]) < next_serve_frame
        ]
        last_contact = max(int(e["frame"]) for e in rally_events)
        end_frame = min(phase_starts) if phase_starts else min(last_frame, last_contact + 1)
        rally_id = f"rally_{rally_number:04d}"
        contextual_events = _possession_context(rally_id, rally_events)

        if next_serve:
            winner_side = next_serve.get("team_side")
            winner_id = TEAM_IDENTITIES.get(winner_side, winner_side)
            score[winner_id] = score.get(winner_id, 0) + 1
            point_status, evidence_frame = "confirmed", int(next_serve["frame"])
        else:
            winner_side = winner_id = evidence_frame = None
            point_status = "unresolved"
        point = {
            "point_id": f"point_{rally_number:04d}", "rally_id": rally_id,
            "status": point_status, "winner_court_side": winner_side,
            "winner_team_id": winner_id, "evidence": "next_serve" if next_serve else "awaiting_next_serve",
            "evidence_frame": evidence_frame, "score_after": dict(score),
        }
        for event in contextual_events:
            event["point_id"] = point["point_id"]
            event["point_status"] = point_status
            event["point_winner_team_id"] = winner_id
        enriched_events.extend(contextual_events)
        rallies.append({
            "rally_id": rally_id, "start_frame": int(serve["frame"]), "end_frame": end_frame,
            "serving_court_side": serve.get("team_side"),
            "serving_team_id": TEAM_IDENTITIES.get(serve.get("team_side"), serve.get("team_side")),
            "receiving_court_side": "team_b" if serve.get("team_side") == "team_a" else "team_a",
            "num_annotated_events": len(contextual_events), "events": contextual_events,
            "point": point,
        })
    return rallies, enriched_events


def build_spotting_labels(rallies: list[dict]) -> dict:
    seen: set[tuple[int, str]] = set()
    labels: list[dict] = []
    for rally in rallies:
        for event in rally["events"]:
            label = SPOTTING_EVENT_MAP[event["event_type"]]
            key = (int(event["frame"]), label)
            if key in seen:
                continue
            seen.add(key)
            labels.append({
                "frame": key[0], "type": label, "label_source": "human",
                "rally_id": rally["rally_id"], "annotation_id": event.get("annotation_id"),
            })
        end_key = (int(rally["end_frame"]), "rally_end")
        if end_key not in seen:
            seen.add(end_key)
            labels.append({
                "frame": end_key[0], "type": "rally_end", "label_source": "human_derived",
                "rally_id": rally["rally_id"], "derived_from": "manual_between_rallies_or_last_contact",
            })
    labels.sort(key=lambda e: (e["frame"], e["type"]))
    return {"schema_version": 1, "label_source": "human", "events": labels}


def build_clip_manifest(
    video: Mapping[str, Any], rallies: list[dict], *, pre_frames: int, post_frames: int,
) -> dict:
    clips: list[dict] = []
    priority = {name: i for i, name in enumerate(("serve", "reception", "attack", "set", "dig", "block", "free_ball", "other"))}
    for rally in rallies:
        grouped: dict[int, list[dict]] = defaultdict(list)
        for event in rally["events"]:
            grouped[int(event["frame"])].append(event)
        for frame, frame_events in sorted(grouped.items()):
            full_types = sorted({str(e["event_type"]) for e in frame_events}, key=lambda x: priority.get(x, 99))
            legacy_types = sorted({LEGACY_EVENT_MAP[t] for t in full_types})
            primary = full_types[0]
            start = max(int(rally["start_frame"]), frame - pre_frames)
            end = min(int(rally["end_frame"]), frame + post_frames)
            clip_id = f"{video['video_id']}_{rally['rally_id']}_f{frame:06d}"
            representative = frame_events[0]
            clips.append({
                "clip_id": clip_id, "path": "", "source_video": video["path"],
                "source_stem": video["video_id"], "start_frame": start, "end_frame": end,
                "contact_frame": frame, "contact_offset": frame - start,
                "post_contact_frames": end - frame, "num_frames": end - start + 1,
                "primary_event_type": primary, "fsm_event_type": primary,
                "event_types": legacy_types, "full_event_types": full_types,
                "annotation_ids": [e.get("annotation_id") for e in frame_events],
                "team": representative.get("team_side"), "court_side": representative.get("team_side"),
                "stable_team_id": representative.get("stable_team_id"), "fps": float(video["fps"]),
                "rally_id": rally["rally_id"], "possession_ids": sorted({e["possession_id"] for e in frame_events}),
                "touch_indices": [e["touch_index"] for e in frame_events],
                "match_phase": representative.get("match_phase"),
                "physical_actions": sorted({str(e.get("physical_action")) for e in frame_events}),
                "context_roles": sorted({str(e.get("context_role")) for e in frame_events}),
                "is_service_reception": any(bool(e.get("is_service_reception")) for e in frame_events),
                "reception_qualities": sorted({str(e.get("reception_quality")) for e in frame_events}),
                "point_winner_team_id": rally["point"]["winner_team_id"],
                "point_status": rally["point"]["status"], "label_source": "human",
                "context_label_source": "human", "feature_schema": "resolution_normalized_v2",
            })
    return {
        "schema_version": 3,
        "label_source": "human",
        "window_policy": {"pre_contact_frames": pre_frames, "post_contact_frames": post_frames, "reason": "post-contact trajectory is required for reception quality"},
        "clips": clips,
    }


def build_dense_targets(
    events: list[dict], rallies: list[dict], segments: list[dict], num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_targets = np.zeros((num_frames, len(EVENT_TYPES)), dtype=np.uint8)
    phase_targets = np.zeros(num_frames, dtype=np.uint8)
    quality_targets = np.full(num_frames, -1, dtype=np.int8)
    event_index = {name: index for index, name in enumerate(EVENT_TYPES)}
    phase_index = {name: index for index, name in enumerate(PHASE_TYPES)}
    quality_names = sorted(RECEPTION_QUALITIES)
    quality_index = {name: index for index, name in enumerate(quality_names)}

    for rally in rallies:
        start, end = int(rally["start_frame"]), min(num_frames - 1, int(rally["end_frame"]))
        if start <= end:
            phase_targets[start:end + 1] = phase_index["rally_live"]
        first_reception = next((int(e["frame"]) for e in rally["events"] if e["event_type"] == "reception"), None)
        serve_end = min(end, first_reception - 1 if first_reception is not None else start)
        phase_targets[start:serve_end + 1] = phase_index["serve_flight"]
    # Manual no-contact segments take precedence except on exact human contacts.
    for segment in segments:
        phase = str(segment.get("match_phase"))
        if phase not in phase_index:
            continue
        start = max(0, int(segment["start_frame"]))
        end = min(num_frames - 1, int(segment["end_frame"]))
        phase_targets[start:end + 1] = phase_index[phase]
    for event in events:
        frame = int(event["frame"])
        if not 0 <= frame < num_frames or event.get("event_type") not in event_index:
            continue
        event_targets[frame, event_index[event["event_type"]]] = 1
        event_phase = "serve_flight" if event["event_type"] == "serve" else "rally_live"
        phase_targets[frame] = phase_index[event_phase]
        if event["event_type"] == "reception":
            quality_targets[frame] = quality_index.get(str(event.get("reception_quality")), -1)
    return event_targets, phase_targets, quality_targets


def export_clip_features_npz(
    source_annotations: Mapping[str, Any], clip_manifest: dict, output_dir: Path,
) -> dict:
    """Export lightweight 18-channel sequence features without Parquet."""
    from action_recognition.rally_spotting.prepare import build_continuous_features

    frame_numbers, spotting_features = build_continuous_features(source_annotations)
    # Spotting adds visibility/prior channels.  The event sequence contract is
    # the original 10 trajectory channels followed by the 8 pose channels.
    event_features = np.concatenate(
        (spotting_features[:, :10], spotting_features[:, 12:20]), axis=1
    ).astype(np.float32, copy=False)
    first_frame = int(frame_numbers[0])
    feature_dir = output_dir / "event_clip_features"
    feature_dir.mkdir(exist_ok=True)
    for clip in clip_manifest["clips"]:
        start = int(clip["start_frame"]) - first_frame
        end = int(clip["end_frame"]) - first_frame + 1
        if start < 0 or end > len(event_features):
            raise ValueError(f"clip {clip['clip_id']} is outside the feature timeline")
        relative_path = Path("event_clip_features") / f"{clip['clip_id']}.npz"
        np.savez_compressed(output_dir / relative_path, features=event_features[start:end])
        clip["features_path"] = relative_path.as_posix()
        clip["features_format"] = "npz"
    schema = {
        "feature_schema": "resolution_normalized_v2", "num_features": 18,
        "layout": "10 trajectory + 8 pose", "storage": "npz key=features",
        "contact_alignment": "contact_offset in event_clip_manifest.json",
    }
    _write_json(feature_dir / "feature_schema.json", schema)
    return schema


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    counts = report["counts"]
    score = report["score"]
    text = f"""# Full-video gold dataset report

Source video: `{report['video_id']}`  
Source annotation SHA-256: `{report['source_sha256']}`

## Reusable supervision

- Raw human event annotations: **{counts.get('raw_human_events', counts['events'])}**
- Derived event annotations after exact-duplicate collapse: **{counts['events']}**
- Unique contact frames / multi-label clips: **{counts['unique_contact_frames']}**
- Complete serve-led rallies: **{counts['rallies']}**
- Confirmed points from the next serve: **{counts['confirmed_points']}**
- Unresolved final point: **{counts['unresolved_points']}**
- Reception-quality labels: **{counts['receptions']}**
- Manually marked no-contact phase segments: **{counts['segments']}**
- Dense labeled video frames: **{counts['video_frames']}**

## Score confirmed by subsequent serves

- Observed / near-camera team: **{score.get('observed_team', 0)}**
- Opponent / far team: **{score.get('opponent_team', 0)}**

## Quality control

The source was not edited. Review items are stored in `qc_issues.json`.
Counts by severity: `{json.dumps(report['qc_by_severity'], ensure_ascii=False)}`.
Concurrent labels on one frame are preserved as multi-label examples.

## Split policy

All clips from this video share one source group. Never randomly split these
clips between train and validation/test; that would leak camera, players, and
adjacent frames. Use this whole video as one group.
"""
    path.write_text(text, encoding="utf-8")


def _write_qc_review(path: Path, issues: Iterable[Mapping[str, Any]], fps: float) -> None:
    actionable = [item for item in issues if item.get("severity") in ("error", "warning", "review")]
    lines = [
        "# Annotation review queue", "",
        "These are review suggestions only. The gold annotations were not changed.", "",
    ]
    for item in actionable:
        frames = item.get("frames") or [item.get("frame")]
        clean_frames = [int(frame) for frame in frames if frame is not None]
        times = [round(frame / fps, 3) for frame in clean_frames]
        lines.extend([
            f"- **{item.get('severity')} · {item.get('code')}** — frames "
            f"{clean_frames}, seconds {times}; event types: {item.get('event_types', item.get('event_type'))}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def export_full_video_annotations(
    project_path: Path, video_id: str, output_dir: Path, *, annotations_path: Path | None = None,
    stats_path: Path | None = None, pre_frames: int = 30, post_frames: int = 45,
) -> dict:
    project_path = project_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project = _read_json(project_path)
    video, raw_events, segments = load_video_annotations(project, video_id)

    source_annotations = None
    if annotations_path:
        source_annotations = _read_json(annotations_path.resolve())
        num_frames = len(source_annotations.get("frames") or [])
        if num_frames <= 0:
            raise ValueError("source annotations contain no frames")
    else:
        num_frames = max(int(e["frame"]) for e in raw_events) + 1
    last_frame = num_frames - 1

    qc = audit_annotations(raw_events, segments)
    derivation_events, exact_duplicates = deduplicate_exact_events(raw_events)
    rallies, events = build_rallies(derivation_events, segments, last_frame)
    spotting = build_spotting_labels(rallies)
    clips = build_clip_manifest(video, rallies, pre_frames=pre_frames, post_frames=post_frames)
    if source_annotations is not None:
        clips["feature_export"] = export_clip_features_npz(source_annotations, clips, output_dir)
    event_targets, phase_targets, quality_targets = build_dense_targets(events, rallies, segments, num_frames)

    _write_json(output_dir / "events_enriched.json", {"schema_version": 1, "events": events})
    _write_json(output_dir / "rallies_and_score.json", {
        "schema_version": 1, "team_identity_map": TEAM_IDENTITIES,
        "rallies": rallies, "score": rallies[-1]["point"]["score_after"],
    })
    _write_json(output_dir / "event_clip_manifest.json", clips)
    _write_json(output_dir / "spotting_labels.json", spotting)
    _write_json(output_dir / "phase_segments_raw.json", {"segments": segments})
    _write_json(output_dir / "qc_issues.json", {"source_unchanged": True, "issues": qc})
    _write_json(output_dir / "derived_exact_duplicates.json", {
        "source_unchanged": True, "policy": "collapsed only in derived datasets",
        "duplicates": exact_duplicates,
    })
    _write_qc_review(output_dir / "QC_REVIEW.md", qc, float(video["fps"]))
    quality_names = sorted(RECEPTION_QUALITIES)
    np.savez_compressed(
        output_dir / "dense_targets.npz", event_targets=event_targets,
        phase_targets=phase_targets, reception_quality_targets=quality_targets,
        frame_numbers=np.arange(num_frames, dtype=np.int64),
    )
    _write_json(output_dir / "dense_targets_schema.json", {
        "event_types": list(EVENT_TYPES), "phase_types": list(PHASE_TYPES),
        "reception_qualities": quality_names, "unlabeled_reception_quality_index": -1,
        "num_frames": num_frames,
    })

    if annotations_path and stats_path:
        _write_json(output_dir / "rally_spotting_source_manifest.json", {"videos": [{
            "source_video": video["path"], "annotations_path": str(annotations_path.resolve()),
            "stats_path": str(stats_path.resolve()), "label_source": "human",
            "labels_path": str((output_dir / "spotting_labels.json").resolve()),
        }]})

    event_counts = Counter(e["event_type"] for e in events)
    qc_counts = Counter(item["severity"] for item in qc)
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "video_id": video_id, "source_project": str(project_path),
        "source_sha256": _sha256(project_path), "source_unchanged": True,
        "counts": {
            "raw_human_events": len(raw_events), "events": len(events),
            "exact_duplicates_collapsed": len(exact_duplicates),
            "unique_contact_frames": len(clips["clips"]),
            "rallies": len(rallies), "confirmed_points": sum(r["point"]["status"] == "confirmed" for r in rallies),
            "unresolved_points": sum(r["point"]["status"] != "confirmed" for r in rallies),
            "receptions": event_counts["reception"], "segments": len(segments),
            "video_frames": num_frames, "events_by_type": dict(event_counts),
            "spotting_labels": len(spotting["events"]),
        },
        "score": rallies[-1]["point"]["score_after"], "qc_by_severity": dict(qc_counts),
        "artifacts": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
    }
    _write_json(output_dir / "dataset_report.json", report)
    _write_report(output_dir / "DATASET_REPORT.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations-path", type=Path)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--pre-frames", type=int, default=30)
    parser.add_argument("--post-frames", type=int, default=45)
    args = parser.parse_args()
    report = export_full_video_annotations(
        args.project, args.video_id, args.output_dir,
        annotations_path=args.annotations_path, stats_path=args.stats_path,
        pre_frames=args.pre_frames, post_frames=args.post_frames,
    )
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
