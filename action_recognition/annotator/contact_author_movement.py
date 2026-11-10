"""Review short pieces of observed player tracks against a human photo roster."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from src.tracking.player_contacts import side


SCHEMA = "human_player_movement_v1"


@lru_cache(maxsize=2)
def movement_segments(path, source_sha256, target_side, fps):
    """Cut raw tracks at missing frames, ambiguous raw IDs and six seconds."""
    frames = json.loads(Path(path).read_bytes())["frames"]
    max_length = max(1, round(fps * 6))
    segments = {}
    active = {}
    for row in frames:
        frame = row["f"]
        poses = row.get("poses") or []
        ids = Counter(pose.get("id") for pose in poses)
        next_active = {}
        for index, pose in enumerate(poses):
            if side(pose.get("team")) != target_side or not pose.get("bb"):
                continue
            raw_id = pose.get("id")
            unique = raw_id is not None and ids[raw_id] == 1
            segment = active.get(raw_id) if unique else None
            if segment is None or frame != segment["end"] + 1 or len(segment["boxes"]) >= max_length:
                key = f"{frame}:{index}"
                segment = {"id": key, "start": frame, "end": frame, "raw_track_id": raw_id,
                           "boxes": [], "pose_indices": [], "predictions": []}
                segments[key] = segment
            segment["end"] = frame
            segment["boxes"].append(pose["bb"])
            segment["pose_indices"].append(index)
            segment["predictions"].append(pose.get("player_id"))
            if unique:
                next_active[raw_id] = segment
        active = next_active
    return segments


def segments_for(metadata):
    source = metadata.get("annotations_source")
    if not source:
        raise ValueError("Saved poses are unavailable for this video")
    return movement_segments(source["path"], source["sha256"], metadata["target_side"], metadata["fps"])


def segment_queue(segments):
    queue = []
    for segment in segments.values():
        predictions = segment["predictions"]
        changes = sum(a != b for a, b in zip(predictions, predictions[1:]))
        queue.append({"id": segment["id"], "start": segment["start"], "end": segment["end"],
                      "priority": [changes, predictions.count(None) / len(predictions)]})
    return queue


def movement_export(project, state):
    videos = {}
    for video_id, reviewed in state.get("movement_reviews", {}).items():
        segments = segments_for(project["videos"][video_id])
        rows = []
        for key, spans in reviewed.items():
            segment = segments[key]
            for span in spans:
                start, end = span["start"], span["end"]
                offset = segment["start"]
                rows.append({**span, "segment_id": key, "raw_track_id": segment["raw_track_id"],
                             "pose_indices": segment["pose_indices"][start-offset:end-offset+1]})
        seen = set()
        for row in sorted(rows, key=lambda item: item["revision"]):
            if row["decision"] != "player":
                continue
            duplicates = []
            for frame in range(row["start"], row["end"] + 1):
                key = (row["player_id"], frame)
                if key in seen:
                    duplicates.append(frame)
                seen.add(key)
            if duplicates:
                row["duplicate_frames"] = duplicates
        videos[video_id] = {"segments": len(segments), "spans": rows,
                            "observations": sum(len(s["boxes"]) for s in segments.values()),
                            "reviewed_observations": sum(row["end"] - row["start"] + 1 for row in rows)}
    return {"schema": SCHEMA, "scope": "observed cached poses on the target court side",
            "pose_index_scope": "zero-based index in source frame poses; one index per frame from start through end",
            "duplicate_frame_scope": "same person, additional pose; omit these frames from one-pose-per-player metrics",
            "videos": videos}
