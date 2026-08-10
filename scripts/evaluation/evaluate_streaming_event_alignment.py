"""Replay a streaming event checkpoint over saved, DP-cleaned annotations.

This separates model quality from first-pass ball noise: GameAnalyzer predicts
events before the global trajectory solver runs, while the saved FSM events are
recomputed afterwards from the cleaned ball path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.inference.streaming_event_predictor import (
    StreamingEventPredictor,
)


ACTIVE_STATES = {"serving", "rally_active"}


def _teams(annotation: dict) -> dict[str, list[tuple[float, float]]]:
    teams = {"team_a": [], "team_b": []}
    for pose in annotation.get("poses") or []:
        bbox = pose.get("bb") or []
        if len(bbox) != 4:
            continue
        point = ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))
        team = "team_a" if pose.get("team") == "a" else "team_b"
        teams[team].append(point)
    if not teams["team_a"] and not teams["team_b"]:
        teams["team_a"] = [tuple(map(float, p)) for p in annotation.get("p") or []]
    return teams


def replay(annotations: dict, checkpoint: str) -> list[dict]:
    predictor = StreamingEventPredictor(checkpoint)
    fps = float(annotations.get("fps") or 30.0)
    frame_size = (
        float(annotations.get("frame_width") or 3840),
        float(annotations.get("frame_height") or 2160),
    )
    net_y = float(annotations["net_y"])
    cooldown = int(round(8.0 * fps))
    sticky = int(round(2.0 * fps))
    last_serve = -cooldown
    last_label = None
    last_label_frame = -sticky
    replayed = []

    for ann in annotations.get("frames") or []:
        frame = int(ann["f"])
        state = ann.get("state") or ""
        ball = tuple(map(float, ann["ball"])) if ann.get("ball") else None
        raw = predictor.add_frame(
            ball,
            _teams(ann),
            net_y,
            fps,
            fsm_state=state,
            pose_features=ann.get("pose_f"),
            frame_size=frame_size,
        )
        display = None
        if raw is not None and state in ACTIVE_STATES:
            if raw == "serve":
                if frame - last_serve >= cooldown:
                    last_serve = frame
                    last_label = raw
                    last_label_frame = frame
                    display = raw
            else:
                last_label = raw
                last_label_frame = frame
                display = raw
        elif (
            raw is not None
            and state not in ACTIVE_STATES
            and last_label
            and last_label != "serve"
            and frame - last_label_frame < sticky
        ):
            display = last_label
        replayed.append({"frame": frame, "label": display, "raw": raw})
    return replayed


def summarize(replayed: list[dict], events: list[dict]) -> dict:
    segments = []
    current = None
    for row in replayed:
        label, frame = row["label"], row["frame"]
        if label:
            if not current or current["label"] != label or frame != current["end"] + 1:
                if current:
                    segments.append(current)
                current = {"label": label, "start": frame, "end": frame}
            else:
                current["end"] = frame
        elif current:
            segments.append(current)
            current = None
    if current:
        segments.append(current)

    within_15 = Counter()
    offsets: dict[str, list[int]] = {}
    for segment in segments:
        segment["duration"] = segment["end"] - segment["start"] + 1
        middle = round((segment["start"] + segment["end"]) / 2)
        matches = [event for event in events if event.get("type") == segment["label"]]
        if matches:
            nearest = min(matches, key=lambda event: abs(int(event["frame"]) - middle))
            offset = middle - int(nearest["frame"])
            segment["nearest_fsm"] = int(nearest["frame"])
            segment["offset_mid"] = offset
            offsets.setdefault(segment["label"], []).append(abs(offset))
            if abs(offset) <= 15:
                within_15[segment["label"]] += 1

    alignment = {}
    for label, values in offsets.items():
        ordered = sorted(values)
        alignment[label] = {
            "segments": len(values),
            "within_15_frames": within_15[label],
            "median_abs_offset_frames": ordered[len(ordered) // 2],
        }
    return {
        "model_frames": sum(row["label"] is not None for row in replayed),
        "segments": segments,
        "alignment": alignment,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    summary = summarize(replay(annotations, args.checkpoint), stats.get("all_events") or [])
    print(json.dumps(summary["alignment"], indent=2))
    print(f"model_frames={summary['model_frames']} segments={len(summary['segments'])}")
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
