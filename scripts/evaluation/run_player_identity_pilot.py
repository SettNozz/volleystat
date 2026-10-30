"""Cache saved player crops and compare roster association on identical detections."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.tracking.player_identity import SCHEMA, appearance, compare, default_seed, overlap


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    # Exclusive create: experiments never overwrite prior artifacts.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def prepare(args: argparse.Namespace) -> None:
    annotations_path = args.annotations.resolve()
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh versioned directory")
    if args.stride < 1 or args.roster_size < 1 or args.start_frame < 0:
        raise ValueError("Stride/roster size must be positive; start frame nonnegative")
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    raw_video = args.video or annotations.get("original_video_path")
    if not raw_video:
        raise ValueError("No original video recorded; pass --video")
    video = Path(raw_video).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    frames = [f for f in annotations["frames"] if f["f"] >= args.start_frame
              and (args.end_frame is None or f["f"] < args.end_frame)
              and (f["f"]-args.start_frame) % args.stride == 0]
    if not frames:
        raise ValueError("Requested window contains no saved frames")
    fps = float(annotations["fps"])
    if fps <= 0:
        raise ValueError("Invalid annotation FPS")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (
            annotations["frame_width"], annotations["frame_height"]):
            raise ValueError("Video dimensions differ from saved analysis")
        if abs(cap.get(cv2.CAP_PROP_FPS)-fps) > .05:
            raise ValueError("Video FPS differs from saved analysis")
        cached, current = [], -1
        for n, saved in enumerate(frames):
            target = int(saved["f"])
            while current < target:
                if not cap.grab():
                    raise ValueError(f"Video ends before source frame {target}")
                current += 1
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode source frame {target}")
            players = []
            all_poses = saved.get("poses") or []
            for i, pose in enumerate(all_poses):
                if pose.get("team") not in (args.side, f"team_{args.side}"):
                    continue
                box = pose["bb"]
                if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"Invalid box at frame {target}, pose {i}")
                features = appearance(image, box)
                players.append({"key": f"{target}:{i}", "pose_index": i,
                                "bb": box, "raw_id": pose.get("id"),
                                "detection_source": pose.get("source", "yolo_pose"),
                                "appearance": [round(v, 6) for v in features] if features is not None else None,
                                "occlusion": max((overlap(box, q["bb"]) for j, q in enumerate(all_poses) if j != i), default=0.)})
            cached.append({"f": target, "players": players})
            if n % 100 == 0:
                print(f"[identity] cached {n+1}/{len(frames)} frames", flush=True)
    finally:
        cap.release()
    observations = {"schema": SCHEMA, "fps": fps, "side": args.side,
                    "width": annotations["frame_width"], "height": annotations["frame_height"],
                    "roster_size": args.roster_size, "stride": args.stride,
                    "source": {"annotations": str(annotations_path), "annotations_sha256": digest(annotations_path),
                               "video": str(video), "video_size": video.stat().st_size,
                               "video_mtime_ns": video.stat().st_mtime_ns,
                               "appearance_schema": "three_body_band_hsv_root_hist_v1"}, "frames": cached}
    seed = default_seed(cached, args.roster_size)
    result = compare(observations, seed)
    args.output.mkdir(parents=True)
    write_json(args.output/"observations.json", observations)
    write_json(args.output/"comparison_0000.json", result)
    print_summary(result)
    print(f"[identity] output: {args.output.resolve()}", flush=True)


def print_summary(result: dict) -> None:
    print(f"[identity] seed frame {result['seed']['frame']}; source={result['seed']['label_source']}")
    for method, run in result["methods"].items():
        print(f"[identity] {method}: {json.dumps({'diagnostics': run['diagnostics'], 'human_checks': run['human_checks']})}")


def rerun(project: Path) -> Path:
    observations_path = project/"observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    labels_path = project/"labels.json"
    seed, checks = default_seed(observations["frames"], observations["roster_size"]), []
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if labels.get("observations_sha256") != digest(observations_path):
            raise ValueError("Observations changed; labels belong to a different cache")
        seed = labels.get("seed") or seed
        checks = labels.get("checks", [])
    result = compare(observations, seed, checks)
    index = 0
    while (target := project/f"comparison_{index:04d}.json").exists():
        index += 1
    write_json(target, result)
    print_summary(result)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Decode original video, cache same saved detections, run three controls")
    prep.add_argument("--annotations", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True, help="New versioned directory; must not exist")
    prep.add_argument("--video", type=Path)
    prep.add_argument("--side", choices=("a", "b"), default="b")
    prep.add_argument("--roster-size", type=int, default=6)
    prep.add_argument("--stride", type=int, default=3)
    prep.add_argument("--start-frame", type=int, default=0)
    prep.add_argument("--end-frame", type=int, help="Exclusive source frame")
    replay = sub.add_parser("compare", help="Rerun from human seed; score separate human checks")
    replay.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        print(rerun(args.project))


if __name__ == "__main__":
    main()
