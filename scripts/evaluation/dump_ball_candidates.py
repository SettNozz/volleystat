"""Dump raw per-frame ball candidates for offline evaluation.

Detection is expensive; scoring is not.  Running YOLO once per configuration
and saving every candidate lets recall/false-positive experiments (confidence
thresholds, ``BallTrackGate`` tuning, model comparison) run in seconds on the
saved JSON instead of minutes on the video, which is the same replay trick the
FSM work uses.

Example (PowerShell):
  .\\volley_env_311\\Scripts\\python.exe scripts\\evaluation\\dump_ball_candidates.py ^
    --video C:\\Users\\illya\\Downloads\\labeling_matherial\\3.mp4 ^
    --start 3000 --count 900 --imgsz 640 1280 1920
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models/yolov8_curated_training/yolov8n_volleyball_curated8/weights/best.pt"


def read_window(video: Path, start: int, count: int, stride: int) -> tuple[list[int], list]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    indices, frames = [], []
    idx = start
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start) % stride == 0:
            indices.append(idx)
            frames.append(frame)
        idx += 1
    cap.release()
    return indices, frames


def detect(model: YOLO, frames: list, imgsz: int, conf: float, iou: float, batch: int) -> list[list]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out: list[list] = []
    for i in range(0, len(frames), batch):
        chunk = frames[i : i + batch]
        results = model(
            chunk,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
            half=(device == "cuda"),
        )
        for res in results:
            boxes = res.boxes
            frame_dets = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].cpu().numpy())
                    frame_dets.append(
                        {
                            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "conf": round(float(box.conf[0].cpu().numpy()), 4),
                        }
                    )
            out.append(frame_dets)
    return out


def tile_rects(w: int, h: int, cols: int, rows: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles so a ball on a seam is whole in at least one tile."""
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    rects = []
    for r in range(rows):
        for c in range(cols):
            rects.append(
                (
                    max(0, int(c * tw - ox)),
                    max(0, int(r * th - oy)),
                    min(w, int((c + 1) * tw + ox)),
                    min(h, int((r + 1) * th + oy)),
                )
            )
    return rects


def detect_tiled(
    model: YOLO,
    frames: list,
    grid: tuple[int, int],
    imgsz: int,
    conf: float,
    iou: float,
    batch: int,
    overlap: float = 0.12,
) -> list[list]:
    """Slice each frame, detect per tile, map boxes back and merge.

    A 20 px ball in a 720p frame is near the floor of what a detector trained on
    downscaled 4K can see. Detecting on tiles keeps the ball at native size while
    the tile itself is what gets resized to ``imgsz`` — the same idea as SAHI.
    """
    h, w = frames[0].shape[:2]
    rects = tile_rects(w, h, grid[0], grid[1], overlap)
    merged: list[list] = [[] for _ in frames]
    for (tx1, ty1, tx2, ty2) in rects:
        tiles = [f[ty1:ty2, tx1:tx2] for f in frames]
        for fi, dets in enumerate(detect(model, tiles, imgsz, conf, iou, batch)):
            for d in dets:
                x1, y1, x2, y2 = d["box"]
                merged[fi].append(
                    {"box": [x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1], "conf": d["conf"]}
                )
    # Tiles overlap, so the same ball can be reported twice; keep the better one.
    for fi, dets in enumerate(merged):
        dets.sort(key=lambda d: -d["conf"])
        kept: list[dict] = []
        for d in dets:
            cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
            if any(
                abs(cx - (k["box"][0] + k["box"][2]) / 2) < 15
                and abs(cy - (k["box"][1] + k["box"][3]) / 2) < 15
                for k in kept
            ):
                continue
            kept.append(d)
        merged[fi] = kept
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=900, help="Number of frames to dump")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280, 1920])
    ap.add_argument(
        "--tiled",
        nargs="*",
        default=[],
        metavar="COLSxROWS@IMGSZ",
        help="Extra sliced runs, e.g. 2x2@640 3x2@640",
    )
    ap.add_argument("--conf", type=float, default=0.05, help="Dump low so thresholds can be swept offline")
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--output-dir", default=str(ROOT / "results" / "ball_eval"))
    args = ap.parse_args()

    video = Path(args.video)
    indices, frames = read_window(video, args.start, args.count, args.stride)
    if not frames:
        raise SystemExit("No frames read")
    h, w = frames[0].shape[:2]
    print(f"{video.name}: {len(frames)} frames @ {w}x{h}, indices {indices[0]}..{indices[-1]}")

    model = YOLO(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "video": str(video),
        "model": args.model,
        "resolution": [w, h],
        "start": args.start,
        "stride": args.stride,
        "conf_floor": args.conf,
        "iou": args.iou,
        "frame_indices": indices,
        "runs": {},
    }
    for imgsz in args.imgsz:
        t0 = time.time()
        dets = detect(model, frames, imgsz, args.conf, args.iou, args.batch)
        dt = time.time() - t0
        n = sum(len(d) for d in dets)
        with_any = sum(1 for d in dets if d)
        print(
            f"imgsz={imgsz:5d}: {n:5d} candidates, {with_any}/{len(dets)} frames non-empty, "
            f"{dt:.1f}s ({len(frames)/dt:.1f} fps)"
        )
        payload["runs"][str(imgsz)] = {"seconds": round(dt, 2), "detections": dets}

    for spec in args.tiled:
        grid_spec, _, imgsz_spec = spec.partition("@")
        cols, rows = (int(v) for v in grid_spec.lower().split("x"))
        imgsz = int(imgsz_spec) if imgsz_spec else 640
        t0 = time.time()
        dets = detect_tiled(model, frames, (cols, rows), imgsz, args.conf, args.iou, args.batch)
        dt = time.time() - t0
        n = sum(len(d) for d in dets)
        with_any = sum(1 for d in dets if d)
        print(
            f"tiled {cols}x{rows}@{imgsz}: {n:5d} candidates, {with_any}/{len(dets)} frames non-empty, "
            f"{dt:.1f}s ({len(frames)/dt:.1f} fps)"
        )
        payload["runs"][f"{cols}x{rows}@{imgsz}"] = {"seconds": round(dt, 2), "detections": dets}

    # The model name is part of the stem so runs of different weights coexist.
    model_tag = Path(args.model).parent.parent.name or Path(args.model).stem
    stem = f"{video.stem}_{args.start}_{args.start + args.count * args.stride}_{model_tag}"
    path = out_dir / f"{stem}_candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
