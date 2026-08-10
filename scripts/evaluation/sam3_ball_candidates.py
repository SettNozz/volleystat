"""Propose ball positions with SAM3 text segmentation (auto-label source).

Runs in the isolated Python 3.12 environment (``.sam3_env``) like the net-post
worker, and writes plain JSON so the production 3.11 pipeline can consume it.

A volleyball is ~20 px wide in a 720p broadcast frame, which is small for a
promptable segmenter, so the frame can be split into overlapping tiles that are
upscaled before being shown to SAM3 (``--tiles 2x2``).  Boxes are mapped back
to full-frame coordinates and merged with NMS.

Example (PowerShell):
  .\\.sam3_env\\Scripts\\python.exe scripts\\evaluation\\sam3_ball_candidates.py ^
    --video C:\\Users\\illya\\Downloads\\labeling_matherial\\3.mp4 ^
    --start 6000 --count 6 --tiles 2x2 --save-overlays
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def parse_tiles(spec: str) -> tuple[int, int]:
    if spec in ("", "1x1", "none"):
        return 1, 1
    cols, rows = spec.lower().split("x")
    return int(cols), int(rows)


def tile_rects(w: int, h: int, cols: int, rows: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles so a ball on a seam is whole in at least one tile."""
    if cols == 1 and rows == 1:
        return [(0, 0, w, h)]
    tw, th = w / cols, h / rows
    ox, oy = tw * overlap, th * overlap
    rects = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tw - ox))
            y1 = max(0, int(r * th - oy))
            x2 = min(w, int((c + 1) * tw + ox))
            y2 = min(h, int((r + 1) * th + oy))
            rects.append((x1, y1, x2, y2))
    return rects


def nms(boxes: list[list[float]], scores: list[float], iou_thr: float = 0.4) -> list[int]:
    if not boxes:
        return []
    arr = np.asarray(boxes, dtype=float)
    sc = np.asarray(scores, dtype=float)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    area = np.maximum(x2 - x1, 1e-6) * np.maximum(y2 - y1, 1e-6)
    order = sc.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (area[i] + area[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--prompt", default="volleyball")
    ap.add_argument("--tiles", default="1x1", help="e.g. 2x2, 3x2")
    ap.add_argument("--tile-overlap", type=float, default=0.12)
    ap.add_argument("--upscale", type=float, default=2.0, help="Tile upscale before SAM3")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max-size", type=float, default=60.0, help="Reject boxes wider than this (px, full frame)")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--output-dir", default=str(ROOT / "results" / "ball_eval"))
    args = ap.parse_args()

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    indices, frames = [], []
    idx = args.start
    while len(frames) < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - args.start) % args.stride == 0:
            indices.append(idx)
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit("No frames read")
    h, w = frames[0].shape[:2]

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise SystemExit("Run this with .sam3_env/Scripts/python.exe") from exc

    cols, rows = parse_tiles(args.tiles)
    rects = tile_rects(w, h, cols, rows, args.tile_overlap)
    print(f"{video.name}: {len(frames)} frames @ {w}x{h}, {len(rects)} tiles, prompt={args.prompt!r}")

    model = build_sam3_image_model(device="cuda")
    processor = Sam3Processor(model, device="cuda", confidence_threshold=args.conf)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame = []
    for fi, (frame_idx, frame) in enumerate(zip(indices, frames)):
        boxes: list[list[float]] = []
        scores: list[float] = []
        for (tx1, ty1, tx2, ty2) in rects:
            tile = frame[ty1:ty2, tx1:tx2]
            if args.upscale != 1.0:
                tile = cv2.resize(
                    tile, None, fx=args.upscale, fy=args.upscale, interpolation=cv2.INTER_CUBIC
                )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = processor.set_image(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)))
                output = processor.set_text_prompt(state=state, prompt=args.prompt)
            tb = output["boxes"].detach().float().cpu().numpy()
            ts = output["scores"].detach().float().cpu().numpy()
            for box, score in zip(tb, ts):
                x1, y1, x2, y2 = (float(v) / args.upscale for v in box)
                bw, bh = x2 - x1, y2 - y1
                if max(bw, bh) > args.max_size or max(bw, bh) < 4:
                    continue
                if bw <= 0 or bh <= 0 or max(bw, bh) / max(min(bw, bh), 1e-6) > 2.0:
                    continue  # a ball is round in the image
                boxes.append([x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1])
                scores.append(float(score))

        keep = nms(boxes, scores)
        dets = [
            {"box": [round(v, 1) for v in boxes[i]], "score": round(scores[i], 4)} for i in keep
        ]
        dets.sort(key=lambda d: -d["score"])
        per_frame.append({"frame": frame_idx, "candidates": dets})
        print(f"  f{frame_idx}: {len(dets)} candidates" + (f" top={dets[0]['score']:.2f}" if dets else ""))

        if args.save_overlays:
            vis = frame.copy()
            for d in dets:
                x1, y1, x2, y2 = (int(v) for v in d["box"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{d['score']:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"sam3_{video.stem}_f{frame_idx}.png"), vis)

    stem = f"{video.stem}_{args.start}_{args.start + args.count * args.stride}"
    path = out_dir / f"{stem}_sam3_{args.tiles}.json"
    path.write_text(
        json.dumps(
            {
                "video": str(video),
                "resolution": [w, h],
                "prompt": args.prompt,
                "tiles": args.tiles,
                "upscale": args.upscale,
                "conf": args.conf,
                "frames": per_frame,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    total = sum(len(f["candidates"]) for f in per_frame)
    nonempty = sum(1 for f in per_frame if f["candidates"])
    print(f"{total} candidates, {nonempty}/{len(per_frame)} frames non-empty -> {path}")


if __name__ == "__main__":
    main()
