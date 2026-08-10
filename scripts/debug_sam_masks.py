#!/usr/bin/env python3
"""
Debug SAM segmentation: show raw masks for each sampled frame.

Usage:
    python scripts/debug_sam_masks.py \
        --video path/to/video.mp4 \
        --calibration path/to/court_calibration.json
"""

import argparse
import base64
import json
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_SAM_CHECKPOINT = str(Path.home() / "Downloads" / "sam_vit_h_4b8939.pth")
_SAM_MODEL_TYPE = "vit_h"
_TARGET_H = 720
_TARGET_W = 1280
_SAMPLE_COUNT = 5
_FIRST_N_SEC = 5.0
_NET_HEIGHT_HINT = 60.0  # px at 720p

MASK_COLORS = [
    (255, 80,  80),   # red
    (80,  255, 80),   # green
    (80,  80,  255),  # blue
]


def load_sam(checkpoint):
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({_SAM_MODEL_TYPE}) on {device} ...")
    sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=checkpoint)
    sam.to(device)
    print("SAM ready")
    return SamPredictor(sam)


def extract_frames(video_path, n, first_n_sec):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frame = int(first_n_sec * fps)
    step = max(1, max_frame // n)
    frames, timestamps = [], []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (_TARGET_W, _TARGET_H)))
        timestamps.append(i * step / fps)
    cap.release()
    return frames, timestamps


def run_sam_on_frame(predictor, frame_bgr, box):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    # multimask_output=True → get all 3 candidate masks
    masks, scores, _ = predictor.predict(box=box, multimask_output=False)
    # Also run point prompts for comparison (center of box)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    pt_masks, pt_scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]], dtype=np.float32),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    return masks, scores, pt_masks, pt_scores


def overlay_mask(img, mask, color, alpha=0.45):
    overlay = img.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_box(img, box, color=(255, 255, 0), thickness=2):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def label(img, text, x, y, color=(255, 255, 255)):
    cv2.putText(img, text, (x, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, max(y, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)


def render_frame_variants(frame_bgr, ts, box, masks, scores, pt_masks, pt_scores):
    """
    Returns list of (title, annotated_image) for one source frame.
    Shows: box mask, 3 point masks side by side.
    """
    results = []

    # --- Box mask ---
    img = frame_bgr.copy()
    draw_box(img, box, (255, 220, 0), 2)
    m = masks[0]
    ys = np.where(m.any(axis=1))[0]
    top_y = int(ys.min()) if len(ys) else -1
    bot_y = int(ys.max()) if len(ys) else -1
    img = overlay_mask(img, m, (0, 140, 255))
    if top_y >= 0:
        cv2.line(img, (0, top_y), (_TARGET_W, top_y), (0, 80, 255), 2)
        cv2.line(img, (0, bot_y), (_TARGET_W, bot_y), (0, 40, 200), 2)
    label(img, f"BOX  score={scores[0]:.3f}  top={top_y} bot={bot_y} h={bot_y-top_y}",
          6, 20, (255, 220, 0))
    results.append((f"t={ts:.2f}s — BOX prompt", img))

    # --- Point masks (3 candidates) ---
    for i, (pm, ps) in enumerate(zip(pt_masks, pt_scores)):
        img2 = frame_bgr.copy()
        draw_box(img2, box, (100, 100, 100), 1)
        ys2 = np.where(pm.any(axis=1))[0]
        ty = int(ys2.min()) if len(ys2) else -1
        by = int(ys2.max()) if len(ys2) else -1
        img2 = overlay_mask(img2, pm, MASK_COLORS[i % len(MASK_COLORS)])
        if ty >= 0:
            cv2.line(img2, (0, ty), (_TARGET_W, ty), MASK_COLORS[i % len(MASK_COLORS)], 2)
            cv2.line(img2, (0, by), (_TARGET_W, by), MASK_COLORS[i % len(MASK_COLORS)], 1)
        # Draw point
        cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        cv2.circle(img2, (cx, cy), 6, (255, 255, 0), -1)
        label(img2, f"POINT mask={i}  score={ps:.3f}  top={ty} bot={by} h={by-ty}",
              6, 20, (255, 255, 100))
        results.append((f"t={ts:.2f}s — POINT mask {i}", img2))

    return results


def to_b64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def build_html(panels):
    cards = ""
    for title, img in panels:
        b64 = to_b64(img)
        cards += f"""
        <div class="card">
            <div class="title">{title}</div>
            <img src="data:image/jpeg;base64,{b64}" />
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>SAM Debug — Raw Masks</title>
<style>
  body {{ background:#0d0d0d; color:#eee; font-family:monospace; padding:14px; margin:0; }}
  h1 {{ color:#ff0; margin-bottom:10px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .card {{ background:#1a1a1a; border-radius:5px; overflow:hidden; border:1px solid #333; }}
  .card img {{ display:block; width:640px; }}
  .title {{ padding:5px 10px; font-size:12px; color:#aaa; }}
</style>
</head>
<body>
<h1>SAM Debug — Raw Masks (box + point variants)</h1>
<p style="color:#888;font-size:12px">
  Yellow box = box prompt region &nbsp;|&nbsp;
  Blue overlay = BOX mask &nbsp;|&nbsp;
  Colored overlays = POINT masks (3 candidates)<br>
  Lines show detected top/bottom Y of each mask.
</p>
<div class="grid">{cards}</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", default=str(
        Path("C:/Users/illya/Documents/volleyball_analitics/volleystat/court_calibration.json")))
    parser.add_argument("--output", default="results/sam_debug")
    parser.add_argument("--sam-checkpoint", default=_SAM_CHECKPOINT)
    args = parser.parse_args()

    with open(args.calibration) as f:
        calib = json.load(f)

    net_pts = calib.get("net_line_points", [])
    frame_w = float(calib.get("frame_width", 3840))
    frame_h = float(calib.get("frame_height", 2160))
    net_y = float(np.mean([p[1] for p in net_pts])) if net_pts else frame_h * 0.5
    net_x_left = float(min(p[0] for p in net_pts)) if net_pts else 0.0
    net_x_right = float(max(p[0] for p in net_pts)) if net_pts else frame_w

    sy = _TARGET_H / frame_h
    sx = _TARGET_W / frame_w
    net_y_720 = net_y * sy
    box = np.array([
        max(0.0, net_x_left * sx - 10),
        max(0.0, net_y_720 - 8),
        min(float(_TARGET_W), net_x_right * sx + 10),
        min(float(_TARGET_H), net_y_720 + _NET_HEIGHT_HINT),
    ])
    print(f"Box prompt (720p): {box}")

    predictor = load_sam(args.sam_checkpoint)

    print(f"Extracting {_SAMPLE_COUNT} frames …")
    frames, timestamps = extract_frames(args.video, _SAMPLE_COUNT, _FIRST_N_SEC)
    print(f"Got {len(frames)} frames. Running SAM …")

    all_panels = []
    for i, (frame, ts) in enumerate(zip(frames, timestamps)):
        print(f"  frame {i}  t={ts:.2f}s …")
        masks, scores, pt_masks, pt_scores = run_sam_on_frame(predictor, frame, box)
        panels = render_frame_variants(frame, ts, box, masks, scores, pt_masks, pt_scores)
        all_panels.extend(panels)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "sam_debug.html"
    html_path.write_text(build_html(all_panels), encoding="utf-8")
    print(f"Saved: {html_path}")
    webbrowser.open(html_path.resolve().as_uri())


if __name__ == "__main__":
    main()
