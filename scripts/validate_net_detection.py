#!/usr/bin/env python3
"""
Visual validator for SAM net detection.

Extracts frames from first 5 s of video, draws calibration overlays,
and opens an HTML page in the browser for manual inspection.

Usage:
    python scripts/validate_net_detection.py \
        --video path/to/clip.mp4 \
        --calibration path/to/court_calibration.json \
        --output results/net_validation
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

from src.detection.net_detector import infer_floor_y_from_net_bounds

TARGET_H = 720
TARGET_W = 1280
SAMPLE_COUNT = 8
FIRST_N_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame_720: np.ndarray, calib: dict) -> np.ndarray:
    img = frame_720.copy()
    h, w = img.shape[:2]  # 720, 1280

    # Scale factors from original calibration resolution to 720p
    orig_w = calib.get("frame_width", 3840)
    orig_h = calib.get("frame_height", 2160)
    sx = w / orig_w
    sy = h / orig_h

    # --- net_line_points (manual calibration, green dashed) ---
    net_pts = calib.get("net_line_points", [])
    if net_pts:
        net_y_orig = float(np.mean([p[1] for p in net_pts]))
        net_y = int(net_y_orig * sy)
        xs = [p[0] for p in net_pts]
        x_left = int(min(xs) * sx)
        x_right = int(max(xs) * sx)
        # Dashed line via segments
        seg = 20
        for x in range(x_left, x_right, seg * 2):
            cv2.line(img, (x, net_y), (min(x + seg, x_right), net_y), (0, 255, 0), 2)
        _label(img, f"net_y={net_y_orig:.0f}px orig ({net_y}px 720p)",
               x_left, net_y - 8, (0, 255, 0))

    # --- SAM net tape ---
    sam = calib.get("net_detection_sam", {})
    net_tape_top_720 = sam.get("net_top_y_720p")
    net_tape_bot_720 = sam.get("net_bottom_y_720p")

    if net_tape_top_720 is not None:
        t = int(net_tape_top_720)
        b = int(net_tape_bot_720)
        x0, x1 = 0, w

        # Filled translucent band
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, t), (x1, b), (0, 180, 255), -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Border lines
        cv2.line(img, (x0, t), (x1, t), (0, 120, 255), 2)
        cv2.line(img, (x0, b), (x1, b), (0, 60, 200), 2)
        _label(img, f"SAM top={net_tape_top_720:.0f}px", x0 + 4, t - 6, (0, 120, 255))
        _label(img, f"SAM bot={net_tape_bot_720:.0f}px  h={net_tape_bot_720-net_tape_top_720:.0f}px",
               x0 + 4, b + 16, (0, 60, 200))

    # --- Homography projection of the regulation court/net intersections ---
    homography = calib.get("perspective_matrix")
    if homography:
        matrix = np.asarray(homography, dtype=np.float64)
        world = np.asarray([[-4.5, 0.0, 1.0], [4.5, 0.0, 1.0]])
        projected = world @ matrix.T
        projected = projected[:, :2] / projected[:, 2:3]
        hp0 = (round(projected[0, 0] * sx), round(projected[0, 1] * sy))
        hp1 = (round(projected[1, 0] * sx), round(projected[1, 1] * sy))
        cv2.line(img, hp0, hp1, (210, 210, 210), 2, cv2.LINE_AA)
        _label(img, "homography y=0", hp0[0] + 8, hp0[1] - 8, (210, 210, 210))

    # --- Physically inferred floor and saved production divider ---
    floor_left = sam.get("net_floor_left_720p")
    floor_right = sam.get("net_floor_right_720p")
    net_height_m = float(sam.get("net_height_m", 2.34))
    if (
        net_tape_top_720 is not None
        and net_tape_bot_720 is not None
        and floor_left
        and floor_right
    ):
        physical_y = infer_floor_y_from_net_bounds(
            net_tape_top_720, net_tape_bot_720, net_height_m,
        )
        physical_a = (round(float(floor_left[0])), round(physical_y))
        physical_b = (round(float(floor_right[0])), round(physical_y))
        for x in range(physical_a[0], physical_b[0], 24):
            cv2.line(
                img, (x, physical_a[1]), (min(x + 12, physical_b[0]), physical_b[1]),
                (255, 255, 0), 1, cv2.LINE_AA,
            )

    if floor_left and floor_right:
        a = tuple(round(float(v)) for v in floor_left)
        b = tuple(round(float(v)) for v in floor_right)
        cv2.line(img, a, b, (255, 0, 255), 3, cv2.LINE_AA)
        for label, point in (("A", a), ("B", b)):
            cv2.line(img, (point[0], int(net_tape_top_720 or 0)), point,
                     (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(img, point, 7, (255, 0, 255), -1, cv2.LINE_AA)
            _label(img, label, point[0] + 9, point[1] - 7, (255, 0, 255))

    # --- net horizontal extent (vertical guides) ---
    if net_pts:
        cv2.line(img, (x_left, 0), (x_left, h), (0, 255, 0), 1)
        cv2.line(img, (x_right, 0), (x_right, h), (0, 255, 0), 1)

    # --- Legend ---
    legend = [
        ("Green dashed line", (0, 255, 0), "net_y from manual calibration"),
        ("Orange band", (0, 180, 255), "SAM net tape bounds"),
        ("Magenta A-B", (255, 0, 255), "production team divider / post bases"),
        ("Cyan dashed", (255, 255, 0),
         f"{net_height_m:.2f}/{net_height_m - 1.0:.2f} m physical estimate"),
        ("Light gray", (210, 210, 210), "homography court y=0 cross-check"),
    ]
    lx, ly = 8, h - 12 - len(legend) * 22
    for label, color, desc in legend:
        cv2.rectangle(img, (lx, ly), (lx + 14, ly + 14), color, -1)
        cv2.putText(img, f"{label}: {desc}", (lx + 20, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ly += 22

    return img


def _label(img, text, x, y, color):
    cv2.putText(img, text, (x, max(y, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, max(y, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_native_floor_overlay(frame: np.ndarray, calib: dict) -> tuple[np.ndarray, list]:
    """Draw the production A/B line at the video's native resolution."""
    img = frame.copy()
    h, w = img.shape[:2]
    sx, sy = w / TARGET_W, h / TARGET_H
    sam = calib.get("net_detection_sam", {})
    left = sam.get("net_floor_left_720p")
    right = sam.get("net_floor_right_720p")
    top = sam.get("net_top_y_720p")
    bottom = sam.get("net_bottom_y_720p")
    if not (left and right and top is not None and bottom is not None):
        return img, []

    a = (round(float(left[0]) * sx), round(float(left[1]) * sy))
    b = (round(float(right[0]) * sx), round(float(right[1]) * sy))
    top_y, bottom_y = round(float(top) * sy), round(float(bottom) * sy)
    cv2.rectangle(img, (a[0], top_y), (b[0], bottom_y), (0, 180, 255), 3)
    cv2.line(img, a, b, (255, 0, 255), 4, cv2.LINE_AA)
    for label, point in (("A", a), ("B", b)):
        cv2.line(img, (point[0], top_y), point, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(img, point, 12, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(
            img, label, (point[0] + 25, point[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 255), 4, cv2.LINE_AA,
        )
    return img, [a, b]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, n: int, first_n_sec: float) -> list:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frame = int(first_n_sec * fps)
    step = max(1, max_frame // n)

    frames = []
    timestamps = []
    for i in range(n):
        idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (TARGET_W, TARGET_H)))
        timestamps.append(idx / fps)
    cap.release()
    return frames, timestamps


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def frame_to_b64(frame_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


def build_html(frames_b64: list, timestamps: list, calib: dict, video_path: str) -> str:
    sam = calib.get("net_detection_sam", {})
    method = sam.get("net_floor_method", sam.get("method", "unknown"))
    top = sam.get("net_top_y_720p", "n/a")
    bot = sam.get("net_bottom_y_720p", "n/a")
    height = (
        bot - top
        if isinstance(top, (int, float)) and isinstance(bot, (int, float))
        else "n/a"
    )
    floor_left = sam.get("net_floor_left_720p", "n/a")
    floor_right = sam.get("net_floor_right_720p", "n/a")
    net_height_m = float(sam.get("net_height_m", 2.34))
    post_spacing_m = sam.get("net_post_spacing_m", "n/a")
    physical_floor = (
        infer_floor_y_from_net_bounds(top, bot, net_height_m)
        if isinstance(top, (int, float)) and isinstance(bot, (int, float))
        else "n/a"
    )

    cards = ""
    for b64, ts in zip(frames_b64, timestamps):
        cards += f"""
        <div class="card">
            <div class="ts">t = {ts:.2f} s</div>
            <img src="data:image/jpeg;base64,{b64}" />
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<title>Net Detection Validator</title>
<style>
  body {{ background:#111; color:#eee; font-family:monospace; margin:0; padding:16px; }}
  h1 {{ color:#4cf; margin-bottom:4px; }}
  .meta {{ background:#1e1e1e; border-left:4px solid #4cf; padding:10px 14px;
           margin-bottom:20px; font-size:13px; line-height:1.8; }}
  .meta span {{ color:#fa0; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:12px; }}
  .card {{ background:#1a1a1a; border-radius:6px; overflow:hidden; }}
  .card img {{ display:block; max-width:640px; width:100%; }}
  .ts {{ padding:4px 10px; font-size:12px; color:#aaa; }}
  .legend {{ margin-top:20px; font-size:12px; }}
  .swatch {{ display:inline-block; width:14px; height:14px;
             vertical-align:middle; margin-right:6px; border-radius:2px; }}
</style>
</head>
<body>
<h1>SAM Net Detection — Validation</h1>
<div class="meta">
  <b>Video:</b> <span>{video_path}</span><br>
  <b>Method:</b> <span>{method}</span><br>
  <b>net_tape_top (720p):</b> <span>{top} px</span><br>
  <b>net_tape_bottom (720p):</b> <span>{bot} px</span><br>
  <b>tape height:</b> <span>{f"{height:.1f}" if isinstance(height, float) else height} px</span><br>
  <b>net height:</b> <span>{net_height_m:.2f} m</span><br>
  <b>physical floor y:</b> <span>{f"{physical_floor:.2f}" if isinstance(physical_floor, float) else physical_floor} px</span><br>
  <b>post spacing:</b> <span>{f"{post_spacing_m:.2f}" if isinstance(post_spacing_m, (int, float)) else post_spacing_m} m (expected 10-11 m)</span><br>
  <b>production A/B:</b> <span>{floor_left} â†’ {floor_right}</span><br>
  <b>Frames shown:</b> <span>first {FIRST_N_SECONDS}s, {len(frames_b64)} samples</span>
</div>
<div class="grid">{cards}</div>
<div class="legend">
  <span class="swatch" style="background:#00ff00"></span>Green dashed = net_y (manual calibration) &nbsp;|&nbsp;
  <span class="swatch" style="background:#00b4ff"></span>Orange band = SAM net tape bounds
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate SAM net detection visually")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--calibration", default=str(_ROOT / "court_calibration.json"),
                        help="Path to court_calibration.json")
    parser.add_argument("--output", default="results/net_validation", help="Output dir")
    parser.add_argument("--frames", type=int, default=SAMPLE_COUNT, help="Frames to sample")
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.calibration) as f:
        calib = json.load(f)

    print(f"Extracting {args.frames} frames from first {FIRST_N_SECONDS}s …")
    frames, timestamps = extract_frames(args.video, args.frames, FIRST_N_SECONDS)
    print(f"  Got {len(frames)} frames")

    print("Drawing overlays …")
    annotated = [draw_overlay(fr, calib) for fr in frames]
    for index, frame in enumerate(annotated):
        cv2.imwrite(str(out_dir / f"frame_{index:02d}.jpg"), frame)
    thumbs = [
        cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        for frame in annotated
    ]
    if thumbs:
        while len(thumbs) % 2:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.hstack(thumbs[i:i + 2]) for i in range(0, len(thumbs), 2)]
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), np.vstack(rows))

    native_cap = cv2.VideoCapture(str(args.video))
    native_ok, native_frame = native_cap.read()
    native_cap.release()
    if native_ok:
        native_overlay, post_points = draw_native_floor_overlay(native_frame, calib)
        cv2.imwrite(str(out_dir / "frame_00_native.jpg"), native_overlay)
        native_h, native_w = native_overlay.shape[:2]
        for label, (x, y) in zip(("left", "right"), post_points):
            x0, x1 = max(0, x - 260), min(native_w, x + 260)
            y0, y1 = max(0, y - 260), min(native_h, y + 180)
            cv2.imwrite(
                str(out_dir / f"frame_00_native_{label}_post.jpg"),
                native_overlay[y0:y1, x0:x1],
            )
    frames_b64 = [frame_to_b64(fr) for fr in annotated]

    html_path = out_dir / "net_validation.html"
    html = build_html(frames_b64, timestamps, calib, args.video)
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved: {html_path}")

    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
        print("Opened in browser.")


if __name__ == "__main__":
    main()
