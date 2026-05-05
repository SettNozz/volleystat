"""
SAM-based volleyball net tape detector.

Processes first 5 seconds of video at 720p, uses Segment Anything Model
with a point prompt at the known net position to segment the net tape,
then returns top/bottom Y coordinates in 720p space.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.utils.logger import get_logger

log = get_logger("NetDetector")

_TARGET_H = 720
_TARGET_W = 1280
_SAMPLE_COUNT = 5        # frames to sample from first 5 seconds
_SAM_CHECKPOINT = str(Path.home() / "Downloads" / "sam_vit_h_4b8939.pth")
_SAM_MODEL_TYPE = "vit_h"


class NetDetector:
    """
    Detects net tape boundaries using SAM (Segment Anything Model).

    All returned Y coordinates are in 720p space so they can be stored in
    calibration JSON and scaled to any target resolution at runtime.
    """

    def __init__(self, sam_checkpoint: str = _SAM_CHECKPOINT):
        self.sam_checkpoint = sam_checkpoint
        self._predictor = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_video(
        self,
        video_path: str,
        net_y_hint: float,
        net_x_left: float,
        net_x_right: float,
        frame_h: float,
        frame_w: float,
        net_height_hint: float = 60.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect net tape top/bottom from first 5 s of *video_path*.

        Args:
            video_path:       Path to video file.
            net_y_hint:       Net Y position in original frame coordinates (from calibration).
            net_x_left:       Left boundary of net in original coords.
            net_x_right:      Right boundary of net in original coords.
            frame_h:          Original frame height (pixels).
            frame_w:          Original frame width (pixels).
            net_height_hint:  Estimated net height in 720p pixels (default 60).

        Returns:
            (net_tape_top_720p, net_tape_bottom_720p) or (None, None) on failure.
        """
        self._ensure_sam_loaded()

        sy = _TARGET_H / frame_h
        sx = _TARGET_W / frame_w
        net_y_720 = net_y_hint * sy
        net_cx_720 = ((net_x_left + net_x_right) / 2.0) * sx
        net_x_left_720 = net_x_left * sx
        net_x_right_720 = net_x_right * sx
        net_width_720 = net_x_right_720 - net_x_left_720

        # Two point prompts: one at top tape, one at estimated bottom tape.
        # Volleyball net has white tape both at top and bottom.
        # net_y_hint is at top tape; bottom tape is ~net_height_hint px below.
        top_pt = np.array([net_cx_720, net_y_720])
        bot_pt = np.array([net_cx_720, net_y_720 + net_height_hint * 0.8])

        log.info(f"SAM point prompts (720p): top=({top_pt[0]:.0f},{top_pt[1]:.0f})  "
                 f"bot=({bot_pt[0]:.0f},{bot_pt[1]:.0f})  net_width={net_width_720:.0f}px")

        frames = self._sample_frames(video_path)
        if not frames:
            log.warning("No frames extracted from video — SAM net detection skipped")
            return None, None

        tops, bottoms = [], []
        for i, frame_bgr in enumerate(frames):
            top, bot = self._segment_net_tape(
                frame_bgr, top_pt, net_y_720, net_height_hint, net_width_720
            )
            if top is not None:
                tops.append(top)
                bottoms.append(bot)
                log.debug(f"  frame {i}: tape top={top:.1f} bot={bot:.1f}")

        if not tops:
            log.warning("SAM could not segment net tape in any sampled frame")
            return None, None

        result_top = float(np.median(tops))
        result_bot = float(np.median(bottoms))
        log.info(f"Net tape (720p): top={result_top:.1f}  bottom={result_bot:.1f}  "
                 f"(height={result_bot - result_top:.1f}px)")
        return result_top, result_bot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_sam_loaded(self):
        if self._predictor is not None:
            return
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            raise ImportError(
                "segment-anything not installed. "
                "Run: pip install segment-anything"
            )
        if not Path(self.sam_checkpoint).exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {self.sam_checkpoint}\n"
                "Download from https://github.com/facebookresearch/segment-anything"
            )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading SAM ({_SAM_MODEL_TYPE}) on {device} ...")
        sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=self.sam_checkpoint)
        sam.to(device)
        self._predictor = SamPredictor(sam)
        log.info("SAM ready")

    def _sample_frames(self, video_path: str) -> list:
        """Extract _SAMPLE_COUNT evenly-spaced frames from first 5 s, resized to 720p."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frame = int(5.0 * fps)
        step = max(1, max_frame // _SAMPLE_COUNT)

        frames = []
        for i in range(_SAMPLE_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break
            frame_720 = cv2.resize(frame, (_TARGET_W, _TARGET_H))
            frames.append(frame_720)
        cap.release()
        return frames

    def _segment_net_tape(
        self,
        frame_bgr: np.ndarray,
        top_pt: np.ndarray,
        net_y_720: float,
        net_height_hint: float,
        net_width_720: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Hybrid detection:
          SAM (point at top tape) → precise top_y.
          HoughLinesP in ROI below top_y → precise bot_y from net mesh boundary.

        SAM handles the top tape reliably (high-contrast white band).
        Classical CV handles the bottom (avoids SAM confusing players/shadows
        with the lower tape when they overlap).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(rgb)

        # --- SAM: top tape ---
        masks_t, scores_t, _ = self._predictor.predict(
            point_coords=top_pt[np.newaxis],
            point_labels=np.array([1]),
            multimask_output=True,
        )
        top_mask = self._pick_widest(masks_t, scores_t, net_width_720)
        top_y = self._contiguous_top(top_mask, int(top_pt[1]))
        if top_y is None:
            return None, None

        # --- Hough: bottom tape ---
        bot_y = self._hough_net_bottom(frame_bgr, top_y, net_height_hint)

        # Fallback: estimate from hint if Hough finds nothing
        if bot_y is None or bot_y <= top_y + 5:
            bot_y = top_y + net_height_hint * 0.85

        # Safety clamp: bottom within [top+10, top+height_hint*1.4]
        bot_y = max(top_y + 10, min(bot_y, top_y + net_height_hint * 1.4))

        return float(top_y), float(bot_y)

    @staticmethod
    def _hough_net_bottom(
        frame_bgr: np.ndarray,
        top_y: int,
        net_height_hint: float,
        min_line_len_frac: float = 0.15,
    ) -> Optional[float]:
        """
        Detect the bottom tape of the net using horizontal Hough lines.

        Searches in ROI [top_y + 10 px, top_y + height_hint * 1.3 px].
        The net bottom tape is a wide white horizontal stripe — it should
        produce a strong horizontal edge in that ROI.
        """
        h, w = frame_bgr.shape[:2]
        roi_top = int(top_y) + 10
        roi_bot = min(h, int(top_y + net_height_hint * 1.3))
        if roi_bot <= roi_top:
            return None

        roi = frame_bgr[roi_top:roi_bot, :]

        # Enhance horizontal edges
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 3), 0)
        edges = cv2.Canny(blurred, 30, 90)

        min_len = int(w * min_line_len_frac)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=min_len,
            maxLineGap=20,
        )
        if lines is None:
            return None

        # Keep nearly horizontal lines (angle < 8°) and collect their Y positions
        ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1:
                continue
            angle = abs(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
            if angle < 8:
                ys.append((y1 + y2) / 2.0 + roi_top)

        if not ys:
            return None

        # Return the BOTTOM-MOST strong horizontal line (= bottom tape)
        return float(max(ys))

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_widest(masks: np.ndarray, scores: np.ndarray, min_width: float) -> np.ndarray:
        """Return the mask with greatest horizontal span (prefers net-shaped masks)."""
        best, best_span = None, -1
        for mask, score in zip(masks, scores):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            span = xs.max() - xs.min()
            if span > best_span:
                best_span = span
                best = mask
        return best if best is not None else masks[int(np.argmax(scores))]

    @staticmethod
    def _contiguous_top(mask: np.ndarray, seed_y: int, max_gap: int = 3) -> Optional[int]:
        """
        Walk upward from seed_y to find the topmost row of the contiguous
        horizontal band containing seed_y. Stops at gaps > max_gap rows.
        """
        active_rows = np.where(mask.any(axis=1))[0]
        if len(active_rows) == 0:
            return None
        # Find rows at or below seed_y
        relevant = active_rows[active_rows <= seed_y + 20]
        if len(relevant) == 0:
            relevant = active_rows
        # Walk upward from the row nearest to seed_y
        start = relevant[np.argmin(np.abs(relevant - seed_y))]
        top = start
        for r in range(start - 1, -1, -1):
            if mask[r].any():
                top = r
            elif start - r > max_gap:
                break
        return int(top)

    @staticmethod
    def _contiguous_bottom(mask: np.ndarray, seed_y: int, max_gap: int = 3) -> Optional[int]:
        """
        Walk downward from seed_y to find the bottommost row of the contiguous
        horizontal band containing seed_y. Stops at gaps > max_gap rows.
        """
        active_rows = np.where(mask.any(axis=1))[0]
        if len(active_rows) == 0:
            return None
        relevant = active_rows[active_rows >= seed_y - 20]
        if len(relevant) == 0:
            relevant = active_rows
        start = relevant[np.argmin(np.abs(relevant - seed_y))]
        bot = start
        h = mask.shape[0]
        for r in range(start + 1, h):
            if mask[r].any():
                bot = r
            elif r - start > max_gap:
                break
        return int(bot)


# ------------------------------------------------------------------
# Calibration JSON helpers (used by GameAnalyzer)
# ------------------------------------------------------------------

def load_sam_net_from_calibration(calib_path: str) -> Tuple[Optional[float], Optional[float]]:
    """Read net_detection_sam from calibration JSON. Returns (top, bottom) in 720p or (None, None)."""
    p = Path(calib_path)
    if not p.exists():
        return None, None
    with open(p) as f:
        data = json.load(f)
    sam = data.get("net_detection_sam", {})
    top = float(sam["net_top_y_720p"]) if sam.get("net_top_y_720p") is not None else None
    bot = float(sam["net_bottom_y_720p"]) if sam.get("net_bottom_y_720p") is not None else None
    return top, bot


def save_sam_net_to_calibration(
    calib_path: str,
    net_top_y_720p: float,
    net_bottom_y_720p: float,
) -> None:
    """Persist SAM net detection results into calibration JSON."""
    p = Path(calib_path)
    data: dict = {}
    if p.exists():
        with open(p) as f:
            data = json.load(f)
    data["net_detection_sam"] = {
        "net_top_y_720p": net_top_y_720p,
        "net_bottom_y_720p": net_bottom_y_720p,
    }
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"SAM net saved to calibration: {p}")
