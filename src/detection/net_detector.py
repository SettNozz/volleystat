"""
SAM-based volleyball net tape detector.

Processes first 5 seconds of video at 720p, uses Segment Anything Model
with a point prompt at the known net position to segment the net tape,
then returns top/bottom Y coordinates in 720p space.
"""

import json
import os
import subprocess
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
_DEFAULT_NET_TOP_HEIGHT_M = 2.34
_NET_MESH_HEIGHT_M = 1.00
_MIN_POST_SPACING_M = 10.0
_MAX_POST_SPACING_M = 11.0


def infer_floor_y_from_net_bounds(
    net_tape_top: float,
    net_tape_bottom: float,
    net_top_height_m: float = _DEFAULT_NET_TOP_HEIGHT_M,
) -> float:
    """Infer the floor under the net from its known physical proportions.

    The segmented mesh/tape band is 1.00 m high.  For the default mixed-game
    height of 2.34 m, its lower tape is 1.34 m above the floor.  Men's 2.43 m
    footage can be selected explicitly.  The result stays in the same pixel
    coordinate system as the inputs.
    """
    mesh_px = float(net_tape_bottom) - float(net_tape_top)
    if mesh_px <= 0:
        raise ValueError("net_tape_bottom must be below net_tape_top")
    bottom_height_m = float(net_top_height_m) - _NET_MESH_HEIGHT_M
    if bottom_height_m <= 0:
        raise ValueError("net_top_height_m must exceed the 1.00 m mesh height")
    return float(net_tape_bottom) + mesh_px * (bottom_height_m / _NET_MESH_HEIGHT_M)


def level_floor_line(
    left: Tuple[float, float],
    right: Tuple[float, float],
    base_y: Optional[float] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Force the net's ground line parallel to the net tape.

    The tape is modelled as a horizontal line (``net_tape_top`` is a single y),
    so the ground line under the net — parallel to the tape in 3D and therefore
    sharing its vanishing point — has to be horizontal in the image too. The two
    post bases are found independently, one per post, so detector noise alone
    tilts the segment by tens of pixels; observed slopes even flip sign between
    videos shot from the same camera position. Since this segment is the near/far
    team boundary, a tilt misassigns players standing near the net.

    ``base_y`` overrides the height with a more robust estimate (e.g. the median
    over all sampled frames); otherwise the two endpoints are averaged.
    """
    y = float(base_y) if base_y is not None else (float(left[1]) + float(right[1])) / 2.0
    return (float(left[0]), y), (float(right[0]), y)


class NetDetector:
    """
    Detects net tape boundaries using SAM (Segment Anything Model).

    All returned Y coordinates are in 720p space so they can be stored in
    calibration JSON and scaled to any target resolution at runtime.
    """

    def __init__(
        self,
        sam_checkpoint: str = _SAM_CHECKPOINT,
        net_height_m: float = _DEFAULT_NET_TOP_HEIGHT_M,
    ):
        self.sam_checkpoint = sam_checkpoint
        self.net_height_m = float(net_height_m)
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

    def detect_floor_line_from_video(
        self,
        video_path: str,
        net_x_left: float,
        net_x_right: float,
        frame_h: float,
        frame_w: float,
        net_tape_top: Optional[float] = None,
        net_tape_bottom: Optional[float] = None,
        floor_x_priors: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Find ground contact points of the two net posts in 720p pixels.

        The tape itself floats above the ground, so it cannot separate teams.
        The post-base line is the actual divider: player feet below it belong
        to the near side, feet above it to the far side.
        """
        sx, sy = _TARGET_W / frame_w, _TARGET_H / frame_h
        expected = floor_x_priors or (net_x_left * sx, net_x_right * sx)
        top_y = net_tape_top * sy if net_tape_top is not None else _TARGET_H * 0.35
        bottom_y = net_tape_bottom * sy if net_tape_bottom is not None else None
        # A calibrated homography puts the search close to the actual
        # post/court intersection.  Keep this ROI tight enough that a player's
        # leg cannot win 100+ pixels away from the post.
        search_half_width = max(55, int(_TARGET_W * 0.055)) if floor_x_priors else None

        left_points, right_points = [], []
        for frame in self._sample_frames(video_path):
            # Get the visible tape span first.  Unlike persisted ``net_x``
            # values, it follows a changed camera crop/pan and excludes the
            # vertical mesh strings that otherwise look like a post.
            tape_span = self._find_tape_span(frame, top_y)
            frame_expected = expected if floor_x_priors else (tape_span or expected)
            left = self._find_post_base(
                frame, frame_expected[0], top_y, outer_direction=-1,
                search_half_width=search_half_width,
            )
            right = self._find_post_base(
                frame, frame_expected[1], top_y, outer_direction=1,
                search_half_width=search_half_width,
            )
            if left is not None:
                left_points.append(left)
            if right is not None:
                right_points.append(right)

        if len(left_points) < 2 or len(right_points) < 2:
            log.warning("Net post-base detector found insufficient vertical post lines")
            return None

        # A and B are physical ground-contact points of the two posts.
        left = tuple(np.median(np.asarray(left_points), axis=0).astype(float))
        right = tuple(np.median(np.asarray(right_points), axis=0).astype(float))
        # Each base is estimated from its own post independently, so detector
        # noise alone tilts the segment. Level it against the tape.
        if bottom_y is not None:
            base_y = infer_floor_y_from_net_bounds(
                top_y, bottom_y, self.net_height_m,
            )
        else:
            base_y = float(np.median([p[1] for p in left_points + right_points]))
        left, right = level_floor_line(left, right, base_y)
        log.info(f"Net floor A/B (720p): A=({left[0]:.0f},{left[1]:.0f}) "
                 f"B=({right[0]:.0f},{right[1]:.0f})")
        return left, right

    def detect_floor_line_sam3(
        self,
        video_path: str,
        calibration_path: str,
        net_tape_top: Optional[float] = None,
        net_tape_bottom: Optional[float] = None,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get net-post ground contacts from the isolated SAM3 worker.

        SAM3 currently runs in a project-local Python 3.12 environment while
        the pipeline intentionally stays on Python 3.11.  The worker writes a
        transparent JSON/PNG artifact; this bridge reads only its validated
        A/B coordinates.  It does not require a Hugging Face token after the
        checkpoint has been cached locally.
        """
        override = os.environ.get("SAM3_PYTHON")
        if override:
            python = Path(override)
        else:
            python = _ROOT / ".sam3_env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        worker = _ROOT / "scripts" / "detection" / "detect_net_posts_sam3.py"
        if not python.exists() or not worker.exists() or not Path(calibration_path).exists():
            log.info("SAM3 post worker unavailable; falling back to Hough post bases")
            return None

        output_dir = _ROOT / "results" / "sam3_net_posts"
        cmd = [
            str(python), str(worker),
            "--video", str(video_path),
            "--calibration", str(calibration_path),
            "--prompt", "volleyball net post",
            "--samples", "3",
            "--output-dir", str(output_dir),
        ]
        try:
            completed = subprocess.run(
                cmd, cwd=str(_ROOT), capture_output=True, text=True,
                timeout=180, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning(f"SAM3 post worker could not run ({exc}); using Hough fallback")
            return None

        stem = Path(video_path).stem
        result_path = output_dir / f"{stem}_sam3_posts.json"
        if completed.returncode != 0 or not result_path.exists():
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            suffix = detail[-1] if detail else f"exit code {completed.returncode}"
            log.warning(f"SAM3 post worker failed ({suffix}); using Hough fallback")
            return None
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            floor = payload.get("net_floor_line_720p") or {}
            left, right = floor.get("left"), floor.get("right")
            if not (left and right and len(left) == 2 and len(right) == 2):
                raise ValueError("missing A/B coordinates")
            base_y = None
            if net_tape_top is not None and net_tape_bottom is not None:
                base_y = infer_floor_y_from_net_bounds(
                    net_tape_top, net_tape_bottom, self.net_height_m,
                )
            result = level_floor_line(
                (float(left[0]), float(left[1])),
                (float(right[0]), float(right[1])),
                base_y=base_y,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            log.warning(f"Invalid SAM3 post result ({exc}); using Hough fallback")
            return None

        log.info(f"SAM3 net floor A/B (720p): A=({result[0][0]:.0f},{result[0][1]:.0f}) "
                 f"B=({result[1][0]:.0f},{result[1][1]:.0f})")
        return result

    @staticmethod
    def project_net_floor_x_priors(
        homography,
        frame_w: float,
        frame_h: float,
    ) -> Optional[Tuple[float, float]]:
        """Project the two court/net intersections into 720p as post priors."""
        if homography is None:
            return None
        try:
            matrix = np.asarray(homography, dtype=np.float64)
            if matrix.shape != (3, 3):
                return None
            world = np.asarray([[-4.5, 0.0], [4.5, 0.0]], dtype=np.float64)
            homogeneous = np.c_[world, np.ones(2)] @ matrix.T
            if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
                return None
            image_x = homogeneous[:, 0] / homogeneous[:, 2]
            result = tuple(sorted(float(x) * _TARGET_W / float(frame_w) for x in image_x))
            if not all(np.isfinite(result)):
                return None
            return result
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def estimate_post_spacing_m(
        left: Tuple[float, float],
        right: Tuple[float, float],
        homography,
        frame_w: float,
        frame_h: float,
    ) -> Optional[float]:
        """Measure a 720p A/B pair in court meters through the homography."""
        if homography is None:
            return None
        try:
            matrix_inv = np.linalg.inv(np.asarray(homography, dtype=np.float64))
            native = np.asarray([
                [left[0] * frame_w / _TARGET_W, left[1] * frame_h / _TARGET_H, 1.0],
                [right[0] * frame_w / _TARGET_W, right[1] * frame_h / _TARGET_H, 1.0],
            ])
            world_h = native @ matrix_inv.T
            if np.any(np.abs(world_h[:, 2]) < 1e-9):
                return None
            world = world_h[:, :2] / world_h[:, 2:3]
            spacing = float(np.linalg.norm(world[1] - world[0]))
            return spacing if np.isfinite(spacing) else None
        except (TypeError, ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return None

    @staticmethod
    def post_spacing_is_plausible(spacing_m: Optional[float]) -> bool:
        """Whether post spacing agrees with the known 10–11 m installation."""
        return (
            spacing_m is None
            or _MIN_POST_SPACING_M <= spacing_m <= _MAX_POST_SPACING_M
        )

    @staticmethod
    def _find_tape_span(
        frame_bgr: np.ndarray,
        net_top_y: float,
    ) -> Optional[Tuple[float, float]]:
        """Return the endpoints of the longest near-horizontal net-tape line."""
        h, w = frame_bgr.shape[:2]
        y0 = max(0, int(net_top_y - h * 0.07))
        y1 = min(h, int(net_top_y + h * 0.07))
        if y1 <= y0:
            return None
        gray = cv2.cvtColor(frame_bgr[y0:y1], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 20, 90)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=max(18, int(w * 0.015)),
            minLineLength=max(80, int(w * 0.10)), maxLineGap=max(20, int(w * 0.04)),
        )
        if lines is None:
            return None

        candidates = []
        for x1, ly1, x2, ly2 in lines[:, 0]:
            dx, dy = abs(float(x2 - x1)), abs(float(ly2 - ly1))
            if dx < max(80.0, w * 0.10) or dy > dx * 0.08:
                continue
            y_mid = (float(ly1 + ly2) / 2.0) + y0
            score = dx - 1.5 * abs(y_mid - net_top_y)
            candidates.append((score, float(min(x1, x2)), float(max(x1, x2)), y_mid))
        if not candidates:
            return None

        # Hough often splits the white tape into adjacent segments.  Merge
        # segments at the best tape height so their outer extrema become the
        # two real post priors rather than an arbitrary middle fragment.
        candidates.sort(reverse=True)
        anchor_y = candidates[0][3]
        aligned = [item for item in candidates if abs(item[3] - anchor_y) <= 7.0]
        left = min(item[1] for item in aligned)
        right = max(item[2] for item in aligned)
        return (left, right)

    @staticmethod
    def _find_post_base(
        frame_bgr: np.ndarray,
        expected_x: float,
        net_top_y: float,
        outer_direction: int,
        search_half_width: Optional[int] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return the bottom end of the strongest vertical line near one post."""
        h, w = frame_bgr.shape[:2]
        # Calibration tape endpoints are only priors.  On a new camera angle
        # they can be displaced by a full post-width (or more), so search a
        # broad strip rather than rejecting the true post for being on the
        # "wrong" side of a stale endpoint.
        half_roi = (
            max(20, int(search_half_width))
            if search_half_width is not None
            else max(55, int(w * 0.12))
        )
        x0, x1 = max(0, int(expected_x - half_roi)), min(w, int(expected_x + half_roi))
        y0 = max(0, int(net_top_y - h * 0.12))
        y1 = min(h, int(net_top_y + h * 0.48))
        if x1 <= x0 or y1 <= y0:
            return None

        gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 110)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=24,
            minLineLength=max(45, int(h * 0.10)), maxLineGap=18,
        )
        if lines is None:
            return None

        best, best_score = None, -float("inf")
        for line in lines[:, 0]:
            x_a, y_a, x_b, y_b = (float(line[0] + x0), float(line[1] + y0),
                                  float(line[2] + x0), float(line[3] + y0))
            dx, dy = abs(x_b - x_a), abs(y_b - y_a)
            if dy < max(45.0, h * 0.10) or dx > dy * 0.20:
                continue
            x_mid, y_top, y_bottom = (x_a + x_b) / 2.0, min(y_a, y_b), max(y_a, y_b)
            # The outward side is a weak tie-breaker, not a hard constraint:
            # the old tape endpoint can be left/right of the real post.
            # Prefer a long upright line that reaches the court, then the
            # closest position to the prior. This rejects most player limbs
            # without losing a valid post after a camera-angle change.
            outward_bonus = 8.0 if (x_mid - expected_x) * outer_direction >= 0 else 0.0
            score = (dy - 1.5 * abs(x_mid - expected_x)
                     - 0.35 * abs(y_top - net_top_y) + outward_bonus)
            # A post has to extend well below the tape.  Short mesh/support
            # segments often terminate just above the court and are not a
            # valid ground-contact point.
            if y_bottom < net_top_y + h * 0.12:
                continue
            if score > best_score:
                best_score, best = score, (x_mid, y_bottom)
        return best

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
    net_floor_line_720p: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    net_floor_method: Optional[str] = None,
    net_height_m: Optional[float] = None,
    net_post_spacing_m: Optional[float] = None,
) -> None:
    """Persist SAM net detection results into calibration JSON."""
    p = Path(calib_path)
    data: dict = {}
    if p.exists():
        with open(p) as f:
            data = json.load(f)
    detected = data.get("net_detection_sam", {})
    detected.update({
        "net_top_y_720p": net_top_y_720p,
        "net_bottom_y_720p": net_bottom_y_720p,
    })
    if net_height_m is not None:
        detected["net_height_m"] = float(net_height_m)
        metadata = data.setdefault("metadata", {})
        court_geometry = metadata.setdefault("court_geometry", {})
        court_geometry["net_height"] = float(net_height_m)
    if net_post_spacing_m is not None:
        detected["net_post_spacing_m"] = float(net_post_spacing_m)
    if net_floor_line_720p is not None:
        left, right = net_floor_line_720p
        detected["net_floor_left_720p"] = [float(left[0]), float(left[1])]
        detected["net_floor_right_720p"] = [float(right[0]), float(right[1])]
        if net_floor_method:
            detected["net_floor_method"] = net_floor_method
    data["net_detection_sam"] = detected
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"SAM net saved to calibration: {p}")
