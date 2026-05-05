"""
Volleyball game analyzer pipeline.

Combines YOLO ball detection + YOLO-pose skeleton + FSM event tracking.
Works entirely in pixel coordinates. No world coordinate transforms needed.
"""

import cv2
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project root
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from src.tracking.integrated_volleyball_fsm import CourtBounds, IntegratedVolleyballFSM
from src.utils.logger import get_logger

log = get_logger("Analyzer")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_BALL_MODEL = "models/yolov8_curated_training/yolov8n_volleyball_curated8/weights/best.pt"
_POSE_MODEL = "models/pretrained/yolo11m-pose.pt"
_POSE_FALLBACK = "yolo11n-pose.pt"  # will be auto-downloaded


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CalibrationData:
    """Key pixel-space references extracted from court_calibration.json."""
    net_y: float        # Net line Y position (pixels)
    net_x_left: float  # Net left X (pixels)
    net_x_right: float # Net right X (pixels)
    court_top: float   # Approx top of play area (pixels)
    court_bottom: float
    court_left: float
    court_right: float
    frame_width: int
    frame_height: int
    # SAM-detected net tape edges (optional, in same coordinate system as net_y)
    net_tape_top: Optional[float] = None
    net_tape_bottom: Optional[float] = None


@dataclass
class RallySummary:
    start_frame: int
    end_frame: int
    duration_frames: int
    events: List[Dict]
    contacts_team_a: int = 0
    contacts_team_b: int = 0


@dataclass
class GameStats:
    video_name: str = ""
    total_frames: int = 0
    processed_frames: int = 0
    ball_detected_frames: int = 0
    fps: float = 30.0
    rallies: List[Dict] = field(default_factory=list)
    all_events: List[Dict] = field(default_factory=list)
    # Per-frame data for HTML viewer overlay (ball pos, players, FSM state)
    frame_annotations: List[Dict] = field(default_factory=list)
    processing_time_sec: float = 0.0

    @property
    def ball_detection_rate(self) -> float:
        return self.ball_detected_frames / max(self.processed_frames, 1)

    def to_dict(self) -> Dict:
        return {
            "video": self.video_name,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "ball_detection_rate": round(self.ball_detection_rate, 3),
            "total_rallies": len(self.rallies),
            "rallies": self.rallies,
            "all_events": self.all_events,
            "processing_time_sec": round(self.processing_time_sec, 1),
        }


# ---------------------------------------------------------------------------
# GameAnalyzer
# ---------------------------------------------------------------------------

class GameAnalyzer:
    """
    Main pipeline:
        video / frames dir → ball + pose detection → FSM → GameStats

    Usage:
        analyzer = GameAnalyzer()
        stats = analyzer.process_video("game.mp4", output_dir="results/")
        # or
        stats = analyzer.process_frames("data/frames/", output_dir="results/")
    """

    # Default event model for overlay (LSTM v4 — best macro_f1=0.377, 1211 clips)
    _DEFAULT_EVENT_CHECKPOINT = "action_recognition/models/checkpoints_lstm_v4/best_event_sequence.pt"

    def __init__(
        self,
        calibration_path: Optional[str] = "court_calibration.json",
        ball_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        output_video: bool = True,
        skip_frames: int = 0,
        net_y_override: Optional[float] = None,
        event_checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            calibration_path: Path to court_calibration.json (relative to project root or absolute)
            ball_model_path:  Override ball detection model
            pose_model_path:  Override pose model
            output_video:     Produce annotated video / frame outputs
            skip_frames:      Process every (skip_frames+1)-th frame for speed
            event_checkpoint_path: Optional path to event sequence model (Transformer v3) for overlay.
        """
        self.output_video = output_video
        self.skip_frames = skip_frames
        self.net_y_override = net_y_override

        self.calibration = self._load_calibration(calibration_path)
        self._init_ball_detector(ball_model_path or str(_ROOT / _BALL_MODEL))
        self._init_pose_model(pose_model_path or str(_ROOT / _POSE_MODEL))
        self._event_predictor = self._init_event_predictor(event_checkpoint_path)
        self._warmup()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_calibration(self, path: Optional[str]) -> Optional[CalibrationData]:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        if not p.exists():
            log.warning(f"Calibration not found: {p} — using defaults")
            return None

        with open(p) as f:
            data = json.load(f)

        net_pts   = data.get("net_line_points", [])
        back_pts  = data.get("back_line_points", [])
        left_pts  = data.get("left_sideline_points", [])
        right_pts = data.get("right_sideline_points", [])

        net_y      = float(np.mean([p[1] for p in net_pts]))   if net_pts   else 800.0
        net_xs     = [p[0] for p in net_pts]
        court_bot  = float(np.mean([p[1] for p in back_pts]))  if back_pts  else 1500.0
        court_left = float(min(p[0] for p in left_pts))        if left_pts  else 0.0
        court_right= float(max(p[0] for p in right_pts))       if right_pts else float(data.get("frame_width", 3840))

        # Load SAM-detected net tape positions if available
        sam = data.get("net_detection_sam", {})
        net_tape_top    = float(sam["net_top_y_720p"])    if sam.get("net_top_y_720p")    else None
        net_tape_bottom = float(sam["net_bottom_y_720p"]) if sam.get("net_bottom_y_720p") else None
        # SAM tape coords are stored in 720p; store the 720p-reference height so we can scale
        sam_ref_h = 720.0  # SAM was run on 720p frames

        return CalibrationData(
            net_y=net_y,
            net_x_left=float(min(net_xs)) if net_xs else court_left,
            net_x_right=float(max(net_xs)) if net_xs else court_right,
            # court_top is set very high in the frame (low y) so that balls in the air
            # (above the net) are NOT flagged as "out of bounds".
            court_top=50.0,
            court_bottom=court_bot,
            court_left=court_left,
            court_right=court_right,
            frame_width=data.get("frame_width", 3840),
            frame_height=data.get("frame_height", 2160),
            net_tape_top=net_tape_top,
            net_tape_bottom=net_tape_bottom,
        )

    def _scale_calibration(self, actual_w: int, actual_h: int) -> "CalibrationData":
        """Return calibration scaled to actual frame size, with optional net_y override."""
        c = self.calibration
        if c is None:
            # No calibration: build defaults with override if provided
            net_y = self.net_y_override if self.net_y_override else actual_h * 0.5
            return CalibrationData(
                net_y=net_y, net_x_left=0.0, net_x_right=float(actual_w),
                court_top=10.0, court_bottom=float(actual_h),
                court_left=0.0, court_right=float(actual_w),
                frame_width=actual_w, frame_height=actual_h,
            )
        if c.frame_width != actual_w or c.frame_height != actual_h:
            sx = actual_w / c.frame_width
            sy = actual_h / c.frame_height
            log.info(f"Scaling calibration {c.frame_width}x{c.frame_height} "
                     f"→ {actual_w}x{actual_h}  (sx={sx:.3f} sy={sy:.3f})")
            # SAM tape coords are stored in 720p; scale to actual frame height
            sam_sy = actual_h / 720.0
            c = CalibrationData(
                net_y=c.net_y * sy,
                net_x_left=c.net_x_left * sx,
                net_x_right=c.net_x_right * sx,
                court_top=c.court_top * sy,
                court_bottom=c.court_bottom * sy,
                court_left=c.court_left * sx,
                court_right=c.court_right * sx,
                frame_width=actual_w,
                frame_height=actual_h,
                net_tape_top=c.net_tape_top * sam_sy if c.net_tape_top else None,
                net_tape_bottom=c.net_tape_bottom * sam_sy if c.net_tape_bottom else None,
            )
        if self.net_y_override is not None:
            log.info(f"net_y override: {c.net_y:.1f} -> {self.net_y_override}")
            c = CalibrationData(
                net_y=self.net_y_override,
                net_x_left=c.net_x_left, net_x_right=c.net_x_right,
                court_top=c.court_top, court_bottom=c.court_bottom,
                court_left=c.court_left, court_right=c.court_right,
                frame_width=c.frame_width, frame_height=c.frame_height,
                net_tape_top=c.net_tape_top, net_tape_bottom=c.net_tape_bottom,
            )
        return c

    def _init_ball_detector(self, model_path: str):
        from detection.detect_ball_on_video import BallDetectorVideo
        self.ball_detector = BallDetectorVideo(model_path)

    def _init_pose_model(self, model_path: str):
        from ultralytics import YOLO
        if Path(model_path).exists():
            self.pose_model = YOLO(model_path)
            log.info(f"Pose model loaded: {Path(model_path).name}")
        else:
            log.warning(f"Pose model not found at {model_path} — downloading {_POSE_FALLBACK}")
            self.pose_model = YOLO(_POSE_FALLBACK)

    def _init_event_predictor(self, event_checkpoint_path: Optional[str]):
        """Load sliding-window event predictor (Transformer v3) for overlay; None if disabled or file missing."""
        if event_checkpoint_path == "":
            return None
        path = event_checkpoint_path if event_checkpoint_path else self._DEFAULT_EVENT_CHECKPOINT
        p = Path(path)
        if not p.is_absolute():
            p = _ROOT / p
        if not p.exists():
            log.info(f"Event overlay disabled: checkpoint not found at {p}")
            return None
        try:
            from action_recognition.inference.streaming_event_predictor import StreamingEventPredictor
            return StreamingEventPredictor(p)
        except Exception as e:
            log.warning(f"Event overlay disabled: {e}")
            return None

    def _warmup(self):
        """
        Run one dummy inference on both models to trigger CUDA JIT / cudnn benchmark.
        torch.backends.cudnn.benchmark=True (set by BallDetectorVideo) profiles all CUDA
        kernels on the first call — this takes 5+ minutes without warmup.
        Running warmup upfront means the main loop starts fast.
        """
        import torch
        if not torch.cuda.is_available():
            return
        log.info("Warming up CUDA kernels (first-time only, may take a few minutes)...")
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        t0 = time.time()
        self.ball_detector.detect_ball(dummy)
        log.info(f"  Ball model warmed up in {time.time()-t0:.1f}s")
        t0 = time.time()
        self.pose_model(dummy, verbose=False, device="cuda", half=True)
        log.info(f"  Pose model warmed up in {time.time()-t0:.1f}s")

    # Frames without ball detection before forcing rally end.
    # At 30fps, 150 frames = 5 seconds. Ball can be temporarily occluded by players
    # or miss detection for several seconds during an active rally — don't end too soon.
    NO_BALL_RALLY_TIMEOUT = 150

    def _make_fsm(self, fps: float = 30.0) -> IntegratedVolleyballFSM:
        if self.calibration:
            c = self.calibration
            bounds = CourtBounds(
                left=c.court_left,
                right=c.court_right,
                top=c.court_top,
                bottom=c.court_bottom,
                net_y=c.net_y,
            )
        else:
            bounds = CourtBounds(left=0, right=3840, top=50, bottom=1700, net_y=830)

        fsm = IntegratedVolleyballFSM(court_bounds=bounds, fps=fps)

        # 4K GoPro: player bbox center is ~300px from hands/feet.
        # Default 40px radius misses every contact — increase substantially.
        fsm.event_tracker.contact_radius = 300.0

        return fsm

    # ------------------------------------------------------------------
    # Per-frame detection
    # ------------------------------------------------------------------

    def _detect_ball(self, frame: np.ndarray) -> Tuple[Optional[Tuple[float, float]], float]:
        """Returns (center_xy, confidence) or (None, 0.0)."""
        raw = self.ball_detector.detect_ball(frame)
        best = None
        for x1, y1, x2, y2, conf in raw:
            if conf < self.ball_detector.min_confidence:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Velocity filter
            if self.ball_detector.last_valid_position is not None:
                lx, ly = self.ball_detector.last_valid_position
                v = np.hypot(cx - lx, cy - ly)
                if v > self.ball_detector.max_velocity:
                    continue

            # Size filter
            size = max(x2 - x1, y2 - y1)
            if self.ball_detector.last_bbox_size is not None:
                ratio = max(size, self.ball_detector.last_bbox_size) / max(
                    min(size, self.ball_detector.last_bbox_size), 1.0
                )
                if ratio > self.ball_detector.size_change_threshold:
                    continue

            if best is None or conf > best[2]:
                best = (cx, cy, conf, size)

        if best:
            cx, cy, conf, size = best
            self.ball_detector.last_valid_position = (cx, cy)
            self.ball_detector.last_bbox_size = size
            self.ball_detector.consecutive_misses = 0
            return (cx, cy), conf

        self.ball_detector.update_no_detection()
        return None, 0.0

    def _detect_players(self, frame: np.ndarray) -> Dict[str, List[Tuple[float, float]]]:
        """
        Returns player centers split by which side of the net they're on:
            {"team_a": [(cx,cy), ...], "team_b": [(cx,cy), ...]}
        team_a = above net (y < net_y) — far side
        team_b = below net (y > net_y) — near side
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        results = self.pose_model(
            frame,
            verbose=False,
            device=device,
            half=(device == "cuda"),
        )

        net_y = self.calibration.net_y if self.calibration else 830.0
        teams: Dict[str, List[Tuple[float, float]]] = {"team_a": [], "team_b": []}

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in boxes:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                team = "team_a" if cy < net_y else "team_b"
                teams[team].append((float(cx), float(cy)))

        return teams

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_frame(
        self,
        frame: np.ndarray,
        ball_pos: Optional[Tuple[float, float]],
        player_teams: Dict[str, List[Tuple[float, float]]],
        fsm_status: Dict,
        event_display: Optional[str] = None,
    ) -> np.ndarray:
        out = frame.copy()

        # Net line
        if self.calibration:
            c = self.calibration
            ny = int(c.net_y)
            cv2.line(out, (int(c.net_x_left), ny), (int(c.net_x_right), ny), (255, 220, 0), 4)

        # Ball
        if ball_pos:
            cx, cy = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(out, (cx, cy), 18, (0, 255, 0), 3)
            cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)

        # Players — team_a=blue, team_b=red
        colors = {"team_a": (255, 80, 0), "team_b": (0, 80, 255)}
        for team, pts in player_teams.items():
            for px, py in pts:
                cv2.circle(out, (int(px), int(py)), 10, colors[team], -1)

        # FSM state overlay (line 1)
        state = fsm_status.get("game_state", "?")
        phase = fsm_status.get("rally_phase") or ""
        team  = fsm_status.get("current_team", "?")
        contacts = fsm_status.get("team_contacts", 0)
        text = f"{state} | {phase} | {team} [{contacts}]"
        cv2.putText(out, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

        # Event model overlay (line 2): sliding-window Transformer v3 prediction
        if event_display is not None and event_display != "-":
            ev_text = f"Event: {event_display}"
            cv2.putText(out, ev_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 200, 0), 4)

        return out

    # ------------------------------------------------------------------
    # Core processing loop
    # ------------------------------------------------------------------

    def _process_frames_iter(
        self,
        frame_iter,           # iterator yielding (frame_idx, frame_np)
        fps: float,
        total: int,
        stats: GameStats,
        out_writer=None,
        out_frames_dir: Optional[Path] = None,
    ) -> GameStats:
        """Shared processing loop used by both process_video and process_frames."""
        fsm = self._make_fsm(fps)

        # Accumulated rallies across FSM resets
        completed_rallies: List[Dict] = []
        current_rally_events: List[Dict] = []
        current_rally_start: Optional[int] = None
        prev_state = None

        # No-ball timeout: if ball not detected for this many frames during RALLY_ACTIVE,
        # force the rally to end (ball went out / rally over).
        frames_since_ball = 0

        # Serve cooldown: suppress repeated "serve" predictions — a real serve can happen
        # at most once per rally. After showing "serve", block it for at least this many
        # frames so the model can't flicker back to "serve" mid-rally.
        _SERVE_COOLDOWN_SEC  = 8.0          # minimum seconds between two serve displays
        _serve_cooldown_frames = int(fps * _SERVE_COOLDOWN_SEC)
        _last_serve_shown_frame: int = -_serve_cooldown_frames  # allow first serve immediately

        # Sticky overlay: after rally ends, keep showing the last in-rally prediction for
        # a short window so the screen doesn't go blank between rallies.
        _STICKY_FRAMES = int(fps * 2.0)     # show last prediction for 2 sec after rally ends
        _last_rally_display: str = ""
        _last_rally_display_frame: int = -_STICKY_FRAMES

        t0 = time.time()

        for frame_idx, frame in frame_iter:
            stats.total_frames += 1

            # Frame skipping
            if self.skip_frames > 0 and frame_idx % (self.skip_frames + 1) != 0:
                if out_writer:
                    out_writer.write(frame)
                continue

            stats.processed_frames += 1
            timestamp = frame_idx / fps

            # --- Detect ---
            ball_pos, ball_conf = self._detect_ball(frame)
            if ball_pos:
                stats.ball_detected_frames += 1
                frames_since_ball = 0
            else:
                frames_since_ball += 1

            # No-ball timeout: end rally/serve if ball gone too long
            fsm_status_pre = fsm.get_current_status()
            cur_state_pre = fsm_status_pre.get("game_state", "")
            if frames_since_ball >= self.NO_BALL_RALLY_TIMEOUT and cur_state_pre in ("rally_active", "serving"):
                log.warning(f"f{frame_idx}: [{cur_state_pre}] ended — no ball for {frames_since_ball} frames (timeout)")
                fsm._end_rally(frame_idx, "no_ball_timeout")
                frames_since_ball = 0

            player_teams = self._detect_players(frame)

            # --- FSM update ---
            fsm_events = fsm.update(
                frame_idx=frame_idx,
                ball_pos=ball_pos,
                player_positions=player_teams,
                timestamp=timestamp,
            )
            fsm_status = fsm.get_current_status()
            cur_state = fsm_status.get("game_state")

            # Track rally boundaries
            if cur_state == "rally_active" and current_rally_start is None:
                current_rally_start = frame_idx

            for ev in fsm_events:
                ev_dict = {
                    "frame": ev.frame_idx,
                    "time": round(ev.timestamp, 2),
                    "type": ev.event_type,
                    "game_state": str(ev.game_state.value),
                    "rally_phase": str(ev.rally_phase.value) if ev.rally_phase else None,
                    "team": ev.team_controlling,
                    "contact_seq": ev.contacts_count,
                }
                stats.all_events.append(ev_dict)
                current_rally_events.append(ev_dict)

            # Rally ended → save summary before FSM resets it
            if cur_state == "rally_end" and prev_state != "rally_end":
                if current_rally_events:
                    # Use first event frame as start if we never entered rally_active
                    r_start = current_rally_start or current_rally_events[0]["frame"]
                    rally_dict = {
                        "start_frame": r_start,
                        "end_frame": frame_idx,
                        "duration_frames": frame_idx - r_start,
                        "duration_sec": round((frame_idx - r_start) / fps, 2),
                        "events": list(current_rally_events),
                        "contacts_team_a": sum(1 for e in current_rally_events if e["team"] == "team_a"),
                        "contacts_team_b": sum(1 for e in current_rally_events if e["team"] == "team_b"),
                    }
                    completed_rallies.append(rally_dict)
                current_rally_start = None
                current_rally_events = []

            prev_state = cur_state

            # --- Event model (Transformer) prediction for overlay and viewer comparison ---
            # FSM gate: only show LSTM prediction when ball is actually in play.
            # During serve_prep / serve_ready players stand in serving position, causing
            # the LSTM to continuously output "serve" between rallies — suppress that.
            _LSTM_ACTIVE_STATES = {"serving", "rally_active"}
            event_display = None
            if self._event_predictor is not None and self.calibration is not None:
                raw_display = self._event_predictor.add_frame(
                    ball_pos, player_teams, self.calibration.net_y, fps,
                    fsm_state=cur_state or "",   # let predictor exclude impossible classes
                )
                # Gate: only pass prediction through when FSM says ball is in play
                if raw_display is not None and cur_state in _LSTM_ACTIVE_STATES:
                    if raw_display == "serve":
                        # Serve cooldown: enforce minimum gap between serve displays.
                        # During rally_active the predictor already suppresses serve, but
                        # "serving" state can still briefly overlap with a false serve
                        # prediction lingering in the smoothing window.
                        frames_since_serve = frame_idx - _last_serve_shown_frame
                        if frames_since_serve >= _serve_cooldown_frames:
                            _last_serve_shown_frame = frame_idx
                            _last_rally_display = raw_display
                            _last_rally_display_frame = frame_idx
                            event_display = raw_display
                        # else: too soon — suppress this serve prediction silently
                    else:
                        _last_rally_display = raw_display
                        _last_rally_display_frame = frame_idx
                        event_display = raw_display
                elif (raw_display is not None
                      and cur_state not in _LSTM_ACTIVE_STATES
                      and _last_rally_display
                      and _last_rally_display != "serve"
                      and frame_idx - _last_rally_display_frame < _STICKY_FRAMES):
                    # Sticky: rally just ended — keep last non-serve prediction briefly visible
                    event_display = _last_rally_display

            # --- Collect per-frame annotation for viewer (FSM + Transformer for comparison) ---
            frame_events_this = [e for e in stats.all_events if e["frame"] == frame_idx]
            ann = {
                "f": frame_idx,
                "t": round(timestamp, 3),
                "ball": [round(ball_pos[0], 1), round(ball_pos[1], 1)] if ball_pos else None,
                "p": [[round(x, 1), round(y, 1)]
                      for x, y in player_teams.get("team_a", []) + player_teams.get("team_b", [])],
                "state": cur_state,
                "phase": fsm_status.get("rally_phase"),
                "ev": [e["type"] for e in frame_events_this],
            }
            if event_display is not None:
                ann["event_model"] = event_display
            stats.frame_annotations.append(ann)

            # --- Annotate output ---
            if self.output_video and (out_writer or out_frames_dir):
                annotated = self._draw_frame(
                    frame, ball_pos, player_teams, fsm_status, event_display=event_display
                )
                if out_writer:
                    out_writer.write(annotated)
                if out_frames_dir:
                    cv2.imwrite(str(out_frames_dir / f"frame_{frame_idx:05d}.jpg"), annotated)

            # Progress log every 5 seconds of video
            log_interval = max(1, int(fps * 5))
            if frame_idx % log_interval == 0 and frame_idx > 0:
                elapsed = time.time() - t0
                pct = frame_idx / max(total, 1) * 100
                fps_proc = stats.processed_frames / max(elapsed, 0.01)
                ball_pct = stats.ball_detected_frames / max(stats.processed_frames, 1) * 100
                log.info(f"[{pct:5.1f}%] f{frame_idx}/{total} | {fps_proc:.1f} fps | "
                         f"state={cur_state} | ball={ball_pct:.0f}%")

        stats.rallies = completed_rallies
        stats.processing_time_sec = time.time() - t0

        # Post-process: remove trajectory jumps (false positives inconsistent with neighbours)
        n_removed = self._filter_ball_outliers(stats.frame_annotations)
        if n_removed:
            log.info(f"Outlier filter: removed {n_removed} false-positive frames")

        # Post-process: fill short gaps in ball detection by linear interpolation
        n_filled = self._interpolate_ball_positions(stats.frame_annotations)
        if n_filled:
            log.info(f"Ball interpolation: filled {n_filled} missing frames")

        return stats

    def _filter_ball_outliers(
        self,
        annotations: List[Dict],
        max_gap: int = 4,
        detour_ratio: float = 2.5,
        min_jump_px: float = 30.0,
    ) -> int:
        """
        Remove ball detections that are inconsistent with neighbouring frames.

        A detection at frame N is a false positive if:
        - It is within `max_gap` frames of both a previous and a next detection
        - The detour distance (prev→curr + curr→next) exceeds `detour_ratio` × the
          direct distance (prev→next) — i.e. the ball "jumps out and back"
        - The jump size is at least `min_jump_px` (ignore tiny wiggles)

        Removed frames are cleared (ball=None) so interpolation can fill them later.
        Returns the number of frames removed.
        """
        detected = [i for i, ann in enumerate(annotations)
                    if ann.get("ball") and not ann.get("ball_interp")]
        removed = 0

        for k in range(1, len(detected) - 1):
            i_prev, i_curr, i_next = detected[k - 1], detected[k], detected[k + 1]

            # Only consider closely-spaced detections
            if i_curr - i_prev > max_gap or i_next - i_curr > max_gap:
                continue

            # Skip if any neighbour was already filtered this pass
            bp = annotations[i_prev]["ball"]
            bc = annotations[i_curr]["ball"]
            bn = annotations[i_next]["ball"]
            if bp is None or bc is None or bn is None:
                continue

            xp, yp = bp
            xc, yc = bc
            xn, yn = bn

            d_in   = np.hypot(xc - xp, yc - yp)   # prev → curr
            d_out  = np.hypot(xn - xc, yn - yc)   # curr → next
            d_skip = np.hypot(xn - xp, yn - yp)   # prev → next (direct)

            if d_skip < 1.0:
                d_skip = 1.0

            jump = max(d_in, d_out)
            if jump < min_jump_px:
                continue  # tiny movement, not an outlier

            if (d_in + d_out) > detour_ratio * d_skip:
                annotations[i_curr]["ball"] = None
                annotations[i_curr]["ball_filtered"] = True
                removed += 1

        return removed

    def _interpolate_ball_positions(self, annotations: List[Dict], max_gap: int = 10) -> int:
        """
        Linearly interpolate ball positions for short detection gaps.

        Only fills gaps of at most `max_gap` consecutive missing frames — longer gaps
        mean the ball is genuinely out of sight.  Interpolated frames are tagged with
        ``"ball_interp": True`` so the viewer can style them differently.

        Returns the number of frames filled.
        """
        if not annotations:
            return 0

        # Indices of annotation entries that have a confirmed ball detection
        detected = [i for i, ann in enumerate(annotations) if ann.get("ball") and not ann.get("ball_interp")]
        filled = 0

        for k in range(len(detected) - 1):
            i0, i1 = detected[k], detected[k + 1]
            gap = i1 - i0 - 1
            if gap <= 0 or gap > max_gap:
                continue
            x0, y0 = annotations[i0]["ball"]
            x1, y1 = annotations[i1]["ball"]
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                annotations[i0 + step]["ball"] = [
                    round(x0 + t * (x1 - x0), 1),
                    round(y0 + t * (y1 - y0), 1),
                ]
                annotations[i0 + step]["ball_interp"] = True
                filled += 1

        return filled

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: str = "results",
        max_duration_sec: Optional[float] = None,
    ) -> GameStats:
        """Process a video file."""
        vp = Path(video_path)
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vp}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_frames = int(max_duration_sec * fps) if max_duration_sec else total

        log.section(f"Video: {vp.name}")
        log.info(f"Resolution={w}x{h}  fps={fps:.1f}  frames={total}  skip={self.skip_frames}")

        # Scale calibration to actual frame resolution (e.g., 4K calib on 720p video)
        saved_calib = self.calibration
        self.calibration = self._scale_calibration(w, h)

        out_writer = None
        if self.output_video:
            out_path = od / f"{vp.stem}_analyzed.mp4"
            # Prefer mp4v first to avoid OpenH264 DLL warning on Windows (avc1/H264 need it)
            for codec in ("mp4v", "avc1", "H264"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                if out_writer.isOpened():
                    log.info(f"VideoWriter codec: {codec}")
                    break
                out_writer = None
            if out_writer is None:
                log.warning("VideoWriter failed — annotated video will not be saved")

        def frame_iter():
            idx = 0
            while idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                yield idx, frame
                idx += 1

        stats = GameStats(video_name=vp.name, fps=fps)
        scaled_calib = self.calibration  # already scaled — needed for _save_and_print
        try:
            self._process_frames_iter(
                frame_iter(), fps, max_frames, stats,
                out_writer=out_writer,
            )
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
            self.calibration = saved_calib  # restore original

        self._save_and_print(stats, od, vp.stem,
                             original_video_path=str(vp.resolve()),
                             calibration_override=scaled_calib)
        return stats

    def process_frames(
        self,
        frames_dir: str,
        output_dir: str = "results",
        fps: float = 30.0,
        max_frames: Optional[int] = None,
    ) -> GameStats:
        """Process a directory of JPEG/PNG frames (e.g. for testing)."""
        fd = Path(frames_dir)
        od = Path(output_dir)
        od.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png"}
        frame_files = sorted(p for p in fd.iterdir() if p.suffix.lower() in exts)
        if max_frames:
            frame_files = frame_files[:max_frames]

        total = len(frame_files)
        log.section(f"Frames dir: {fd.name}")
        log.info(f"frames={total}  fps={fps:.1f}")

        out_frames_dir = None
        if self.output_video:
            out_frames_dir = od / "annotated_frames"
            out_frames_dir.mkdir(exist_ok=True)

        def frame_iter():
            for idx, fp in enumerate(frame_files):
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                yield idx, frame

        stats = GameStats(video_name=fd.name)
        self._process_frames_iter(
            frame_iter(), fps, total, stats,
            out_frames_dir=out_frames_dir,
        )

        self._save_and_print(stats, od, fd.name)
        return stats

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_and_print(self, stats: GameStats, output_dir: Path, stem: str,
                        original_video_path: str = "",
                        calibration_override: Optional["CalibrationData"] = None):
        out_path = output_dir / f"{stem}_stats.json"
        with open(out_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2, default=str)

        # Use the scaled calibration that was actually used during processing
        calib = calibration_override or self.calibration

        # Save compact per-frame annotations separately (for HTML viewer overlay)
        ann_path = output_dir / f"{stem}_annotations.json"
        ann_data = {
            "fps": stats.fps,
            "video": stats.video_name,
            "original_video_path": original_video_path,
            "frame_width": calib.frame_width if calib else 1280,
            "frame_height": calib.frame_height if calib else 720,
            "net_y": calib.net_y if calib else 370.0,
            "net_x_left": calib.net_x_left if calib else 0.0,
            "net_x_right": calib.net_x_right if calib else 1280.0,
            "net_tape_top": calib.net_tape_top if calib else None,
            "net_tape_bottom": calib.net_tape_bottom if calib else None,
            "frames": stats.frame_annotations,
        }
        with open(ann_path, "w") as f:
            json.dump(ann_data, f, separators=(",", ":"),
                      default=lambda o: float(o) if hasattr(o, "__float__") else str(o))

        log.section(f"Done: {stem}")
        log.info(f"Frames processed : {stats.processed_frames}/{stats.total_frames}")
        log.info(f"Ball detected    : {stats.ball_detection_rate*100:.1f}%")
        log.info(f"Rallies detected : {len(stats.rallies)}")
        log.info(f"Total events     : {len(stats.all_events)}")
        log.info(f"Processing time  : {stats.processing_time_sec:.1f}s")
        log.info(f"Stats saved to   : {out_path}")
