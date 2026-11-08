"""Result and calibration data structures produced by GameAnalyzer."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _EventOverlayState:
    """Per-pass display policy shared by provisional and final predictions."""
    fps: float
    last_serve_frame: Optional[int] = None
    last_rally_display: str = ""
    last_rally_frame: int = 0

    def update(self, raw_display: Optional[str], state: str, frame: int) -> Optional[str]:
        if raw_display is not None and state in {"serving", "rally_active"}:
            if raw_display == "serve":
                if (self.last_serve_frame is not None
                        and frame - self.last_serve_frame < int(self.fps * 8.0)):
                    return None
                self.last_serve_frame = frame
            self.last_rally_display = raw_display
            self.last_rally_frame = frame
            return raw_display
        if (raw_display is not None
                and state not in {"serving", "rally_active"}
                and self.last_rally_display
                and self.last_rally_display != "serve"
                and frame - self.last_rally_frame < int(self.fps * 2.0)):
            return self.last_rally_display
        return None


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
    # Ground-level line between the two net-post bases. Unlike the tape, this
    # line lies on the court and can reliably divide the teams.
    net_floor_left: Optional[Tuple[float, float]] = None
    net_floor_right: Optional[Tuple[float, float]] = None
    # SAM3 post X plus the configured physical net-height geometry is
    # trusted; older Hough values are refreshed before becoming the divider.
    net_floor_method: Optional[str] = None
    net_height_m: float = 2.34
    net_post_spacing_m: Optional[float] = None
    # Homography world (court meters) → image pixels, scaled to current frame.
    # Enables the on-court player filter and court-half team split.
    homography: Optional[List[List[float]]] = None
    # Full court outline in image pixels:
    # front-left, front-right, back-right, back-left.
    court_corners: Optional[List[Tuple[float, float]]] = None
    net_tape_endpoint_y: Optional[Tuple[float, float]] = None


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
    # Confirmed playable rallies only. Service attempts that end before a
    # controlled follow-up are kept separately instead of inflating this list.
    rallies: List[Dict] = field(default_factory=list)
    service_attempts: List[Dict] = field(default_factory=list)
    rally_lifecycle: Dict[str, int] = field(default_factory=dict)
    match_analytics: Dict[str, Any] = field(default_factory=dict)
    all_events: List[Dict] = field(default_factory=list)
    # With the dense contact spotter enabled it owns ``all_events``, and the
    # FSM's own contacts are kept here so nothing is lost by the swap.
    all_events_fsm: List[Dict] = field(default_factory=list)
    # Likewise the spotter's serve-anchored rallies replace the FSM's, which
    # stay here. Empty when the spotter did not run.
    rallies_fsm: List[Dict] = field(default_factory=list)
    service_attempts_fsm: List[Dict] = field(default_factory=list)
    rally_lifecycle_fsm: Dict[str, int] = field(default_factory=dict)
    # Spotter diagnostics, provenance and decoded rally phases.
    contact_spotter: Dict[str, Any] = field(default_factory=dict)
    player_identity: Dict[str, Any] = field(default_factory=dict)
    player_contacts: Dict[str, Any] = field(default_factory=dict)
    model_cache: Dict[str, Any] = field(default_factory=dict)
    # Per-frame data for HTML viewer overlay (ball pos, players, FSM state)
    frame_annotations: List[Dict] = field(default_factory=list)
    # Trajectory-only contact/attack candidates aggregated into court zones.
    attack_heatmap: Dict[str, Any] = field(default_factory=dict)
    ball_reconstruction: Dict[str, Any] = field(default_factory=dict)
    # Window used, points moved, and the jitter measured before/after.
    ball_smoothing: Dict[str, Any] = field(default_factory=dict)
    # Fixed objects the detector fired on, and what was dropped because of it.
    ball_static_suppression: Dict[str, Any] = field(default_factory=dict)
    # Which weights produced this result: registry name, version, source.
    models: Dict[str, Any] = field(default_factory=dict)
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
            "service_attempts": self.service_attempts,
            "rally_lifecycle": self.rally_lifecycle,
            "match_analytics": self.match_analytics,
            "all_events": self.all_events,
            "all_events_fsm": self.all_events_fsm,
            "rallies_fsm": self.rallies_fsm,
            "service_attempts_fsm": self.service_attempts_fsm,
            "rally_lifecycle_fsm": self.rally_lifecycle_fsm,
            "contact_spotter": self.contact_spotter,
            "player_identity": self.player_identity,
            "player_contacts": self.player_contacts,
            "model_cache": self.model_cache,
            "attack_heatmap": self.attack_heatmap,
            "ball_reconstruction": self.ball_reconstruction,
            "ball_smoothing": self.ball_smoothing,
            "ball_static_suppression": self.ball_static_suppression,
            "models": self.models,
            "processing_time_sec": round(self.processing_time_sec, 1),
        }
