"""
Integrated Volleyball FSM — state-aware event classification.

Valid rally scenarios (per user spec):
  serve → out (serve_out)
  serve → ace (no reception, point for server)
  serve → reception → out
  serve → reception → pass → out
  serve → reception → pass → attack
  ... after attack: block OR goal/out
  ... after block: dig → pass → attack OR out OR goal

Contact type is determined by FSM STATE, NOT by ball physics alone.
This prevents impossible sequences like serve→serve or two consecutive serves.

State transitions:
  SERVE_READY → (contact detected) → SERVING         [event: serve]
  SERVING     → (receiving team contact) → RALLY_ACTIVE  [event: reception]
  SERVING     → (ball out of frame) → RALLY_END       [event: serve_out]
  RALLY_ACTIVE → (contact #1) → RALLY_ACTIVE          [event: dig]
  RALLY_ACTIVE → (contact #2) → RALLY_ACTIVE          [event: pass]
  RALLY_ACTIVE → (contact #3) → RALLY_ACTIVE          [event: attack]
  RALLY_ACTIVE → (near-net opposing team) → RALLY_ACTIVE  [event: block]
  RALLY_ACTIVE → (ball out) → RALLY_END               [event: out]
  RALLY_END   → immediately → SERVE_READY
"""

from enum import Enum
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import math

try:
    from .enhanced_event_tracker import EnhancedEventTracker, ContactType, BallContact
except ImportError:
    from enhanced_event_tracker import EnhancedEventTracker, ContactType, BallContact

try:
    from src.utils.logger import get_logger
except ImportError:
    class _FallbackLogger:
        def __init__(self, n): self.name = n
        def info(self, m): print(f"[FSM] {m}")
        def warning(self, m): print(f"[FSM] WARN {m}")
        def debug(self, m): pass
    def get_logger(n): return _FallbackLogger(n)

_log = get_logger("FSM")


# ---------------------------------------------------------------------------
# Enums & Data classes
# ---------------------------------------------------------------------------

class GameState(Enum):
    SERVE_PREP   = "serve_prep"   # NEW: between-rally preparation phase
    SERVE_READY  = "serve_ready"  # Ready — next contact = serve
    SERVING      = "serving"
    RALLY_ACTIVE = "rally_active"
    RALLY_END    = "rally_end"
    TIMEOUT      = "timeout"     # legacy / unused
    SET_END      = "set_end"     # legacy / unused
    MATCH_END    = "match_end"   # legacy / unused


class RallyPhase(Enum):
    SERVE_RECEPTION     = "serve_reception"
    FIRST_CONTACT       = "first_contact"
    SECOND_CONTACT      = "second_contact"
    THIRD_CONTACT       = "third_contact"
    DEFENSE             = "defense"
    RALLY_CONTINUATION  = "rally_continuation"


@dataclass
class CourtBounds:
    """Volleyball court boundaries in pixel coordinates."""
    left:   float
    right:  float
    top:    float
    bottom: float
    net_y:  float

    def is_ball_in_bounds(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def get_ball_side(self, y: float) -> str:
        """Return which team's side the ball is on."""
        return "team_a" if y < self.net_y else "team_b"


@dataclass
class RallyEvent:
    """Single rally event with full context."""
    frame_idx:       int
    timestamp:       float
    event_type:      str
    ball_contact:    Optional[BallContact]
    game_state:      GameState
    rally_phase:     Optional[RallyPhase]
    team_controlling: str
    contacts_count:  int


# ---------------------------------------------------------------------------
# Main FSM
# ---------------------------------------------------------------------------

class IntegratedVolleyballFSM:
    """
    Volleyball FSM with state-aware event classification.

    Key design principle
    --------------------
    Contact TYPE is determined by the current FSM STATE and contact sequence
    within the current possession — NOT by ball physics/speed/direction alone.

    This prevents impossible sequences such as:
      • serve → serve (two consecutive serves)
      • attack → serve (serve in the middle of a rally)
      • reception → reception (double reception)

    Physics (from EnhancedEventTracker) are only used for *detecting WHEN*
    a contact occurs (speed/direction change thresholds). The *classification*
    of WHAT that contact is belongs to the FSM.
    """

    # Margin around net_y to consider a contact as a potential block (px)
    BLOCK_NET_MARGIN = 80.0

    # ── Out-of-play detection thresholds ───────────────────────────────
    # Ball moving slower than this (px/frame, averaged over last 4 detections)
    # is considered rolling or stationary → not in active play.
    ROLLING_SPEED_PX_PER_FRAME: float = 6.0
    # When ball is within this many pixels of the near/far baseline AND moving
    # nearly horizontally at moderate speed → also considered out of play.
    FLOOR_MARGIN_PX: float = 70.0
    # Number of consecutive "out of play" detections required before acting.
    # Prevents single-frame false negatives from the detector.
    OUT_OF_PLAY_CONFIRM_FRAMES: int = 3

    def __init__(self, court_bounds: CourtBounds, fps: float = 30.0):
        self.court_bounds = court_bounds
        self.fps = fps

        # ── FSM state ──────────────────────────────────────────────────
        # Start in SERVE_PREP so the first min_serve_prep_frames ignore contacts
        # (avoids false "serve" from detector noise at video start or right after rally end)
        self.current_state = GameState.SERVE_PREP
        self.serve_prep_start_frame = 0
        self.current_rally_phase: Optional[RallyPhase] = None

        # ── Enhanced event tracker (contact DETECTION only) ────────────
        self.event_tracker = EnhancedEventTracker(court_bounds, fps)

        # ── State-aware rally tracking (NEW) ───────────────────────────
        self.serve_done: bool = False
        self.serving_team: Optional[str] = None          # Team that served
        self.possessing_team: Optional[str] = None       # Team with ball now
        self.possessing_team_contacts: int = 0           # 0-3 contacts this possession

        # ── Ball trajectory (for net-crossing detection) ───────────────
        self.rally_events: List[RallyEvent] = []
        self.ball_trajectory: List[Tuple[int, float, float]] = []
        self.last_net_crossing_frame: int = -1
        self.min_net_crossing_interval: int = 15         # frames debounce

        # ── Serve parameters ───────────────────────────────────────────
        self.min_serve_speed: float = 300.0   # px/s (legacy, kept for compat)
        self.max_contacts_per_team: int = 3
        self.min_serving_frames: int = 8
        # Max frames to stay in SERVING before forcing rally end.
        # At 30fps: 150 frames = 5 s — longer than any real serve flight.
        # Prevents the FSM from getting stuck in SERVING when reception
        # is missed (e.g. ball detector lost the ball after it crossed the net).
        self.max_serving_frames: int = 150
        self.serving_start_frame: int = -1

        # ── Serve-prep parameters ───────────────────────────────────────
        # serve_prep_start_frame already set to 0 above (reset to frame_idx in _handle_rally_end)
        # Minimum frames to stay in SERVE_PREP before allowing a serve.
        # At 30fps: 45 frames ≈ 1.5s  (prevents false serve-detections right
        # after a rally ends while the ball is still bouncing/being passed back)
        self.min_serve_prep_frames: int = 45

        # ── Out-of-play detection state ─────────────────────────────────
        # Counts consecutive frames where ball appears out of active play.
        # Only triggers once it reaches OUT_OF_PLAY_CONFIRM_FRAMES.
        self._out_of_play_consecutive: int = 0

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        ball_pos: Optional[Tuple[float, float]],
        player_positions: Dict[str, List[Tuple[float, float]]],
        timestamp: float,
    ) -> List[RallyEvent]:
        """Main per-frame update. Returns new rally events for this frame."""
        new_events: List[RallyEvent] = []

        if ball_pos is not None:
            self.ball_trajectory.append((frame_idx, ball_pos[0], ball_pos[1]))

            # ── Out-of-play check ───────────────────────────────────────
            ball_in_play = self._is_ball_in_active_play(frame_idx)
            if ball_in_play:
                self._out_of_play_consecutive = 0
            else:
                self._out_of_play_consecutive += 1

            # Only detect contacts while ball is in active play
            if ball_in_play:
                # Physics-based contact detection (WHEN a contact happens)
                new_contacts = self.event_tracker.update_trajectory(
                    frame_idx, ball_pos, timestamp, player_positions
                )

                # Classify each detected contact using FSM state (WHAT it is)
                for contact in new_contacts:
                    event = self._process_ball_contact(contact, frame_idx, timestamp)
                    if event is not None:
                        new_events.append(event)
                        self.rally_events.append(event)

            # FSM transitions (net crossing, out of bounds, etc.)
            out_of_play_confirmed = (
                self._out_of_play_consecutive >= self.OUT_OF_PLAY_CONFIRM_FRAMES
            )
            self._update_fsm_state(frame_idx, ball_pos, timestamp, out_of_play_confirmed)

        return new_events

    # ──────────────────────────────────────────────────────────────────
    # Contact classification (core of the redesign)
    # ──────────────────────────────────────────────────────────────────

    def _process_ball_contact(
        self,
        contact: BallContact,
        frame_idx: int,
        timestamp: float,
    ) -> Optional[RallyEvent]:
        """Classify a detected contact using the current FSM state."""
        event_type = self._determine_event_type(contact, frame_idx)
        if event_type is None:
            return None

        self.current_rally_phase = self._event_to_phase(event_type)

        return RallyEvent(
            frame_idx=frame_idx,
            timestamp=timestamp,
            event_type=event_type,
            ball_contact=contact,
            game_state=self.current_state,
            rally_phase=self.current_rally_phase,
            team_controlling=self.possessing_team or contact.team,
            contacts_count=self.possessing_team_contacts,
        )

    def _determine_event_type(
        self, contact: BallContact, frame_idx: int
    ) -> Optional[str]:
        """
        State-aware contact classification.

        Returns the event type string, or None if this contact should be ignored.
        Also performs state transitions (SERVE_READY→SERVING, SERVING→RALLY_ACTIVE).
        """
        state = self.current_state
        ball_y = contact.ball_position[1]
        ball_side = self.court_bounds.get_ball_side(ball_y)

        # ── SERVE_READY: contact with sufficient speed = serve ─────────────
        if state == GameState.SERVE_READY:
            # Require minimum post-contact speed to avoid noise/flicker as serve
            if contact.ball_speed_after < self.min_serve_speed:
                _log.debug(
                    f"f{frame_idx}: contact ignored (speed {contact.ball_speed_after:.0f} < {self.min_serve_speed})"
                )
                return None
            self.serving_team = ball_side
            self.serve_done = True
            self.possessing_team = None
            self.possessing_team_contacts = 0

            # Transition: SERVE_READY → SERVING
            self.current_state = GameState.SERVING
            self.serving_start_frame = frame_idx

            _log.info(f"f{frame_idx}: SERVE by {self.serving_team}")
            return "serve"

        # ── SERVING: receiving-team contact = reception ─────────────────
        elif state == GameState.SERVING:
            receiving_team = (
                "team_b" if self.serving_team == "team_a" else "team_a"
            )
            if ball_side == receiving_team:
                # Reception!
                self.possessing_team = receiving_team
                self.possessing_team_contacts = 1

                # Transition: SERVING → RALLY_ACTIVE
                self.current_state = GameState.RALLY_ACTIVE

                _log.info(f"f{frame_idx}: RECEPTION by {receiving_team}")
                return "reception"
            else:
                # Serving team touched again (noise / fault) → ignore
                return None

        # ── RALLY_ACTIVE: sequence-based classification ─────────────────
        elif state == GameState.RALLY_ACTIVE:

            # Check for BLOCK: near net, OTHER team touching, after ≥2 contacts
            near_net = abs(ball_y - self.court_bounds.net_y) < self.BLOCK_NET_MARGIN
            if (
                near_net
                and self.possessing_team is not None
                and ball_side != self.possessing_team
                and self.possessing_team_contacts >= 2
            ):
                # In volleyball a block does NOT count as one of the 3 contacts.
                # Blocking team gets fresh possession with 3 contacts available.
                _log.info(
                    f"f{frame_idx}: BLOCK by {ball_side} "
                    f"(defender had {self.possessing_team_contacts} contacts)"
                )
                self.possessing_team = ball_side
                self.possessing_team_contacts = 0  # Block doesn't count
                return "block"

            # Normal rally contact
            if ball_side != self.possessing_team:
                # Team changed (net crossing not caught by trajectory watcher)
                self.possessing_team = ball_side
                self.possessing_team_contacts = 0

            self.possessing_team_contacts += 1

            if self.possessing_team_contacts == 1:
                ev = "dig"
            elif self.possessing_team_contacts == 2:
                ev = "pass"
            else:
                ev = "attack"
                # After attack: reset count so next contact by same team restarts as
                # dig (handles ball bouncing back, blocked ball, etc.)
                self.possessing_team_contacts = 0

            _log.info(
                f"f{frame_idx}: {ev.upper()} by {ball_side} "
                f"(#{self.possessing_team_contacts if ev != 'attack' else 3})"
            )
            return ev

        # Any other state → ignore contact
        return None

    @staticmethod
    def _event_to_phase(event_type: str) -> Optional[RallyPhase]:
        return {
            "serve":     None,
            "reception": RallyPhase.SERVE_RECEPTION,
            "dig":       RallyPhase.FIRST_CONTACT,
            "pass":      RallyPhase.SECOND_CONTACT,
            "attack":    RallyPhase.THIRD_CONTACT,
            "block":     RallyPhase.DEFENSE,
        }.get(event_type)

    # ──────────────────────────────────────────────────────────────────
    # FSM state machine (transitions & timeouts)
    # ──────────────────────────────────────────────────────────────────

    def _update_fsm_state(
        self,
        frame_idx: int,
        ball_pos: Tuple[float, float],
        timestamp: float,
        out_of_play_confirmed: bool = False,
    ) -> None:
        if self.current_state == GameState.SERVE_PREP:
            self._handle_serve_prep(frame_idx, ball_pos)

        elif self.current_state == GameState.SERVE_READY:
            # No speed-based transition here.
            # State change happens via _determine_event_type on contact.
            pass

        elif self.current_state == GameState.SERVING:
            self._handle_serving(frame_idx, ball_pos, out_of_play_confirmed)

        elif self.current_state == GameState.RALLY_ACTIVE:
            self._handle_rally_active(frame_idx, ball_pos, out_of_play_confirmed)

        elif self.current_state == GameState.RALLY_END:
            self._handle_rally_end(frame_idx, ball_pos)

    def _handle_serving(
        self,
        frame_idx: int,
        ball_pos: Tuple[float, float],
        out_of_play_confirmed: bool = False,
    ) -> None:
        """SERVING state: watch for net crossing & out-of-bounds.

        NOTE: out_of_play_confirmed is intentionally NOT used here.
        A volleyball in flight naturally slows at the apex of its arc; applying
        the rolling/stationary heuristic here causes false rally endings.
        Ball disappearance is handled separately by NO_BALL_RALLY_TIMEOUT and
        MAX_SERVING_FRAMES in the game analyzer / FSM timeout.
        """
        if self._ball_crossed_net():
            # Ball is now on receiving side — wait for reception contact
            _log.debug(f"f{frame_idx}: Serve crossed net (awaiting reception)")
            return

        frames_serving = frame_idx - self.serving_start_frame
        if frames_serving < self.min_serving_frames:
            return

        # Serve timeout: if serve has been in flight too long without reception,
        # end the rally (handles cases where reception was never detected).
        if frames_serving >= self.max_serving_frames:
            _log.info(f"f{frame_idx}: Serve timeout after {frames_serving} frames → rally end")
            self._end_rally(frame_idx, "serve_timeout")
            return

        # Serve out of frame bounds
        x, y = ball_pos
        w, h = self.court_bounds.right, self.court_bounds.bottom
        if x < -30 or x > w + 30 or y < -30 or y > h + 30:
            self._end_rally(frame_idx, "serve_out")

    def _handle_rally_active(
        self,
        frame_idx: int,
        ball_pos: Tuple[float, float],
        out_of_play_confirmed: bool = False,
    ) -> None:
        """RALLY_ACTIVE state: watch net crossings & out-of-bounds."""
        if self._ball_crossed_net():
            self._handle_net_crossing(frame_idx, ball_pos)

        # Ball rolling/stationary → end the rally
        if out_of_play_confirmed:
            _log.info(f"f{frame_idx}: Ball out of play → rally end")
            self._end_rally(frame_idx, "ball_out_of_play")
            return

        x, y = ball_pos
        w, h = self.court_bounds.right, self.court_bounds.bottom
        if x < -50 or x > w + 50 or y < -50 or y > h + 50:
            self._end_rally(frame_idx, "ball_out")

    def _handle_rally_end(
        self, frame_idx: int, ball_pos: Tuple[float, float]
    ) -> None:
        """RALLY_END: reset and enter serve-preparation phase."""
        self._reset_rally_state()
        self.current_state = GameState.SERVE_PREP
        self.serve_prep_start_frame = frame_idx
        _log.info(f"f{frame_idx}: SERVE_PREP (preparing for next serve)")

    def _handle_serve_prep(
        self, frame_idx: int, ball_pos: Tuple[float, float]
    ) -> None:
        """SERVE_PREP: wait minimum frames, then become SERVE_READY.

        During this window contacts are ignored (return None from
        _determine_event_type) so false ball detections immediately after a
        rally end don't accidentally register as a new serve.
        """
        frames_elapsed = frame_idx - self.serve_prep_start_frame
        if frames_elapsed >= self.min_serve_prep_frames:
            self.current_state = GameState.SERVE_READY
            _log.info(
                f"f{frame_idx}: SERVE_READY "
                f"(prep complete after {frames_elapsed} frames)"
            )

    # ──────────────────────────────────────────────────────────────────
    # Out-of-play detection
    # ──────────────────────────────────────────────────────────────────

    def _is_ball_in_active_play(self, frame_idx: int) -> bool:
        """
        Heuristic: is the ball currently in active play (flying / being hit)?

        Returns False when the ball appears to be:
          • rolling or stationary on the floor  (speed < ROLLING_SPEED_PX_PER_FRAME)
          • sitting near a baseline and moving nearly horizontally at low speed
            (being carried / walked back after a point)

        Uses the last 4 trajectory points so momentary detector noise doesn't
        incorrectly suppress contacts.
        """
        traj = self.ball_trajectory
        if len(traj) < 4:
            return True  # not enough history → assume active

        recent = traj[-4:]
        f0, x0, y0 = recent[0]
        fn, xn, yn = recent[-1]
        frame_span = fn - f0
        if frame_span == 0:
            return True

        dist = math.sqrt((xn - x0) ** 2 + (yn - y0) ** 2)
        speed = dist / frame_span  # px / frame

        # Very slow → rolling or stationary
        if speed < self.ROLLING_SPEED_PX_PER_FRAME:
            return False

        # Near the baseline AND mostly horizontal → being carried
        cb = self.court_bounds
        near_floor = (
            yn > cb.bottom - self.FLOOR_MARGIN_PX
            or yn < cb.top + self.FLOOR_MARGIN_PX
        )
        if near_floor and speed < 20.0:
            dx = abs(xn - x0)
            dy = abs(yn - y0)
            if dx > 0 and dy / (dx + 1e-6) < 0.25:
                return False

        return True

    # ──────────────────────────────────────────────────────────────────
    # Net crossing helpers
    # ──────────────────────────────────────────────────────────────────

    def _ball_crossed_net(self) -> bool:
        if len(self.ball_trajectory) < 2:
            return False

        current_frame = self.ball_trajectory[-1][0]
        if current_frame - self.last_net_crossing_frame < self.min_net_crossing_interval:
            return False

        last_y  = self.ball_trajectory[-1][2]
        prev_y  = self.ball_trajectory[-2][2]
        net_y   = self.court_bounds.net_y

        crossed = (
            (prev_y < net_y and last_y > net_y) or
            (prev_y > net_y and last_y < net_y)
        )
        if crossed:
            self.last_net_crossing_frame = current_frame
        return crossed

    def _handle_net_crossing(
        self, frame_idx: int, ball_pos: Tuple[float, float]
    ) -> None:
        """Ball crossed net during RALLY_ACTIVE → switch possession."""
        new_side = self.court_bounds.get_ball_side(ball_pos[1])
        if new_side != self.possessing_team:
            old = self.possessing_team
            self.possessing_team = new_side
            self.possessing_team_contacts = 0
            _log.info(f"f{frame_idx}: Net crossing {old} -> {new_side}")

    # ──────────────────────────────────────────────────────────────────
    # Rally management
    # ──────────────────────────────────────────────────────────────────

    def _end_rally(self, frame_idx: int, reason: str) -> None:
        """Transition to RALLY_END."""
        self.current_state = GameState.RALLY_END
        self.current_rally_phase = None
        _log.warning(f"f{frame_idx}: Rally ended -> {reason}")

    def _reset_rally_state(self) -> None:
        """Reset all per-rally tracking."""
        self.current_rally_phase = None
        self.rally_events.clear()
        self.ball_trajectory.clear()
        self.last_net_crossing_frame = -1
        self.event_tracker.reset_rally()

        # State-aware tracking reset
        self.serve_done = False
        self.serving_team = None
        self.possessing_team = None
        self.possessing_team_contacts = 0
        self._out_of_play_consecutive = 0

    # ──────────────────────────────────────────────────────────────────
    # Status / legacy compatibility
    # ──────────────────────────────────────────────────────────────────

    def get_current_status(self) -> Dict[str, Any]:
        traj  = self.event_tracker.get_trajectory_analysis()
        recnt = self.event_tracker.get_recent_contacts(3)
        return {
            "game_state":         self.current_state.value,
            "rally_phase":        self.current_rally_phase.value if self.current_rally_phase else None,
            "current_team":       self.possessing_team or self.serving_team or "team_a",
            "team_contacts":      self.possessing_team_contacts,
            "max_contacts_reached": self.possessing_team_contacts >= 3,
            "serving_team":       self.serving_team,
            "possessing_team":    self.possessing_team,
            "recent_contacts": [
                {
                    "type":       c.contact_type.value,
                    "sequence":   c.contact_sequence,
                    "team":       c.team,
                    "confidence": c.contact_confidence,
                }
                for c in recnt
            ],
            "trajectory": traj,
        }

    def get_rally_summary(self) -> Dict[str, Any]:
        if not self.rally_events:
            return {}

        event_counts: Dict[str, int] = {}
        team_contacts = {"team_a": 0, "team_b": 0}

        for ev in self.rally_events:
            event_counts[ev.event_type] = event_counts.get(ev.event_type, 0) + 1
            if ev.team_controlling in team_contacts:
                team_contacts[ev.team_controlling] += 1

        return {
            "total_events":    len(self.rally_events),
            "event_types":     event_counts,
            "team_contacts":   team_contacts,
            "rally_duration_frames": len(self.ball_trajectory),
            "current_phase":   self.current_rally_phase.value if self.current_rally_phase else None,
        }

    # Legacy property kept for backward compatibility
    @property
    def current_team(self) -> str:
        return self.possessing_team or self.serving_team or "team_a"
