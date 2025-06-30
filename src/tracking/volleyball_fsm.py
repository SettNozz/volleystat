from enum import Enum
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass


class GameState(Enum):
    """Main volleyball game states"""
    SERVE_READY = "serve_ready"
    SERVING = "serving"
    RALLY_ACTIVE = "rally_active"
    RALLY_END = "rally_end"
    TIMEOUT = "timeout"
    SET_END = "set_end"
    MATCH_END = "match_end"


class RallyPhase(Enum):
    """Rally phases during RALLY_ACTIVE state"""
    RECEPTION = "reception"          # Serve reception
    SETTING = "setting"              # Setting/passing
    ATTACKING = "attacking"          # Attack
    BLOCKING = "blocking"            # Blocking
    DEFENSE = "defense"              # Defense
    EMERGENCY_PLAY = "emergency_play" # Emergency play


class ActionType(Enum):
    """Types of ball actions"""
    SERVE = "serve"
    PASS = "pass"
    SET = "set"
    ATTACK = "attack"
    BLOCK = "block"
    DIG = "dig"


@dataclass
class CourtBounds:
    """Volleyball court boundaries"""
    left: float
    right: float
    top: float
    bottom: float
    net_y: float  # Y-coordinate of the net
    
    def is_ball_in_bounds(self, x: float, y: float) -> bool:
        """Check if ball is within court boundaries"""
        return (self.left <= x <= self.right and 
                self.top <= y <= self.bottom)
    
    def get_ball_side(self, y: float) -> str:
        """Determine which side of the court the ball is on"""
        return "team_a" if y < self.net_y else "team_b"


@dataclass
class GameEvent:
    """Game event data structure"""
    timestamp: float
    frame_idx: int
    event_type: str
    ball_position: Tuple[float, float]
    player_positions: List[Tuple[float, float]]
    team: str
    action_type: Optional[ActionType] = None
    contact_count: int = 0


class VolleyballFSM:
    """Finite State Machine for volleyball game analysis"""
    
    def __init__(self, court_bounds: CourtBounds):
        self.current_state = GameState.SERVE_READY
        self.current_rally_phase = None
        self.court_bounds = court_bounds
        
        # Rally state tracking
        self.current_team = "team_a"  # Team controlling the ball
        self.contact_count = 0
        self.ball_trajectory = []  # (frame_idx, x, y)
        self.player_positions = {}  # {team: [(x, y), ...]}
        
        # Event history
        self.events = []
        self.rally_events = []  # Current rally events
        
        # Net crossing tracking
        self.last_net_crossing_frame = -1  # Track when net was last crossed
        
        # Rally analyzer for detailed phase detection
        try:
            from rally_analyzer import RallyAnalyzer
            self.rally_analyzer = RallyAnalyzer(court_bounds)
        except ImportError:
            self.rally_analyzer = None
            print("WARNING: RallyAnalyzer not available, using basic phase detection")
        
        # Analysis parameters
        self.min_ball_speed_threshold = 5.0  # Minimum speed for action detection
        self.net_crossing_threshold = 10.0  # Threshold for net crossing detection
        
    def update(self, frame_idx: int, ball_pos: Optional[Tuple[float, float]], 
               player_positions: Dict[str, List[Tuple[float, float]]]) -> None:
        """Update FSM state based on new data"""
        
        if ball_pos is not None:
            self.ball_trajectory.append((frame_idx, ball_pos[0], ball_pos[1]))
        
        self.player_positions = player_positions
        
        # Analyze current situation
        self._analyze_current_situation(frame_idx, ball_pos)
        
    def _analyze_current_situation(self, frame_idx: int, 
                                   ball_pos: Optional[Tuple[float, float]]) -> None:
        """Analyze current situation and update state"""
        
        if ball_pos is None:
            return
            
        # Main state transition logic
        if self.current_state == GameState.SERVE_READY:
            self._handle_serve_ready(frame_idx, ball_pos)
        elif self.current_state == GameState.SERVING:
            self._handle_serving(frame_idx, ball_pos)
        elif self.current_state == GameState.RALLY_ACTIVE:
            self._handle_rally_active(frame_idx, ball_pos)
        elif self.current_state == GameState.RALLY_END:
            self._handle_rally_end(frame_idx, ball_pos)
    
    def _handle_serve_ready(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Handle serve ready state"""
        # If ball moves with sufficient speed, serve begins
        ball_speed = self._calculate_ball_speed()
        if ball_speed > self.min_ball_speed_threshold:
            self._transition_to_serving(frame_idx, ball_pos)
    
    def _handle_serving(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Handle serving state"""
        # Check if ball crossed net or went out of bounds
        if not self.court_bounds.is_ball_in_bounds(ball_pos[0], ball_pos[1]):
            self._transition_to_rally_end(frame_idx, ball_pos, "serve_out")
        elif self._ball_crossed_net():
            self._transition_to_rally_active(frame_idx, ball_pos, RallyPhase.RECEPTION)
    
    def _handle_rally_active(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Handle active rally state"""
        # Main rally phase analysis logic
        current_side = self.court_bounds.get_ball_side(ball_pos[1])
        
        # Check for ball side change
        if self._ball_crossed_net():
            self._handle_net_crossing(frame_idx, ball_pos)
        
        # Check if ball went out of bounds
        if not self.court_bounds.is_ball_in_bounds(ball_pos[0], ball_pos[1]):
            self._transition_to_rally_end(frame_idx, ball_pos, "ball_out")
        
        # Analyze rally phases
        self._analyze_rally_phase(frame_idx, ball_pos)
    
    def _handle_rally_end(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Handle rally end state"""
        # Prepare for next serve
        self._reset_rally_state()
        self.current_state = GameState.SERVE_READY
    
    def _calculate_ball_speed(self) -> float:
        """Calculate ball speed based on recent positions"""
        if len(self.ball_trajectory) < 2:
            return 0.0
            
        last_pos = self.ball_trajectory[-1]
        prev_pos = self.ball_trajectory[-2]
        
        dx = last_pos[1] - prev_pos[1]
        dy = last_pos[2] - prev_pos[2]
        
        return np.sqrt(dx*dx + dy*dy)
    
    def _ball_crossed_net(self) -> bool:
        """Check if ball crossed the net"""
        if len(self.ball_trajectory) < 2:
            return False
            
        current_frame = self.ball_trajectory[-1][0]
        
        # Don't check same frame twice
        if current_frame == self.last_net_crossing_frame:
            return False
            
        last_y = self.ball_trajectory[-1][2]
        prev_y = self.ball_trajectory[-2][2]
        net_y = self.court_bounds.net_y
        
        # Check if ball crossed net line
        crossed = ((prev_y < net_y and last_y > net_y) or 
                   (prev_y > net_y and last_y < net_y))
        
        # Debug information
        if crossed:
            print(f"DEBUG: Ball crossed net! prev_y={prev_y:.1f}, last_y={last_y:.1f}, net_y={net_y}")
            self.last_net_crossing_frame = current_frame
        
        return crossed
    
    def _transition_to_serving(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Transition to serving state"""
        self.current_state = GameState.SERVING
        self._add_event(frame_idx, "serve_start", ball_pos, ActionType.SERVE)
    
    def _transition_to_rally_active(self, frame_idx: int, ball_pos: Tuple[float, float], 
                                   phase: RallyPhase) -> None:
        """Transition to active rally state"""
        self.current_state = GameState.RALLY_ACTIVE
        self.current_rally_phase = phase
        self.contact_count = 0
        self._add_event(frame_idx, "rally_start", ball_pos)
    
    def _transition_to_rally_end(self, frame_idx: int, ball_pos: Tuple[float, float], 
                                reason: str) -> None:
        """Transition to rally end state"""
        self.current_state = GameState.RALLY_END
        self._add_event(frame_idx, f"rally_end_{reason}", ball_pos)
    
    def _handle_net_crossing(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Handle ball crossing the net"""
        # Switch controlling team
        self.current_team = "team_b" if self.current_team == "team_a" else "team_a"
        self.contact_count = 0
        
        # Determine new phase - always RECEPTION when ball crosses net
        self.current_rally_phase = RallyPhase.RECEPTION
        
        self._add_event(frame_idx, "net_crossing", ball_pos)
    
    def _analyze_rally_phase(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Analyze current rally phase"""
        if self.rally_analyzer is not None:
            # Use detailed rally analyzer
            new_phase, action_type = self.rally_analyzer.analyze_rally_phase(
                frame_idx, ball_pos, self.player_positions, 
                self.current_team, self.contact_count
            )
            
            # Update phase if it changed
            if new_phase != self.current_rally_phase:
                old_phase = self.current_rally_phase
                self.current_rally_phase = new_phase
                
                # Add phase transition event
                event_type = f"phase_change_{old_phase.value if old_phase else 'none'}_to_{new_phase.value}"
                self._add_event(frame_idx, event_type, ball_pos, action_type)
                
                print(f"DEBUG: Phase changed from {old_phase} to {new_phase} with action {action_type}")
            
            # Add action event if detected
            if action_type is not None:
                self._add_event(frame_idx, f"action_{action_type.value}", ball_pos, action_type)
        else:
            # Basic phase determination based on ball position and contact count
            self._basic_phase_analysis(frame_idx, ball_pos)
    
    def _basic_phase_analysis(self, frame_idx: int, ball_pos: Tuple[float, float]) -> None:
        """Basic rally phase analysis when detailed analyzer is not available"""
        ball_side = self.court_bounds.get_ball_side(ball_pos[1])
        
        # Simple phase logic based on ball position
        if abs(ball_pos[1] - self.court_bounds.net_y) < 50:  # Near net
            if self.current_rally_phase != RallyPhase.ATTACKING:
                self.current_rally_phase = RallyPhase.ATTACKING
                self._add_event(frame_idx, "phase_change_to_attacking", ball_pos)
        elif self.contact_count == 0:
            if self.current_rally_phase != RallyPhase.RECEPTION:
                self.current_rally_phase = RallyPhase.RECEPTION
                self._add_event(frame_idx, "phase_change_to_reception", ball_pos)
        else:
            if self.current_rally_phase != RallyPhase.SETTING:
                self.current_rally_phase = RallyPhase.SETTING
                self._add_event(frame_idx, "phase_change_to_setting", ball_pos)
    
    def _add_event(self, frame_idx: int, event_type: str, ball_pos: Tuple[float, float], 
                   action_type: Optional[ActionType] = None) -> None:
        """Add event to history"""
        event = GameEvent(
            timestamp=frame_idx,  # Can be converted to real time
            frame_idx=frame_idx,
            event_type=event_type,
            ball_position=ball_pos,
            player_positions=list(self.player_positions.get(self.current_team, [])),
            team=self.current_team,
            action_type=action_type,
            contact_count=self.contact_count
        )
        
        self.events.append(event)
        self.rally_events.append(event)
    
    def _reset_rally_state(self) -> None:
        """Reset rally state"""
        self.current_rally_phase = None
        self.contact_count = 0
        self.rally_events = []
        self.last_net_crossing_frame = -1
        
        # Reset rally analyzer if available
        if self.rally_analyzer is not None:
            self.rally_analyzer.reset()
    
    def get_current_state(self) -> GameState:
        """Get current game state"""
        return self.current_state
    
    def get_current_rally_phase(self) -> Optional[RallyPhase]:
        """Get current rally phase"""
        return self.current_rally_phase
    
    def get_events(self) -> List[GameEvent]:
        """Get all game events"""
        return self.events
    
    def get_rally_events(self) -> List[GameEvent]:
        """Get current rally events"""
        return self.rally_events 