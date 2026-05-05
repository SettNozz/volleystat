import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

try:
    from .volleyball_fsm import ActionType, RallyPhase
except ImportError:
    from volleyball_fsm import ActionType, RallyPhase


@dataclass
class PlayerContact:
    """Represents a player contact with the ball"""
    frame_idx: int
    player_position: Tuple[float, float]
    ball_position: Tuple[float, float]
    contact_type: ActionType
    team: str
    player_id: Optional[int] = None


class ContactDetector:
    """Detects player contacts with the ball"""
    
    def __init__(self, contact_distance_threshold: float = 50.0):
        self.contact_distance_threshold = contact_distance_threshold
        self.last_contact_frame = -1
        self.contact_cooldown = 5  # Minimum frames between contacts
    
    def detect_contact(self, frame_idx: int, ball_pos: Tuple[float, float], 
                      player_positions: Dict[str, List[Tuple[float, float]]]) -> Optional[PlayerContact]:
        """Detect if any player is in contact with the ball"""
        
        # Skip if too soon after last contact
        if frame_idx - self.last_contact_frame < self.contact_cooldown:
            return None
        
        min_distance = float('inf')
        closest_player = None
        closest_team = None
        closest_idx = None
        
        # Check all players from both teams
        for team, positions in player_positions.items():
            for idx, player_pos in enumerate(positions):
                distance = self._calculate_distance(ball_pos, player_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_player = player_pos
                    closest_team = team
                    closest_idx = idx
        
        # If closest player is within threshold, register contact
        if min_distance <= self.contact_distance_threshold:
            self.last_contact_frame = frame_idx
            
            # Determine contact type based on context (simplified for now)
            contact_type = self._determine_contact_type(ball_pos, closest_player)
            
            return PlayerContact(
                frame_idx=frame_idx,
                player_position=closest_player,
                ball_position=ball_pos,
                contact_type=contact_type,
                team=closest_team,
                player_id=closest_idx
            )
        
        return None
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _determine_contact_type(self, ball_pos: Tuple[float, float], 
                               player_pos: Tuple[float, float]) -> ActionType:
        """Determine the type of contact based on positions (simplified)"""
        # This is a simplified version - in reality, you'd use more sophisticated logic
        # including ball velocity, player movement, court position, etc.
        
        # For now, return basic action types
        return ActionType.PASS  # Default to pass


class RallyAnalyzer:
    """Analyzes rally phases and determines appropriate actions"""
    
    def __init__(self, court_bounds, net_proximity_threshold: float = 100.0):
        self.court_bounds = court_bounds
        self.net_proximity_threshold = net_proximity_threshold
        self.contact_detector = ContactDetector()
        
        # Rally state tracking
        self.team_contacts = {"team_a": 0, "team_b": 0}
        self.last_contact_team = None
        self.current_team_contact_count = 0
        
        # Ball trajectory analysis
        self.ball_velocity_history = []
        self.ball_height_threshold = 200.0  # Threshold for determining ball height
    
    def analyze_rally_phase(self, frame_idx: int, ball_pos: Tuple[float, float],
                           player_positions: Dict[str, List[Tuple[float, float]]],
                           current_team: str, contact_count: int) -> Tuple[RallyPhase, Optional[ActionType]]:
        """Analyze current rally phase and determine action type"""
        
        # Detect player contact
        contact = self.contact_detector.detect_contact(frame_idx, ball_pos, player_positions)
        
        if contact:
            # Update contact tracking
            if contact.team != self.last_contact_team:
                # New team touched the ball
                self.last_contact_team = contact.team
                self.current_team_contact_count = 1
            else:
                # Same team continues
                self.current_team_contact_count += 1
            
            # Determine rally phase based on contact count and context
            rally_phase = self._determine_rally_phase(contact, ball_pos)
            action_type = self._refine_action_type(contact, ball_pos, player_positions)
            
            return rally_phase, action_type
        
        # No contact detected - analyze ball movement
        ball_velocity = self._calculate_ball_velocity(frame_idx, ball_pos)
        
        # Determine phase based on ball movement and position
        if self._is_ball_near_net(ball_pos):
            if ball_velocity > 15.0:  # Fast ball near net
                return RallyPhase.ATTACKING, ActionType.ATTACK
            else:
                return RallyPhase.SETTING, ActionType.SET
        
        # Default phases based on ball position and team
        ball_side = self.court_bounds.get_ball_side(ball_pos[1])
        if contact_count == 0:
            return RallyPhase.RECEPTION, None
        elif contact_count == 1:
            return RallyPhase.SETTING, None
        else:
            return RallyPhase.ATTACKING, None
    
    def _determine_rally_phase(self, contact: PlayerContact, ball_pos: Tuple[float, float]) -> RallyPhase:
        """Determine rally phase based on contact and ball position"""
        
        # Check if player is near net (potential block/attack)
        if self._is_player_near_net(contact.player_position):
            if self._is_ball_high(ball_pos):
                return RallyPhase.ATTACKING
            else:
                return RallyPhase.BLOCKING
        
        # Check ball trajectory and contact sequence
        if self.current_team_contact_count == 1:
            return RallyPhase.RECEPTION
        elif self.current_team_contact_count == 2:
            return RallyPhase.SETTING
        elif self.current_team_contact_count == 3:
            return RallyPhase.ATTACKING
        else:
            return RallyPhase.EMERGENCY_PLAY
    
    def _refine_action_type(self, contact: PlayerContact, ball_pos: Tuple[float, float],
                           player_positions: Dict[str, List[Tuple[float, float]]]) -> ActionType:
        """Refine action type based on detailed analysis"""
        
        ball_velocity = self._get_current_ball_velocity()
        
        # High velocity near net = attack
        if ball_velocity > 20.0 and self._is_ball_near_net(ball_pos):
            return ActionType.ATTACK
        
        # Player near net with ball crossing = block
        if (self._is_player_near_net(contact.player_position) and 
            self._is_ball_crossing_net(ball_pos)):
            return ActionType.BLOCK
        
        # Low ball = dig
        if ball_pos[1] > self.ball_height_threshold:
            return ActionType.DIG
        
        # Ball going up = set
        if self._is_ball_going_up():
            return ActionType.SET
        
        # Default = pass
        return ActionType.PASS
    
    def _calculate_ball_velocity(self, frame_idx: int, ball_pos: Tuple[float, float]) -> float:
        """Calculate current ball velocity"""
        if len(self.ball_velocity_history) < 2:
            self.ball_velocity_history.append((frame_idx, ball_pos))
            return 0.0
        
        # Keep only recent history
        if len(self.ball_velocity_history) > 10:
            self.ball_velocity_history.pop(0)
        
        self.ball_velocity_history.append((frame_idx, ball_pos))
        
        # Calculate velocity from last two points
        last_frame, last_pos = self.ball_velocity_history[-1]
        prev_frame, prev_pos = self.ball_velocity_history[-2]
        
        if last_frame == prev_frame:
            return 0.0
        
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        dt = last_frame - prev_frame
        
        return np.sqrt(dx*dx + dy*dy) / dt if dt > 0 else 0.0
    
    def _get_current_ball_velocity(self) -> float:
        """Get current ball velocity from history"""
        if len(self.ball_velocity_history) < 2:
            return 0.0
        
        last_frame, last_pos = self.ball_velocity_history[-1]
        prev_frame, prev_pos = self.ball_velocity_history[-2]
        
        if last_frame == prev_frame:
            return 0.0
        
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        dt = last_frame - prev_frame
        
        return np.sqrt(dx*dx + dy*dy) / dt if dt > 0 else 0.0
    
    def _is_ball_near_net(self, ball_pos: Tuple[float, float]) -> bool:
        """Check if ball is near the net"""
        return abs(ball_pos[1] - self.court_bounds.net_y) < self.net_proximity_threshold
    
    def _is_player_near_net(self, player_pos: Tuple[float, float]) -> bool:
        """Check if player is near the net"""
        return abs(player_pos[1] - self.court_bounds.net_y) < self.net_proximity_threshold
    
    def _is_ball_high(self, ball_pos: Tuple[float, float]) -> bool:
        """Check if ball is at attacking height (simplified - in reality would use 3D coordinates)"""
        # This is a placeholder - in real implementation you'd have ball height from 3D tracking
        return True  # Assume ball is always at playable height for now
    
    def _is_ball_crossing_net(self, ball_pos: Tuple[float, float]) -> bool:
        """Check if ball is currently crossing the net"""
        return abs(ball_pos[1] - self.court_bounds.net_y) < 20.0  # Very close to net
    
    def _is_ball_going_up(self) -> bool:
        """Check if ball trajectory is upward (simplified)"""
        if len(self.ball_velocity_history) < 3:
            return False
        
        # Compare recent Y positions to determine if ball is rising
        recent_positions = [pos[1][1] for _, pos in self.ball_velocity_history[-3:]]
        return recent_positions[-1] < recent_positions[0]  # Y decreasing = going up (inverted coordinates)
    
    def reset(self) -> None:
        """Reset analyzer state for new rally"""
        self.team_contacts = {"team_a": 0, "team_b": 0}
        self.last_contact_team = None
        self.current_team_contact_count = 0
        self.ball_velocity_history = []
        self.contact_detector.last_contact_frame = -1 