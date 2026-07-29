from enum import Enum
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
import math


class ContactType(Enum):
    """Types of ball contacts"""
    PASS = "pass"
    ATTACK = "attack"
    SERVE = "serve"
    BLOCK = "block"
    DIG = "dig"
    SET = "set"


@dataclass
class BallContact:
    """Ball contact event data"""
    frame_idx: int
    timestamp: float
    ball_position: Tuple[float, float]
    contact_type: ContactType
    player_id: Optional[int]
    team: str
    contact_sequence: int  # Number of contact in current rally (1-3)
    
    # Trajectory analysis
    ball_speed_before: float
    ball_speed_after: float
    direction_change: float  # Angle change in degrees
    speed_change: float     # Speed change magnitude
    
    # Contact quality metrics
    contact_confidence: float  # 0-1 confidence score


@dataclass
class TrajectoryPoint:
    """Single point in ball trajectory"""
    frame_idx: int
    x: float
    y: float
    timestamp: float
    speed: float
    direction: float  # Direction angle in radians
    interp: bool = False  # True when the position came from gap interpolation


class EnhancedEventTracker:
    """Enhanced event tracking system for volleyball"""
    
    def __init__(self, court_bounds, fps: float = 30.0):
        self.court_bounds = court_bounds
        self.fps = fps
        
        # Trajectory tracking
        self.trajectory_history: List[TrajectoryPoint] = []
        self.max_trajectory_length = 50
        
        # Contact tracking
        self.contacts: List[BallContact] = []
        self.current_team_contacts = 0  # Contacts by current team (max 3)
        self.current_team = "team_a"
        
        # ── Contact detection: velocity-vector turn ────────────────────
        # A contact is a sharp turn of the ball velocity between the frames
        # before and after it, with both sides moving fast and coherently.
        # These are the same physics (and thresholds) as AttackHeatmap, which
        # is validated against real footage; keeping them aligned means the
        # heatmap and the FSM agree on what counts as a hit.
        #
        # Speeds are px/SECOND everywhere in this class.
        self.window_frames = 3            # frames each side of the candidate
        self.min_speed_px_per_sec = 120.0  # both sides must exceed this
        self.min_direction_change = 50.0   # degrees between in/out velocity
        self.min_velocity_consistency = 0.72  # 1.0 = perfectly straight segment
        self.contact_radius = 40.0       # Player contact detection radius (overridden in GameAnalyzer)
        # At 30fps: 10 frames ≈ 0.33 s minimum between contacts — one physical
        # hit can bend the trajectory across several adjacent windows.
        self.min_contact_interval = 10   # Minimum frames between contacts

        self.last_contact_frame = -1

    def update_trajectory(self, frame_idx: int, ball_pos: Tuple[float, float],
                         timestamp: float, player_positions: Dict[str, List[Tuple[float, float]]],
                         ball_interp: bool = False) -> List[BallContact]:
        """Update trajectory and detect contacts"""

        # Add new trajectory point
        trajectory_point = self._create_trajectory_point(frame_idx, ball_pos, timestamp)
        trajectory_point.interp = bool(ball_interp)
        self.trajectory_history.append(trajectory_point)
        
        # Limit trajectory history size
        if len(self.trajectory_history) > self.max_trajectory_length:
            self.trajectory_history.pop(0)
        
        # Detect contacts
        new_contacts = self._detect_contacts(trajectory_point, player_positions)
        
        # Update contact counting
        for contact in new_contacts:
            self._update_contact_sequence(contact)
            self.contacts.append(contact)
        
        return new_contacts
    
    def _create_trajectory_point(self, frame_idx: int, ball_pos: Tuple[float, float], 
                               timestamp: float) -> TrajectoryPoint:
        """Create trajectory point with calculated metrics"""
        
        x, y = ball_pos
        speed = 0.0
        direction = 0.0
        
        if len(self.trajectory_history) >= 1:
            prev_point = self.trajectory_history[-1]
            
            # Calculate speed
            dx = x - prev_point.x
            dy = y - prev_point.y
            dt = timestamp - prev_point.timestamp
            
            if dt > 0:
                distance = math.sqrt(dx*dx + dy*dy)
                speed = distance / dt
                direction = math.atan2(dy, dx)
        
        return TrajectoryPoint(
            frame_idx=frame_idx,
            x=x,
            y=y,
            timestamp=timestamp,
            speed=speed,
            direction=direction
        )
    
    def _detect_contacts(self, current_point: TrajectoryPoint,
                        player_positions: Dict[str, List[Tuple[float, float]]]) -> List[BallContact]:
        """Detect ball contacts as sharp turns of the ball velocity vector.

        The candidate is the point `window_frames` back in the history, so a
        contact is reported with that much latency — both the incoming and the
        outgoing leg must be observed before a turn can be measured at all.
        """

        contacts = []
        w = self.window_frames

        # Need the full window on both sides of the candidate
        if len(self.trajectory_history) < 2 * w + 1:
            return contacts

        segment = self.trajectory_history[-(2 * w + 1):]
        centre = segment[w]

        # Prevent detecting contacts too frequently
        if centre.frame_idx - self.last_contact_frame < self.min_contact_interval:
            return contacts

        # Interpolated points are straight by construction: a "turn" at the
        # edge of a filled gap is an artifact of the fill, not a real hit.
        if any(p.interp for p in segment):
            return contacts

        incoming, speed_before, consistency_before = self._segment_velocity(segment[:w + 1])
        outgoing, speed_after, consistency_after = self._segment_velocity(segment[w:])

        if min(consistency_before, consistency_after) < self.min_velocity_consistency:
            return contacts
        if min(speed_before, speed_after) < self.min_speed_px_per_sec:
            return contacts

        cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (speed_before * speed_after)
        direction_change = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if direction_change < self.min_direction_change:
            return contacts

        speed_change = abs(speed_after - speed_before)

        # Find closest player for contact attribution
        closest_player = self._find_closest_player(centre, player_positions)

        # If no player found within contact_radius, fall back to team attribution
        # by ball position relative to net (court_bounds.net_y).
        # This ensures contacts are detected even when pose estimation is imperfect.
        if not closest_player:
            team = "team_a" if centre.y < self.court_bounds.net_y else "team_b"
            closest_player = {
                'team': team,
                'player_id': -1,      # -1 = unknown player
                'position': (centre.x, centre.y),
                'distance': float('inf'),
            }

        # Determine contact type
        contact_type = self._determine_contact_type(centre, speed_change, direction_change)

        # Create contact event
        contact = self._create_contact_event(
            centre, contact_type, closest_player, speed_change, direction_change,
            speed_before, speed_after,
        )

        contacts.append(contact)
        self.last_contact_frame = centre.frame_idx

        return contacts

    def _segment_velocity(self, points: List[TrajectoryPoint]) -> Tuple[Tuple[float, float], float, float]:
        """Mean velocity (px/s) plus directional coherence for a short segment.

        Consistency is the length of the mean velocity divided by the mean of
        the step lengths: 1.0 when every step points the same way, near 0 for
        detector jitter around a stationary ball.
        """
        vectors = []
        for prev, cur in zip(points, points[1:]):
            df = cur.frame_idx - prev.frame_idx
            if df <= 0:
                continue
            vectors.append(((cur.x - prev.x) * self.fps / df,
                            (cur.y - prev.y) * self.fps / df))
        if not vectors:
            return (0.0, 0.0), 0.0, 0.0

        mx = sum(v[0] for v in vectors) / len(vectors)
        my = sum(v[1] for v in vectors) / len(vectors)
        speed = math.hypot(mx, my)
        mean_magnitude = sum(math.hypot(*v) for v in vectors) / len(vectors)
        consistency = speed / max(mean_magnitude, 1e-6)
        return (mx, my), speed, consistency


    def _find_closest_player(self, trajectory_point: TrajectoryPoint, 
                           player_positions: Dict[str, List[Tuple[float, float]]]) -> Optional[Dict]:
        """Find closest player to ball position"""
        
        closest_player = None
        min_distance = float('inf')
        
        ball_x, ball_y = trajectory_point.x, trajectory_point.y
        
        for team, positions in player_positions.items():
            for player_idx, (px, py) in enumerate(positions):
                distance = math.sqrt((ball_x - px)**2 + (ball_y - py)**2)
                
                if distance < self.contact_radius and distance < min_distance:
                    min_distance = distance
                    closest_player = {
                        'team': team,
                        'player_id': player_idx,
                        'position': (px, py),
                        'distance': distance
                    }
        
        return closest_player
    
    def _determine_contact_type(self, current_point: TrajectoryPoint,
                              speed_change: float, direction_change: float) -> ContactType:
        """Rough physics label for a contact.

        Only advisory: the FSM assigns the authoritative event type from rally
        state, because physics alone cannot tell a serve from an attack.
        Thresholds are px/SECOND, matching TrajectoryPoint.speed.
        """

        speed = current_point.speed

        # High speed + significant direction change = Attack
        if speed > 750.0 and direction_change > 45.0:
            return ContactType.ATTACK

        # Moderate speed + controlled trajectory = Pass
        elif 240.0 <= speed <= 600.0 and direction_change <= 30.0:
            return ContactType.PASS

        # Low speed from serve position = Serve
        elif speed > 450.0 and self.current_team_contacts == 0:
            return ContactType.SERVE

        # Low speed + big direction change = Dig
        elif speed <= 240.0 and direction_change > 35.0:
            return ContactType.DIG

        # Default to pass
        else:
            return ContactType.PASS
    
    def _create_contact_event(self, trajectory_point: TrajectoryPoint,
                            contact_type: ContactType, player_data: Dict,
                            speed_change: float, direction_change: float,
                            speed_before: float, speed_after: float) -> BallContact:
        """Create ball contact event"""

        # Calculate contact confidence
        confidence = self._calculate_contact_confidence(speed_change, direction_change, player_data['distance'])
        
        return BallContact(
            frame_idx=trajectory_point.frame_idx,
            timestamp=trajectory_point.timestamp,
            ball_position=(trajectory_point.x, trajectory_point.y),
            contact_type=contact_type,
            player_id=player_data['player_id'],
            team=player_data['team'],
            contact_sequence=self.current_team_contacts + 1,
            ball_speed_before=speed_before,
            ball_speed_after=speed_after,
            direction_change=direction_change,
            speed_change=speed_change,
            contact_confidence=confidence
        )
    
    def _calculate_contact_confidence(self, speed_change: float, direction_change: float,
                                    player_distance: float) -> float:
        """Calculate confidence score for contact detection"""

        confidence = 0.0

        # Distance factor (closer = higher confidence; inf distance = 0 factor)
        if math.isfinite(player_distance):
            distance_factor = max(0.0, 1.0 - (player_distance / self.contact_radius))
        else:
            distance_factor = 0.0
        confidence += distance_factor * 0.4
        
        # Trajectory change factor
        speed_factor = min(speed_change / 20.0, 1.0)
        direction_factor = min(direction_change / 60.0, 1.0)
        confidence += (speed_factor + direction_factor) * 0.3
        
        return min(confidence, 1.0)
    
    def _update_contact_sequence(self, contact: BallContact) -> None:
        """Update contact sequence counting"""
        
        # If team changes, reset counter
        if contact.team != self.current_team:
            self.current_team = contact.team
            self.current_team_contacts = 0
        
        # Increment contact counter
        self.current_team_contacts += 1
        contact.contact_sequence = self.current_team_contacts
        
        # Reset if max contacts reached (3 touches rule)
        if self.current_team_contacts >= 3:
            self.current_team_contacts = 0
    
    def is_max_contacts_reached(self) -> bool:
        """Check if current team reached maximum contacts (3)"""
        return self.current_team_contacts >= 3
    
    def get_recent_contacts(self, max_count: int = 10) -> List[BallContact]:
        """Get recent contacts"""
        return self.contacts[-max_count:] if self.contacts else []
    
    def get_current_team_contacts(self) -> int:
        """Get number of contacts by current team"""
        return self.current_team_contacts
    
    def get_trajectory_analysis(self) -> Dict[str, Any]:
        """Get current trajectory analysis"""
        
        if len(self.trajectory_history) < 2:
            return {}
        
        current = self.trajectory_history[-1]
        
        return {
            'current_speed': current.speed,
            'current_direction': math.degrees(current.direction),
            'trajectory_length': len(self.trajectory_history),
            'recent_contacts': len(self.get_recent_contacts(5)),
            'team_contacts': self.current_team_contacts,
            'current_team': self.current_team,
            'max_contacts_reached': self.is_max_contacts_reached()
        }
    
    def reset_rally(self) -> None:
        """Reset tracking for new rally"""
        self.current_team_contacts = 0
        self.current_team = "team_a"
        self.last_contact_frame = -1
        self.trajectory_history.clear()
        self.contacts.clear() 