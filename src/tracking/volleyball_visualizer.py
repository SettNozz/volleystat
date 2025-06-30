import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

try:
    from .volleyball_fsm import VolleyballFSM, GameState, RallyPhase, ActionType, CourtBounds
except ImportError:
    from volleyball_fsm import VolleyballFSM, GameState, RallyPhase, ActionType, CourtBounds


@dataclass
class PlayerKeypoints:
    """Player keypoints data structure"""
    player_id: int
    team: str
    center: Tuple[float, float]  # Main position
    head: Optional[Tuple[float, float]] = None
    shoulders: Optional[List[Tuple[float, float]]] = None
    arms: Optional[List[Tuple[float, float]]] = None
    hands: Optional[List[Tuple[float, float]]] = None
    confidence: float = 1.0


@dataclass
class VisualizationFrame:
    """Complete frame data for visualization"""
    frame_idx: int
    ball_position: Optional[Tuple[float, float]]
    ball_confidence: float
    player_keypoints: List[PlayerKeypoints]
    fsm_state: GameState
    rally_phase: Optional[RallyPhase]
    current_team: str
    contact_count: int
    ball_velocity: float
    events: List[str]
    trajectory: List[Tuple[float, float]]


class VolleyballVisualizer:
    """Comprehensive volleyball game visualizer"""
    
    def __init__(self, court_bounds: CourtBounds, window_size: Tuple[int, int] = (1200, 800)):
        self.court_bounds = court_bounds
        self.window_size = window_size
        
        # Color schemes (BGR format for OpenCV)
        self.colors = {
            'court': (34, 139, 34),      # Forest green
            'net': (255, 255, 255),      # White
            'lines': (255, 255, 255),    # White
            'ball': (0, 255, 255),       # Yellow
            'ball_trail': (100, 255, 255), # Light yellow
            'team_a': (100, 100, 255),   # Light red
            'team_b': (255, 100, 100),   # Light blue
            'background': (0, 100, 0),   # Dark green
            'text': (255, 255, 255),     # White
            'contact_zone': (0, 255, 255) # Yellow
        }
        
        # Visualization settings
        self.ball_radius = 8
        self.player_radius = 15
        self.keypoint_radius = 3
        self.trail_length = 20
        self.contact_zone_radius = 50
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # State colors
        self.state_colors = {
            GameState.SERVE_READY: (255, 100, 100),
            GameState.SERVING: (100, 100, 255),
            GameState.RALLY_ACTIVE: (100, 255, 100),
            GameState.RALLY_END: (100, 255, 255),
            GameState.TIMEOUT: (128, 128, 128),
        }
        
        self.phase_colors = {
            RallyPhase.RECEPTION: (100, 200, 255),
            RallyPhase.SETTING: (200, 255, 100),
            RallyPhase.ATTACKING: (100, 100, 255),
            RallyPhase.BLOCKING: (255, 100, 200),
            RallyPhase.DEFENSE: (255, 200, 100),
            RallyPhase.EMERGENCY_PLAY: (100, 255, 255),
        }
    
    def create_visualization_frame(self, frame_idx: int, ball_pos: Optional[Tuple[float, float]],
                                  player_keypoints: List[PlayerKeypoints], fsm: VolleyballFSM,
                                  ball_velocity: float = 0.0) -> np.ndarray:
        """Create a complete visualization frame"""
        
        # Create blank canvas
        canvas = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        canvas[:] = self.colors['background']
        
        # Draw court
        self._draw_court(canvas)
        
        # Draw ball trajectory
        if len(fsm.ball_trajectory) > 1:
            self._draw_ball_trajectory(canvas, fsm.ball_trajectory)
        
        # Draw ball
        if ball_pos:
            self._draw_ball(canvas, ball_pos, ball_velocity)
        
        # Draw players
        self._draw_players(canvas, player_keypoints)
        
        # Draw contact zones
        if ball_pos:
            self._draw_contact_zones(canvas, ball_pos, player_keypoints)
        
        # Draw FSM information
        self._draw_fsm_info(canvas, fsm, frame_idx, ball_velocity)
        
        # Draw events
        recent_events = [e.event_type for e in fsm.get_events() if e.frame_idx >= frame_idx - 30]
        self._draw_events(canvas, recent_events)
        
        return canvas
    
    def _draw_court(self, canvas: np.ndarray) -> None:
        """Draw volleyball court with proper dimensions"""
        # Court boundaries
        court_left = int(100)
        court_right = int(self.window_size[0] - 100)
        court_top = int(100)
        court_bottom = int(self.window_size[1] - 100)
        
        # Draw court background
        cv2.rectangle(canvas, (court_left, court_top), (court_right, court_bottom), 
                     self.colors['court'], -1)
        
        # Draw court lines
        cv2.rectangle(canvas, (court_left, court_top), (court_right, court_bottom), 
                     self.colors['lines'], 3)
        
        # Draw net
        net_y = int(court_top + (court_bottom - court_top) * 
                   (self.court_bounds.net_y - self.court_bounds.top) / 
                   (self.court_bounds.bottom - self.court_bounds.top))
        
        cv2.line(canvas, (court_left, net_y), (court_right, net_y), 
                self.colors['net'], 4)
        
        # Draw center line
        center_x = (court_left + court_right) // 2
        cv2.line(canvas, (center_x, court_top), (center_x, court_bottom), 
                self.colors['lines'], 2)
        
        # Add court labels
        cv2.putText(canvas, "Team A", (center_x - 50, court_top - 20), 
                   self.font, 0.8, self.colors['team_a'], 2)
        cv2.putText(canvas, "Team B", (center_x - 50, court_bottom + 40), 
                   self.font, 0.8, self.colors['team_b'], 2)
        
        # Store court coordinates for later use
        self.court_coords = {
            'left': court_left,
            'right': court_right,
            'top': court_top,
            'bottom': court_bottom,
            'net_y': net_y
        }
    
    def _world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        if not hasattr(self, 'court_coords'):
            return (int(world_pos[0]), int(world_pos[1]))
        
        # Scale and translate coordinates
        screen_x = int(self.court_coords['left'] + 
                      (world_pos[0] - self.court_bounds.left) * 
                      (self.court_coords['right'] - self.court_coords['left']) / 
                      (self.court_bounds.right - self.court_bounds.left))
        
        screen_y = int(self.court_coords['top'] + 
                      (world_pos[1] - self.court_bounds.top) * 
                      (self.court_coords['bottom'] - self.court_coords['top']) / 
                      (self.court_bounds.bottom - self.court_bounds.top))
        
        return (screen_x, screen_y)
    
    def _draw_ball(self, canvas: np.ndarray, ball_pos: Tuple[float, float], 
                   velocity: float) -> None:
        """Draw ball with velocity indicator"""
        screen_pos = self._world_to_screen(ball_pos)
        
        # Draw ball
        cv2.circle(canvas, screen_pos, self.ball_radius, self.colors['ball'], -1)
        cv2.circle(canvas, screen_pos, self.ball_radius + 2, self.colors['text'], 2)
        
        # Draw velocity indicator
        if velocity > 5.0:
            velocity_length = min(int(velocity * 2), 50)
            end_pos = (screen_pos[0] + velocity_length, screen_pos[1])
            cv2.arrowedLine(canvas, screen_pos, end_pos, (0, 255, 255), 2)
            
            # Velocity text
            cv2.putText(canvas, f"{velocity:.1f}", 
                       (screen_pos[0] + 10, screen_pos[1] - 10),
                       self.font, 0.5, self.colors['text'], 1)
    
    def _draw_ball_trajectory(self, canvas: np.ndarray, 
                             trajectory: List[Tuple[int, float, float]]) -> None:
        """Draw ball trajectory trail"""
        if len(trajectory) < 2:
            return
        
        # Draw trail
        recent_positions = trajectory[-self.trail_length:]
        for i in range(1, len(recent_positions)):
            pos1 = self._world_to_screen((recent_positions[i-1][1], recent_positions[i-1][2]))
            pos2 = self._world_to_screen((recent_positions[i][1], recent_positions[i][2]))
            
            # Fade trail
            alpha = i / len(recent_positions)
            color = tuple(int(c * alpha) for c in self.colors['ball_trail'])
            cv2.line(canvas, pos1, pos2, color, 2)
    
    def _draw_players(self, canvas: np.ndarray, 
                     player_keypoints: List[PlayerKeypoints]) -> None:
        """Draw players with keypoints"""
        for player in player_keypoints:
            screen_pos = self._world_to_screen(player.center)
            
            # Choose color based on team
            color = self.colors['team_a'] if player.team == 'team_a' else self.colors['team_b']
            
            # Draw player center
            cv2.circle(canvas, screen_pos, self.player_radius, color, -1)
            cv2.circle(canvas, screen_pos, self.player_radius + 2, self.colors['text'], 2)
            
            # Draw player ID
            cv2.putText(canvas, str(player.player_id), 
                       (screen_pos[0] - 5, screen_pos[1] + 5),
                       self.font, 0.5, self.colors['text'], 1)
            
            # Draw keypoints if available
            self._draw_player_keypoints(canvas, player, color)
    
    def _draw_player_keypoints(self, canvas: np.ndarray, player: PlayerKeypoints, 
                              base_color: Tuple[int, int, int]) -> None:
        """Draw detailed player keypoints"""
        # Draw head
        if player.head:
            head_pos = self._world_to_screen(player.head)
            cv2.circle(canvas, head_pos, self.keypoint_radius, base_color, -1)
        
        # Draw shoulders
        if player.shoulders:
            for shoulder in player.shoulders:
                shoulder_pos = self._world_to_screen(shoulder)
                cv2.circle(canvas, shoulder_pos, self.keypoint_radius, base_color, -1)
        
        # Draw arms
        if player.arms:
            for arm in player.arms:
                arm_pos = self._world_to_screen(arm)
                cv2.circle(canvas, arm_pos, self.keypoint_radius, base_color, -1)
        
        # Draw hands
        if player.hands:
            for hand in player.hands:
                hand_pos = self._world_to_screen(hand)
                cv2.circle(canvas, hand_pos, self.keypoint_radius + 1, (0, 255, 255), -1)
    
    def _draw_contact_zones(self, canvas: np.ndarray, ball_pos: Tuple[float, float],
                           player_keypoints: List[PlayerKeypoints]) -> None:
        """Draw contact zones around players near the ball"""
        ball_screen = self._world_to_screen(ball_pos)
        
        for player in player_keypoints:
            player_screen = self._world_to_screen(player.center)
            distance = np.sqrt((ball_screen[0] - player_screen[0])**2 + 
                             (ball_screen[1] - player_screen[1])**2)
            
            # If player is close to ball, draw contact zone
            if distance < self.contact_zone_radius * 2:
                # Create overlay for semi-transparent circle
                overlay = canvas.copy()
                cv2.circle(overlay, player_screen, self.contact_zone_radius, 
                          self.colors['contact_zone'], -1)
                cv2.addWeighted(canvas, 0.8, overlay, 0.2, 0, canvas)
    
    def _draw_fsm_info(self, canvas: np.ndarray, fsm: VolleyballFSM, 
                       frame_idx: int, ball_velocity: float) -> None:
        """Draw FSM state information"""
        # Info panel background
        info_height = 200
        info_width = 300
        
        # Create semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (info_width, info_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0, canvas)
        
        y_offset = 30
        line_height = 25
        
        # Frame info
        cv2.putText(canvas, f"Frame: {frame_idx}", (20, y_offset), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        y_offset += line_height
        
        # Game state
        state_color = self.state_colors.get(fsm.current_state, self.colors['text'])
        cv2.putText(canvas, f"State: {fsm.current_state.value}", (20, y_offset), 
                   self.font, self.font_scale, state_color, self.font_thickness)
        y_offset += line_height
        
        # Rally phase
        if fsm.current_rally_phase:
            phase_color = self.phase_colors.get(fsm.current_rally_phase, self.colors['text'])
            cv2.putText(canvas, f"Phase: {fsm.current_rally_phase.value}", (20, y_offset), 
                       self.font, self.font_scale, phase_color, self.font_thickness)
        y_offset += line_height
        
        # Current team
        team_color = self.colors['team_a'] if fsm.current_team == 'team_a' else self.colors['team_b']
        cv2.putText(canvas, f"Team: {fsm.current_team}", (20, y_offset), 
                   self.font, self.font_scale, team_color, self.font_thickness)
        y_offset += line_height
        
        # Contacts
        cv2.putText(canvas, f"Contacts: {fsm.contact_count}", (20, y_offset), 
                   self.font, self.font_scale, self.colors['text'], self.font_thickness)
        y_offset += line_height
        
        # Ball velocity
        velocity_color = (0, 255, 0) if ball_velocity < 10 else (0, 255, 255) if ball_velocity < 20 else (0, 0, 255)
        cv2.putText(canvas, f"Ball Speed: {ball_velocity:.1f}", (20, y_offset), 
                   self.font, self.font_scale, velocity_color, self.font_thickness)
        y_offset += line_height
        
        # Ball position
        if len(fsm.ball_trajectory) > 0:
            last_pos = fsm.ball_trajectory[-1]
            ball_side = fsm.court_bounds.get_ball_side(last_pos[2])
            cv2.putText(canvas, f"Ball Side: {ball_side}", (20, y_offset), 
                       self.font, self.font_scale, self.colors['text'], self.font_thickness)
    
    def _draw_events(self, canvas: np.ndarray, events: List[str]) -> None:
        """Draw recent events"""
        if not events:
            return
        
        # Events panel
        events_width = 400
        events_height = min(len(events) * 25 + 20, 150)
        start_x = self.window_size[0] - events_width - 10
        
        # Background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (start_x, 10), (start_x + events_width, 10 + events_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0, canvas)
        
        # Events title
        cv2.putText(canvas, "Recent Events:", (start_x + 10, 30), 
                   self.font, 0.6, self.colors['text'], 2)
        
        # Draw events (last 5)
        recent_events = events[-5:] if len(events) > 5 else events
        for i, event in enumerate(recent_events):
            y_pos = 55 + i * 20
            # Color code events
            event_color = self.colors['text']
            if 'serve' in event:
                event_color = (100, 100, 255)
            elif 'rally' in event:
                event_color = (100, 255, 100)
            elif 'net_crossing' in event:
                event_color = (100, 255, 255)
            elif 'phase_change' in event:
                event_color = (255, 255, 100)
            
            cv2.putText(canvas, f"• {event}", (start_x + 10, y_pos), 
                       self.font, 0.4, event_color, 1)
    
    def save_frame(self, canvas: np.ndarray, filename: str) -> None:
        """Save visualization frame to file"""
        cv2.imwrite(filename, canvas)
    
    def display_frame(self, canvas: np.ndarray, window_name: str = "Volleyball Analysis") -> bool:
        """Display frame in window, return False if user wants to quit"""
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        return key != ord('q') and key != 27  # 27 is ESC


def create_sample_player_keypoints(frame_idx: int) -> List[PlayerKeypoints]:
    """Create sample player keypoints for testing"""
    keypoints = []
    
    # Team A players
    team_a_positions = [(200, 120), (400, 120), (600, 120), (250, 220), (450, 220), (650, 220)]
    for i, (x, y) in enumerate(team_a_positions):
        # Add some animation
        movement = np.sin(frame_idx * 0.1 + i) * 10
        keypoints.append(PlayerKeypoints(
            player_id=i + 1,
            team='team_a',
            center=(x + movement, y),
            head=(x + movement, y - 20),
            shoulders=[(x + movement - 15, y - 10), (x + movement + 15, y - 10)],
            arms=[(x + movement - 25, y), (x + movement + 25, y)],
            hands=[(x + movement - 30, y + 10), (x + movement + 30, y + 10)],
            confidence=0.9
        ))
    
    # Team B players
    team_b_positions = [(200, 480), (400, 480), (600, 480), (250, 380), (450, 380), (650, 380)]
    for i, (x, y) in enumerate(team_b_positions):
        movement = np.sin(frame_idx * 0.1 + i + 6) * 10
        keypoints.append(PlayerKeypoints(
            player_id=i + 1,
            team='team_b',
            center=(x + movement, y),
            head=(x + movement, y - 20),
            shoulders=[(x + movement - 15, y - 10), (x + movement + 15, y - 10)],
            arms=[(x + movement - 25, y), (x + movement + 25, y)],
            hands=[(x + movement - 30, y + 10), (x + movement + 30, y + 10)],
            confidence=0.9
        ))
    
    return keypoints


def demo_visualization() -> None:
    """Demonstrate the visualization system"""
    try:
        from fsm_example import create_sample_court_bounds, generate_realistic_rally_trajectory
    except ImportError:
        print("Error: Cannot import fsm_example. Make sure it's in the same directory.")
        return
    
    print("Starting Volleyball Visualization Demo...")
    print("Press 'q' or ESC to quit the visualization")
    
    # Setup
    court_bounds = create_sample_court_bounds()
    fsm = VolleyballFSM(court_bounds)
    visualizer = VolleyballVisualizer(court_bounds)
    
    # Get trajectory
    ball_trajectory = generate_realistic_rally_trajectory()
    
    # Process and visualize each frame
    for frame_idx, ball_x, ball_y in ball_trajectory:
        # Update FSM
        player_keypoints_data = create_sample_player_keypoints(frame_idx)
        player_positions = {
            'team_a': [kp.center for kp in player_keypoints_data if kp.team == 'team_a'],
            'team_b': [kp.center for kp in player_keypoints_data if kp.team == 'team_b']
        }
        
        fsm.update(frame_idx, (ball_x, ball_y), player_positions)
        
        # Calculate ball velocity
        ball_velocity = 0.0
        if len(fsm.ball_trajectory) >= 2:
            last_pos = fsm.ball_trajectory[-1]
            prev_pos = fsm.ball_trajectory[-2]
            dx = last_pos[1] - prev_pos[1]
            dy = last_pos[2] - prev_pos[2]
            ball_velocity = np.sqrt(dx*dx + dy*dy)
        
        # Create visualization
        canvas = visualizer.create_visualization_frame(
            frame_idx, (ball_x, ball_y), player_keypoints_data, fsm, ball_velocity
        )
        
        # Display frame
        if not visualizer.display_frame(canvas):
            break
    
    cv2.destroyAllWindows()
    print("Visualization demo completed!")


if __name__ == "__main__":
    demo_visualization() 