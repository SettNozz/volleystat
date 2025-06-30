import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.tracking.volleyball_fsm import VolleyballFSM, GameState, RallyPhase, ActionType, CourtBounds
    from src.tracking.volleyball_visualizer import VolleyballVisualizer, PlayerKeypoints
except ImportError:
    # Fallback import paths
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'tracking'))
    from volleyball_fsm import VolleyballFSM, GameState, RallyPhase, ActionType, CourtBounds
    from volleyball_visualizer import VolleyballVisualizer, PlayerKeypoints


@dataclass
class VolleyballEvent:
    """Structure for volleyball event"""
    event_type: str  # "spike", "set", "receive", "block", "out"
    frame_idx: int
    ball_position: Tuple[float, float]
    ball_velocity: Tuple[float, float]
    ball_speed: float
    confidence: float
    near_net: bool = False
    trajectory_change: bool = False


class VolleyballEventDetector:
    """Volleyball event detector based on ball trajectory analysis"""
    
    def __init__(self, net_y_position: float = 0.5):
        self.net_y_position = net_y_position  # Relative net position (0-1)
        self.ball_history = []  # Ball position history
        self.velocity_history = []  # Velocity history
        self.events = []  # Detected events
        
        # Detection parameters
        self.min_spike_speed = 100.0  # Minimum speed for spike (px/frame)
        self.min_trajectory_change = 30.0  # Minimum angle change for block
        self.net_proximity_threshold = 0.15  # Net proximity (relative)
        self.min_upward_velocity = 50.0  # Minimum upward velocity for set
        
        self.current_event = None
        self.event_confidence = 0.0
        
    def add_ball_position(self, frame_idx: int, position: Tuple[float, float], 
                         frame_width: float, frame_height: float) -> Optional[VolleyballEvent]:
        """Add new ball position and analyze events"""
        
        # Normalize coordinates (0-1)
        norm_x = position[0] / frame_width
        norm_y = position[1] / frame_height
        
        self.ball_history.append((frame_idx, norm_x, norm_y))
        
        # Keep only last 10 positions
        if len(self.ball_history) > 10:
            self.ball_history.pop(0)
            
        # Need minimum 3 points for analysis
        if len(self.ball_history) < 3:
            return None
            
        # Calculate velocity and direction
        velocity = self._calculate_velocity()
        if velocity:
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > 5:
                self.velocity_history.pop(0)
                
        # Analyze events
        event = self._analyze_events(frame_idx, (norm_x, norm_y))
        if event:
            self.events.append(event)
            self.current_event = event
            
        return event
        
    def _calculate_velocity(self) -> Optional[Tuple[float, float, float]]:
        """Calculate ball velocity and direction"""
        if len(self.ball_history) < 2:
            return None
            
        current = self.ball_history[-1]
        previous = self.ball_history[-2]
        
        # Velocity in pixels per frame (normalized)
        dx = current[1] - previous[1]  # norm_x
        dy = current[2] - previous[2]  # norm_y
        
        speed = np.sqrt(dx*dx + dy*dy)
        
        return (dx, dy, speed)
        
    def _analyze_events(self, frame_idx: int, position: Tuple[float, float]) -> Optional[VolleyballEvent]:
        """Analyze events based on trajectory"""
        
        if len(self.velocity_history) < 2:
            return None
            
        current_vel = self.velocity_history[-1]
        prev_vel = self.velocity_history[-2] if len(self.velocity_history) > 1 else current_vel
        
        dx, dy, speed = current_vel
        prev_dx, prev_dy, prev_speed = prev_vel
        
        # Check proximity to net
        near_net = abs(position[1] - self.net_y_position) < self.net_proximity_threshold
        
        # Calculate trajectory change
        angle_change = self._calculate_angle_change(prev_vel, current_vel)
        trajectory_change = abs(angle_change) > self.min_trajectory_change
        
        # SPIKE detection
        if self._detect_spike(dx, dy, speed, near_net):
            return VolleyballEvent(
                event_type="spike",
                frame_idx=frame_idx,
                ball_position=position,
                ball_velocity=(dx, dy),
                ball_speed=speed,
                confidence=0.8,
                near_net=near_net,
                trajectory_change=trajectory_change
            )
            
        # BLOCK detection
        elif self._detect_block(current_vel, prev_vel, near_net, trajectory_change):
            return VolleyballEvent(
                event_type="block",
                frame_idx=frame_idx,
                ball_position=position,
                ball_velocity=(dx, dy),
                ball_speed=speed,
                confidence=0.7,
                near_net=near_net,
                trajectory_change=trajectory_change
            )
            
        # SET detection
        elif self._detect_set(dx, dy, speed):
            return VolleyballEvent(
                event_type="set",
                frame_idx=frame_idx,
                ball_position=position,
                ball_velocity=(dx, dy),
                ball_speed=speed,
                confidence=0.6,
                near_net=False,
                trajectory_change=False
            )
            
        # RECEIVE detection
        elif self._detect_receive(dx, dy, speed, position):
            return VolleyballEvent(
                event_type="receive",
                frame_idx=frame_idx,
                ball_position=position,
                ball_velocity=(dx, dy),
                ball_speed=speed,
                confidence=0.5,
                near_net=False,
                trajectory_change=False
            )
            
        return None
        
    def _detect_spike(self, dx: float, dy: float, speed: float, near_net: bool) -> bool:
        """Spike detection: ball moves sharply downward with high speed"""
        # Ball moves downward (dy > 0) with high speed
        return (dy > 0.02 and  # Downward movement
                speed > 0.05 and  # High speed
                abs(dx) < 0.1)  # Not too much horizontal movement
                
    def _detect_block(self, current_vel: Tuple[float, float, float], 
                     prev_vel: Tuple[float, float, float], 
                     near_net: bool, trajectory_change: bool) -> bool:
        """Block detection: direction change near net"""
        return (near_net and 
                trajectory_change and 
                current_vel[2] > 0.02)  # Sufficient speed
                
    def _detect_set(self, dx: float, dy: float, speed: float) -> bool:
        """Set detection: ball moves upward with moderate speed"""
        return (dy < -0.01 and  # Upward movement
                speed > 0.02 and speed < 0.08 and  # Moderate speed
                abs(dx) > 0.01)  # Horizontal movement (pass)
                
    def _detect_receive(self, dx: float, dy: float, speed: float, position: Tuple[float, float]) -> bool:
        """Receive detection: ball changes trajectory in back court"""
        # Ball in back court and changes direction upward
        return (position[1] > 0.7 and  # Back court
                dy < -0.005 and  # Slight upward movement
                speed > 0.015)  # Moderate speed
                
    def _calculate_angle_change(self, prev_vel: Tuple[float, float, float], 
                               current_vel: Tuple[float, float, float]) -> float:
        """Calculate trajectory angle change"""
        if prev_vel[2] == 0 or current_vel[2] == 0:
            return 0
            
        # Angles in degrees
        prev_angle = np.degrees(np.arctan2(prev_vel[1], prev_vel[0]))
        current_angle = np.degrees(np.arctan2(current_vel[1], current_vel[0]))
        
        # Angle change
        angle_diff = current_angle - prev_angle
        
        # Normalize angle to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
            
        return angle_diff
        
    def get_current_event_info(self) -> str:
        """Get current event information"""
        if not self.current_event:
            return "Observing..."
            
        event_names = {
            "spike": "🏐 SPIKE (Attack)",
            "block": "🛡️ BLOCK",
            "set": "🤲 SET (Pass)",
            "receive": "📥 RECEIVE",
            "out": "❌ OUT"
        }
        
        event_name = event_names.get(self.current_event.event_type, self.current_event.event_type)
        confidence = int(self.current_event.confidence * 100)
        
        return f"{event_name} ({confidence}%)"
        
    def get_event_details(self) -> List[str]:
        """Get detailed event information"""
        if not self.current_event:
            return ["No active events"]
            
        details = []
        event = self.current_event
        
        details.append(f"Type: {event.event_type}")
        details.append(f"Speed: {event.ball_speed*1000:.1f} px/frame")
        details.append(f"Direction: ({event.ball_velocity[0]*100:.1f}, {event.ball_velocity[1]*100:.1f})")
        
        if event.near_net:
            details.append("Near net: YES")
        if event.trajectory_change:
            details.append("Trajectory change: YES")
            
        return details


class YOLOBallDetector:
    """Advanced YOLO-based ball detector with tracking"""
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        
        # Try to load best trained volleyball model
        model_paths = [
            r"C:\Users\illya\Documents\volleyball_analitics\volleystat\models\yolov8_curated_training\yolov8n_volleyball_curated4\weights\best.pt",
            # "C:\Users\illya\Documents\volleyball_analitics\volleystat\models\yolov8_volleyball_training\yolov8n_volleyball\weights\best.pt",
            "volleystat/models/best/yolo11n.pt",
            "yolov8n.pt"  # Fallback
        ]
        
        self.model = None
        for model_path in model_paths:
            try:
                from ultralytics import YOLO
                if Path(model_path).exists():
                    self.model = YOLO(model_path)
                    print(f"✅ Loaded YOLO model: {model_path}")
                    # Print model info
                    if hasattr(self.model, 'names'):
                        print(f"📋 Model classes: {self.model.names}")
                    print(f"🎯 Model ready for ball detection")
                    break
            except Exception as e:
                print(f"❌ Failed to load {model_path}: {e}")
                continue
                
        if self.model is None:
            print("❌ No YOLO model available, falling back to color detection")
            self.use_color_fallback = True
        else:
            self.use_color_fallback = False
            self._print_model_info()
            
        # Tracking state for stability
        self.last_detection = None
        self.detection_history = []  # Last N detections
        self.max_history = 5
        self.max_jump_distance = 200  # Max pixels ball can move between frames
        self.min_confidence = 0.2
        self.stability_threshold = 3  # Require N consistent detections
        
    def _print_model_info(self) -> None:
        """Print information about loaded YOLO model"""
        if self.model and hasattr(self.model, 'names'):
            print(f"📋 Model classes ({len(self.model.names)}): {list(self.model.names.values())}")
            # Check for volleyball/ball related classes
            ball_classes = []
            for idx, name in self.model.names.items():
                if any(keyword in name.lower() for keyword in ['ball', 'volleyball', 'sport']):
                    ball_classes.append(f"{idx}: {name}")
            if ball_classes:
                print(f"⚽ Ball-related classes: {ball_classes}")
            else:
                print("⚠️  No ball-specific classes found, will use general object detection")
        print(f"🎯 YOLO ball detector ready with tracking enabled")
        
    def _color_detection_fallback(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Fallback color detection method"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask = cv2.bitwise_or(mask_white, mask_yellow)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_detection = None
            best_score = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 5000:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if 5 <= radius <= 50:
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            score = circularity * min(1.0, area / 1000.0)
                            
                            if score > best_score and score > self.confidence_threshold:
                                best_score = score
                                best_detection = (x, y, min(0.9, score))
            
            return best_detection
            
        except Exception as e:
            print(f"Color detection error: {e}")
            return None
            
    def _is_valid_detection(self, detection: Tuple[float, float, float]) -> bool:
        """Check if detection is valid based on tracking history"""
        x, y, conf = detection
        
        # Confidence check
        if conf < self.min_confidence:
            return False
            
        # Distance check with last detection
        if self.last_detection is not None:
            last_x, last_y, _ = self.last_detection
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if distance > self.max_jump_distance:
                return False
                
        return True
        
    def _smooth_detection(self, detection: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply temporal smoothing to detection"""
        if len(self.detection_history) == 0:
            return detection
            
        # Weighted average with history (higher weight for recent detections)
        weights = np.exp(np.linspace(-1, 0, len(self.detection_history) + 1))
        weights = weights / np.sum(weights)
        
        all_detections = self.detection_history + [detection]
        
        smooth_x = np.sum([det[0] * w for det, w in zip(all_detections, weights)])
        smooth_y = np.sum([det[1] * w for det, w in zip(all_detections, weights)])
        smooth_conf = detection[2]  # Keep original confidence
        
        return (smooth_x, smooth_y, smooth_conf)
        
    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Detect ball with YOLO and tracking
        Returns: (x, y, confidence) or None
        """
        best_detection = None
        
        if self.use_color_fallback or self.model is None:
            best_detection = self._color_detection_fallback(frame)
        else:
            try:
                # YOLO detection
                results = self.model(frame, conf=self.min_confidence, verbose=False)
                
                best_conf = 0
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            # Check if this is a ball-related class
                            is_ball_class = False
                            if hasattr(self.model, 'names') and class_id in self.model.names:
                                class_name = self.model.names[class_id].lower()
                                is_ball_class = any(keyword in class_name for keyword in 
                                                  ['ball', 'volleyball', 'sport'])
                            else:
                                # For COCO models, class 32 is 'sports ball'
                                is_ball_class = (class_id == 32 or class_id == 0)
                            
                            if is_ball_class and confidence > best_conf:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                best_detection = (center_x, center_y, confidence)
                                best_conf = confidence
                                
            except Exception as e:
                print(f"YOLO detection error: {e}")
                best_detection = self._color_detection_fallback(frame)
        
        # Apply tracking validation
        if best_detection and self._is_valid_detection(best_detection):
            # Apply smoothing
            smoothed_detection = self._smooth_detection(best_detection)
            
            # Update history
            self.detection_history.append(best_detection)
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)
                
            self.last_detection = smoothed_detection
            return smoothed_detection
        else:
            # No valid detection - clear history gradually
            if len(self.detection_history) > 0:
                self.detection_history.pop(0)
            return None


class VideoFSMDemo:
    """Demo class for video processing with FSM"""
    
    def __init__(self):
        # Initialize court bounds (standard volleyball court - official dimensions)
        self.court_bounds = CourtBounds(
            left=-4.5, right=4.5,    # 9m width total (-4.5 to +4.5)
            top=0, bottom=9.0,       # 9m depth (net at 0, back line at 9)
            net_y=0                  # Net at Y=0 (camera behind court)
        )
        
        # Camera calibration for perspective correction
        self.net_height_meters = 2.43  # Standard volleyball net height
        self.net_detected = False
        self.net_top_y = None
        self.net_bottom_y = None
        self.height_scale_factor = 1.0
        self.calibration_frame_count = 0
        
        # Manual antenna calibration
        self.left_antenna_points = []   # Will store [(x1, y1), (x2, y2)] - two points on left antenna
        self.right_antenna_points = []  # Will store [(x1, y1), (x2, y2)] - two points on right antenna
        self.back_line_points = []  # Will store [(x1, y1), (x2, y2)] - two points on back line
        self.left_sideline_points = []  # Will store [(x1, y1), (x2, y2)] - two points on left sideline
        self.right_sideline_points = []  # Will store [(x1, y1), (x2, y2)] - two points on right sideline
        self.perspective_matrix = None  # Perspective transformation matrix
        self.calibration_completed = False
        self.court_calibration_completed = False  # Full court calibration with perspective
        self.ground_level_y = None  # Will be calculated based on antenna positions
        
        # Court geometry (official volleyball court dimensions)
        self.court_width_meters = 9.0   # Width of volleyball court (9m)
        self.court_half_length_meters = 9.0  # Distance from net to back line (9m)
        
        # Initialize FSM and visualizer
        self.fsm = VolleyballFSM(self.court_bounds)
        self.visualizer = VolleyballVisualizer(self.court_bounds)
        
        # Initialize ball detector with YOLO
        self.ball_detector = YOLOBallDetector(confidence_threshold=0.4)
        
        # Processing state
        self.frame_count = 0
        self.ball_detections = 0
        self.ball_trajectory = []
        
        # Detection quality statistics
        self.yolo_detections = 0
        self.color_fallback_detections = 0
        self.filtered_detections = 0  # Detections filtered out by tracking
        self.smoothed_detections = 0
        self.confidence_history = []
        self.max_confidence_history = 100
        self.current_detection_method = "NONE"  # Track current frame detection method
        
    def calibrate_with_antennas(self, first_frame: np.ndarray) -> bool:
        """Interactive full court perspective calibration"""
        print("\n🏐 Full Court Perspective Calibration")
        print("=" * 60)
        print("This will calibrate court perspective using 8 points")
        print("Stage 1: Antenna calibration")
        print("1. Click on TWO POINTS on LEFT antenna (top and bottom)")
        print("2. Click on TWO POINTS on RIGHT antenna (top and bottom)")
        print("3. Press ENTER to proceed to next stage")
        print("4. Press ESC to skip calibration")
        print("=" * 60)
        
        # Create calibration window
        cv2.namedWindow('Antenna Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Antenna Calibration', 1200, 700)
        
        # Copy frame for drawing
        calibration_frame = first_frame.copy()
        self.left_antenna_points = []
        self.right_antenna_points = []
        total_clicks = 0
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal total_clicks
            if event == cv2.EVENT_LBUTTONDOWN and total_clicks < 4:
                total_clicks += 1
                
                if total_clicks <= 2:
                    # Left antenna points
                    self.left_antenna_points.append((x, y))
                    color = (0, 255, 0)  # Green for left antenna
                    point_num = len(self.left_antenna_points)
                    label = f"Left Ant P{point_num}"
                    print(f"✅ Left antenna point {point_num}: ({x}, {y})")
                else:
                    # Right antenna points
                    self.right_antenna_points.append((x, y))
                    color = (255, 0, 0)  # Red for right antenna
                    point_num = len(self.right_antenna_points)
                    label = f"Right Ant P{point_num}"
                    print(f"✅ Right antenna point {point_num}: ({x}, {y})")
                
                # Draw point
                cv2.circle(calibration_frame, (x, y), 8, color, -1)
                cv2.circle(calibration_frame, (x, y), 15, color, 3)
                cv2.putText(calibration_frame, label, (x + 20, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw line if we have 2 points for an antenna
                if len(self.left_antenna_points) == 2:
                    cv2.line(calibration_frame, self.left_antenna_points[0], 
                            self.left_antenna_points[1], (0, 255, 0), 3)
                    print(f"📏 Left antenna line drawn")
                    
                if len(self.right_antenna_points) == 2:
                    cv2.line(calibration_frame, self.right_antenna_points[0], 
                            self.right_antenna_points[1], (255, 0, 0), 3)
                    print(f"📏 Right antenna line drawn")
                
                # Complete calibration when all 4 points are marked
                if total_clicks == 4:
                    self._calculate_calibration_from_antennas(calibration_frame.shape[0])
                    print(f"🎯 Antenna calibration completed! Scale: {self.height_scale_factor:.4f} m/pixel")
        
        cv2.setMouseCallback('Antenna Calibration', mouse_callback)
        
        # Show calibration interface
        while True:
            # Draw instructions on frame
            overlay = calibration_frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, calibration_frame, 0.2, 0, calibration_frame)
            
            instructions = [
                f"Clicks: {total_clicks}/4",
                "Left click: Mark antenna points",
                "Left antenna: 2 points, Right antenna: 2 points",
                "ENTER: Complete calibration", 
                "ESC: Skip calibration"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(calibration_frame, instruction, (15, 30 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                           
            cv2.imshow('Antenna Calibration', calibration_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                break
            elif key == 27:  # ESC
                print("❌ Calibration skipped")
                cv2.destroyWindow('Antenna Calibration')
                return False
                
        cv2.destroyWindow('Antenna Calibration')
        
        if len(self.left_antenna_points) == 2 and len(self.right_antenna_points) == 2:
            self.calibration_completed = True
            print(f"✅ Stage 1 completed!")
            print(f"📏 Left antenna line: {self.left_antenna_points}")
            print(f"📏 Right antenna line: {self.right_antenna_points}")
            print(f"📐 Height scale: {self.height_scale_factor:.4f} m/pixel")
            
            # Proceed to full court calibration
            if self._calibrate_court_lines(first_frame):
                print(f"✅ Full court perspective calibration completed!")
                return True
            else:
                print("⚠️ Court line calibration not completed, using antenna-only calibration")
                return True
        else:
            print("⚠️ Need to mark all 4 antenna points (2 per antenna)")
            return False
            
    def _calculate_calibration_from_antennas(self, frame_height: int) -> None:
        """Calculate height scale from antenna lines"""
        if len(self.left_antenna_points) != 2 or len(self.right_antenna_points) != 2:
            return
            
        # Calculate average Y position from all 4 antenna points (net level)
        all_antenna_y_coords = [
            self.left_antenna_points[0][1], self.left_antenna_points[1][1],
            self.right_antenna_points[0][1], self.right_antenna_points[1][1]
        ]
        avg_antenna_y = sum(all_antenna_y_coords) / len(all_antenna_y_coords)
        
        print(f"🎯 Net level calculation:")
        print(f"   Left antenna Y coords: {self.left_antenna_points[0][1]}, {self.left_antenna_points[1][1]}")
        print(f"   Right antenna Y coords: {self.right_antenna_points[0][1]}, {self.right_antenna_points[1][1]}")
        print(f"   Average net level Y: {avg_antenna_y:.1f}")
        
        # Estimate ground level (assuming camera is positioned at reasonable height)
        # For volleyball court, antennas are typically visible in upper 1/3 of frame
        # Ground would be at the bottom of the frame or slightly above
        self.ground_level_y = frame_height - 50  # 50 pixels from bottom as safety margin
        
        # Calculate height scale: antenna height in pixels = real height in meters
        antenna_height_pixels = self.ground_level_y - avg_antenna_y
        
        if antenna_height_pixels > 0:
            self.height_scale_factor = self.net_height_meters / antenna_height_pixels
            self.net_top_y = int(avg_antenna_y - 10)  # Net top slightly below antennas
            self.net_bottom_y = int(avg_antenna_y + 10)  # Net bottom slightly above antennas
            self.net_detected = True
        else:
            print("⚠️ Calibration error: antennas should be above ground level")
            
    def get_ball_height_meters_calibrated(self, ball_y: int) -> float:
        """Calculate ball height using antenna calibration"""
        if not self.calibration_completed or self.ground_level_y is None:
            return 0.0
            
        # Height above ground level
        height_pixels = self.ground_level_y - ball_y
        height_meters = max(0.0, height_pixels * self.height_scale_factor)
        
        return height_meters
        
    def get_net_clearance_calibrated(self, ball_y: int) -> float:
        """Calculate ball clearance over net using antenna calibration"""
        if not self.calibration_completed or len(self.left_antenna_points) != 2 or len(self.right_antenna_points) != 2:
            return 0.0
            
        # Net height is at antenna level (2.43m) - average of all antenna points
        all_antenna_y_coords = [
            self.left_antenna_points[0][1], self.left_antenna_points[1][1],
            self.right_antenna_points[0][1], self.right_antenna_points[1][1]
        ]
        avg_antenna_y = sum(all_antenna_y_coords) / len(all_antenna_y_coords)
        clearance_pixels = avg_antenna_y - ball_y
        clearance_meters = clearance_pixels * self.height_scale_factor
        
        return clearance_meters
        
    def is_ball_above_net_height_calibrated(self, ball_y: int) -> bool:
        """Check if ball is above net height using antenna calibration"""
        ball_height = self.get_ball_height_meters_calibrated(ball_y)
        return ball_height > self.net_height_meters
        
    def _calibrate_court_lines(self, frame: np.ndarray) -> bool:
        """Calibrate all court lines for full perspective"""
        # Stage 2: Back line
        if not self._calibrate_line(frame, "back line", "Back Line Calibration"):
            return False
        
        # Stage 3: Left sideline
        if not self._calibrate_line(frame, "left sideline", "Left Sideline Calibration"):
            return False
            
        # Stage 4: Right sideline
        if not self._calibrate_line(frame, "right sideline", "Right Sideline Calibration"):
            return False
            
        # Calculate perspective transformation
        self._calculate_perspective_matrix()
        return True
        
    def _calibrate_line(self, frame: np.ndarray, line_name: str, window_name: str) -> bool:
        """Generic method to calibrate any court line"""
        stage_num = {"back line": 2, "left sideline": 3, "right sideline": 4}[line_name]
        
        print(f"\n📐 Stage {stage_num}: {line_name.title()} calibration")
        print("=" * 60)
        print("Instructions:")
        print(f"1. Click on ANY TWO POINTS on the {line_name}")
        print("2. Points can be anywhere along the line")
        print("3. Press ENTER to continue")
        print("4. Press ESC to skip (will abort full calibration)")
        print("=" * 60)
        
        # Create calibration window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 700)
        
        # Copy frame for drawing
        calibration_frame = frame.copy()
        current_points = []
        
        # Draw already calibrated elements
        self._draw_calibrated_elements(calibration_frame)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(current_points) < 2:
                current_points.append((x, y))
                colors = [(0, 255, 255), (255, 0, 255)]  # Cyan, Magenta
                color = colors[len(current_points) - 1]
                
                cv2.circle(calibration_frame, (x, y), 8, color, -1)
                cv2.circle(calibration_frame, (x, y), 15, color, 3)
                
                label = f"{line_name.title()} P{len(current_points)}"
                cv2.putText(calibration_frame, label, (x + 20, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                print(f"✅ Marked {label} at ({x}, {y})")
                
                if len(current_points) == 2:
                    # Draw line between points
                    cv2.line(calibration_frame, current_points[0], current_points[1], 
                            (255, 255, 255), 3)
                    print(f"🎯 {line_name.title()} marked!")
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Show calibration interface
        while True:
            # Draw instructions on frame
            overlay = calibration_frame.copy()
            cv2.rectangle(overlay, (10, 10), (600, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, calibration_frame, 0.2, 0, calibration_frame)
            
            instructions = [
                f"{line_name.title()} Points: {len(current_points)}/2",
                f"Left click: Mark points on {line_name}",
                "ENTER: Continue to next stage", 
                "ESC: Skip (abort calibration)"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(calibration_frame, instruction, (15, 30 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                           
            cv2.imshow(window_name, calibration_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                break
            elif key == 27:  # ESC
                print(f"❌ {line_name.title()} calibration skipped")
                cv2.destroyWindow(window_name)
                return False
                
        cv2.destroyWindow(window_name)
        
        if len(current_points) == 2:
            # Store points in appropriate list
            if line_name == "back line":
                self.back_line_points = current_points
            elif line_name == "left sideline":
                self.left_sideline_points = current_points
            elif line_name == "right sideline":
                self.right_sideline_points = current_points
                
            print(f"✅ {line_name.title()} calibrated!")
            print(f"📏 Points: {current_points}")
            return True
        else:
            print(f"⚠️ Need to mark two points on {line_name}")
            return False
            
    def _draw_calibrated_elements(self, frame: np.ndarray) -> None:
        """Draw already calibrated elements on calibration frame"""
        # Draw antenna lines
        if len(self.left_antenna_points) == 2:
            cv2.line(frame, self.left_antenna_points[0], self.left_antenna_points[1], (0, 255, 0), 3)
            for i, point in enumerate(self.left_antenna_points):
                cv2.circle(frame, point, 8, (0, 255, 0), -1)
                cv2.putText(frame, f"L{i+1}", (point[0] - 8, point[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if len(self.right_antenna_points) == 2:
            cv2.line(frame, self.right_antenna_points[0], self.right_antenna_points[1], (255, 0, 0), 3)
            for i, point in enumerate(self.right_antenna_points):
                cv2.circle(frame, point, 8, (255, 0, 0), -1)
                cv2.putText(frame, f"R{i+1}", (point[0] - 8, point[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw net line if both antenna lines are complete
        if len(self.left_antenna_points) == 2 and len(self.right_antenna_points) == 2:
            all_antenna_y_coords = [
                self.left_antenna_points[0][1], self.left_antenna_points[1][1],
                self.right_antenna_points[0][1], self.right_antenna_points[1][1]
            ]
            avg_net_y = int(sum(all_antenna_y_coords) / len(all_antenna_y_coords))
            net_left_x = min(self.left_antenna_points[0][0], self.left_antenna_points[1][0]) - 30
            net_right_x = max(self.right_antenna_points[0][0], self.right_antenna_points[1][0]) + 30
            cv2.line(frame, (net_left_x, avg_net_y), (net_right_x, avg_net_y), (255, 255, 0), 3)
                       
        # Draw back line if calibrated
        if len(self.back_line_points) == 2:
            cv2.line(frame, self.back_line_points[0], self.back_line_points[1], 
                    (255, 255, 255), 3)
            for i, point in enumerate(self.back_line_points):
                cv2.circle(frame, point, 6, (0, 255, 255), -1)
                cv2.putText(frame, f"B{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                           
        # Draw left sideline if calibrated
        if len(self.left_sideline_points) == 2:
            cv2.line(frame, self.left_sideline_points[0], self.left_sideline_points[1], 
                    (0, 255, 0), 3)
            for i, point in enumerate(self.left_sideline_points):
                cv2.circle(frame, point, 6, (0, 255, 0), -1)
                cv2.putText(frame, f"L{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                           
        # Draw right sideline if calibrated
        if len(self.right_sideline_points) == 2:
            cv2.line(frame, self.right_sideline_points[0], self.right_sideline_points[1], 
                    (255, 0, 0), 3)
            for i, point in enumerate(self.right_sideline_points):
                cv2.circle(frame, point, 6, (255, 0, 0), -1)
                cv2.putText(frame, f"R{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
    def _calculate_perspective_matrix(self) -> None:
        """Calculate perspective transformation matrix from 6 calibrated points"""
        if (len(self.left_antenna_points) != 2 or len(self.right_antenna_points) != 2 or
            len(self.back_line_points) != 2 or 
            len(self.left_sideline_points) != 2 or len(self.right_sideline_points) != 2):
            print("❌ Error: All lines must be calibrated for perspective calculation")
            return
            
        print("🔄 Calculating perspective transformation...")
        
        # Step 1: Calculate intersection points of lines to get court corners
        court_corners_screen = self._calculate_court_corners()
        
        if len(court_corners_screen) != 4:
            print("❌ Error: Could not calculate court corners")
            return
        
        # Step 2: Define real-world court coordinates (in meters)  
        # Standard volleyball court: 9m wide, 9m deep from net (half court visible)
        # Camera is behind the court, so net is at Y=0, back line at Y=9
        # Court width: 9m total, so X from -4.5 to +4.5
        court_corners_world = np.array([
            [-4.5, 0.0],    # Net-left (left antenna) - left side of net
            [4.5, 0.0],     # Net-right (right antenna) - right side of net
            [4.5, 9.0],     # Back-right corner - right side of back line
            [-4.5, 9.0]     # Back-left corner - left side of back line
        ], dtype=np.float32)
        
        print(f"🗺️ Volleyball court coordinate system (official dimensions):")
        print(f"   Court width: 9m (-4.5m to +4.5m)")
        print(f"   Court depth: 9m (net at 0m, back line at 9m)")
        print(f"   🎯 Net corners: antenna level ∩ sidelines (antennas set height reference)")
        print(f"   Net-left: {court_corners_world[0]} (net level intersects left sideline)")
        print(f"   Net-right: {court_corners_world[1]} (net level intersects right sideline)")
        print(f"   Back-right: {court_corners_world[2]} (back line intersects right sideline)")
        print(f"   Back-left: {court_corners_world[3]} (back line intersects left sideline)")
        
        # Step 3: Arrange screen coordinates to match world coordinates order (clockwise)
        court_corners_screen_ordered = np.array([
            court_corners_screen['net_left'],      # Top-left
            court_corners_screen['net_right'],     # Top-right
            court_corners_screen['back_right'],    # Bottom-right
            court_corners_screen['back_left']      # Bottom-left
        ], dtype=np.float32)
        
        # Step 3.5: Validate and fix corner order if needed
        court_corners_screen_fixed = self._validate_corner_order(court_corners_screen_ordered)
        
        # Step 4: Calculate perspective transformation matrix
        self.perspective_matrix = cv2.getPerspectiveTransform(
            court_corners_screen_fixed, court_corners_world
        )
        
        # Store court corners for visualization
        self.court_corners_screen = court_corners_screen
        self.court_corners_world = court_corners_world
        
        print("✅ Perspective transformation calculated!")
        print(f"📐 Court corners in screen coordinates:")
        for corner_name, point in court_corners_screen.items():
            print(f"   {corner_name}: {point}")
        
        print(f"📐 Transformation mapping:")
        print(f"   Screen -> World coordinate pairs:")
        for i, (screen_pt, world_pt) in enumerate(zip(court_corners_screen_fixed, court_corners_world)):
            print(f"   {i+1}: {screen_pt} -> {world_pt}")
            
        # Store fixed corners for visualization
        self.court_corners_screen = court_corners_screen_fixed
            
        # Set calibration completed flag before testing
        self.court_calibration_completed = True
        
        # Test transformation with one point
        test_screen = court_corners_screen_fixed[0]
        test_world = self.screen_to_court_coordinates(test_screen)
        print(f"📐 Test transformation: {test_screen} -> {test_world} (should be close to {court_corners_world[0]})")
        
        # Additional validation
        expected_world = court_corners_world[0]
        distance = np.sqrt((test_world[0] - expected_world[0])**2 + (test_world[1] - expected_world[1])**2)
        if distance < 1.0:  # Within 1 meter tolerance
            print(f"✅ Transformation accuracy good: distance = {distance:.2f}m")
        else:
            print(f"⚠️ Transformation may be inaccurate: distance = {distance:.2f}m")
        
    def _validate_corner_order(self, corners):
        """Validate that corners form a proper quadrilateral without self-intersections"""
        print("🔍 Validating corner order...")
        
        # Check if lines intersect (which would indicate wrong order)
        def lines_intersect(p1, p2, p3, p4):
            """Check if line p1-p2 intersects with line p3-p4"""
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
        
        # Check diagonal lines - they should intersect (normal for quadrilateral)
        diagonal1_intersects_sides = False
        diagonal2_intersects_sides = False
        
        # Check if sides intersect each other (bad - means wrong order)
        side_intersections = 0
        sides = [(0,1), (1,2), (2,3), (3,0)]  # Adjacent sides
        
        for i, (a, b) in enumerate(sides):
            for j, (c, d) in enumerate(sides):
                if abs(i - j) > 1 and not (i == 0 and j == 3):  # Skip adjacent sides
                    if lines_intersect(corners[a], corners[b], corners[c], corners[d]):
                        side_intersections += 1
                        print(f"⚠️ Side {i} intersects with side {j}")
        
        if side_intersections > 0:
            print(f"❌ Found {side_intersections} side intersections - corner order is wrong!")
            print("🔄 Attempting to fix corner order...")
            return self._fix_corner_order(corners)
        else:
            print("✅ Corner order is correct - no side intersections")
            return corners
            
    def _fix_corner_order(self, corners):
        """Attempt to fix corner order by trying different arrangements"""
        print("🔧 Trying to fix corner order...")
        
        # Try different orderings
        orderings = [
            [0, 1, 2, 3],  # Original
            [0, 3, 2, 1],  # Reverse order
            [1, 2, 3, 0],  # Rotate
            [3, 0, 1, 2],  # Rotate other way
            [0, 2, 1, 3],  # Swap diagonals
            [1, 3, 0, 2]   # Other diagonal swap
        ]
        
        for i, order in enumerate(orderings):
            test_corners = np.array([corners[j] for j in order], dtype=np.float32)
            print(f"🧪 Testing order {i+1}: {order}")
            
            # Test this ordering
            if self._test_corner_order(test_corners):
                print(f"✅ Fixed! Using order {i+1}")
                return test_corners
                
        print("❌ Could not fix corner order automatically")
        return corners
        
    def _test_corner_order(self, corners):
        """Test if corner order is valid (no side intersections)"""
        def lines_intersect(p1, p2, p3, p4):
            def ccw(A, B, C):
                return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
        
        sides = [(0,1), (1,2), (2,3), (3,0)]
        for i, (a, b) in enumerate(sides):
            for j, (c, d) in enumerate(sides):
                if abs(i - j) > 1 and not (i == 0 and j == 3):
                    if lines_intersect(corners[a], corners[b], corners[c], corners[d]):
                        return False
        return True
            
    def _calculate_court_corners(self) -> dict:
        """Calculate the 4 court corners from calibrated lines"""
        # Get line intersection helper
        def line_intersection(line1_points, line2_points):
            """Calculate intersection of two lines defined by 2 points each"""
            x1, y1 = line1_points[0]
            x2, y2 = line1_points[1] 
            x3, y3 = line2_points[0]
            x4, y4 = line2_points[1]
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:  # Lines are parallel
                return None
                
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            
            # Calculate intersection point
            ix = x1 + t*(x2-x1)
            iy = y1 + t*(y2-y1)
            return (int(ix), int(iy))
        
        corners = {}
        
        # Net line (horizontal line at antenna level)
        # Calculate net level from antenna midpoints
        left_antenna_mid_y = (self.left_antenna_points[0][1] + self.left_antenna_points[1][1]) / 2
        right_antenna_mid_y = (self.right_antenna_points[0][1] + self.right_antenna_points[1][1]) / 2
        net_level_y = (left_antenna_mid_y + right_antenna_mid_y) / 2
        
        # Net line extends across the frame width at net level
        net_line = [(0, int(net_level_y)), (10000, int(net_level_y))]  # Horizontal line across frame
        
        # Calculate intersections
        # Back-left corner: intersection of back line and left sideline
        back_left = line_intersection(self.back_line_points, self.left_sideline_points)
        if back_left:
            corners['back_left'] = back_left
            print(f"✅ Back-left corner calculated: {back_left}")
        else:
            print("❌ Could not calculate back-left corner intersection")
            
        # Back-right corner: intersection of back line and right sideline  
        back_right = line_intersection(self.back_line_points, self.right_sideline_points)
        if back_right:
            corners['back_right'] = back_right
            print(f"✅ Back-right corner calculated: {back_right}")
        else:
            print("❌ Could not calculate back-right corner intersection")
            
        # Net corners: find intersections of antenna level (net height) with sidelines
        # Get Y-level from antenna lines (average Y of all 4 antenna points = net level)
        all_antenna_y_coords = [
            self.left_antenna_points[0][1], self.left_antenna_points[1][1],
            self.right_antenna_points[0][1], self.right_antenna_points[1][1]
        ]
        net_level_y = sum(all_antenna_y_coords) / len(all_antenna_y_coords)
        
        print(f"🎯 Net level Y: {net_level_y:.1f} (average of all antenna points)")
        
        # Create horizontal line at net level to intersect with sidelines
        # Horizontal line from far left to far right at net level
        net_level_line = [(0, int(net_level_y)), (10000, int(net_level_y))]
        
        # Net-left corner: intersection of net level with left sideline
        net_left = line_intersection(net_level_line, self.left_sideline_points)
        if net_left:
            corners['net_left'] = net_left
            print(f"✅ Net-left corner calculated: {net_left} (net level ∩ left sideline)")
        else:
            # Fallback: use average of left antenna points
            left_antenna_mid = ((self.left_antenna_points[0][0] + self.left_antenna_points[1][0]) // 2,
                              (self.left_antenna_points[0][1] + self.left_antenna_points[1][1]) // 2)
            corners['net_left'] = left_antenna_mid
            print(f"⚠️ Using left antenna midpoint as fallback: {corners['net_left']}")
            
        # Net-right corner: intersection of net level with right sideline
        net_right = line_intersection(net_level_line, self.right_sideline_points)
        if net_right:
            corners['net_right'] = net_right
            print(f"✅ Net-right corner calculated: {net_right} (net level ∩ right sideline)")
        else:
            # Fallback: use average of right antenna points
            right_antenna_mid = ((self.right_antenna_points[0][0] + self.right_antenna_points[1][0]) // 2,
                               (self.right_antenna_points[0][1] + self.right_antenna_points[1][1]) // 2)
            corners['net_right'] = right_antenna_mid
            print(f"⚠️ Using right antenna midpoint as fallback: {corners['net_right']}")
            
        print(f"📐 Total corners calculated: {len(corners)}")
        return corners
        
    def screen_to_court_coordinates(self, screen_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Convert screen coordinates to court coordinates using perspective transformation"""
        if not self.court_calibration_completed or self.perspective_matrix is None:
            return (0.0, 0.0)
            
        x_screen, y_screen = screen_pos
        
        # Apply perspective transformation
        screen_point = np.array([[[x_screen, y_screen]]], dtype=np.float32)
        court_point = cv2.perspectiveTransform(screen_point, self.perspective_matrix)
        
        # Extract transformed coordinates
        x_court, y_court = court_point[0][0]
        
        return (float(x_court), float(y_court))
        
    def court_to_screen_coordinates(self, court_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert court coordinates back to screen coordinates"""
        if not self.court_calibration_completed or self.perspective_matrix is None:
            return (0, 0)
            
        x_court, y_court = court_pos
        
        # Apply inverse perspective transformation
        court_point = np.array([[[x_court, y_court]]], dtype=np.float32) 
        inverse_matrix = cv2.invert(self.perspective_matrix)[1]
        screen_point = cv2.perspectiveTransform(court_point, inverse_matrix)
        
        # Extract screen coordinates
        x_screen, y_screen = screen_point[0][0]
        
        return (int(x_screen), int(y_screen))
        
    def screen_to_world(self, screen_pos: Tuple[float, float], 
                       frame_width: int, frame_height: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        x_screen, y_screen = screen_pos
        
        # Simple linear mapping (assumes camera shows full court)
        x_world = (x_screen / frame_width) * self.court_bounds.right
        y_world = (y_screen / frame_height) * self.court_bounds.bottom
        
        return (x_world, y_world)
        
    def world_to_screen(self, world_pos: Tuple[float, float], 
                       frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        x_world, y_world = world_pos
        
        x_screen = int((x_world / self.court_bounds.right) * frame_width)
        y_screen = int((y_world / self.court_bounds.bottom) * frame_height)
        
        return (x_screen, y_screen)
        
    def detect_net_in_frame(self, frame: np.ndarray) -> bool:
        """Detect volleyball net for camera calibration"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 100, 10)
            
            if lines is None:
                return False
                
            # Filter for horizontal lines (net top and bottom)
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                
                # Keep nearly horizontal lines
                if angle < 15 and abs(x2 - x1) > 50:  # Long horizontal lines
                    y_avg = (y1 + y2) // 2
                    horizontal_lines.append(y_avg)
            
            if len(horizontal_lines) < 2:
                return False
                
            # Sort lines by Y coordinate and take top/bottom
            horizontal_lines.sort()
            self.net_top_y = horizontal_lines[0]
            self.net_bottom_y = horizontal_lines[-1]
            
            # Calculate height scale factor
            net_pixel_height = abs(self.net_bottom_y - self.net_top_y)
            if net_pixel_height > 10:  # Minimum reasonable net height
                self.height_scale_factor = self.net_height_meters / net_pixel_height
                self.net_detected = True
                
                if self.calibration_frame_count == 0:  # First time calibration
                    print(f"📐 Net detected! Height: {net_pixel_height}px = {self.net_height_meters}m")
                    print(f"📐 Height scale: {self.height_scale_factor:.4f} m/pixel")
                
                self.calibration_frame_count += 1
                return True
                
            return False
            
        except Exception as e:
            print(f"Net detection error: {e}")
            return False
            
    def get_ball_height_meters(self, ball_y: int) -> float:
        """Calculate ball height in meters using net as reference"""
        if not self.net_detected or self.net_bottom_y is None:
            return 0.0
            
        # Calculate height above ground (net bottom as reference level)
        height_pixels = self.net_bottom_y - ball_y
        height_meters = max(0.0, height_pixels * self.height_scale_factor)
        
        return height_meters
        
    def get_net_clearance(self, ball_y: int) -> float:
        """Calculate ball clearance over net"""
        if not self.net_detected or self.net_top_y is None:
            return 0.0
            
        # Clearance above net top
        clearance_pixels = self.net_top_y - ball_y
        clearance_meters = clearance_pixels * self.height_scale_factor
        
        return clearance_meters
        
    def is_ball_above_net_height(self, ball_y: int) -> bool:
        """Check if ball is above net height"""
        ball_height = self.get_ball_height_meters(ball_y)
        return ball_height > self.net_height_meters
        
    def screen_to_world_with_perspective(self, screen_pos: Tuple[float, float], 
                                       frame_width: int, frame_height: int) -> Tuple[float, float, float]:
        """Convert screen coordinates to world coordinates with height"""
        x_screen, y_screen = screen_pos
        
        # Basic 2D mapping for ground position
        x_world = (x_screen / frame_width) * self.court_bounds.right
        y_world = (y_screen / frame_height) * self.court_bounds.bottom
        
        # Calculate height using net calibration
        z_world = self.get_ball_height_meters(int(y_screen))
        
        # Perspective correction for Y coordinate (depth)
        # Camera behind team means closer = larger Y values
        # Apply simple perspective scaling
        if frame_height > 0:
            perspective_factor = y_screen / frame_height
            # Far objects (small y_screen) should map to far Y world coordinates
            y_world_corrected = self.court_bounds.bottom * (1.0 - perspective_factor)
            y_world = y_world_corrected
            
        return (x_world, y_world, z_world)
        
    def draw_calibration_overlay(self, frame: np.ndarray) -> None:
        """Draw calibration information on frame"""
        h, w = frame.shape[:2]
        
        if self.calibration_completed and len(self.left_antenna_points) == 2 and len(self.right_antenna_points) == 2:
            # Draw antenna lines and points
            
            # Draw left antenna line and points
            cv2.line(frame, self.left_antenna_points[0], self.left_antenna_points[1], (0, 255, 0), 4)
            for i, point in enumerate(self.left_antenna_points):
                cv2.circle(frame, point, 8, (0, 255, 0), -1)
                cv2.circle(frame, point, 12, (0, 255, 0), 3)
                cv2.putText(frame, f"L{i+1}", (point[0] - 8, point[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw right antenna line and points
            cv2.line(frame, self.right_antenna_points[0], self.right_antenna_points[1], (255, 0, 0), 4)
            for i, point in enumerate(self.right_antenna_points):
                cv2.circle(frame, point, 8, (255, 0, 0), -1)
                cv2.circle(frame, point, 12, (255, 0, 0), 3)
                cv2.putText(frame, f"R{i+1}", (point[0] - 8, point[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw net line (horizontal line at average antenna level)
            all_antenna_y_coords = [
                self.left_antenna_points[0][1], self.left_antenna_points[1][1],
                self.right_antenna_points[0][1], self.right_antenna_points[1][1]
            ]
            avg_net_y = int(sum(all_antenna_y_coords) / len(all_antenna_y_coords))
            
            # Net line spans across the visible court area
            net_left_x = min(self.left_antenna_points[0][0], self.left_antenna_points[1][0]) - 50
            net_right_x = max(self.right_antenna_points[0][0], self.right_antenna_points[1][0]) + 50
            cv2.line(frame, (net_left_x, avg_net_y), (net_right_x, avg_net_y), (255, 255, 0), 4)
            
            # Draw court lines if full perspective calibration is completed
            if self.court_calibration_completed and hasattr(self, 'court_corners_screen'):
                self._draw_perspective_court(frame)
            
            # Draw ground level line if available
            if self.ground_level_y:
                cv2.line(frame, (0, int(self.ground_level_y)), (w, int(self.ground_level_y)), 
                        (100, 100, 100), 2)
                cv2.putText(frame, "Ground Level", (10, int(self.ground_level_y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Add calibration info
            calib_status = "FULL COURT CALIBRATED" if self.court_calibration_completed else "ANTENNA CALIBRATED"
            calib_color = (0, 255, 0) if self.court_calibration_completed else (255, 255, 0)
            cv2.putText(frame, f"{calib_status} | Scale: {self.height_scale_factor:.4f} m/px", 
                       (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 2)
            
        elif self.net_detected and self.net_top_y and self.net_bottom_y:
            # Fallback to automatic net detection display
            cv2.line(frame, (0, self.net_top_y), (w, self.net_top_y), (0, 255, 0), 4)
            cv2.line(frame, (0, self.net_bottom_y), (w, self.net_bottom_y), (0, 255, 0), 4)
            cv2.rectangle(frame, (50, self.net_top_y), (w-50, self.net_bottom_y), (255, 0, 0), 4)
            
            net_center_y = (self.net_top_y + self.net_bottom_y) // 2
            cv2.putText(frame, f"Net: {self.net_height_meters}m", 
                       (w//2 - 60, net_center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 3)
            
            cv2.putText(frame, f"AUTO | Scale: {self.height_scale_factor:.4f} m/px", 
                       (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # No calibration
            cv2.putText(frame, "NOT CALIBRATED - Heights may be inaccurate", 
                       (10, h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
    def _draw_perspective_court(self, frame: np.ndarray) -> None:
        """Draw court with proper perspective using calibrated transformation"""
        if not hasattr(self, 'court_corners_screen') or not hasattr(self, 'perspective_matrix'):
            return
            
        corners = self.court_corners_screen
        
        # Draw court corners - corners is now a numpy array
        corner_colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 0, 255)]  # Cyan, Magenta, Green, Red
        corner_labels = ['TL', 'TR', 'BR', 'BL']  # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        
        for i, (point, color, label) in enumerate(zip(corners, corner_colors, corner_labels)):
            point_int = tuple(map(int, point))
            cv2.circle(frame, point_int, 10, color, -1)
            cv2.circle(frame, point_int, 15, color, 3)
            
            # Label corners
            cv2.putText(frame, label, 
                       (point_int[0] + 20, point_int[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw court boundary lines using fixed corners if available
        if hasattr(self, 'court_corners_screen'):
            # Use the validated and fixed corners
            boundary_points = self.court_corners_screen.tolist()
            
            # Draw court boundary (white lines) - connect points in order
            for i in range(len(boundary_points)):
                pt1 = tuple(map(int, boundary_points[i]))
                pt2 = tuple(map(int, boundary_points[(i + 1) % len(boundary_points)]))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 4)
        else:
            # Fallback to original method
            court_lines = [
                (corners['back_left'], corners['back_right']),      # Back line
                (corners['back_left'], corners['net_left']),        # Left sideline
                (corners['back_right'], corners['net_right']),      # Right sideline  
                (corners['net_left'], corners['net_right'])         # Net line
            ]
            
            for line_start, line_end in court_lines:
                cv2.line(frame, line_start, line_end, (255, 255, 255), 4)
            
        # Draw internal court lines with perspective
        self._draw_internal_court_lines(frame)
        
        # Draw calibration reference points
        self._draw_calibration_reference_points(frame)
        
    def _draw_internal_court_lines(self, frame: np.ndarray) -> None:
        """Draw internal court lines (center line, attack lines) with perspective"""
        if self.perspective_matrix is None:
            return
            
        # Define court lines in world coordinates (meters) - official volleyball court
        internal_lines_world = [
            # Attack line (3m from net) - official volleyball dimension
            ((-4.5, 3.0), (4.5, 3.0)),  # Attack line across full court width
            # Center line down the middle of court (dividing left/right sides)
            ((0.0, 0.0), (0.0, 9.0))    # Center line from net to back line
        ]
        
        for line_start_world, line_end_world in internal_lines_world:
            # Convert world coordinates to screen coordinates
            start_screen = self.court_to_screen_coordinates(line_start_world)
            end_screen = self.court_to_screen_coordinates(line_end_world)
            
            # Draw line
            cv2.line(frame, start_screen, end_screen, (100, 100, 100), 2)
            
    def _draw_calibration_reference_points(self, frame: np.ndarray) -> None:
        """Draw the original calibration points for reference"""
        # Draw user-marked back line points
        if len(self.back_line_points) == 2:
            for i, point in enumerate(self.back_line_points):
                cv2.circle(frame, point, 6, (255, 255, 0), -1)  # Yellow
                cv2.putText(frame, f"B{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                           
        # Draw user-marked left sideline points
        if len(self.left_sideline_points) == 2:
            for i, point in enumerate(self.left_sideline_points):
                cv2.circle(frame, point, 6, (0, 255, 0), -1)  # Green
                cv2.putText(frame, f"L{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                           
        # Draw user-marked right sideline points
        if len(self.right_sideline_points) == 2:
            for i, point in enumerate(self.right_sideline_points):
                cv2.circle(frame, point, 6, (255, 0, 0), -1)  # Red
                cv2.putText(frame, f"R{i+1}", (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    def create_analysis_visualization(self, frame_shape: tuple) -> np.ndarray:
        """Create separate analysis visualization window"""
        h, w = frame_shape[:2]
        
        # Create black background for analysis
        analysis_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw court representation
        self._draw_court_diagram(analysis_frame)
        
        # Draw FSM information
        self._draw_fsm_info_detailed(analysis_frame)
        
        # Draw ball trajectory and statistics
        self._draw_ball_analysis(analysis_frame)
        
        # Draw detection statistics
        self._draw_detection_stats(analysis_frame)
        
        return analysis_frame
    
    def _draw_court_diagram(self, frame: np.ndarray) -> None:
        """Draw volleyball court diagram for analysis"""
        h, w = frame.shape[:2]
        
        # Court dimensions in the analysis view (scaled to fit) - enlarged
        court_margin = 40
        court_width = w - 2 * court_margin
        court_height = int(court_width * 0.6)  # Increased court height for better visibility
        
        # Court position
        court_x = court_margin
        court_y = court_margin
        
        # Draw court boundary - thickened lines
        court_rect = [(court_x, court_y), (court_x + court_width, court_y + court_height)]
        cv2.rectangle(frame, court_rect[0], court_rect[1], (100, 100, 100), 4)  # Increased thickness
        
        # Draw net line (horizontal line in middle) - thicker line
        net_y = court_y
        cv2.line(frame, (court_x, net_y), (court_x + court_width, net_y), (255, 255, 0), 6)  # Increased thickness
        
        # Draw center line (vertical) - thicker line
        center_x = court_x + court_width // 2
        cv2.line(frame, (center_x, court_y), (center_x, court_y + court_height), (100, 100, 100), 3)  # Increased thickness
        
        # Draw attack line (3m from net) - thicker line
        attack_y = court_y + int(court_height * 3 / 9)  # 3m from net
        cv2.line(frame, (court_x, attack_y), (court_x + court_width, attack_y), (100, 100, 100), 3)  # Increased thickness
        
        # Court labels - increased text size
        cv2.putText(frame, "Net", (center_x - 30, net_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 3)  # Increased
        cv2.putText(frame, "Attack Line", (court_x + 15, attack_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)  # Increased
        cv2.putText(frame, "Back Line", (court_x + 15, court_y + court_height - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)  # Increased
        
        # Store court coordinates for ball position mapping
        self.analysis_court_bounds = {
            'x': court_x, 'y': court_y, 
            'width': court_width, 'height': court_height
        }
    
    def _draw_fsm_info_detailed(self, frame: np.ndarray) -> None:
        """Draw detailed FSM information"""
        if not hasattr(self, 'analysis_court_bounds'):
            return
            
        bounds = self.analysis_court_bounds
        info_x = bounds['x']
        info_y = bounds['y'] + bounds['height'] + 30
        
        # FSM State information
        # Calculate score from events (rally end events)
        rally_end_events = [e for e in self.fsm.events if e.event_type == 'rally_end']
        team_a_score = len([e for e in rally_end_events if e.team == 'team_a'])
        team_b_score = len([e for e in rally_end_events if e.team == 'team_b'])
        
        state_info = [
            f"Volleyball FSM Analysis",
            f"Frame: {self.frame_count}",
            f"Game State: {self.fsm.current_state.name if self.fsm.current_state is not None else 'Unknown'}",
            f"Rally Phase: {self.fsm.current_rally_phase.name if hasattr(self.fsm, 'current_rally_phase') and self.fsm.current_rally_phase is not None else 'N/A'}",
            f"Team A Score: {team_a_score}",
            f"Team B Score: {team_b_score}",
            f"Contacts: {self.fsm.contact_count if hasattr(self.fsm, 'contact_count') else 0}",
            f"Current Team: {self.fsm.current_team if hasattr(self.fsm, 'current_team') else 'Unknown'}",
        ]
        
        for i, info in enumerate(state_info):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)  # Yellow for title, white for others
            font_size = 1.0 if i == 0 else 0.8  # Increased text size
            thickness = 3 if i == 0 else 2  # Increased thickness
            
            cv2.putText(frame, info, (info_x, info_y + i * 30),  # Increased spacing between lines
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    
    def _draw_ball_analysis(self, frame: np.ndarray) -> None:
        """Draw ball position and trajectory analysis"""
        if not hasattr(self, 'analysis_court_bounds'):
            return
            
        bounds = self.analysis_court_bounds
        
        # Draw ball trajectory on court diagram (simplified without calibration)
        if len(self.ball_trajectory) > 1:
            # Map screen coordinates directly to court diagram
            for i in range(1, min(len(self.ball_trajectory), 20)):  # Last 20 points
                ball_screen = self.ball_trajectory[-(i+1)]
                
                # Simple mapping from screen to court diagram (assuming 1920x1080 screen)
                # Map to court bounds proportionally
                screen_x, screen_y = ball_screen
                
                # Normalize screen coordinates (0-1)
                norm_x = screen_x / 1920.0 if screen_x < 3000 else screen_x / 3840.0  # Handle different resolutions
                norm_y = screen_y / 1080.0 if screen_y < 2000 else screen_y / 2160.0
                
                # Map to court diagram
                analysis_x = bounds['x'] + int(norm_x * bounds['width'])
                analysis_y = bounds['y'] + int(norm_y * bounds['height'])
                
                # Ensure coordinates are within bounds
                analysis_x = max(bounds['x'], min(bounds['x'] + bounds['width'], analysis_x))
                analysis_y = max(bounds['y'], min(bounds['y'] + bounds['height'], analysis_y))
                
                # Draw trajectory point (fade with age) - increased point size
                alpha = 1.0 - (i / 20.0)
                color_intensity = int(255 * alpha)
                color = (0, color_intensity, 0) if i == 1 else (0, color_intensity // 2, 0)
                radius = 10 if i == 1 else max(4, 10 - i // 2)  # Increased point radius
                cv2.circle(frame, (analysis_x, analysis_y), radius, color, -1)
        
        # Ball information panel - moved down
        ball_info_x = bounds['x'] + bounds['width'] + 30  # Slightly further right
        ball_info_y = bounds['y'] + 40
        
        # Calculate ball speed from trajectory
        ball_speed = 0.0
        if len(self.ball_trajectory) >= 2:
            last_pos = self.ball_trajectory[-1]
            prev_pos = self.ball_trajectory[-2]
            dx = last_pos[0] - prev_pos[0]  
            dy = last_pos[1] - prev_pos[1]
            ball_speed = np.sqrt(dx*dx + dy*dy) * 30.0  # Assuming 30 FPS
        
        # Determine ball side from current position (simple screen-based)
        ball_side = "Unknown"
        if len(self.ball_trajectory) > 0:
            current_ball_pos = self.ball_trajectory[-1]
            screen_y = current_ball_pos[1]
            
            # Simple assumption: top half = Team A, bottom half = Team B
            # This works for most volleyball camera angles
            frame_height = 1080 if current_ball_pos[1] < 2000 else 2160  # Detect resolution
            ball_side = "Team A" if screen_y < frame_height * 0.5 else "Team B"
        
        ball_info = [
            "Ball Analysis",
            f"Trajectory Points: {len(self.ball_trajectory)}",
            f"Speed: {ball_speed:.1f} px/s",
            f"Side: {ball_side}",
            f"Events: {len(self.fsm.events)}",
        ]
        
        for i, info in enumerate(ball_info):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            font_size = 0.9 if i == 0 else 0.7  # Increased text size
            thickness = 3 if i == 0 else 2  # Increased thickness
            
            cv2.putText(frame, info, (ball_info_x, ball_info_y + i * 30),  # Increased spacing
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    
    def _draw_detection_stats(self, frame: np.ndarray) -> None:
        """Draw detection quality statistics"""
        if not hasattr(self, 'analysis_court_bounds'):
            return
            
        bounds = self.analysis_court_bounds
        stats_x = bounds['x'] + bounds['width'] + 30  # Slightly further right
        stats_y = bounds['y'] + 220  # Slightly lower for better positioning
        
        # Calculate success rates
        total_detections = self.yolo_detections + self.color_fallback_detections
        yolo_rate = (self.yolo_detections / total_detections * 100) if total_detections > 0 else 0
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0
        
        detection_stats = [
            "Detection Stats",
            f"Total Detections: {total_detections}",
            f"YOLO Success: {yolo_rate:.1f}%",
            f"Current Method: {self.current_detection_method}",
            f"Avg Confidence: {avg_confidence:.2f}",
            f"Smoothed: {self.smoothed_detections}",
        ]
        
        for i, stat in enumerate(detection_stats):
            color = (255, 165, 0) if i == 0 else (255, 255, 255)  # Orange for title
            font_size = 0.9 if i == 0 else 0.7  # Increased text size
            thickness = 3 if i == 0 else 2  # Increased thickness
            
            cv2.putText(frame, stat, (stats_x, stats_y + i * 30),  # Increased spacing
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process single frame with FSM and return both original and analysis frames"""
        h, w = frame.shape[:2]
        timestamp = self.frame_count / 30.0  # Assume 30 FPS
        
        # Skip automatic net detection if manual calibration is completed
        if not self.calibration_completed and self.frame_count < 100:  # Calibrate in first 100 frames
            self.detect_net_in_frame(frame)
            
        # Detect ball
        ball_detection = self.ball_detector.detect(frame)
        ball_world_pos = None
        ball_height_info = None
        
        if ball_detection:
            x, y, confidence = ball_detection
            self.ball_detections += 1
            
            # Track detection statistics
            if hasattr(self.ball_detector, 'model') and self.ball_detector.model is not None and not self.ball_detector.use_color_fallback:
                self.yolo_detections += 1
                detection_method = "YOLO"
            else:
                self.color_fallback_detections += 1
                detection_method = "COLOR"
                
            # Store detection method for this frame
            self.current_detection_method = detection_method
                
            # Track confidence history
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > self.max_confidence_history:
                self.confidence_history.pop(0)
            
            # Get detailed height information using calibrated or fallback methods
            if self.calibration_completed:
                ball_height_info = {
                    'height_meters': self.get_ball_height_meters_calibrated(int(y)),
                    'net_clearance': self.get_net_clearance_calibrated(int(y)),
                    'above_net': self.is_ball_above_net_height_calibrated(int(y))
                }
            else:
                ball_height_info = {
                    'height_meters': self.get_ball_height_meters(int(y)),
                    'net_clearance': self.get_net_clearance(int(y)),
                    'above_net': self.is_ball_above_net_height(int(y))
                }
            
            # Convert to world coordinates with best available method
            if self.court_calibration_completed:
                # Use precise court coordinates
                ball_world_pos = self.screen_to_court_coordinates((x, y))
            elif self.calibration_completed:
                # Use antenna-based calibration with perspective correction
                ball_world_3d = self.screen_to_world_with_perspective((x, y), w, h)
                ball_world_pos = (ball_world_3d[0], ball_world_3d[1])  # 2D for FSM
            else:
                # Fallback to simple screen-to-world mapping
                ball_world_pos = self.screen_to_world((x, y), w, h)
            
            # Update FSM with correct parameters
            sample_player_positions = {"team_a": [], "team_b": []}  # Empty for now
            self.fsm.update(self.frame_count, ball_world_pos, sample_player_positions)
            
            # Store trajectory
            self.ball_trajectory.append((x, y))
            if len(self.ball_trajectory) > 50:  # Keep last 50 points
                self.ball_trajectory.pop(0)
                
            # Draw ball detection with height info
            center = (int(x), int(y))
            
            # Color based on height
            if ball_height_info and ball_height_info['above_net']:
                ball_color = (0, 255, 255)  # Yellow if above net
            else:
                ball_color = (0, 255, 0)    # Green if below net
                
            # Enlarged circles for ball
            cv2.circle(frame, center, 12, ball_color, -1)  # Increased from 8 to 12
            cv2.circle(frame, center, 18, ball_color, 3)   # Increased from 12 to 18, thickness from 2 to 3
            
            # Ball info text with detection method
            method_text = f"[{self.current_detection_method}]"
            method_color = (0, 255, 0) if self.current_detection_method == "YOLO" else (0, 255, 255)
            
            if ball_height_info:
                height_text = f"H: {ball_height_info['height_meters']:.1f}m"
                clearance_text = f"Net: {ball_height_info['net_clearance']:.1f}m"
                
                cv2.putText(frame, height_text, 
                           (center[0] + 25, center[1] - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball_color, 2)
                           
                cv2.putText(frame, clearance_text, 
                           (center[0] + 25, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
                
                # Show court coordinates if available
                if self.court_calibration_completed and ball_world_pos:
                    court_x, court_y = ball_world_pos
                    court_text = f"Court: ({court_x:.1f}, {court_y:.1f})m"
                    cv2.putText(frame, court_text, 
                               (center[0] + 25, center[1] + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
                           
                # Detection method indicator
                cv2.putText(frame, method_text, 
                           (center[0] + 25, center[1] + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, method_color, 2)
            else:
                cv2.putText(frame, f"Ball: {confidence:.2f}", 
                           (center[0] + 25, center[1] - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, ball_color, 2)
                
                # Show court coordinates if available
                if self.court_calibration_completed and ball_world_pos:
                    court_x, court_y = ball_world_pos
                    court_text = f"Court: ({court_x:.1f}, {court_y:.1f})m"
                    cv2.putText(frame, court_text, 
                               (center[0] + 25, center[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
                           
                # Detection method indicator
                cv2.putText(frame, method_text, 
                           (center[0] + 25, center[1] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, method_color, 2)
        else:
            # No ball detected in this frame
            self.current_detection_method = "NONE"
            # Update FSM with no ball position
            sample_player_positions = {"team_a": [], "team_b": []}
            self.fsm.update(self.frame_count, None, sample_player_positions)
                       
        # Draw ball trajectory
        if len(self.ball_trajectory) > 1:
            points = np.array(self.ball_trajectory, dtype=np.int32)
            for i in range(1, len(points)):
                alpha = i / len(points)
                thickness = max(2, int(5 * alpha))  # Increased thickness from 1-3 to 2-5
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), (255, 0, 0), thickness)
                
        # Create FSM visualization overlay
        try:
            # Create sample players for visualization
            sample_players = self._create_sample_players(self.frame_count)
            
            # Create visualization frame using correct method
            ball_velocity = 0.0
            if len(self.fsm.ball_trajectory) >= 2:
                last_pos = self.fsm.ball_trajectory[-1]
                prev_pos = self.fsm.ball_trajectory[-2]
                dx = last_pos[1] - prev_pos[1]
                dy = last_pos[2] - prev_pos[2]
                ball_velocity = np.sqrt(dx*dx + dy*dy)
            
            vis_frame = self.visualizer.create_visualization_frame(
                frame_idx=self.frame_count,
                ball_pos=ball_world_pos,
                player_keypoints=sample_players,
                fsm=self.fsm,
                ball_velocity=ball_velocity
            )
            
            # Resize visualization to match frame size
            vis_frame_resized = cv2.resize(vis_frame, (w, h))
            
            # Blend with original frame
            result_frame = cv2.addWeighted(frame, 0.7, vis_frame_resized, 0.3, 0)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            result_frame = frame
            
        # Draw FSM state information
        # Create separate analysis visualization
        analysis_frame = self.create_analysis_visualization(frame.shape)
        
        # Draw minimal overlay on original frame (only ball detection if any)
        # Skip calibration overlay since we're not using interactive calibration
        
        # Update frame counter
        self.frame_count += 1
        
        return result_frame, analysis_frame
        
    def _create_sample_players(self, frame_idx: int) -> List[PlayerKeypoints]:
        """Create sample player positions for visualization"""
        players = []
        
        # Team A positions (top half)
        team_a_positions = [(4, 2), (9, 2), (14, 2), (4, 4), (9, 4), (14, 4)]
        for i, (x, y) in enumerate(team_a_positions):
            # Add some movement animation
            movement = np.sin(frame_idx * 0.05 + i) * 0.5
            players.append(PlayerKeypoints(
                player_id=i + 1,
                team='team_a',
                center=(x + movement, y),
                head=(x + movement, y - 0.2),
                confidence=0.8
            ))
        
        # Team B positions (bottom half)  
        team_b_positions = [(4, 7), (9, 7), (14, 7), (4, 5), (9, 5), (14, 5)]
        for i, (x, y) in enumerate(team_b_positions):
            movement = np.sin(frame_idx * 0.05 + i + 6) * 0.5
            players.append(PlayerKeypoints(
                player_id=i + 1,
                team='team_b',
                center=(x + movement, y),
                head=(x + movement, y - 0.2),
                confidence=0.8
            ))
            
        return players
        
    def _draw_fsm_info(self, frame: np.ndarray) -> None:
        """Draw FSM state information on frame"""
        # Calculate detection rates and statistics
        detection_rate = (self.ball_detections / max(1, self.frame_count)) * 100
        yolo_rate = (self.yolo_detections / max(1, self.ball_detections)) * 100 if self.ball_detections > 0 else 0
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0
        
        # Create info panel (bigger for more stats)
        panel_height = 240
        panel_width = 480
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # FSM state information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # Increased from 0.5
        color = (255, 255, 255)
        y_offset = 35
        
        # Detection method indicator
        detection_method = "YOLO" if (hasattr(self.ball_detector, 'model') and 
                                    self.ball_detector.model is not None and 
                                    not self.ball_detector.use_color_fallback) else "COLOR"
        
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Game State: {self.fsm.current_state.name if self.fsm.current_state is not None else 'Unknown'}",
            f"Rally Phase: {self.fsm.current_rally_phase.name if self.fsm.current_rally_phase else 'None'}",
            f"Ball Detections: {self.ball_detections} ({detection_rate:.1f}%)",
            f"Detection Method: {detection_method}",
            f"YOLO/Color: {self.yolo_detections}/{self.color_fallback_detections}",
            f"YOLO Success: {yolo_rate:.1f}%",
            f"Avg Confidence: {avg_confidence:.2f}",
            f"Camera: {'AUTO NET DETECTED' if self.net_detected else 'AUTO MODE (NO CALIBRATION)'}",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (15, y_offset + i * 22), 
                       font, font_scale, color, 2)  # Thickness increased from 1 to 2
                       
        # Draw court visualization legend
        legend_y = frame.shape[0] - 185  # Increased from 160 for additional line
        legend_font_scale = 0.6  # Increased from 0.4
        cv2.putText(frame, "Legend:", (15, legend_y), font, font_scale, (255, 255, 255), 2)
        cv2.putText(frame, "Green: Ball Below Net | Yellow: Ball Above Net", 
                   (15, legend_y + 25), font, legend_font_scale, (255, 255, 255), 2)
        cv2.putText(frame, "Red Trail: Ball Trajectory | Green Lines: Net", 
                   (15, legend_y + 50), font, legend_font_scale, (255, 255, 255), 2)
        cv2.putText(frame, "H: Height in meters | Net: Clearance over net", 
                   (15, legend_y + 75), font, legend_font_scale, (255, 255, 255), 2)
        cv2.putText(frame, "[YOLO]: Green | [COLOR]: Yellow | Detection Methods", 
                   (15, legend_y + 100), font, legend_font_scale, (255, 255, 255), 2)
        cv2.putText(frame, "L/R: Antenna markers | Yellow line: Net height (2.43m)", 
                   (15, legend_y + 125), font, legend_font_scale, (255, 255, 255), 2)
        cv2.putText(frame, "White lines: Court perspective | B/L/R: Calibration points", 
                   (15, legend_y + 150), font, legend_font_scale, (255, 255, 255), 2)
                   
    def process_video(self, input_path: str, output_path: Optional[str] = None,
                     start_sec: float = 0, duration_sec: Optional[float] = None,
                     show_preview: bool = True) -> Dict[str, Any]:
        """
        Process video file with FSM visualization
        
        Args:
            input_path: Path to input video
            output_path: Path to output video (optional)
            start_sec: Start time in seconds
            duration_sec: Duration to process (None for full video)
            show_preview: Whether to show live preview
            
        Returns:
            Processing statistics
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
        
        # Calculate frame range
        start_frame = int(start_sec * fps)
        if duration_sec:
            end_frame = start_frame + int(duration_sec * fps)
        else:
            end_frame = total_frames
            
        end_frame = min(end_frame, total_frames)
        frames_to_process = end_frame - start_frame
        
        print(f"🎬 Processing frames {start_frame} to {end_frame} ({frames_to_process} frames)")
        
        # Set start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Skip interactive calibration - use automatic detection
        ret, first_frame = cap.read()
        if ret:
            # Try automatic net detection for basic calibration
            self.detect_net_in_frame(first_frame)
            # Reset position after detection
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Setup output writer if needed
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
        # Reset processing state
        self.frame_count = 0
        self.ball_detections = 0
        self.ball_trajectory = []
        self.fsm = VolleyballFSM(self.court_bounds)  # Reset FSM
        
        # Reset detection statistics
        self.yolo_detections = 0
        self.color_fallback_detections = 0
        self.filtered_detections = 0
        self.smoothed_detections = 0
        self.confidence_history = []
        
        start_time = time.time()
        
        try:
            for frame_idx in range(frames_to_process):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame (returns original + analysis)
                original_frame, analysis_frame = self.process_frame(frame)
                
                # Write to output if specified (only original frame)
                if out:
                    out.write(original_frame)
                    
                # Show preview - display both frames side by side
                if show_preview:
                    # Resize frames for display (larger for better visibility)
                    display_original = cv2.resize(original_frame, (960, 540))  # Increased from 800x450
                    display_analysis = cv2.resize(analysis_frame, (960, 540))   # Increased from 800x450
                    
                    # Combine frames horizontally
                    combined_display = np.hstack([display_original, display_analysis])
                    
                    cv2.imshow('Volleyball FSM Demo - Original | Analysis', combined_display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Stopping processing (user requested)")
                        break
                    elif key == ord('s'):
                        # Save screenshots (both original and analysis)
                        original_path = f"fsm_original_{frame_idx}.jpg"
                        analysis_path = f"fsm_analysis_{frame_idx}.jpg"
                        combined_path = f"fsm_combined_{frame_idx}.jpg"
                        cv2.imwrite(original_path, original_frame)
                        cv2.imwrite(analysis_path, analysis_frame)
                        cv2.imwrite(combined_path, combined_display)
                        print(f"Screenshots saved: {original_path}, {analysis_path}, {combined_path}")
                    elif key == ord('f'):
                        # Toggle fullscreen
                        cv2.destroyWindow('Volleyball FSM Demo - Original | Analysis')
                        cv2.namedWindow('Volleyball FSM Demo - Original | Analysis', cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty('Volleyball FSM Demo - Original | Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("Switched to fullscreen (press 'f' again to exit)")
                    elif key == ord('w'):
                        # Toggle windowed mode
                        cv2.destroyWindow('Volleyball FSM Demo - Original | Analysis')
                        cv2.namedWindow('Volleyball FSM Demo - Original | Analysis', cv2.WINDOW_AUTOSIZE)
                        print("Switched to windowed mode")
                        
                # Progress update
                if frame_idx % 30 == 0:  # Every second
                    progress = (frame_idx + 1) / frames_to_process * 100
                    elapsed = time.time() - start_time
                    fps_current = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                    detection_rate = (self.ball_detections / max(1, frame_idx + 1)) * 100
                    
                    yolo_rate = (self.yolo_detections / max(1, self.ball_detections)) * 100 if self.ball_detections > 0 else 0
                    avg_conf = np.mean(self.confidence_history) if self.confidence_history else 0
                    
                    print(f"⚡ Progress: {progress:.1f}% | "
                          f"FPS: {fps_current:.1f} | "
                          f"Ball: {detection_rate:.1f}% | "
                          f"YOLO: {yolo_rate:.1f}% | "
                          f"Conf: {avg_conf:.2f} | "
                          f"State: {self.fsm.current_state.name if self.fsm.current_state is not None else 'Unknown'}")
                          
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
                
        # Final statistics
        total_time = time.time() - start_time
        detection_rate = (self.ball_detections / max(1, self.frame_count)) * 100
        yolo_rate = (self.yolo_detections / max(1, self.ball_detections)) * 100 if self.ball_detections > 0 else 0
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0
        
        stats = {
            'total_frames': self.frame_count,
            'ball_detections': self.ball_detections,
            'detection_rate': detection_rate,
            'yolo_detections': self.yolo_detections,
            'color_detections': self.color_fallback_detections,
            'yolo_success_rate': yolo_rate,
            'avg_confidence': avg_confidence,
            'processing_time': total_time,
            'fps': self.frame_count / total_time if total_time > 0 else 0,
            'final_state': self.fsm.current_state.name if self.fsm.current_state is not None else 'Unknown',
            'total_events': len(self.fsm.events)
        }
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Processed: {stats['total_frames']} frames in {stats['processing_time']:.1f}s")
        print(f"📊 Average FPS: {stats['fps']:.1f}")
        print(f"📊 Ball detections: {stats['ball_detections']} ({stats['detection_rate']:.1f}%)")
        print(f"📊 YOLO detections: {stats['yolo_detections']} ({stats['yolo_success_rate']:.1f}%)")
        print(f"📊 Color fallback: {stats['color_detections']}")
        print(f"📊 Average confidence: {stats['avg_confidence']:.2f}")
        print(f"📊 Final FSM state: {stats['final_state']}")
        print(f"📊 Total FSM events: {stats['total_events']}")
        
        if output_path:
            print(f"📊 Output saved: {output_path}")
            
        return stats


def main():
    """Main demo function"""
    print("🏐 Volleyball FSM Video Demo")
    print("=" * 50)
    
    # Create demo processor
    demo = VideoFSMDemo()
    
    # Example usage
    try:
        # You can replace this with any volleyball video path
        test_videos = [
            "data/test_video.mp4",
            "data/volleyball_sample.mp4", 
            "volleystat/data/pipeline_result/GX010373_processed.avi",
            # Add your video paths here
        ]
        
        input_video = None
        for video_path in test_videos:
            if Path(video_path).exists():
                input_video = video_path
                break
                
        if input_video:
            print(f"🎥 Processing video: {input_video}")
            
            # Process video
            stats = demo.process_video(
                input_path=input_video,
                output_path="results/fsm_testing/fsm_demo_output_upd.mp4",
                start_sec=0,        # Start at 30 seconds
                duration_sec=120,    # Process 2 minutes
                show_preview=True
            )
            
            print(f"\n🎯 Demo completed successfully!")
            
        else:
            print("❌ No test video found. Please provide a video path.")
            print("💡 You can run the demo with: python video_fsm_demo.py <video_path>")
            
            # If video path provided as argument
            if len(sys.argv) > 1:
                video_path = sys.argv[1]
                if Path(video_path).exists():
                    print(f"🎥 Using provided video: {video_path}")
                    stats = demo.process_video(
                        input_path=video_path,
                        show_preview=True
                    )
                else:
                    print(f"❌ Video not found: {video_path}")
                    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 