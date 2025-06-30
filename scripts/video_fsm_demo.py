import sys
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

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
        # Initialize court bounds (standard volleyball court)
        self.court_bounds = CourtBounds(
            left=0, right=18.0,      # 18m width
            top=0, bottom=9.0,       # 9m height  
            net_y=4.5                # Net at center (Y coordinate)
        )
        
        # Camera calibration for perspective correction
        self.net_height_meters = 2.43  # Standard volleyball net height
        self.net_detected = False
        self.net_top_y = None
        self.net_bottom_y = None
        self.height_scale_factor = 1.0
        self.calibration_frame_count = 0
        
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
        if self.net_detected and self.net_top_y and self.net_bottom_y:
            h, w = frame.shape[:2]
            
            # Draw net lines
            cv2.line(frame, (0, self.net_top_y), (w, self.net_top_y), (0, 255, 0), 2)
            cv2.line(frame, (0, self.net_bottom_y), (w, self.net_bottom_y), (0, 255, 0), 2)
            
            # Draw net rectangle
            cv2.rectangle(frame, (50, self.net_top_y), (w-50, self.net_bottom_y), (255, 0, 0), 2)
            
            # Add height annotations
            net_center_y = (self.net_top_y + self.net_bottom_y) // 2
            cv2.putText(frame, f"Net: {self.net_height_meters}m", 
                       (w//2 - 60, net_center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Height scale info
            cv2.putText(frame, f"Scale: {self.height_scale_factor:.4f} m/px", 
                       (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame with FSM"""
        h, w = frame.shape[:2]
        timestamp = self.frame_count / 30.0  # Assume 30 FPS
        
        # Try to detect and calibrate using net (first few frames)
        if self.frame_count < 100:  # Calibrate in first 100 frames
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
            
            # Get detailed height information
            ball_height_info = {
                'height_meters': self.get_ball_height_meters(int(y)),
                'net_clearance': self.get_net_clearance(int(y)),
                'above_net': self.is_ball_above_net_height(int(y))
            }
            
            # Convert to world coordinates with perspective correction
            if self.net_detected:
                ball_world_3d = self.screen_to_world_with_perspective((x, y), w, h)
                ball_world_pos = (ball_world_3d[0], ball_world_3d[1])  # 2D for FSM
            else:
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
                
            cv2.circle(frame, center, 8, ball_color, -1)
            cv2.circle(frame, center, 12, ball_color, 2)
            
            # Ball info text with detection method
            method_text = f"[{self.current_detection_method}]"
            method_color = (0, 255, 0) if self.current_detection_method == "YOLO" else (0, 255, 255)
            
            if ball_height_info:
                height_text = f"H: {ball_height_info['height_meters']:.1f}m"
                clearance_text = f"Net: {ball_height_info['net_clearance']:.1f}m"
                
                cv2.putText(frame, height_text, 
                           (center[0] + 15, center[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)
                           
                cv2.putText(frame, clearance_text, 
                           (center[0] + 15, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_color, 1)
                           
                # Detection method indicator
                cv2.putText(frame, method_text, 
                           (center[0] + 15, center[1] + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, method_color, 2)
            else:
                cv2.putText(frame, f"Ball: {confidence:.2f}", 
                           (center[0] + 15, center[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)
                           
                # Detection method indicator
                cv2.putText(frame, method_text, 
                           (center[0] + 15, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, method_color, 2)
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
                thickness = max(1, int(3 * alpha))
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
        self._draw_fsm_info(result_frame)
        
        # Draw calibration overlay
        self.draw_calibration_overlay(result_frame)
        
        # Update frame counter
        self.frame_count += 1
        
        return result_frame
        
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
        panel_height = 180
        panel_width = 380
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # FSM state information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        y_offset = 30
        
        # Detection method indicator
        detection_method = "YOLO" if (hasattr(self.ball_detector, 'model') and 
                                    self.ball_detector.model is not None and 
                                    not self.ball_detector.use_color_fallback) else "COLOR"
        
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Game State: {self.fsm.current_state.name}",
            f"Rally Phase: {self.fsm.current_rally_phase.name if self.fsm.current_rally_phase else 'None'}",
            f"Ball Detections: {self.ball_detections} ({detection_rate:.1f}%)",
            f"Detection Method: {detection_method}",
            f"YOLO/Color: {self.yolo_detections}/{self.color_fallback_detections}",
            f"YOLO Success: {yolo_rate:.1f}%",
            f"Avg Confidence: {avg_confidence:.2f}",
            f"Camera: {'CALIBRATED' if self.net_detected else 'NOT CALIBRATED'}",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (15, y_offset + i * 18), 
                       font, font_scale, color, 1)
                       
        # Draw court visualization legend
        legend_y = frame.shape[0] - 100
        cv2.putText(frame, "Legend:", (15, legend_y), font, font_scale, (255, 255, 255), 1)
        cv2.putText(frame, "Green: Ball Below Net | Yellow: Ball Above Net", 
                   (15, legend_y + 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Red Trail: Ball Trajectory | Green Lines: Net", 
                   (15, legend_y + 40), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "H: Height in meters | Net: Clearance over net", 
                   (15, legend_y + 60), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "[YOLO]: Green | [COLOR]: Yellow | Detection Methods", 
                   (15, legend_y + 80), font, 0.4, (255, 255, 255), 1)
                   
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
                    
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write to output if specified
                if out:
                    out.write(processed_frame)
                    
                # Show preview
                if show_preview:
                    # Resize for display
                    display_frame = cv2.resize(processed_frame, (960, 540))
                    cv2.imshow('Volleyball FSM Demo', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Stopping processing (user requested)")
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        screenshot_path = f"fsm_screenshot_{frame_idx}.jpg"
                        cv2.imwrite(screenshot_path, processed_frame)
                        print(f"Screenshot saved: {screenshot_path}")
                        
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
                          f"State: {self.fsm.current_state.name}")
                          
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
            'final_state': self.fsm.current_state.name,
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