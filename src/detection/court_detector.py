import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging
from enum import Enum


class DetectionMethod(Enum):
    """Available detection methods for court elements"""
    HOUGH_LINES = "hough_lines"
    CONTOUR_ANALYSIS = "contour_analysis"
    COLOR_SEGMENTATION = "color_segmentation"
    EDGE_DETECTION = "edge_detection"
    PERSPECTIVE_CALIBRATION = "perspective_calibration"


@dataclass
class CourtBounds:
    """Volleyball court boundaries and reference points"""
    left: float
    right: float
    top: float
    bottom: float
    net_y: float  # Y-coordinate of the net
    net_height: float = 2.43  # Standard volleyball net height in meters
    
    def is_ball_in_bounds(self, x: float, y: float) -> bool:
        """Check if ball position is within court boundaries"""
        return (self.left <= x <= self.right and 
                self.top <= y <= self.bottom)
    
    def get_ball_side(self, y: float) -> str:
        """Determine which side of the court the ball is on"""
        return "team_a" if y < self.net_y else "team_b"


@dataclass
class NetDetection:
    """Net detection result with reference points"""
    detected: bool = False
    top_line: Optional[List[Tuple[int, int]]] = None
    bottom_line: Optional[List[Tuple[int, int]]] = None
    left_antenna: Optional[Tuple[int, int]] = None
    right_antenna: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    height_pixels: int = 0
    

@dataclass
class CourtDetection:
    """Court detection result with field boundaries"""
    detected: bool = False
    field_mask: Optional[np.ndarray] = None
    boundary_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    corners: List[Tuple[int, int]] = None
    perspective_matrix: Optional[np.ndarray] = None
    confidence: float = 0.0


@dataclass
class CourtGeometry:
    """Standard volleyball court geometry parameters"""
    # Official volleyball court dimensions
    length: float = 18.0      # Full court length (meters)
    width: float = 9.0        # Court width (meters)
    half_length: float = 9.0  # Half court length from net to back line
    attack_line_distance: float = 3.0  # Distance from net to attack line
    net_height: float = 2.43  # Net height in meters
    
    # Camera positioning assumptions
    camera_height: float = 2.0    # Estimated camera height
    camera_distance: float = 10.0 # Estimated distance from court


class VolleyballCourtDetector:
    """
    Comprehensive volleyball court and net detection system
    
    This class implements multiple detection methods to identify:
    1. Net position and boundaries (horizontal line at ~2.43m height)
    2. Field boundaries (trapezoidal shape due to perspective)
    3. Court lines (attack lines, sidelines, back line)
    4. Perspective calibration for accurate measurements
    """
    
    def __init__(self, 
                 court_geometry: Optional[CourtGeometry] = None,
                 detection_methods: List[DetectionMethod] = None):
        """
        Initialize court detector
        
        Args:
            court_geometry: Court dimensions and parameters
            detection_methods: List of detection methods to use
        """
        self.court_geometry = court_geometry or CourtGeometry()
        self.detection_methods = detection_methods or [
            DetectionMethod.HOUGH_LINES,
            DetectionMethod.COLOR_SEGMENTATION,
            DetectionMethod.EDGE_DETECTION
        ]
        
        # Detection parameters
        self.net_params = {
            'hough_threshold': 50,
            'min_line_length': 100,
            'max_line_gap': 10,
            'canny_low': 50,
            'canny_high': 150,
            'horizontal_angle_tolerance': 15  # degrees
        }
        
        self.field_params = {
            'color_lower_green': np.array([35, 40, 40]),   # HSV lower bound for green court
            'color_upper_green': np.array([85, 255, 255]), # HSV upper bound for green court
            'morphology_kernel_size': 5,
            'min_contour_area': 10000,
            'approx_epsilon_factor': 0.02
        }
        
        # Calibration state
        self.is_calibrated = False
        self.perspective_matrix: Optional[np.ndarray] = None
        self.height_scale_factor: float = 1.0
        self.court_bounds: Optional[CourtBounds] = None
        
        # Detection history for stability
        self.detection_history: List[Dict[str, Any]] = []
        self.max_history_length = 10
        
        self.logger = logging.getLogger(__name__)
    
    def detect_net(self, frame: np.ndarray) -> NetDetection:
        """
        Detect volleyball net in frame using multiple methods
        
        The net appears as a horizontal line in the upper portion of the frame
        due to camera positioning behind the court. Uses Hough line detection
        to find horizontal lines that match net characteristics.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            NetDetection object with detection results
        """
        detection = NetDetection()
        
        try:
            # Convert to grayscale for line detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection for line finding
            edges = cv2.Canny(
                blurred, 
                self.net_params['canny_low'], 
                self.net_params['canny_high']
            )
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.net_params['hough_threshold'],
                minLineLength=self.net_params['min_line_length'],
                maxLineGap=self.net_params['max_line_gap']
            )
            
            if lines is None:
                return detection
            
            # Filter for horizontal lines (net characteristics)
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                if abs(x2 - x1) > 0:
                    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                else:
                    angle = 90
                
                # Keep nearly horizontal lines with sufficient length
                if (angle < self.net_params['horizontal_angle_tolerance'] and 
                    abs(x2 - x1) > self.net_params['min_line_length']):
                    
                    y_avg = (y1 + y2) // 2
                    x_start = min(x1, x2)
                    x_end = max(x1, x2)
                    horizontal_lines.append({
                        'y': y_avg,
                        'x_start': x_start,
                        'x_end': x_end,
                        'length': abs(x2 - x1),
                        'points': [(x1, y1), (x2, y2)]
                    })
            
            if len(horizontal_lines) >= 1:
                # Sort by Y coordinate and length
                horizontal_lines.sort(key=lambda x: (x['y'], -x['length']))
                
                # Take the most prominent horizontal line as net
                net_line = horizontal_lines[0]
                detection.top_line = net_line['points']
                detection.detected = True
                detection.confidence = min(1.0, net_line['length'] / frame.shape[1])
                
                # Estimate net height in pixels (simple approach)
                detection.height_pixels = int(frame.shape[0] * 0.05)  # ~5% of frame height
                
                # Try to find antennas (vertical lines at net ends)
                detection.left_antenna, detection.right_antenna = self._detect_antennas(
                    edges, net_line, frame.shape
                )
                
                self.logger.info(f"Net detected with confidence: {detection.confidence:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in net detection: {e}")
            
        return detection
    
    def detect_court_field(self, frame: np.ndarray) -> CourtDetection:
        """
        Detect volleyball court field boundaries and create field mask
        
        The court appears as a trapezoidal shape due to perspective projection.
        Uses color segmentation (green court surface) and contour analysis
        to identify the playing field boundaries.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            CourtDetection object with field boundaries and mask
        """
        detection = CourtDetection()
        
        try:
            # Method 1: Color-based court detection
            court_mask = self._detect_court_by_color(frame)
            
            # Method 2: Contour analysis for field boundaries
            boundaries = self._detect_court_boundaries(court_mask, frame.shape)
            
            if boundaries is not None:
                detection.detected = True
                detection.field_mask = court_mask
                detection.boundary_lines = boundaries['lines']
                detection.corners = boundaries['corners']
                detection.confidence = boundaries['confidence']
                
                # Calculate perspective transformation if we have enough points
                if len(detection.corners) >= 4:
                    detection.perspective_matrix = self._calculate_perspective_transform(
                        detection.corners, frame.shape
                    )
                
                self.logger.info(f"Court field detected with confidence: {detection.confidence:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in court field detection: {e}")
            
        return detection
    
    def detect_complete_court(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Perform complete court detection including net and field
        
        This is the main detection method that combines net and field detection
        to provide a comprehensive understanding of the volleyball court layout.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Dictionary with complete detection results
        """
        results = {
            'net': self.detect_net(frame),
            'field': self.detect_court_field(frame),
            'calibrated': False,
            'court_bounds': None,
            'timestamp': cv2.getTickCount() / cv2.getTickFrequency()
        }
        
        # If both net and field are detected, perform calibration
        if results['net'].detected and results['field'].detected:
            calibration_success = self._calibrate_court(
                results['net'], 
                results['field'], 
                frame.shape
            )
            
            if calibration_success:
                results['calibrated'] = True
                results['court_bounds'] = self.court_bounds
                self.is_calibrated = True
        
        # Store in detection history for stability analysis
        self._update_detection_history(results)
        
        return results
    
    def _detect_court_by_color(self, frame: np.ndarray) -> np.ndarray:
        """Detect court surface using color segmentation"""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green court surface
        mask = cv2.inRange(
            hsv, 
            self.field_params['color_lower_green'], 
            self.field_params['color_upper_green']
        )
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.field_params['morphology_kernel_size'], 
             self.field_params['morphology_kernel_size'])
        )
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _detect_court_boundaries(self, mask: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Extract court boundaries from segmentation mask"""
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (assumed to be the court)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < self.field_params['min_contour_area']:
            return None
        
        # Approximate contour to get court corners
        epsilon = self.field_params['approx_epsilon_factor'] * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Extract corner points
        corners = [tuple(point[0]) for point in approx]
        
        # Generate boundary lines from corners
        lines = []
        for i in range(len(corners)):
            start_point = corners[i]
            end_point = corners[(i + 1) % len(corners)]
            lines.append((start_point, end_point))
        
        # Calculate confidence based on contour area and regularity
        area_ratio = cv2.contourArea(largest_contour) / (frame_shape[0] * frame_shape[1])
        confidence = min(1.0, area_ratio * 2)  # Normalize to reasonable range
        
        return {
            'lines': lines,
            'corners': corners,
            'contour': largest_contour,
            'confidence': confidence
        }
    
    def _detect_antennas(self, edges: np.ndarray, net_line: Dict[str, Any], 
                        frame_shape: Tuple[int, int]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Detect antenna positions at net ends"""
        left_antenna = None
        right_antenna = None
        
        try:
            # Search for vertical lines near net ends
            net_y = net_line['y']
            search_radius = 50  # pixels
            
            # Detect vertical lines in antenna regions
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=30,
                minLineLength=50,
                maxLineGap=5
            )
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Check if line is vertical
                    if abs(y2 - y1) > abs(x2 - x1):  # More vertical than horizontal
                        line_x = (x1 + x2) // 2
                        line_y = (y1 + y2) // 2
                        
                        # Check if near net level
                        if abs(line_y - net_y) < search_radius:
                            # Determine if left or right antenna
                            if line_x < frame_shape[1] // 3:  # Left third of frame
                                if left_antenna is None or line_x > left_antenna[0]:
                                    left_antenna = (line_x, line_y)
                            elif line_x > 2 * frame_shape[1] // 3:  # Right third of frame
                                if right_antenna is None or line_x < right_antenna[0]:
                                    right_antenna = (line_x, line_y)
            
        except Exception as e:
            self.logger.warning(f"Antenna detection failed: {e}")
        
        return left_antenna, right_antenna
    
    def _calculate_perspective_transform(self, corners: List[Tuple[int, int]], 
                                       frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Calculate perspective transformation matrix from court corners"""
        if len(corners) < 4:
            return None
        
        try:
            # Sort corners to match expected order: top-left, top-right, bottom-right, bottom-left
            corners_sorted = self._sort_corners(corners)
            
            # Define real-world court coordinates (in meters)
            # Standard volleyball court: 9m width × 9m depth (half court from net)
            real_world_points = np.array([
                [-4.5, 0.0],    # Net-left
                [4.5, 0.0],     # Net-right  
                [4.5, 9.0],     # Back-right
                [-4.5, 9.0]     # Back-left
            ], dtype=np.float32)
            
            # Image corner points
            image_points = np.array(corners_sorted, dtype=np.float32)
            
            # Calculate homography matrix
            homography, mask = cv2.findHomography(real_world_points, image_points)
            
            return homography
            
        except Exception as e:
            self.logger.error(f"Perspective transform calculation failed: {e}")
            return None
    
    def _sort_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Sort corners in consistent order: top-left, top-right, bottom-right, bottom-left"""
        if len(corners) != 4:
            return corners
        
        # Convert to numpy array for easier manipulation
        points = np.array(corners)
        
        # Sort by y-coordinate to get top and bottom pairs
        sorted_by_y = points[np.argsort(points[:, 1])]
        
        # Top two points
        top_points = sorted_by_y[:2]
        # Bottom two points  
        bottom_points = sorted_by_y[2:]
        
        # Sort top points by x-coordinate (left, right)
        top_sorted = top_points[np.argsort(top_points[:, 0])]
        # Sort bottom points by x-coordinate (left, right)
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        
        # Return in order: top-left, top-right, bottom-right, bottom-left
        return [
            tuple(top_sorted[0]),      # top-left
            tuple(top_sorted[1]),      # top-right
            tuple(bottom_sorted[1]),   # bottom-right
            tuple(bottom_sorted[0])    # bottom-left
        ]
    
    def _calibrate_court(self, net_detection: NetDetection, 
                        field_detection: CourtDetection, 
                        frame_shape: Tuple[int, int]) -> bool:
        """Calibrate court coordinate system using detected net and field"""
        try:
            if not (net_detection.detected and field_detection.detected):
                return False
            
            # Calculate height scale factor using net
            if net_detection.height_pixels > 0:
                self.height_scale_factor = self.court_geometry.net_height / net_detection.height_pixels
            
            # Set up court bounds based on detections
            if field_detection.corners and len(field_detection.corners) >= 4:
                corners = field_detection.corners
                
                # Calculate court bounds in image coordinates
                x_coords = [c[0] for c in corners]
                y_coords = [c[1] for c in corners]
                
                # Estimate net position (typically in upper portion of field)
                net_y_estimate = min(y_coords) + (max(y_coords) - min(y_coords)) * 0.2
                
                self.court_bounds = CourtBounds(
                    left=min(x_coords),
                    right=max(x_coords),
                    top=min(y_coords),
                    bottom=max(y_coords),
                    net_y=net_y_estimate,
                    net_height=self.court_geometry.net_height
                )
                
                # Store perspective matrix
                self.perspective_matrix = field_detection.perspective_matrix
                
                return True
                
        except Exception as e:
            self.logger.error(f"Court calibration failed: {e}")
            
        return False
    
    def _update_detection_history(self, results: Dict[str, Any]) -> None:
        """Update detection history for stability analysis"""
        self.detection_history.append({
            'timestamp': results['timestamp'],
            'net_detected': results['net'].detected,
            'field_detected': results['field'].detected,
            'calibrated': results['calibrated'],
            'net_confidence': results['net'].confidence,
            'field_confidence': results['field'].confidence
        })
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history_length:
            self.detection_history.pop(0)
    
    def get_detection_stability(self) -> Dict[str, float]:
        """Analyze detection stability over recent frames"""
        if len(self.detection_history) < 3:
            return {'net': 0.0, 'field': 0.0, 'overall': 0.0}
        
        recent_frames = self.detection_history[-5:]  # Last 5 frames
        
        net_detections = sum(1 for frame in recent_frames if frame['net_detected'])
        field_detections = sum(1 for frame in recent_frames if frame['field_detected'])
        
        net_stability = net_detections / len(recent_frames)
        field_stability = field_detections / len(recent_frames)
        overall_stability = (net_stability + field_stability) / 2
        
        return {
            'net': net_stability,
            'field': field_stability,
            'overall': overall_stability
        }
    
    def world_to_screen(self, world_pos: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to screen coordinates"""
        if self.perspective_matrix is None:
            return None
        
        try:
            # Convert world point to homogeneous coordinates
            world_point = np.array([[world_pos]], dtype=np.float32)
            
            # Apply perspective transformation
            screen_point = cv2.perspectiveTransform(world_point, self.perspective_matrix)
            
            x, y = screen_point[0][0]
            return (int(x), int(y))
            
        except Exception as e:
            self.logger.error(f"World to screen conversion failed: {e}")
            return None
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Convert screen coordinates to world coordinates"""
        if self.perspective_matrix is None:
            return None
        
        try:
            # Invert perspective matrix
            inv_matrix = cv2.invert(self.perspective_matrix)[1]
            
            # Convert screen point to homogeneous coordinates
            screen_point = np.array([[screen_pos]], dtype=np.float32)
            
            # Apply inverse transformation
            world_point = cv2.perspectiveTransform(screen_point, inv_matrix)
            
            x, y = world_point[0][0]
            return (float(x), float(y))
            
        except Exception as e:
            self.logger.error(f"Screen to world conversion failed: {e}")
            return None


def create_volleyball_court_detector() -> VolleyballCourtDetector:
    """Factory function to create volleyball court detector with standard parameters"""
    court_geometry = CourtGeometry(
        length=18.0,
        width=9.0,
        half_length=9.0,
        attack_line_distance=3.0,
        net_height=2.43
    )
    
    detection_methods = [
        DetectionMethod.HOUGH_LINES,
        DetectionMethod.COLOR_SEGMENTATION,
        DetectionMethod.EDGE_DETECTION
    ]
    
    return VolleyballCourtDetector(court_geometry, detection_methods) 