import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class NetReference:
    """Net reference points for calibration"""
    # Net parameters (volleyball standard)
    net_height: float = 2.43  # meters
    net_width: float = 9.5    # meters
    post_height: float = 2.55 # meters
    
    # Detected net points in image
    top_left: Optional[Tuple[int, int]] = None
    top_right: Optional[Tuple[int, int]] = None
    bottom_left: Optional[Tuple[int, int]] = None  
    bottom_right: Optional[Tuple[int, int]] = None
    
    # Net line in image (for easier detection)
    net_line_top: Optional[List[Tuple[int, int]]] = None
    net_line_bottom: Optional[List[Tuple[int, int]]] = None


@dataclass
class CourtGeometry:
    """Real volleyball court geometry"""
    length: float = 18.0     # meters
    width: float = 9.0       # meters
    attack_line_distance: float = 3.0  # from net
    net_height: float = 2.43 # meters
    
    # Camera position (behind team)
    camera_height: float = 2.0  # estimated camera height
    camera_distance: float = 10.0  # estimated distance from court


class CameraCalibrator:
    """Camera calibration using volleyball net as reference"""
    
    def __init__(self, court_geometry: CourtGeometry = None):
        self.court_geometry = court_geometry or CourtGeometry()
        self.net_reference = NetReference()
        
        # Calibration state
        self.is_calibrated = False
        self.homography_matrix = None
        self.height_scale_factor = 1.0
        self.perspective_points = None
        
        # Net detection parameters
        self.net_detection_params = {
            'hough_threshold': 50,
            'min_line_length': 100,
            'max_line_gap': 10,
            'canny_low': 50,
            'canny_high': 150
        }
        
    def detect_net_in_frame(self, frame: np.ndarray) -> bool:
        """
        Detect volleyball net in frame
        
        Args:
            frame: Input video frame
            
        Returns:
            True if net detected successfully
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 
                            self.net_detection_params['canny_low'], 
                            self.net_detection_params['canny_high'])
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges,
                1,  # rho resolution
                np.pi/180,  # theta resolution
                threshold=self.net_detection_params['hough_threshold'],
                minLineLength=self.net_detection_params['min_line_length'],
                maxLineGap=self.net_detection_params['max_line_gap']
            )
            
            if lines is None:
                return False
                
            # Filter for horizontal lines (net top and bottom)
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                
                # Keep nearly horizontal lines (net top/bottom)
                if angle < 15:  # Within 15 degrees of horizontal
                    horizontal_lines.append(((x1, y1), (x2, y2)))
            
            if len(horizontal_lines) < 2:
                return False
                
            # Sort lines by Y coordinate
            horizontal_lines.sort(key=lambda line: (line[0][1] + line[1][1]) / 2)
            
            # Take top and bottom lines as net boundaries
            net_top_line = horizontal_lines[0]
            net_bottom_line = horizontal_lines[-1]
            
            # Store net reference points
            self.net_reference.net_line_top = [net_top_line[0], net_top_line[1]]
            self.net_reference.net_line_bottom = [net_bottom_line[0], net_bottom_line[1]]
            
            # Calculate net corners (approximate)
            frame_height, frame_width = frame.shape[:2]
            
            # Find leftmost and rightmost points
            all_x = [p[0] for line in [net_top_line, net_bottom_line] for p in line]
            left_x = min(all_x)
            right_x = max(all_x)
            
            # Interpolate Y coordinates for corners
            top_y = (net_top_line[0][1] + net_top_line[1][1]) // 2
            bottom_y = (net_bottom_line[0][1] + net_bottom_line[1][1]) // 2
            
            self.net_reference.top_left = (left_x, top_y)
            self.net_reference.top_right = (right_x, top_y)
            self.net_reference.bottom_left = (left_x, bottom_y)
            self.net_reference.bottom_right = (right_x, bottom_y)
            
            return True
            
        except Exception as e:
            print(f"Net detection error: {e}")
            return False
            
    def calibrate_from_net(self, frame: np.ndarray) -> bool:
        """
        Calibrate camera using detected net
        
        Args:
            frame: Frame with visible net
            
        Returns:
            True if calibration successful
        """
        if not self.detect_net_in_frame(frame):
            print("Cannot detect net for calibration")
            return False
            
        try:
            # Calculate height scale factor using net height
            net_pixel_height = abs(self.net_reference.top_left[1] - self.net_reference.bottom_left[1])
            if net_pixel_height > 0:
                self.height_scale_factor = self.net_reference.net_height / net_pixel_height
                print(f"Height scale factor: {self.height_scale_factor:.4f} m/pixel")
            
            # Set up perspective correction points
            # Assume camera sees court from behind one team
            frame_height, frame_width = frame.shape[:2]
            
            # Source points (current perspective)
            src_points = np.float32([
                [0, frame_height],           # Bottom left (close)
                [frame_width, frame_height], # Bottom right (close)
                [0, 0],                     # Top left (far)
                [frame_width, 0]            # Top right (far)
            ])
            
            # Destination points (corrected top-down view)
            # Using court proportions
            court_pixel_width = 400  # pixels for visualization
            court_pixel_length = int(court_pixel_width * self.court_geometry.length / self.court_geometry.width)
            
            dst_points = np.float32([
                [0, court_pixel_length],                    # Bottom left
                [court_pixel_width, court_pixel_length],    # Bottom right  
                [0, 0],                                     # Top left
                [court_pixel_width, 0]                      # Top right
            ])
            
            # Calculate homography matrix
            self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            self.perspective_points = (src_points, dst_points)
            
            self.is_calibrated = True
            print("Camera calibration successful!")
            return True
            
        except Exception as e:
            print(f"Calibration error: {e}")
            return False
            
    def screen_to_world_with_height(self, screen_pos: Tuple[int, int], 
                                   frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """
        Convert screen coordinates to world coordinates including height
        
        Args:
            screen_pos: (x, y) in screen coordinates
            frame_shape: (height, width) of frame
            
        Returns:
            (x, y, z) in world coordinates (meters)
        """
        if not self.is_calibrated:
            # Fallback to simple 2D mapping
            x_world = (screen_pos[0] / frame_shape[1]) * self.court_geometry.length
            y_world = (screen_pos[1] / frame_shape[0]) * self.court_geometry.width
            return (x_world, y_world, 0.0)
            
        try:
            # Use homography for ground position
            point = np.array([[screen_pos[0], screen_pos[1]]], dtype=np.float32)
            point = point.reshape(-1, 1, 2)
            
            ground_pos = cv2.perspectiveTransform(point, self.homography_matrix)
            x_ground, y_ground = ground_pos[0][0]
            
            # Calculate height using net reference
            if self.net_reference.top_left and self.net_reference.bottom_left:
                # Interpolate net height at this X position
                net_left_x = self.net_reference.top_left[0]
                net_right_x = self.net_reference.top_right[0]
                
                if net_left_x <= screen_pos[0] <= net_right_x:
                    # Ball is at net area - use net height as reference
                    net_top_y = self.net_reference.top_left[1]
                    net_bottom_y = self.net_reference.bottom_left[1]
                    
                    # Height above net bottom
                    height_pixels = net_bottom_y - screen_pos[1]
                    height_meters = height_pixels * self.height_scale_factor
                    
                    # Add ground level (net bottom is not at ground)
                    z_world = max(0.0, height_meters)
                else:
                    # Ball is away from net - estimate height
                    # Assume net bottom represents ground level in that area
                    net_ground_y = self.net_reference.bottom_left[1]
                    height_pixels = net_ground_y - screen_pos[1]
                    z_world = max(0.0, height_pixels * self.height_scale_factor)
            else:
                z_world = 0.0
                
            # Convert to real court coordinates
            x_world = (x_ground / 400.0) * self.court_geometry.length
            y_world = (y_ground / (400.0 * self.court_geometry.length / self.court_geometry.width)) * self.court_geometry.width
            
            return (x_world, y_world, z_world)
            
        except Exception as e:
            print(f"Coordinate conversion error: {e}")
            # Fallback
            x_world = (screen_pos[0] / frame_shape[1]) * self.court_geometry.length
            y_world = (screen_pos[1] / frame_shape[0]) * self.court_geometry.width
            return (x_world, y_world, 0.0)
            
    def world_to_screen(self, world_pos: Tuple[float, float, float], 
                       frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert world coordinates back to screen coordinates
        
        Args:
            world_pos: (x, y, z) in world coordinates
            frame_shape: (height, width) of frame
            
        Returns:
            (x, y) in screen coordinates
        """
        x_world, y_world, z_world = world_pos
        
        if not self.is_calibrated:
            # Simple mapping
            x_screen = int((x_world / self.court_geometry.length) * frame_shape[1])
            y_screen = int((y_world / self.court_geometry.width) * frame_shape[0])
            return (x_screen, y_screen)
            
        try:
            # Convert to court pixel coordinates
            x_court = (x_world / self.court_geometry.length) * 400.0
            y_court = (y_world / self.court_geometry.width) * (400.0 * self.court_geometry.length / self.court_geometry.width)
            
            # Apply inverse homography
            point = np.array([[x_court, y_court]], dtype=np.float32)
            point = point.reshape(-1, 1, 2)
            
            screen_pos = cv2.perspectiveTransform(point, np.linalg.inv(self.homography_matrix))
            x_screen, y_screen = screen_pos[0][0]
            
            # Adjust for height
            if z_world > 0 and self.height_scale_factor > 0:
                height_pixels = z_world / self.height_scale_factor
                y_screen -= height_pixels  # Higher objects appear higher on screen
                
            return (int(x_screen), int(y_screen))
            
        except Exception as e:
            print(f"World to screen conversion error: {e}")
            # Fallback
            x_screen = int((x_world / self.court_geometry.length) * frame_shape[1])
            y_screen = int((y_world / self.court_geometry.width) * frame_shape[0])
            return (x_screen, y_screen)
            
    def draw_calibration_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw calibration information on frame"""
        result = frame.copy()
        
        # Draw detected net
        if self.net_reference.net_line_top and self.net_reference.net_line_bottom:
            # Draw net lines
            for line in [self.net_reference.net_line_top, self.net_reference.net_line_bottom]:
                cv2.line(result, line[0], line[1], (0, 255, 0), 2)
                
            # Draw net rectangle
            if all([self.net_reference.top_left, self.net_reference.top_right,
                   self.net_reference.bottom_left, self.net_reference.bottom_right]):
                
                cv2.rectangle(result, 
                            self.net_reference.top_left, 
                            self.net_reference.bottom_right, 
                            (255, 0, 0), 2)
                            
                # Add height annotation
                net_center_x = (self.net_reference.top_left[0] + self.net_reference.top_right[0]) // 2
                net_center_y = (self.net_reference.top_left[1] + self.net_reference.bottom_left[1]) // 2
                
                cv2.putText(result, f"Net: {self.net_reference.net_height}m", 
                           (net_center_x - 50, net_center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw calibration status
        status_text = "CALIBRATED" if self.is_calibrated else "NOT CALIBRATED"
        status_color = (0, 255, 0) if self.is_calibrated else (0, 0, 255)
        
        cv2.putText(result, f"Camera: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if self.is_calibrated:
            cv2.putText(result, f"Height Scale: {self.height_scale_factor:.4f} m/px", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        return result
        
    def get_ball_height_info(self, ball_screen_pos: Tuple[int, int], 
                           frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Get detailed height information for ball position
        
        Args:
            ball_screen_pos: Ball position in screen coordinates
            frame_shape: Frame dimensions
            
        Returns:
            Dictionary with height information
        """
        world_pos = self.screen_to_world_with_height(ball_screen_pos, frame_shape)
        
        info = {
            'height_meters': world_pos[2],
            'height_above_net': world_pos[2] - self.net_reference.net_height,
            'is_above_net': world_pos[2] > self.net_reference.net_height,
            'ground_distance_x': world_pos[0],
            'ground_distance_y': world_pos[1],
            'net_clearance': world_pos[2] - self.net_reference.net_height if world_pos[2] > self.net_reference.net_height else 0
        }
        
        return info


def create_calibrator_for_volleyball() -> CameraCalibrator:
    """Create calibrator with standard volleyball court parameters"""
    court = CourtGeometry(
        length=18.0,    # Standard court length
        width=9.0,      # Standard court width
        net_height=2.43, # Standard net height
        attack_line_distance=3.0
    )
    
    return CameraCalibrator(court)


if __name__ == "__main__":
    # Test calibrator
    calibrator = create_calibrator_for_volleyball()
    print("Camera calibrator created with volleyball court parameters")
    print(f"Court: {calibrator.court_geometry.length}x{calibrator.court_geometry.width}m")
    print(f"Net height: {calibrator.court_geometry.net_height}m") 