"""
Demo script for using saved court calibration

This script demonstrates how to use previously saved court calibration
to analyze volleyball videos with accurate coordinate conversion.

Usage:
    python scripts/demo_calibrated_court.py --video video.mp4 --calibration court_calibration.json
"""

import sys
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.interactive_court_calibration import CalibrationData, load_calibration
from src.detection.court_detector import create_volleyball_court_detector


class CalibratedCourtDemo:
    """Demo using saved court calibration"""
    
    def __init__(self, video_path: str, calibration_path: str):
        """
        Initialize demo with saved calibration
        
        Args:
            video_path: Path to video file
            calibration_path: Path to calibration JSON file
        """
        self.video_path = video_path
        self.calibration_path = calibration_path
        
        # Load calibration data
        self.calibration_data = load_calibration(calibration_path)
        if self.calibration_data is None:
            raise ValueError(f"Не вдалося завантажити калібровку: {calibration_path}")
        
        # Initialize court detector
        self.detector = create_volleyball_court_detector()
        
        # Apply loaded calibration
        self._apply_calibration()
        
        # Demo settings
        self.show_grid = True
        self.show_measurements = True
        self.show_ball_tracking = True
        self.debug_mode = False  # Show detection masks and debug info
        
        # Display settings - make window smaller
        self.display_width = 1200  # Maximum display width
        self.scale_factor = 1.0
        
        # Ball tracking (simple color-based) with height estimation
        self.ball_positions = []  # List of (x, y, height, contour_area)
        self.max_ball_history = 30
        
        # Height estimation parameters
        self.reference_ball_size = 2000  # Reference contour area for ground level
        self.net_height = 2.43  # Volleyball net height in meters
        
    def _apply_calibration(self) -> None:
        """Apply loaded calibration to detector"""
        if self.calibration_data.perspective_matrix:
            self.detector.perspective_matrix = np.array(self.calibration_data.perspective_matrix)
            self.detector.is_calibrated = True
            
        self.detector.height_scale_factor = self.calibration_data.height_scale_factor
        print(f"✅ Калібровка застосована: масштаб {self.detector.height_scale_factor:.4f} м/піксель")
    
    def run(self) -> None:
        """Run calibrated court demo"""
        print("🏐 Демо з калібровкою поля")
        print("=" * 50)
        print(f"📹 Відео: {self.video_path}")
        print(f"📐 Калібровка: {self.calibration_path}")
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Не вдалося відкрити відео: {self.video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📊 FPS: {fps}, Кадрів: {total_frames}")
        
        print("\n🎮 Керування:")
        print("  G - перемикач сітки координат")
        print("  M - перемикач вимірювань")
        print("  B - перемикач трекінгу м'яча")
        print("  D - дебаг режим (показ детекції)")
        print("  SPACE - пауза")
        print("  ESC/Q - вихід")
        print("\n" + "=" * 50)
        
        frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Calculate scale factor for first frame
                    if frame_count == 1:
                        self.scale_factor = min(self.display_width / frame.shape[1], 1.0)
                        window_height = int(frame.shape[0] * self.scale_factor)
                        window_width = int(frame.shape[1] * self.scale_factor)
                        cv2.namedWindow('Калібровка поля - Демо', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Калібровка поля - Демо', window_width, window_height)
                    
                    # Process frame
                    processed_frame = self._process_frame(frame)
                    
                    # Resize frame if needed
                    if self.scale_factor < 1.0:
                        new_width = int(processed_frame.shape[1] * self.scale_factor)
                        new_height = int(processed_frame.shape[0] * self.scale_factor)
                        processed_frame = cv2.resize(processed_frame, (new_width, new_height))
                    
                    # Add debug info to scaled frame
                    if self.debug_mode:
                        self._add_debug_info_to_frame(processed_frame)
                
                # Display frame
                cv2.imshow('Калібровка поля - Демо', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - pause
                    paused = not paused
                    print(f"{'⏸️ Пауза' if paused else '▶️ Відтворення'}")
                    
                elif key == ord('g'):  # G - toggle grid
                    self.show_grid = not self.show_grid
                    print(f"Сітка координат: {'ON' if self.show_grid else 'OFF'}")
                    
                elif key == ord('m'):  # M - toggle measurements
                    self.show_measurements = not self.show_measurements
                    print(f"Вимірювання: {'ON' if self.show_measurements else 'OFF'}")
                    
                elif key == ord('b'):  # B - toggle ball tracking
                    self.show_ball_tracking = not self.show_ball_tracking
                    print(f"Трекінг м'яча: {'ON' if self.show_ball_tracking else 'OFF'}")
                    
                elif key == ord('d'):  # D - toggle debug mode
                    self.debug_mode = not self.debug_mode
                    print(f"Дебаг режим: {'ON' if self.debug_mode else 'OFF'}")
                    
                    # Close debug windows when turning off debug mode
                    if not self.debug_mode:
                        cv2.destroyWindow('Debug: Detection Mask')
                        cv2.destroyWindow('Debug: Cleaned Mask') 
                        cv2.destroyWindow('Debug: Contours')
                    
                elif key == 27 or key == ord('q'):  # ESC or Q - quit
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Зупинено користувачем")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Explicitly close debug windows
            try:
                cv2.destroyWindow('Debug: Detection Mask')
                cv2.destroyWindow('Debug: Cleaned Mask') 
                cv2.destroyWindow('Debug: Contours')
            except:
                pass
            
        print(f"\n📊 Оброблено кадрів: {frame_count}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with calibrated court analysis"""
        vis_frame = frame.copy()
        
        # Draw calibrated court elements
        if self.show_grid:
            self._draw_court_grid(vis_frame)
        
        # Draw court boundaries from calibration
        self._draw_calibrated_boundaries(vis_frame)
        
        # Ball detection and tracking
        if self.show_ball_tracking:
            ball_data = self._detect_ball(frame)
            if ball_data:
                self._update_ball_tracking(ball_data)
                self._draw_ball_analysis(vis_frame)
        
        # Show debug windows if enabled (but don't add debug info to frame yet)
        if self.debug_mode:
            self._show_debug_windows(frame)
        
        # Draw measurements and coordinates
        if self.show_measurements:
            self._draw_measurements(vis_frame)
        
        return vis_frame
    
    def _show_debug_windows(self, frame: np.ndarray) -> None:
        """Show debug windows for ball detection analysis"""
        if not hasattr(self, 'debug_mask'):
            return
        
        # Create debug visualization
        debug_frame = frame.copy()
        
        # Draw all detected contours in red
        if hasattr(self, 'debug_contours') and self.debug_contours:
            cv2.drawContours(debug_frame, self.debug_contours, -1, (0, 0, 255), 2)
            
            # Draw contour info
            for i, contour in enumerate(self.debug_contours):
                area = cv2.contourArea(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(debug_frame, f"A:{int(area)}", (cx-20, cy-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw valid contours in green
        if hasattr(self, 'debug_valid_contours') and self.debug_valid_contours:
            for contour, area, cx, cy in self.debug_valid_contours:
                cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), 3)
                cv2.circle(debug_frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(debug_frame, f"VALID: {int(area)}", (cx+10, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate debug window size (1/3 of original frame size)
        h, w = frame.shape[:2]
        debug_size = (w // 3, h // 3)
        
        # Show mask windows
        mask_resized = cv2.resize(self.debug_mask, debug_size)
        mask_cleaned_resized = cv2.resize(self.debug_mask_cleaned, debug_size)
        debug_resized = cv2.resize(debug_frame, debug_size)
        
        cv2.imshow('Debug: Detection Mask', mask_resized)
        cv2.imshow('Debug: Cleaned Mask', mask_cleaned_resized)
        cv2.imshow('Debug: Contours', debug_resized)
    
    def _add_debug_info_to_frame(self, frame: np.ndarray) -> None:
        """Add debug information overlay to main frame"""
        # Use original dimensions - frame will be scaled later
        overlay_width = 400
        overlay_height = 120
        margin = 10
        
        # Create debug overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, frame.shape[0] - overlay_height), 
                     (overlay_width, frame.shape[0] - margin), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Use original font and positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 15
        text_x = 20
        
        # Debug info starting position
        y_offset = frame.shape[0] - overlay_height + 20
        color = (0, 255, 255)
        
        # Detection counts
        total_contours = len(self.debug_contours) if hasattr(self, 'debug_contours') else 0
        valid_contours = len(self.debug_valid_contours) if hasattr(self, 'debug_valid_contours') else 0
        
        cv2.putText(frame, f"DEBUG MODE - Ball Detection", (text_x, y_offset), font, font_scale, color, 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Total contours found: {total_contours}", (text_x, y_offset), font, font_scale, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Valid ball candidates: {valid_contours}", (text_x, y_offset), font, font_scale, (0, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Ball tracking: {len(self.ball_positions)} positions", (text_x, y_offset), font, font_scale, (255, 255, 255), 1)
        
        # Show current detection details
        if valid_contours > 0:
            largest_contour, area, cx, cy = self.debug_valid_contours[0]
            y_offset += line_height
            # Use original coordinates - they will be scaled with the frame
            cv2.putText(frame, f"Selected: pos=({cx},{cy}) area={int(area)}", 
                       (text_x, y_offset), font, font_scale, (0, 255, 0), 1)
    
    def _draw_court_grid(self, frame: np.ndarray) -> None:
        """Draw coordinate grid using calibration"""
        if self.detector.perspective_matrix is None:
            return
        
        try:
            H = self.detector.perspective_matrix
            
            # Draw grid lines every meter
            grid_color = (100, 100, 100)
            line_thickness = 1
            
            # Horizontal lines (parallel to net) - FULL COURT
            for y in range(-9, 10):  # -9 to +9 meters (full 18m court)
                points = []
                for x in np.linspace(-4.5, 4.5, 20):
                    world_point = np.array([[x, y]], dtype=np.float32)
                    screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
                    # Use original coordinates - frame will be scaled later
                    points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
                
                # Draw line segments
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], grid_color, line_thickness)
            
            # Vertical lines (parallel to sidelines) - FULL COURT
            for x in np.linspace(-4.5, 4.5, 10):  # Across court width
                points = []
                for y in range(-9, 10):  # Full court length
                    world_point = np.array([[x, y]], dtype=np.float32)
                    screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
                    # Use original coordinates - frame will be scaled later
                    points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
                
                # Draw line segments
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i + 1], grid_color, line_thickness)
            
            # Draw special lines
            self._draw_special_court_lines(frame, H)
            
        except Exception as e:
            print(f"Попередження: не вдалося намалювати сітку: {e}")
    
    def _draw_special_court_lines(self, frame: np.ndarray, H: np.ndarray) -> None:
        """Draw special court lines (net, attack lines, etc.) for FULL COURT"""
        # Net line (Y=0)
        net_points = []
        for x in np.linspace(-4.5, 4.5, 20):
            world_point = np.array([[x, 0.0]], dtype=np.float32)
            screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
            # Use original coordinates - frame will be scaled later
            net_points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
        
        for i in range(len(net_points) - 1):
            cv2.line(frame, net_points[i], net_points[i + 1], (0, 255, 255), 4)  # Yellow net
        
        # Attack line Team A side (Y=-3m)
        attack_a_points = []
        for x in np.linspace(-4.5, 4.5, 20):
            world_point = np.array([[x, -3.0]], dtype=np.float32)
            screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
            # Use original coordinates - frame will be scaled later
            attack_a_points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
        
        for i in range(len(attack_a_points) - 1):
            cv2.line(frame, attack_a_points[i], attack_a_points[i + 1], (0, 165, 255), 2)  # Orange attack A
        
        # Attack line Team B side (Y=+3m)
        attack_b_points = []
        for x in np.linspace(-4.5, 4.5, 20):
            world_point = np.array([[x, 3.0]], dtype=np.float32)
            screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
            # Use original coordinates - frame will be scaled later
            attack_b_points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
        
        for i in range(len(attack_b_points) - 1):
            cv2.line(frame, attack_b_points[i], attack_b_points[i + 1], (0, 165, 255), 2)  # Orange attack B
        
        # Center line (X=0) - FULL COURT
        center_points = []
        for y in range(-9, 10):
            world_point = np.array([[0.0, y]], dtype=np.float32)
            screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
            # Use original coordinates - frame will be scaled later
            center_points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
        
        for i in range(len(center_points) - 1):
            cv2.line(frame, center_points[i], center_points[i + 1], (128, 128, 128), 1)  # Gray center
    
    def _draw_calibrated_boundaries(self, frame: np.ndarray) -> None:
        """Draw court boundaries from calibration data"""
        if not self.calibration_data.court_corners:
            return
        
        corners = self.calibration_data.court_corners
        
        # Draw court boundary using original coordinates - frame will be scaled later
        for i in range(len(corners)):
            start = corners[i]
            end = corners[(i + 1) % len(corners)]
            cv2.line(frame, start, end, (255, 255, 255), 3)
        
        # Draw corner points
        for i, corner in enumerate(corners):
            cv2.circle(frame, corner, 8, (0, 255, 0), -1)
            cv2.putText(frame, f"C{i+1}", (corner[0] + 10, corner[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw calibration points
        self._draw_calibration_points(frame)
    
    def _draw_calibration_points(self, frame: np.ndarray) -> None:
        """Draw original calibration points"""
        point_sets = [
            (self.calibration_data.back_line_points, (0, 0, 255), "Back"),
            (self.calibration_data.left_sideline_points, (255, 0, 0), "Left"),
            (self.calibration_data.right_sideline_points, (0, 255, 0), "Right"),
            (self.calibration_data.net_line_points, (0, 255, 255), "Net")
        ]
        
        for points, color, label in point_sets:
            if points:
                for i, point in enumerate(points):
                    # Use original coordinates - frame will be scaled later
                    cv2.circle(frame, point, 4, color, -1)
                    cv2.putText(frame, f"{label[0]}{i+1}", 
                               (point[0] + 8, point[1] - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Ball detection with height estimation and improved filtering"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # More restrictive white volleyball (reduce false positives)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # More restrictive yellow volleyball 
        lower_yellow = np.array([22, 150, 150])
        upper_yellow = np.array([28, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        
        # Store debug mask for visualization
        self.debug_mask = mask.copy()
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Store cleaned mask
        self.debug_mask_cleaned = mask.copy()
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store debug info
        self.debug_contours = contours
        self.debug_valid_contours = []
        
        if contours:
            # Filter contours by size and shape
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Increased minimum area to reduce false positives
                if area > 300:  # Increased from 100
                    # Check circularity (ball should be roughly circular)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Only accept roughly circular objects (0.3 to 1.2 range)
                        if 0.3 < circularity < 1.2:
                            # Check aspect ratio of bounding rectangle
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h
                            
                            # Ball should have reasonable aspect ratio (0.5 to 2.0)
                            if 0.5 < aspect_ratio < 2.0:
                                # Check if position is reasonable (not in extreme corners)
                                frame_h, frame_w = frame.shape[:2]
                                cx = x + w // 2
                                cy = y + h // 2
                                
                                # Exclude detections too close to edges (likely false positives)
                                margin = min(frame_w, frame_h) * 0.05  # 5% margin from edges
                                if (margin < cx < frame_w - margin and 
                                    margin < cy < frame_h - margin):
                                    valid_contours.append((contour, area, cx, cy))
            
            # Store for debug
            self.debug_valid_contours = valid_contours
            
            if valid_contours:
                # Sort by area and take the largest valid contour
                valid_contours.sort(key=lambda x: x[1], reverse=True)
                largest_contour, contour_area, cx, cy = valid_contours[0]
                
                # Estimate height based on contour size
                height = self._estimate_ball_height(contour_area, cy)
                
                return (cx, cy, height)
        
        return None
    
    def _estimate_ball_height(self, contour_area: float, screen_y: int) -> float:
        """Estimate ball height based on contour size and screen position"""
        # Basic height estimation using contour size
        # Smaller contour generally means higher ball
        
        # Scale factor based on contour size relative to reference
        size_ratio = self.reference_ball_size / max(contour_area, 100)
        
        # Base height estimation (0.1m to 5m range)
        # Higher ratio = smaller contour = higher ball
        estimated_height = min(5.0, max(0.1, size_ratio * 0.8))
        
        # Adjust based on screen position (lower in screen = closer to camera = potentially higher)
        frame_height = 2160 if hasattr(self, 'calibration_data') else 1080
        y_factor = 1.0 - (screen_y / frame_height)  # 0 to 1, higher values for top of screen
        
        # Balls in upper part of screen are more likely to be high
        height_boost = y_factor * 1.5
        
        final_height = estimated_height + height_boost
        
        # Reasonable bounds for volleyball
        return max(0.1, min(6.0, final_height))
    
    def _update_ball_tracking(self, ball_data: Tuple[int, int, float]) -> None:
        """Update ball position history with height data"""
        self.ball_positions.append(ball_data)
        
        # Keep only recent positions
        if len(self.ball_positions) > self.max_ball_history:
            self.ball_positions.pop(0)
    
    def _draw_ball_analysis(self, frame: np.ndarray) -> None:
        """Draw ball analysis with coordinate conversion and height display"""
        if not self.ball_positions:
            return
        
        current_data = self.ball_positions[-1]  # (x, y, height)
        current_pos = (current_data[0], current_data[1])
        current_height = current_data[2]
        
        # Use original positions - frame will be scaled later
        ball_positions = [(data[0], data[1]) for data in self.ball_positions]
        
        # Draw ball (adjust size based on height - higher ball = smaller circle)
        height_factor = max(0.5, 1.0 - (current_height / 6.0))  # Scale based on height
        ball_radius = max(4, int(12 * height_factor))
        ball_border = max(6, int(16 * height_factor))
        
        # Color based on height - higher = more red, lower = more yellow
        if current_height > 3.0:
            ball_color = (0, 100, 255)  # More red for high balls
        elif current_height > 1.5:
            ball_color = (0, 200, 255)  # Orange for medium height
        else:
            ball_color = (0, 255, 255)  # Yellow for low balls
        
        cv2.circle(frame, current_pos, ball_radius, ball_color, -1)
        cv2.circle(frame, current_pos, ball_border, (255, 255, 255), 2)
        
        # Draw trajectory
        if len(ball_positions) > 1:
            for i in range(1, len(ball_positions)):
                alpha = i / len(ball_positions)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, ball_positions[i-1], ball_positions[i],
                        (0, 255, 255), thickness)
        
        # Convert to world coordinates and display
        world_pos = self._screen_to_world(current_pos)  # Use original coordinates
        if world_pos:
            # Font settings - use original scale
            font_scale = 0.6
            text_offset_x = 20
            text_offset_y = 20
            
            # Position text (English)
            position_text = f"Pos: ({world_pos[0]:.1f}m, {world_pos[1]:.1f}m)"
            cv2.putText(frame, position_text, 
                       (current_pos[0] + text_offset_x, current_pos[1] - text_offset_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
            
            # Height text (English) - Most prominent
            height_text = f"Height: {current_height:.1f}m"
            cv2.putText(frame, height_text, 
                       (current_pos[0] + text_offset_x, current_pos[1] - text_offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 255, 100), 2)
            
            # Court side and height category
            side = "Team A" if world_pos[1] < 0 else "Team B"
            height_category = self._get_height_category(current_height)
            
            side_text = f"{side} - {height_category}"
            cv2.putText(frame, side_text, 
                       (current_pos[0] + text_offset_x, current_pos[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), 1)
    
    def _get_height_category(self, height: float) -> str:
        """Get height category in English"""
        if height < 0.5:
            return "Ground"
        elif height < 1.0:
            return "Low"
        elif height < 2.0:
            return "Medium"
        elif height < 3.0:
            return "High"
        elif height < 4.0:
            return "Very High"
        else:
            return "Spike/Serve"
    
    def _screen_to_world(self, screen_pos: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """Convert screen coordinates to world coordinates"""
        if self.detector.perspective_matrix is None:
            return None
        
        try:
            # Invert perspective matrix
            inv_matrix = cv2.invert(self.detector.perspective_matrix)[1]
            
            # Convert point
            screen_point = np.array([[screen_pos]], dtype=np.float32)
            world_point = cv2.perspectiveTransform(screen_point, inv_matrix)
            
            x, y = world_point[0][0]
            return (float(x), float(y))
            
        except Exception:
            return None
    
    def _draw_measurements(self, frame: np.ndarray) -> None:
        """Draw measurements and coordinate information"""
        # Use original dimensions - frame will be scaled later
        overlay_width = 350
        overlay_height = 150
        margin = 10
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, margin), (overlay_width, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Use original font and positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 20
        text_x = 20
        
        # Calibration info starting position
        y_offset = margin + 25
        color = (255, 255, 255)
        
        cv2.putText(frame, "Калібровка поля:", (text_x, y_offset), font, font_scale, color, 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Масштаб: {self.detector.height_scale_factor:.4f} м/піксель", 
                   (text_x, y_offset), font, font_scale, color, 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Розмір кадру: {self.calibration_data.frame_width}x{self.calibration_data.frame_height}", 
                   (text_x, y_offset), font, font_scale, color, 1)
        y_offset += line_height
        
        # Court dimensions
        cv2.putText(frame, "Розміри поля:", (text_x, y_offset), font, font_scale, color, 1)
        y_offset += line_height
        
        cv2.putText(frame, "18m × 9m (повне поле)", (text_x, y_offset), font, font_scale, color, 1)
        y_offset += line_height
        
        # Ball info with height (English)
        if self.ball_positions:
            ball_data = self.ball_positions[-1]  # (x, y, height)
            world_pos = self._screen_to_world((ball_data[0], ball_data[1]))  # Use original coordinates
            if world_pos:
                small_line = 15
                cv2.putText(frame, f"Ball: ({world_pos[0]:.1f}, {world_pos[1]:.1f})m", 
                           (text_x, y_offset), font, font_scale, (0, 255, 255), 1)
                y_offset += small_line
                cv2.putText(frame, f"Height: {ball_data[2]:.1f}m - {self._get_height_category(ball_data[2])}", 
                           (text_x, y_offset), font, font_scale, (0, 255, 100), 1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Демо з калібровкою поля')
    parser.add_argument('--video', '-v', required=True,
                       help='Шлях до відеофайлу')
    parser.add_argument('--calibration', '-c', required=True,
                       help='Шлях до файлу калібровки (.json)')
    
    args = parser.parse_args()
    
    # Validate files
    if not Path(args.video).exists():
        print(f"❌ Відеофайл не знайдено: {args.video}")
        return
    
    if not Path(args.calibration).exists():
        print(f"❌ Файл калібровки не знайдено: {args.calibration}")
        return
    
    try:
        # Create and run demo
        demo = CalibratedCourtDemo(args.video, args.calibration)
        demo.run()
        
    except Exception as e:
        print(f"❌ Помилка: {e}")


if __name__ == "__main__":
    main() 