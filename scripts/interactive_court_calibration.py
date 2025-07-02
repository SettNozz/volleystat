"""
Interactive Volleyball Court Calibration Tool

This tool allows manual calibration of volleyball court boundaries by clicking points with the mouse.
The user can mark:
- 2 points on the back line
- 2 points on the left sideline  
- 2 points on the right sideline
- Points on the net top line

Usage:
    python scripts/interactive_court_calibration.py --input video_file.mp4
    python scripts/interactive_court_calibration.py --input 0  # Use webcam
"""

import sys
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.detection.court_detector import CourtGeometry


class CalibrationStage(Enum):
    """Calibration stages for full court"""
    BACK_LINE = "back_line"
    LEFT_SIDELINE = "left_sideline"  
    RIGHT_SIDELINE = "right_sideline"
    NET_LINE = "net_line"
    FRONT_LINE = "front_line"
    COMPLETED = "completed"


@dataclass
class CalibrationData:
    """Store calibration points and results for full court"""
    back_line_points: List[Tuple[int, int]] = None
    left_sideline_points: List[Tuple[int, int]] = None
    right_sideline_points: List[Tuple[int, int]] = None
    net_line_points: List[Tuple[int, int]] = None
    front_line_points: List[Tuple[int, int]] = None
    perspective_matrix: Optional[List[List[float]]] = None
    court_corners: Optional[List[Tuple[int, int]]] = None
    height_scale_factor: float = 1.0
    frame_width: int = 0
    frame_height: int = 0
    
    def __post_init__(self):
        # Initialize empty lists if None
        if self.back_line_points is None:
            self.back_line_points = []
        if self.left_sideline_points is None:
            self.left_sideline_points = []
        if self.right_sideline_points is None:
            self.right_sideline_points = []
        if self.net_line_points is None:
            self.net_line_points = []
        if self.front_line_points is None:
            self.front_line_points = []


class InteractiveCourtCalibration:
    """Interactive tool for volleyball court calibration using mouse clicks"""
    
    def __init__(self, input_source: str, output_calibration: Optional[str] = None):
        """
        Initialize interactive calibration tool
        
        Args:
            input_source: Video file path or camera index
            output_calibration: Path to save calibration data
        """
        self.input_source = input_source
        self.output_calibration = output_calibration or "court_calibration.json"
        
        # Calibration state
        self.current_stage = CalibrationStage.BACK_LINE
        self.calibration_data = CalibrationData()
        self.current_frame = None
        self.display_frame = None
        
        # Court geometry
        self.court_geometry = CourtGeometry()
        
        # Stage configuration for full court
        self.stage_config = {
            CalibrationStage.BACK_LINE: {
                'name': 'Back court line',
                'description': 'Click 2 points on back court line (left to right)',
                'required_points': 2,
                'color': (0, 0, 255),  # Red
                'points_attr': 'back_line_points'
            },
            CalibrationStage.LEFT_SIDELINE: {
                'name': 'Left sideline',
                'description': 'Click 2 points on left sideline (full length from back to front)',
                'required_points': 2,
                'color': (255, 0, 0),  # Blue
                'points_attr': 'left_sideline_points'
            },
            CalibrationStage.RIGHT_SIDELINE: {
                'name': 'Right sideline',
                'description': 'Click 2 points on right sideline (full length from back to front)',
                'required_points': 2,
                'color': (0, 255, 0),  # Green
                'points_attr': 'right_sideline_points'
            },
            CalibrationStage.NET_LINE: {
                'name': 'Net line',
                'description': 'Click points along the net line (minimum 2)',
                'required_points': 2,
                'color': (0, 255, 255),  # Yellow
                'points_attr': 'net_line_points'
            },
            CalibrationStage.FRONT_LINE: {
                'name': 'Front court line',
                'description': 'Click 2 points on front court line (left to right)',
                'required_points': 2,
                'color': (255, 0, 255),  # Magenta
                'points_attr': 'front_line_points'
            }
        }
        
        # Mouse callback state
        self.mouse_callback_active = False
        
    def run(self) -> None:
        """Run interactive calibration"""
        print("🏐 Інтерактивна калібровка волейбольного поля")
        print("=" * 60)
        
        # Open video source
        cap = self._open_video_source()
        if cap is None:
            return
        
        # Get first frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Помилка: не вдалося прочитати кадр з відео")
            return
        
        self.current_frame = frame.copy()
        self.calibration_data.frame_width = frame.shape[1]
        self.calibration_data.frame_height = frame.shape[0]
        
        print(f"📹 Розмір кадру: {frame.shape[1]}x{frame.shape[0]}")
        print(f"📁 Джерело: {self.input_source}")
        print(f"💾 Збереження: {self.output_calibration}")
        
        # Setup window and mouse callback
        cv2.namedWindow('Калібровка поля', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Калібровка поля', 1200, 800)
        cv2.setMouseCallback('Калібровка поля', self._mouse_callback)
        self.mouse_callback_active = True
        
        print("\n🎮 Керування:")
        print("  Ліва кнопка миші - додати точку")
        print("  U - скасувати останню точку")
        print("  ENTER - перейти до наступного етапу")
        print("  S - зберегти калібровку")
        print("  ESC/Q - вийти")
        print("\n" + "=" * 60)
        
        # Start calibration process
        self._show_current_stage_info()
        
        try:
            while True:
                # Update display
                self._update_display()
                cv2.imshow('Калібровка поля', self.display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 13:  # ENTER - next stage
                    if self._can_proceed_to_next_stage():
                        self._proceed_to_next_stage()
                    else:
                        print(f"⚠️  Потрібно відмітити мінімум {self._get_required_points()} точок")
                        
                elif key == ord('u'):  # U - undo last point
                    self._undo_last_point()
                    
                elif key == ord('s'):  # S - save calibration
                    self._save_calibration()
                    
                elif key == 27 or key == ord('q'):  # ESC or Q - quit
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Зупинено користувачем")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        # Final calibration if completed
        if self.current_stage == CalibrationStage.COMPLETED:
            print("\n✅ Калібровка завершена!")
            self._calculate_final_calibration()
            self._save_calibration()
    
    def _open_video_source(self) -> Optional[cv2.VideoCapture]:
        """Open video source"""
        try:
            if self.input_source.isdigit():
                cap = cv2.VideoCapture(int(self.input_source))
            else:
                cap = cv2.VideoCapture(self.input_source)
            
            if not cap.isOpened():
                print(f"❌ Помилка: не вдалося відкрити відео: {self.input_source}")
                return None
                
            return cap
            
        except Exception as e:
            print(f"❌ Помилка відкриття відео: {e}")
            return None
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse events"""
        if not self.mouse_callback_active:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_stage != CalibrationStage.COMPLETED:
                self._add_point((x, y))
    
    def _add_point(self, point: Tuple[int, int]) -> None:
        """Add calibration point"""
        stage_config = self.stage_config[self.current_stage]
        points_attr = stage_config['points_attr']
        current_points = getattr(self.calibration_data, points_attr)
        
        current_points.append(point)
        
        print(f"➕ Додано точку {len(current_points)}/{stage_config['required_points']}: {point}")
        
        # Check if stage is complete
        if len(current_points) >= stage_config['required_points']:
            print(f"✅ Етап '{stage_config['name']}' завершено!")
            print("   Натисніть ENTER для переходу до наступного етапу")
    
    def _undo_last_point(self) -> None:
        """Remove last added point"""
        if self.current_stage == CalibrationStage.COMPLETED:
            return
            
        stage_config = self.stage_config[self.current_stage]
        points_attr = stage_config['points_attr']
        current_points = getattr(self.calibration_data, points_attr)
        
        if current_points:
            removed_point = current_points.pop()
            print(f"↩️  Скасовано точку: {removed_point}")
        else:
            print("⚠️  Немає точок для скасування")
    
    def _can_proceed_to_next_stage(self) -> bool:
        """Check if can proceed to next stage"""
        if self.current_stage == CalibrationStage.COMPLETED:
            return False
            
        stage_config = self.stage_config[self.current_stage]
        points_attr = stage_config['points_attr']
        current_points = getattr(self.calibration_data, points_attr)
        
        return len(current_points) >= stage_config['required_points']
    
    def _get_required_points(self) -> int:
        """Get required points for current stage"""
        if self.current_stage == CalibrationStage.COMPLETED:
            return 0
        return self.stage_config[self.current_stage]['required_points']
    
    def _proceed_to_next_stage(self) -> None:
        """Move to next calibration stage"""
        # Define stage progression for full court
        stage_order = [
            CalibrationStage.BACK_LINE,
            CalibrationStage.LEFT_SIDELINE,
            CalibrationStage.RIGHT_SIDELINE,
            CalibrationStage.NET_LINE,
            CalibrationStage.FRONT_LINE,
            CalibrationStage.COMPLETED
        ]
        
        current_index = stage_order.index(self.current_stage)
        if current_index < len(stage_order) - 1:
            self.current_stage = stage_order[current_index + 1]
            self._show_current_stage_info()
        
        if self.current_stage == CalibrationStage.COMPLETED:
            print("\n🎉 Всі етапи калібровки завершено!")
            print("   Розраховую перспективну трансформацію...")
    
    def _show_current_stage_info(self) -> None:
        """Show information about current calibration stage"""
        if self.current_stage == CalibrationStage.COMPLETED:
            return
            
        stage_config = self.stage_config[self.current_stage]
        print(f"\n📍 Етап: {stage_config['name']}")
        print(f"   {stage_config['description']}")
        print(f"   Потрібно точок: {stage_config['required_points']}")
    
    def _update_display(self) -> None:
        """Update display frame with current calibration state"""
        self.display_frame = self.current_frame.copy()
        
        # Draw all calibration points
        self._draw_calibration_points()
        
        # Draw current stage info
        self._draw_stage_info()
        
        # Draw lines if enough points
        self._draw_calibration_lines()
        
        # Draw coordinate grid if calibrated
        if self.current_stage == CalibrationStage.COMPLETED:
            self._draw_coordinate_grid()
    
    def _draw_calibration_points(self) -> None:
        """Draw all calibration points for full court"""
        for stage in [CalibrationStage.BACK_LINE, CalibrationStage.LEFT_SIDELINE, 
                     CalibrationStage.RIGHT_SIDELINE, CalibrationStage.NET_LINE,
                     CalibrationStage.FRONT_LINE]:
            
            if stage not in self.stage_config:
                continue
                
            stage_config = self.stage_config[stage]
            points_attr = stage_config['points_attr']
            points = getattr(self.calibration_data, points_attr)
            color = stage_config['color']
            
            # Draw points
            for i, point in enumerate(points):
                # Draw point
                cv2.circle(self.display_frame, point, 8, color, -1)
                cv2.circle(self.display_frame, point, 12, (255, 255, 255), 2)
                
                # Draw point number
                cv2.putText(self.display_frame, f"{i+1}", 
                          (point[0] + 15, point[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_calibration_lines(self) -> None:
        """Draw lines connecting calibration points"""
        # Back line (red)
        if len(self.calibration_data.back_line_points) >= 2:
            cv2.line(self.display_frame, 
                    self.calibration_data.back_line_points[0],
                    self.calibration_data.back_line_points[1],
                    (0, 0, 255), 3)
        
        # Left sideline (blue)
        if len(self.calibration_data.left_sideline_points) >= 2:
            cv2.line(self.display_frame,
                    self.calibration_data.left_sideline_points[0],
                    self.calibration_data.left_sideline_points[1],
                    (255, 0, 0), 3)
        
        # Right sideline (green)
        if len(self.calibration_data.right_sideline_points) >= 2:
            cv2.line(self.display_frame,
                    self.calibration_data.right_sideline_points[0],
                    self.calibration_data.right_sideline_points[1],
                    (0, 255, 0), 3)
        
        # Net line (yellow)
        if len(self.calibration_data.net_line_points) >= 2:
            for i in range(len(self.calibration_data.net_line_points) - 1):
                cv2.line(self.display_frame,
                        self.calibration_data.net_line_points[i],
                        self.calibration_data.net_line_points[i + 1],
                        (0, 255, 255), 4)
        
        # Front line (magenta)
        if len(self.calibration_data.front_line_points) >= 2:
            cv2.line(self.display_frame, 
                    self.calibration_data.front_line_points[0],
                    self.calibration_data.front_line_points[1],
                    (255, 0, 255), 3)
        
        # Draw complete court trapezoid if corners are calculated
        if self.calibration_data.court_corners and len(self.calibration_data.court_corners) == 4:
            self._draw_court_trapezoid()
    
    def _draw_court_trapezoid(self) -> None:
        """Draw complete full court trapezoid connecting all 4 corners"""
        corners = self.calibration_data.court_corners
        
        # Full court corners order: front-left, front-right, back-right, back-left
        front_left = corners[0]
        front_right = corners[1] 
        back_right = corners[2]
        back_left = corners[3]
        
        # Draw trapezoid sides with thick lines for visibility
        line_thickness = 4
        
        # Front line (closest to camera) - magenta
        cv2.line(self.display_frame, front_left, front_right, (255, 0, 255), line_thickness)
        
        # Back line (farthest from camera) - red  
        cv2.line(self.display_frame, back_left, back_right, (0, 0, 255), line_thickness)
        
        # Left sideline - blue
        cv2.line(self.display_frame, front_left, back_left, (255, 0, 0), line_thickness)
        
        # Right sideline - green
        cv2.line(self.display_frame, front_right, back_right, (0, 255, 0), line_thickness)
        
        # Draw net line across the center if net points exist
        if len(self.calibration_data.net_line_points) >= 2:
            # Draw net line in yellow
            for i in range(len(self.calibration_data.net_line_points) - 1):
                cv2.line(self.display_frame,
                        self.calibration_data.net_line_points[i],
                        self.calibration_data.net_line_points[i + 1],
                        (0, 255, 255), line_thickness)
        
        # Draw small circles at corners for clarity
        corner_color = (255, 255, 255)
        for corner in corners:
            cv2.circle(self.display_frame, corner, 6, corner_color, -1)
            cv2.circle(self.display_frame, corner, 8, (0, 0, 0), 2)  # Black outline
    
    def _draw_stage_info(self) -> None:
        """Draw current stage information on display"""
        if self.current_stage == CalibrationStage.COMPLETED:
            # Draw completion message
            cv2.putText(self.display_frame, "Калібровка завершена!", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            return
        
        stage_config = self.stage_config[self.current_stage]
        points_attr = stage_config['points_attr']
        current_points = getattr(self.calibration_data, points_attr)
        
        # Create semi-transparent overlay
        overlay = self.display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)
        
        # Stage title
        cv2.putText(self.display_frame, f"Етап: {stage_config['name']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Description
        cv2.putText(self.display_frame, stage_config['description'], 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Progress
        progress_text = f"Точок: {len(current_points)}/{stage_config['required_points']}"
        cv2.putText(self.display_frame, progress_text, 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   (0, 255, 0) if len(current_points) >= stage_config['required_points'] else (0, 255, 255), 2)
        
        # Instructions
        if len(current_points) >= stage_config['required_points']:
            cv2.putText(self.display_frame, "Натисніть ENTER для продовження", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def _calculate_final_calibration(self) -> None:
        """Calculate final calibration parameters"""
        try:
            print("🔄 Step 1: Calculating court corners...")
            # Calculate court corners from line intersections
            self._calculate_court_corners()
            
            print("🔄 Step 2: Calculating perspective matrix...")
            # Calculate perspective transformation
            self._calculate_perspective_matrix()
            
            print("🔄 Step 3: Calculating height scale...")
            # Calculate height scale factor from net
            self._calculate_height_scale()
            
            print("✅ Calibration calculated successfully!")
            
        except Exception as e:
            print(f"❌ Error calculating calibration: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_court_corners(self) -> None:
        """Calculate full court corners from line intersections"""
        try:
            print("   Validating and ordering calibration points...")
            # Validate and fix point ordering
            self._validate_point_ordering()
            
            print("   Getting line equations for full court...")
            # Get line equations
            back_line = self._get_line_equation(self.calibration_data.back_line_points)
            front_line = self._get_line_equation(self.calibration_data.front_line_points)
            left_line = self._get_line_equation(self.calibration_data.left_sideline_points)
            right_line = self._get_line_equation(self.calibration_data.right_sideline_points)
            
            print("   Calculating full court intersections...")
            # Calculate all 4 corners of the full court
            back_left = self._line_intersection(back_line, left_line)
            back_right = self._line_intersection(back_line, right_line)
            front_left = self._line_intersection(front_line, left_line)
            front_right = self._line_intersection(front_line, right_line)
            
            # Validate corners are within reasonable bounds
            frame_w, frame_h = self.calibration_data.frame_width, self.calibration_data.frame_height
            corners = [front_left, front_right, back_right, back_left]
            
            for i, (x, y) in enumerate(corners):
                if x < -frame_w * 0.5 or x > frame_w * 1.5 or y < -frame_h * 0.5 or y > frame_h * 1.5:
                    print(f"⚠️ Warning: Corner {i} out of bounds: ({x:.1f}, {y:.1f})")
            
            # Store corners in order: front-left, front-right, back-right, back-left
            # This creates a complete trapezoid covering the full court
            self.calibration_data.court_corners = [
                (int(front_left[0]), int(front_left[1])),
                (int(front_right[0]), int(front_right[1])),
                (int(back_right[0]), int(back_right[1])),
                (int(back_left[0]), int(back_left[1]))
            ]
            
            print(f"🔶 Calculated full court corners: {self.calibration_data.court_corners}")
            
        except Exception as e:
            print(f"❌ Error calculating full court corners: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: use calibration points directly as approximate corners
            print("   Using fallback method with direct points...")
            front_points = self.calibration_data.front_line_points
            back_points = self.calibration_data.back_line_points
            
            if len(front_points) >= 2 and len(back_points) >= 2:
                # Ensure proper left-to-right ordering
                front_left = min(front_points, key=lambda p: p[0])
                front_right = max(front_points, key=lambda p: p[0])
                back_left = min(back_points, key=lambda p: p[0])
                back_right = max(back_points, key=lambda p: p[0])
                
                self.calibration_data.court_corners = [
                    front_left,   # Front-left
                    front_right,  # Front-right
                    back_right,   # Back-right
                    back_left     # Back-left
                ]
                print(f"🔶 Fallback full court corners: {self.calibration_data.court_corners}")
            else:
                raise ValueError("Insufficient points for fallback corner calculation")
    
    def _validate_point_ordering(self) -> None:
        """Validate and fix point ordering for consistent calculations"""
        # Fix front line points - should be left to right
        if len(self.calibration_data.front_line_points) >= 2:
            points = self.calibration_data.front_line_points
            if points[0][0] > points[1][0]:  # First point is to the right of second
                self.calibration_data.front_line_points = [points[1], points[0]]
                print("   Fixed front line point ordering: left to right")
        
        # Fix back line points - should be left to right  
        if len(self.calibration_data.back_line_points) >= 2:
            points = self.calibration_data.back_line_points
            if points[0][0] > points[1][0]:  # First point is to the right of second
                self.calibration_data.back_line_points = [points[1], points[0]]
                print("   Fixed back line point ordering: left to right")
        
        # Fix left sideline points - should be from front to back (smaller Y to larger Y)
        if len(self.calibration_data.left_sideline_points) >= 2:
            points = self.calibration_data.left_sideline_points
            if points[0][1] > points[1][1]:  # First point is below second
                self.calibration_data.left_sideline_points = [points[1], points[0]]
                print("   Fixed left sideline point ordering: front to back")
        
        # Fix right sideline points - should be from front to back (smaller Y to larger Y)
        if len(self.calibration_data.right_sideline_points) >= 2:
            points = self.calibration_data.right_sideline_points
            if points[0][1] > points[1][1]:  # First point is below second
                self.calibration_data.right_sideline_points = [points[1], points[0]]
                print("   Fixed right sideline point ordering: front to back")
        
        # Fix net line points - should be left to right
        if len(self.calibration_data.net_line_points) >= 2:
            # Sort all net points from left to right
            self.calibration_data.net_line_points.sort(key=lambda p: p[0])
            print("   Fixed net line point ordering: left to right")
    
    def _calculate_perspective_matrix(self) -> None:
        """Calculate perspective transformation matrix"""
        if not self.calibration_data.court_corners or len(self.calibration_data.court_corners) != 4:
            raise ValueError("Need 4 court corners for perspective calculation")
        
        try:
            print("   Setting up world coordinates for full court...")
            # Real-world full court coordinates (meters)
            # Full volleyball court: 18m length, 9m width
            real_world_points = np.array([
                [-4.5, -9.0],   # Front-left (closest to camera)
                [4.5, -9.0],    # Front-right (closest to camera)  
                [4.5, 9.0],     # Back-right (farthest from camera)
                [-4.5, 9.0]     # Back-left (farthest from camera)
            ], dtype=np.float32)
            
            # Image corner points
            image_points = np.array(self.calibration_data.court_corners, dtype=np.float32)
            
            print(f"   World points: {real_world_points}")
            print(f"   Image points: {image_points}")
            
            print("   Computing homography...")
            # Calculate homography
            homography, mask = cv2.findHomography(real_world_points, image_points, cv2.RANSAC, 5.0)
            
            if homography is None:
                raise ValueError("Failed to compute homography matrix")
            
            # Store as list for JSON serialization
            self.calibration_data.perspective_matrix = homography.tolist()
            
            print("🔄 Perspective matrix calculated successfully")
            
            # Test the transformation with known points
            self._test_perspective_transformation(homography)
            
        except Exception as e:
            print(f"❌ Error calculating perspective matrix: {e}")
            # Set identity matrix as fallback
            self.calibration_data.perspective_matrix = np.eye(3).tolist()
            print("   Using identity matrix as fallback")
    
    def _test_perspective_transformation(self, homography: np.ndarray) -> None:
        """Test perspective transformation accuracy"""
        try:
            print("   Testing perspective transformation accuracy...")
            
            # Test points: center of net and center of court
            test_world_points = np.array([
                [0.0, 0.0],     # Center of net
                [0.0, 4.5],     # Center of back half
                [-2.25, 0.0],   # Left side of net
                [2.25, 0.0]     # Right side of net
            ], dtype=np.float32)
            
            # Transform to screen coordinates
            test_screen_points = cv2.perspectiveTransform(
                test_world_points.reshape(-1, 1, 2), homography
            )
            
            # Transform back to world coordinates
            inv_homography = cv2.invert(homography)[1]
            back_to_world = cv2.perspectiveTransform(test_screen_points, inv_homography)
            
            # Calculate errors
            max_error = 0.0
            for i, (original, recovered) in enumerate(zip(test_world_points, back_to_world.reshape(-1, 2))):
                error = np.sqrt((original[0] - recovered[0])**2 + (original[1] - recovered[1])**2)
                max_error = max(max_error, error)
                
                if error > 0.5:  # More than 50cm error
                    print(f"⚠️ High transformation error at test point {i}: {error:.2f}m")
                else:
                    print(f"✅ Test point {i} error: {error:.3f}m")
            
            if max_error < 0.2:  # Less than 20cm error
                print(f"✅ Perspective transformation accuracy: EXCELLENT (max error: {max_error:.3f}m)")
            elif max_error < 0.5:  # Less than 50cm error
                print(f"⚠️ Perspective transformation accuracy: GOOD (max error: {max_error:.3f}m)")
            else:
                print(f"❌ Perspective transformation accuracy: POOR (max error: {max_error:.3f}m)")
                print("   Consider recalibrating for better accuracy")
                
        except Exception as e:
            print(f"⚠️ Could not test perspective transformation: {e}")
    
    def _calculate_height_scale(self) -> None:
        """Calculate height scale factor from net height"""
        if len(self.calibration_data.net_line_points) < 2:
            print("   Warning: Not enough net points for height calculation")
            return
        
        try:
            print("   Calculating height scale from net...")
            # Estimate net pixel height (simple approach)
            net_points_y = [p[1] for p in self.calibration_data.net_line_points]
            
            # Assume ground level is at bottom of image
            ground_y = self.calibration_data.frame_height - 50  # 50px margin
            net_y = np.mean(net_points_y)
            
            net_height_pixels = ground_y - net_y
            
            print(f"   Ground level Y: {ground_y}")
            print(f"   Net level Y: {net_y}")
            print(f"   Net height in pixels: {net_height_pixels}")
            
            if net_height_pixels > 0:
                self.calibration_data.height_scale_factor = self.court_geometry.net_height / net_height_pixels
                print(f"📏 Height scale: {self.calibration_data.height_scale_factor:.4f} m/pixel")
            else:
                print("   Warning: Invalid net height, using default scale")
                self.calibration_data.height_scale_factor = 0.001  # Default fallback
                
        except Exception as e:
            print(f"❌ Error calculating height scale: {e}")
            self.calibration_data.height_scale_factor = 0.001  # Default fallback
    
    def _get_line_equation(self, points: List[Tuple[int, int]]) -> np.ndarray:
        """Get line equation in homogeneous coordinates"""
        if len(points) < 2:
            raise ValueError("Need minimum 2 points for line")
        
        p1 = np.array([points[0][0], points[0][1], 1])
        p2 = np.array([points[1][0], points[1][1], 1])
        
        # Line equation as cross product
        line = np.cross(p1, p2)
        return line / np.linalg.norm(line[:2])  # Normalize
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Tuple[float, float]:
        """Calculate intersection of two lines"""
        point = np.cross(line1, line2)
        if abs(point[2]) < 1e-10:
            raise ValueError("Lines are parallel")
        
        x = point[0] / point[2] 
        y = point[1] / point[2]
        return (x, y)
    
    def _find_point_on_line_at_y(self, line: np.ndarray, y: float) -> Tuple[float, float]:
        """Find point on line at given Y coordinate"""
        # Line equation: ax + by + c = 0
        a, b, c = line
        
        if abs(b) < 1e-10:
            raise ValueError("Line is vertical")
        
        x = -(c + b * y) / a
        return (x, y)
    
    def _draw_coordinate_grid(self) -> None:
        """Draw coordinate grid on calibrated full court"""
        if not self.calibration_data.perspective_matrix:
            return
        
        try:
            H = np.array(self.calibration_data.perspective_matrix)
            
            # Draw grid lines every 2 meters for full court
            for y in range(-8, 10, 2):  # Full court: -9 to +9 meters (front to back)
                for x_offset in [-4, -2, 0, 2, 4]:  # Every 2 meters across court
                    world_point = np.array([[x_offset, y]], dtype=np.float32)
                    screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
                    x, y_screen = screen_point[0][0]
                    
                    cv2.circle(self.display_frame, (int(x), int(y_screen)), 3, (100, 100, 100), -1)
            
            # Draw net line (center line at y=0)
            net_center_points = []
            for x_offset in np.linspace(-4.5, 4.5, 10):
                world_point = np.array([[x_offset, 0.0]], dtype=np.float32)
                screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
                net_center_points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
            
            for i in range(len(net_center_points) - 1):
                cv2.line(self.display_frame, net_center_points[i], net_center_points[i + 1], 
                        (0, 255, 255), 2)  # Yellow center line
            
            # Draw attack lines (3m from net on both sides)
            for attack_y in [-3.0, 3.0]:  # Attack lines on both sides
                attack_points = []
                for x_offset in np.linspace(-4.5, 4.5, 10):
                    world_point = np.array([[x_offset, attack_y]], dtype=np.float32)
                    screen_point = cv2.perspectiveTransform(world_point.reshape(-1, 1, 2), H)
                    attack_points.append((int(screen_point[0][0][0]), int(screen_point[0][0][1])))
                
                for i in range(len(attack_points) - 1):
                    cv2.line(self.display_frame, attack_points[i], attack_points[i + 1], 
                            (0, 165, 255), 2)  # Orange attack lines
                        
        except Exception as e:
            print(f"Warning: failed to draw coordinate grid: {e}")
    
    def _save_calibration(self) -> None:
        """Save calibration data to JSON file"""
        try:
            # Convert calibration data to dictionary
            calibration_dict = asdict(self.calibration_data)
            
            # Add metadata
            calibration_dict['metadata'] = {
                'created_at': cv2.getTickCount() / cv2.getTickFrequency(),
                'input_source': self.input_source,
                'court_geometry': asdict(self.court_geometry),
                'calibration_completed': self.current_stage == CalibrationStage.COMPLETED
            }
            
            # Save to JSON
            with open(self.output_calibration, 'w', encoding='utf-8') as f:
                json.dump(calibration_dict, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Калібровку збережено: {self.output_calibration}")
            
        except Exception as e:
            print(f"❌ Помилка збереження: {e}")


def load_calibration(calibration_file: str) -> Optional[CalibrationData]:
    """Load calibration data from JSON file"""
    try:
        with open(calibration_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Remove metadata before creating CalibrationData
        data.pop('metadata', None)
        
        return CalibrationData(**data)
        
    except Exception as e:
        print(f"❌ Помилка завантаження калібровки: {e}")
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Інтерактивна калібровка волейбольного поля')
    parser.add_argument('--input', '-i', required=True,
                       help='Відеофайл або індекс камери (0 для веб-камери)')
    parser.add_argument('--output', '-o', default='court_calibration.json',
                       help='Файл для збереження калібровки')
    
    args = parser.parse_args()
    
    # Create and run calibration tool
    calibration = InteractiveCourtCalibration(args.input, args.output)
    calibration.run()


if __name__ == "__main__":
    main() 