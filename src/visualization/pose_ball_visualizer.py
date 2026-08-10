import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import colorsys
import time

from ..detection.simple_pose_estimator import BasePoseEstimator, Pose, KeyPoint
from ..tracking.ball_tracker import BallTracker


class PoseBallVisualizer:
    """Visualizer for poses and ball tracking combined"""
    
    def __init__(self, 
                 pose_estimator: BasePoseEstimator,
                 ball_tracker: Optional[BallTracker] = None,
                 show_pose_bbox: bool = True,
                 show_keypoint_labels: bool = False,
                 trajectory_length: int = 30):
        """
        Initialize pose and ball visualizer
        
        Args:
            pose_estimator: Pose estimation model
            ball_tracker: Ball tracker instance
            show_pose_bbox: Whether to show pose bounding boxes
            show_keypoint_labels: Whether to show keypoint labels
            trajectory_length: Maximum length of ball trajectory to show
        """
        self.pose_estimator = pose_estimator
        self.ball_tracker = ball_tracker or BallTracker()
        self.show_pose_bbox = show_pose_bbox
        self.show_keypoint_labels = show_keypoint_labels
        self.trajectory_length = trajectory_length
        
        # Ball trajectory storage
        self.ball_positions: List[Tuple[int, int]] = []
        
        # Colors for visualization
        self.colors = {
            'ball': (0, 255, 0),           # Green for ball
            'ball_trajectory': (255, 0, 0), # Red for ball trajectory
            'pose_bbox': (255, 255, 0),    # Cyan for pose bbox
            'keypoints': (0, 255, 255),    # Yellow for keypoints
            'skeleton': (255, 0, 255),     # Magenta for skeleton
            'text': (255, 255, 255),       # White for text
        }
        
        # Keypoint names for MediaPipe (33 landmarks)
        self.mp_keypoint_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # COCO keypoint names (17 keypoints)
        self.coco_keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def generate_person_colors(self, num_persons: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for different persons"""
        colors = []
        for i in range(num_persons):
            hue = i / num_persons
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors
    
    def draw_keypoint(self, frame: np.ndarray, keypoint: KeyPoint, 
                     color: Tuple[int, int, int], radius: int = 3,
                     label: Optional[str] = None) -> None:
        """Draw single keypoint on frame"""
        if keypoint.confidence > 0.5:
            x, y = int(keypoint.x), int(keypoint.y)
            cv2.circle(frame, (x, y), radius, color, -1)
            
            if label and self.show_keypoint_labels:
                cv2.putText(frame, label, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.colors['text'], 1)
    
    def draw_skeleton(self, frame: np.ndarray, pose: Pose, color: Tuple[int, int, int]) -> None:
        """Draw skeleton connections on frame"""
        connections = self.pose_estimator.get_pose_connections()
        
        for connection in connections:
            start_idx, end_idx = connection
            
            if (start_idx < len(pose.keypoints) and 
                end_idx < len(pose.keypoints)):
                
                start_kp = pose.keypoints[start_idx]
                end_kp = pose.keypoints[end_idx]
                
                if (start_kp.confidence > 0.5 and 
                    end_kp.confidence > 0.5):
                    
                    start_point = (int(start_kp.x), int(start_kp.y))
                    end_point = (int(end_kp.x), int(end_kp.y))
                    
                    cv2.line(frame, start_point, end_point, color, 2)
    
    def draw_pose_bbox(self, frame: np.ndarray, pose: Pose, 
                      color: Tuple[int, int, int], person_id: int) -> None:
        """Draw pose bounding box"""
        if pose.bbox and self.show_pose_bbox:
            x1, y1, x2, y2 = pose.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add person label
            label = f"Person {person_id} ({pose.confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
    
    def draw_poses(self, frame: np.ndarray, poses: List[Pose]) -> None:
        """Draw all poses on frame"""
        if not poses:
            return
        
        # Generate colors for different persons
        person_colors = self.generate_person_colors(len(poses))
        
        # Get keypoint names based on number of keypoints
        if poses[0].keypoints:
            num_keypoints = len(poses[0].keypoints)
            if num_keypoints == 33:  # MediaPipe
                keypoint_names = self.mp_keypoint_names
            elif num_keypoints == 17:  # COCO
                keypoint_names = self.coco_keypoint_names
            else:
                keypoint_names = [f"kp_{i}" for i in range(num_keypoints)]
        else:
            keypoint_names = []
        
        for person_id, (pose, color) in enumerate(zip(poses, person_colors)):
            # Draw skeleton
            self.draw_skeleton(frame, pose, color)
            
            # Draw keypoints
            for kp_idx, keypoint in enumerate(pose.keypoints):
                label = keypoint_names[kp_idx] if kp_idx < len(keypoint_names) else None
                self.draw_keypoint(frame, keypoint, color, label=label)
            
            # Draw bounding box
            self.draw_pose_bbox(frame, pose, color, person_id + 1)
    
    def update_ball_trajectory(self, ball_center: Tuple[int, int]) -> None:
        """Update ball trajectory"""
        self.ball_positions.append(ball_center)
        
        # Keep only recent positions
        if len(self.ball_positions) > self.trajectory_length:
            self.ball_positions.pop(0)
    
    def draw_ball_trajectory(self, frame: np.ndarray) -> None:
        """Draw ball trajectory"""
        if len(self.ball_positions) < 2:
            return
        
        # Draw trajectory lines with fading effect
        for i in range(1, len(self.ball_positions)):
            # Calculate alpha based on position in trajectory
            alpha = i / len(self.ball_positions)
            thickness = max(1, int(3 * alpha))
            
            # Draw line segment
            cv2.line(frame, 
                    self.ball_positions[i-1], 
                    self.ball_positions[i],
                    self.colors['ball_trajectory'], 
                    thickness)
    
    def draw_ball(self, frame: np.ndarray, ball_detections: List[Tuple[float, float, float, float, float]]) -> None:
        """Draw ball detections and trajectory"""
        for detection in ball_detections:
            x1, y1, x2, y2, confidence = detection
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         self.colors['ball'], 2)
            
            # Draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(frame, (center_x, center_y), 5, self.colors['ball'], -1)
            
            # Update trajectory
            self.update_ball_trajectory((center_x, center_y))
            
            # Add confidence label
            label = f"Ball: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for text
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), self.colors['ball'], -1)
            
            # Text
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Draw trajectory
        self.draw_ball_trajectory(frame)
    
    def add_performance_info(self, frame: np.ndarray, fps: float, 
                           frame_num: int, num_poses: int, num_balls: int) -> None:
        """Add performance and detection information to frame"""
        info_lines = [
            f"Frame: {frame_num}",
            f"FPS: {fps:.1f}",
            f"Poses: {num_poses}",
            f"Balls: {num_balls}"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            y_offset += 25
    
    def process_frame(self, frame: np.ndarray, 
                     ball_detections: Optional[List[Tuple[float, float, float, float, float]]] = None,
                     frame_num: int = 0,
                     fps: float = 0.0) -> np.ndarray:
        """
        Process single frame with pose and ball detection
        
        Args:
            frame: Input frame
            ball_detections: List of ball detections (x1, y1, x2, y2, confidence)
            frame_num: Frame number for display
            fps: Current FPS for display
            
        Returns:
            Processed frame with visualizations
        """
        # Make a copy of the frame
        output_frame = frame.copy()
        
        # Detect poses
        start_time = time.time()
        poses = self.pose_estimator.detect_poses(frame)
        pose_time = time.time() - start_time
        
        # Draw poses
        self.draw_poses(output_frame, poses)
        
        # Draw balls if provided
        ball_count = 0
        if ball_detections:
            self.draw_ball(output_frame, ball_detections)
            ball_count = len(ball_detections)
        
        # Add performance information
        self.add_performance_info(output_frame, fps, frame_num, len(poses), ball_count)
        
        return output_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current visualization statistics"""
        return {
            'trajectory_length': len(self.ball_positions),
            'max_trajectory_length': self.trajectory_length,
            'pose_estimator_type': type(self.pose_estimator).__name__,
        } 