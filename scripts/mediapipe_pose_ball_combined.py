#!/usr/bin/env python3
"""
Combined MediaPipe Pose Estimation + YOLO Ball Detection for Volleyball Analytics
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from ultralytics import YOLO
import torch


class PoseSmoothing:
    """Pose smoothing for stable keypoint visualization"""
    
    def __init__(self, smoothing_factor: float = 0.1, min_confidence: float = 0.5):
        self.smoothing_factor = smoothing_factor  # 0.1 = very smooth, 0.9 = less smooth
        self.min_confidence = min_confidence
        self.pose_history: Dict[int, List[List[List[float]]]] = {}  # person_id -> [keypoints_history]
        self.max_history_length = 5
    
    def smooth_pose(self, person_id: int, keypoints: List[List[float]]) -> List[List[float]]:
        """Apply temporal smoothing to keypoints"""
        if person_id not in self.pose_history:
            self.pose_history[person_id] = []
        
        # Add current keypoints to history
        self.pose_history[person_id].append([kp.copy() for kp in keypoints])
        if len(self.pose_history[person_id]) > self.max_history_length:
            self.pose_history[person_id].pop(0)
        
        # If we don't have enough history, return original
        if len(self.pose_history[person_id]) < 2:
            return keypoints
        
        # Apply exponential moving average smoothing
        smoothed_keypoints = []
        history = self.pose_history[person_id]
        
        for i, current_kp in enumerate(keypoints):
            if len(current_kp) >= 3 and current_kp[2] > self.min_confidence:
                # Initialize with current position
                weighted_x = current_kp[0] * (1.0 - self.smoothing_factor)
                weighted_y = current_kp[1] * (1.0 - self.smoothing_factor)
                
                # Add weighted contributions from previous frames
                weight_sum = 1.0 - self.smoothing_factor
                
                for j, prev_frame in enumerate(reversed(history[:-1])):
                    if i < len(prev_frame) and len(prev_frame[i]) >= 3 and prev_frame[i][2] > self.min_confidence:
                        # Weight decreases exponentially with time distance
                        frame_weight = self.smoothing_factor * (0.7 ** j)
                        weighted_x += prev_frame[i][0] * frame_weight
                        weighted_y += prev_frame[i][1] * frame_weight
                        weight_sum += frame_weight
                
                # Normalize by total weight
                smoothed_x = weighted_x / weight_sum if weight_sum > 0 else current_kp[0]
                smoothed_y = weighted_y / weight_sum if weight_sum > 0 else current_kp[1]
                
                smoothed_keypoints.append([smoothed_x, smoothed_y, current_kp[2]])
            else:
                # Low confidence or invalid keypoint - keep original
                smoothed_keypoints.append(current_kp.copy())
        
        return smoothed_keypoints
    
    def reset_person(self, person_id: int):
        """Reset history for a person (when they disappear and reappear)"""
        if person_id in self.pose_history:
            del self.pose_history[person_id]
    
    def cleanup_old_people(self, active_person_ids: List[int]):
        """Remove pose history for people who are no longer tracked"""
        to_remove = []
        for person_id in self.pose_history:
            if person_id not in active_person_ids:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.pose_history[person_id]


class PersonTracker:
    """Simple person tracker for consistent colors between frames"""
    
    def __init__(self, max_distance: float = 80, max_missing_frames: int = 10):
        self.tracked_people = {}  # stable_id -> {bbox, center, last_seen, velocity}
        self.next_stable_id = 0
        self.max_distance = max_distance  # Maximum distance to consider same person
        self.max_missing_frames = max_missing_frames  # Frames before removing person
        self.current_frame = 0
        
    def update(self, people: List[Dict[str, Any]], frame_count: int) -> List[Dict[str, Any]]:
        """Update tracker with new detections and assign stable IDs"""
        self.current_frame = frame_count
        
        # Calculate centers and sizes for all detected people
        for person in people:
            bbox = person['bbox']
            person['center'] = (
                (bbox[0] + bbox[2]) / 2,  # center_x
                (bbox[1] + bbox[3]) / 2   # center_y
            )
            person['size'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # area
        
        # Match new detections to existing tracked people
        matched_people = []
        used_stable_ids = set()
        
        for person in people:
            best_match_id = None
            min_score = float('inf')
            
            # Find best match among tracked people
            for stable_id, tracked in self.tracked_people.items():
                if stable_id in used_stable_ids:
                    continue
                    
                # Calculate distance between centers
                dx = person['center'][0] - tracked['center'][0]
                dy = person['center'][1] - tracked['center'][1]
                distance = (dx**2 + dy**2)**0.5
                
                # Size similarity check (avoid matching very different sizes)
                size_ratio = min(person['size'], tracked['size']) / max(person['size'], tracked['size'])
                
                # Combine distance and size - lower score is better
                score = distance * (2.0 - size_ratio)  # Penalize size differences
                
                # Disable motion prediction to avoid bbox shifting issues
                # Simple distance-based matching is more reliable for pose estimation
                
                if (distance < self.max_distance and size_ratio > 0.6 and score < min_score):
                    min_score = score
                    best_match_id = stable_id
            
            # Assign stable ID
            if best_match_id is not None:
                person['stable_id'] = best_match_id
                used_stable_ids.add(best_match_id)
                
                # Simplified tracking without velocity calculation to avoid bbox shifts
                velocity = (0, 0)
                
                # Update tracked person
                self.tracked_people[best_match_id] = {
                    'bbox': person['bbox'],
                    'center': person['center'],
                    'size': person['size'],
                    'last_seen': frame_count,
                    'velocity': velocity
                }
            else:
                # New person - assign new stable ID
                person['stable_id'] = self.next_stable_id
                self.tracked_people[self.next_stable_id] = {
                    'bbox': person['bbox'],
                    'center': person['center'],
                    'size': person['size'],
                    'last_seen': frame_count,
                    'velocity': (0, 0)
                }
                self.next_stable_id += 1
            
            matched_people.append(person)
        
        # Remove people who haven't been seen for too long
        to_remove = []
        for stable_id, tracked in self.tracked_people.items():
            if frame_count - tracked['last_seen'] > self.max_missing_frames:
                to_remove.append(stable_id)
        
        for stable_id in to_remove:
            del self.tracked_people[stable_id]
        
        return matched_people


class BallTracker:
    """Ball tracker for trajectory visualization with false positive filtering"""
    
    def __init__(self, max_trajectory_length: int = 50):
        self.ball_positions: List[Tuple[int, int]] = []
        self.max_trajectory_length = max_trajectory_length
        self.last_valid_position: Optional[Tuple[int, int]] = None
        self.consecutive_misses = 0
        self.last_bbox_size: Optional[float] = None
        
        # Filtering parameters
        self.max_velocity = 300  # Maximum pixels per frame the ball can move
        self.min_confidence = 0.5  # Minimum confidence to consider
        self.max_missing_frames = 10  # Max frames before resetting tracking
        self.size_change_threshold = 3.0  # Max ratio change in bbox size
        
        # Statistics
        self.total_detections = 0
        self.valid_detections = 0
        self.rejected_low_confidence = 0
        self.rejected_velocity = 0
        self.rejected_size_change = 0
        
        self.colors = {
            'ball': (0, 255, 255),      # Yellow for ball
            'bbox': (0, 255, 255),      # Yellow for bbox
            'text': (255, 255, 255),    # White for text
            'trajectory': (0, 165, 255) # Orange for trajectory
        }
    
    def add_detection(self, x1: float, y1: float, x2: float, y2: float, confidence: float) -> Optional[Tuple[int, int]]:
        """Add ball detection with validation and return center coordinates if valid"""
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        bbox_size = (x2 - x1) * (y2 - y1)
        
        self.total_detections += 1
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            self.consecutive_misses += 1
            self.rejected_low_confidence += 1
            return None
        
        # If we have previous position, validate the movement
        if self.last_valid_position is not None:
            last_x, last_y = self.last_valid_position
            
            # Check velocity constraint
            distance = ((center_x - last_x)**2 + (center_y - last_y)**2)**0.5
            if distance > self.max_velocity:
                self.consecutive_misses += 1
                self.rejected_velocity += 1
                return None
            
            # Check size consistency
            if self.last_bbox_size is not None:
                size_ratio = max(bbox_size, self.last_bbox_size) / min(bbox_size, self.last_bbox_size)
                if size_ratio > self.size_change_threshold:
                    self.consecutive_misses += 1
                    self.rejected_size_change += 1
                    return None
        
        # Valid detection - add to trajectory
        self.ball_positions.append((center_x, center_y))
        if len(self.ball_positions) > self.max_trajectory_length:
            self.ball_positions.pop(0)
        
        # Update tracking state
        self.last_valid_position = (center_x, center_y)
        self.last_bbox_size = bbox_size
        self.consecutive_misses = 0
        self.valid_detections += 1
        
        return center_x, center_y
    
    def update_no_detection(self):
        """Call when no ball detected in frame"""
        self.consecutive_misses += 1
        
        # Reset tracking if ball missing too long
        if self.consecutive_misses > self.max_missing_frames:
            self.reset_tracking()
    
    def reset_tracking(self):
        """Reset tracking state for new ball sequence"""
        self.last_valid_position = None
        self.last_bbox_size = None
        self.consecutive_misses = 0
        # Keep existing trajectory points but stop adding new ones
    
    def draw_trajectory(self, frame: np.ndarray) -> np.ndarray:
        """Draw ball trajectory on frame"""
        if len(self.ball_positions) < 2:
            return frame
        
        # Draw trajectory lines with fading effect
        for i in range(1, len(self.ball_positions)):
            # Transparency increases for newer points
            alpha = i / len(self.ball_positions)
            thickness = max(1, int(4 * alpha))
            
            # Create color with fading effect
            color = tuple(int(c * alpha) for c in self.colors['trajectory'])
            
            cv2.line(
                frame,
                self.ball_positions[i-1],
                self.ball_positions[i],
                color,
                thickness
            )
        
        return frame


class CombinedPoseBallDetector:
    """Combined MediaPipe Pose Estimation + YOLO Ball Detection"""
    
    def __init__(self, ball_model_path: str, max_people: int = 12):
        """
        Initialize combined detector
        
        Args:
            ball_model_path: Path to YOLO ball detection model
            max_people: Maximum number of people to detect
        """
        self.max_people = max_people
        
        # Initialize MediaPipe components
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Setup MediaPipe object detection and pose estimation
        self._setup_object_detection()
        self._setup_pose_estimation()
        
        # Initialize YOLO ball detector
        self._setup_ball_detection(ball_model_path)
        
        # Initialize trackers
        self.person_tracker = PersonTracker(max_distance=120, max_missing_frames=20)
        self.ball_tracker = BallTracker(max_trajectory_length=50)
        self.pose_smoother = PoseSmoothing(smoothing_factor=0.3, min_confidence=0.5)
        
        # Colors for different people (expanded palette)
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (128, 0, 255),  # Purple
            (255, 128, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Light Green
            (255, 0, 128),  # Pink
            (128, 128, 255), # Light Purple
            (64, 128, 128),  # Teal
            (128, 64, 128),  # Purple-ish
            (128, 128, 64),  # Olive
            (192, 64, 64),   # Brown-ish
            (64, 192, 64),   # Forest green
            (64, 64, 192),   # Navy
            (192, 192, 64),  # Yellow-green
            (192, 64, 192),  # Magenta-ish
            (128, 192, 192), # Light teal
        ]
        
        print(f"Combined Pose+Ball Detector initialized")
        print(f"  - Max people: {max_people}")
        print(f"  - Ball model: {ball_model_path}")
    
    def _setup_object_detection(self):
        """Setup MediaPipe Object Detection for finding people"""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download EfficientDet model if needed
            model_path = "models/efficientdet_lite0.tflite"
            os.makedirs("models", exist_ok=True)
            
            if not os.path.exists(model_path):
                print("Downloading EfficientDet model for object detection...")
                model_url = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite'
                urllib.request.urlretrieve(model_url, model_path)
                print(f"Object detection model downloaded to: {model_path}")
            
            # Create object detector for finding people with lower threshold
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                max_results=20,  # Increase max results
                score_threshold=0.25,  # Lower threshold for more detections
                category_allowlist=['person']  # Only detect people!
            )
            
            self.object_detector = vision.ObjectDetector.create_from_options(options)
            self.object_detection_available = True
            print("MediaPipe Object Detection initialized (person detection only)")
            
        except Exception as e:
            print(f"Object Detection setup failed: {e}")
            self.object_detector = None
            self.object_detection_available = False
    
    def _setup_pose_estimation(self):
        """Setup MediaPipe Pose Estimation"""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Download pose model if needed
            pose_model_path = "models/pose_landmarker_full.task"
            
            if not os.path.exists(pose_model_path):
                print("Downloading MediaPipe pose model...")
                pose_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
                urllib.request.urlretrieve(pose_url, pose_model_path)
                print(f"Pose model downloaded to: {pose_model_path}")
            
            # Create pose landmarker
            base_options = python.BaseOptions(model_asset_path=pose_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=self.max_people,
                min_pose_detection_confidence=0.6,
                min_pose_presence_confidence=0.6,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            self.pose_estimation_available = True
            print("MediaPipe Pose Landmarker initialized")
            
        except Exception as e:
            print(f"Pose Estimation setup failed: {e}")
            # Fallback to legacy pose estimation
            self.pose_landmarker = None
            self.pose_estimation_available = False
            print("Using legacy MediaPipe Pose API")
        
        # Always setup legacy pose as fallback for per-region detection
        self.legacy_pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
    
    def _setup_ball_detection(self, model_path: str):
        """Setup YOLO ball detection"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Ball detection model not found: {model_path}")
        
        print(f"Loading YOLO ball detection model: {model_path}")
        self.ball_model = YOLO(str(model_path))
        
        # Configure for GPU if available
        if torch.cuda.is_available():
            print(f"Using GPU for ball detection: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
        else:
            print("Using CPU for ball detection")
        
        # Ball detection parameters
        self.ball_confidence_threshold = 0.4
        self.ball_nms_threshold = 0.4
        
        print("YOLO ball detection initialized")
    
    def detect_people_objects(self, frame: np.ndarray, timestamp_ms: int) -> List[Dict[str, Any]]:
        """Detect people using MediaPipe Object Detection"""
        if not self.object_detection_available:
            # Fallback: return full frame as single detection
            h, w = frame.shape[:2]
            return [{
                'bbox': (0, 0, w, h),
                'confidence': 0.5,
                'category': 'person'
            }]
        
        try:
            # Convert frame to MediaPipe Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect objects (people only)
            detection_result = self.object_detector.detect_for_video(mp_image, timestamp_ms)
            
            people = []
            if detection_result.detections:
                h, w = frame.shape[:2]
                
                for detection in detection_result.detections:
                    # Get bounding box
                    bbox = detection.bounding_box
                    x1 = int(bbox.origin_x)
                    y1 = int(bbox.origin_y)
                    x2 = int(x1 + bbox.width)
                    y2 = int(y1 + bbox.height)
                    
                    # Ensure bbox is within frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Filter by minimum size (allow distant players)
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    min_size = 25  # Lower minimum for distant volleyball players
                    
                    if bbox_width >= min_size and bbox_height >= min_size:
                        confidence = detection.categories[0].score if detection.categories else 0.5
                        
                        people.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'category': 'person'
                        })
            
            return people
            
        except Exception as e:
            print(f"Object detection failed: {e}")
            return []
    
    def detect_ball(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Detect ball using YOLO model"""
        try:
            # Run YOLO detection
            results = self.ball_model(
                frame,
                conf=self.ball_confidence_threshold,
                iou=self.ball_nms_threshold,
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                half=True  # Use FP16 for speed
            )
            
            detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        detections.append((x1, y1, x2, y2, confidence))
            
            return detections
            
        except Exception as e:
            print(f"Ball detection failed: {e}")
            return []
    
    def detect_poses_for_people(self, frame: np.ndarray, people: List[Dict[str, Any]], 
                                timestamp_ms: int) -> List[Dict[str, Any]]:
        """Detect poses for found people with duplicate elimination"""
        poses = []
        detected_people_masks = [False] * len(people)  # Track which people have poses
        
        if self.pose_estimation_available and self.pose_landmarker:
            # Use new PoseLandmarker API
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                
                if result.pose_landmarks:
                    h, w = frame.shape[:2]
                    
                    for i, pose_landmarks in enumerate(result.pose_landmarks):
                        # Convert landmarks to pixel coordinates
                        keypoints = []
                        for landmark in pose_landmarks:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            keypoints.append([x, y, landmark.visibility])
                        
                        # Calculate bounding box from keypoints
                        visible_points = [(kp[0], kp[1]) for kp in keypoints if kp[2] > 0.5]
                        if visible_points:
                            x_coords = [p[0] for p in visible_points]
                            y_coords = [p[1] for p in visible_points]
                            bbox = (
                                max(0, min(x_coords) - 20), 
                                max(0, min(y_coords) - 20),
                                min(w, max(x_coords) + 20), 
                                min(h, max(y_coords) + 20)
                            )
                        else:
                            bbox = (0, 0, w, h)
                        
                        # Match pose to closest person
                        matched_person_idx = self._match_pose_to_person(bbox, people, detected_people_masks)
                        
                        if matched_person_idx is not None:
                            detected_people_masks[matched_person_idx] = True
                            poses.append({
                                'landmarks': pose_landmarks,
                                'keypoints': np.array(keypoints),
                                'bbox': bbox,
                                'confidence': 0.8,
                                'person_id': matched_person_idx,
                                'source': 'new_api'
                            })
            
            except Exception as e:
                print(f"New pose API failed: {e}, trying per-person detection")
        
        # Try per-person detection only for people without poses
        for i, person in enumerate(people):
            if not detected_people_masks[i]:  # Only for people without poses
                pose = self._detect_pose_in_region(frame, person['bbox'], i)
                if pose:
                    detected_people_masks[i] = True
                    poses.append(pose)
        
        return poses
    
    def _match_pose_to_person(self, pose_bbox: tuple, people: List[Dict[str, Any]], 
                              detected_masks: List[bool]) -> Optional[int]:
        """Match detected pose to the closest undetected person"""
        pose_x1, pose_y1, pose_x2, pose_y2 = pose_bbox
        pose_center_x = (pose_x1 + pose_x2) / 2
        pose_center_y = (pose_y1 + pose_y2) / 2
        
        best_match_idx = None
        min_distance = float('inf')
        
        for i, person in enumerate(people):
            if detected_masks[i]:  # Skip already detected people
                continue
                
            person_bbox = person['bbox']
            person_x1, person_y1, person_x2, person_y2 = person_bbox
            person_center_x = (person_x1 + person_x2) / 2
            person_center_y = (person_y1 + person_y2) / 2
            
            # Calculate distance between centers
            distance = ((pose_center_x - person_center_x)**2 + (pose_center_y - person_center_y)**2)**0.5
            
            # Check if pose bbox overlaps significantly with person bbox
            overlap_x1 = max(pose_x1, person_x1)
            overlap_y1 = max(pose_y1, person_y1)
            overlap_x2 = min(pose_x2, person_x2)
            overlap_y2 = min(pose_y2, person_y2)
            
            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                pose_area = (pose_x2 - pose_x1) * (pose_y2 - pose_y1)
                person_area = (person_x2 - person_x1) * (person_y2 - person_y1)
                
                # Calculate overlap ratio
                overlap_ratio = overlap_area / min(pose_area, person_area) if min(pose_area, person_area) > 0 else 0
                
                # Prefer matches with good overlap and close distance (increased threshold for more accuracy)
                if overlap_ratio > 0.5 and distance < min_distance:
                    min_distance = distance
                    best_match_idx = i
        
        # If no good overlap, match to closest person within stricter distance
        if best_match_idx is None:
            for i, person in enumerate(people):
                if detected_masks[i]:
                    continue
                    
                person_bbox = person['bbox']
                person_center_x = (person_bbox[0] + person_bbox[2]) / 2
                person_center_y = (person_bbox[1] + person_bbox[3]) / 2
                
                distance = ((pose_center_x - person_center_x)**2 + (pose_center_y - person_center_y)**2)**0.5
                
                if distance < 120 and distance < min_distance:  # Reduced from 200 to 120 pixels for more accurate matching
                    min_distance = distance
                    best_match_idx = i
        
        return best_match_idx
    
    def _remove_duplicate_poses(self, poses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate poses based on keypoint overlap"""
        if len(poses) <= 1:
            return poses
        
        unique_poses = []
        
        for pose in poses:
            is_duplicate = False
            
            # Check against existing unique poses
            for existing_pose in unique_poses:
                if self._poses_overlap(pose, existing_pose):
                    # Keep the pose with higher confidence or prefer new_api
                    if (pose.get('confidence', 0) > existing_pose.get('confidence', 0) or
                        (pose.get('source') == 'new_api' and existing_pose.get('source') == 'legacy_api')):
                        # Replace existing with current
                        unique_poses.remove(existing_pose)
                        unique_poses.append(pose)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_poses.append(pose)
        
        return unique_poses
    
    def _poses_overlap(self, pose1: Dict[str, Any], pose2: Dict[str, Any]) -> bool:
        """Check if two poses represent the same person based on keypoint proximity"""
        keypoints1 = pose1.get('keypoints', [])
        keypoints2 = pose2.get('keypoints', [])
        
        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return False
        
        # Count overlapping keypoints
        overlap_count = 0
        total_valid_pairs = 0
        overlap_threshold = 30  # pixels (reduced for more strict duplicate detection)
        
        for i, (kp1, kp2) in enumerate(zip(keypoints1, keypoints2)):
            # Skip face keypoints for overlap check (0-10)
            if i <= 10:
                continue
                
            # Check if both keypoints have good confidence
            if (len(kp1) >= 3 and len(kp2) >= 3 and 
                kp1[2] > 0.5 and kp2[2] > 0.5):
                
                total_valid_pairs += 1
                
                # Calculate distance between keypoints
                distance = ((kp1[0] - kp2[0])**2 + (kp1[1] - kp2[1])**2)**0.5
                
                if distance < overlap_threshold:
                    overlap_count += 1
        
        # Consider poses overlapping if >70% of valid keypoints are close
        if total_valid_pairs > 0:
            overlap_ratio = overlap_count / total_valid_pairs
            return overlap_ratio > 0.7
        
        return False
    
    def _detect_pose_in_region(self, frame: np.ndarray, bbox: tuple, person_id: int) -> Optional[Dict[str, Any]]:
        """Detect pose in specific person region using legacy API"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Try with different padding sizes for better detection
        padding_sizes = [50, 30, 70, 20]
        
        for pad in padding_sizes:
            # Extract and expand region
            x1_pad = max(0, x1 - pad)
            y1_pad = max(0, y1 - pad)
            x2_pad = min(w, x2 + pad)
            y2_pad = min(h, y2 + pad)
            
            roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if roi.size == 0 or roi.shape[0] < 50 or roi.shape[1] < 50:
                continue
            
            # Convert to RGB and process
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.legacy_pose.process(rgb_roi)
            
            if results.pose_landmarks:
                # Check if we have enough visible landmarks
                visible_landmarks = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.3)
                
                if visible_landmarks >= 10:  # At least 10 visible points for good pose
                    # Convert landmarks to full frame coordinates
                    roi_h, roi_w = roi.shape[:2]
                    keypoints = []
                    
                    for landmark in results.pose_landmarks.landmark:
                        # Convert from ROI coordinates to full frame coordinates
                        x = int(landmark.x * roi_w) + x1_pad
                        y = int(landmark.y * roi_h) + y1_pad
                        keypoints.append([x, y, landmark.visibility])
                    
                    return {
                        'landmarks': results.pose_landmarks,
                        'keypoints': np.array(keypoints),
                        'bbox': bbox,
                        'confidence': 0.7,
                        'person_id': person_id,
                        'source': 'legacy_api',
                        'roi_offset': (x1_pad, y1_pad)
                    }
        
        return None
    
    def draw_results(self, frame: np.ndarray, people: List[Dict[str, Any]], 
                     poses: List[Dict[str, Any]], ball_detections: List[Tuple[float, float, float, float, float]]) -> None:
        """Draw all detection results on frame with false positive filtering"""
        
        # Draw ball detections first (so they appear behind poses)
        if not ball_detections:
            # No ball detections in this frame
            self.ball_tracker.update_no_detection()
        else:
            valid_ball_detections = 0
            for x1, y1, x2, y2, confidence in ball_detections:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Try to add detection with validation
                center_coords = self.ball_tracker.add_detection(x1, y1, x2, y2, confidence)
                
                if center_coords is not None:
                    # Valid detection - draw it
                    center_x, center_y = center_coords
                    valid_ball_detections += 1
                    
                    # Draw ball bbox
                    cv2.rectangle(
                        frame,
                        (x1, y1), (x2, y2),
                        self.ball_tracker.colors['bbox'],
                        2
                    )
                    
                    # Draw ball center
                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        6,
                        self.ball_tracker.colors['ball'],
                        -1
                    )
                    
                    # Ball confidence text with validation mark
                    label = f"Ball: {confidence:.2f} ✓"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Text background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        self.ball_tracker.colors['bbox'],
                        -1
                    )
                    
                    # Text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.ball_tracker.colors['text'],
                        2
                    )
                else:
                    # Invalid detection - draw as rejected
                    cv2.rectangle(
                        frame,
                        (x1, y1), (x2, y2),
                        (0, 0, 255),  # Red for rejected
                        1
                    )
                    
                    # Mark as rejected
                    label = f"Rejected: {confidence:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1
                    )
            
            # Update tracker if no valid detections
            if valid_ball_detections == 0:
                self.ball_tracker.update_no_detection()
        
        # Draw ball trajectory
        frame = self.ball_tracker.draw_trajectory(frame)
        
        # Draw poses with sequential colors (no tracking between frames)
        for i, pose in enumerate(poses):
            # Use sequential color for current frame only
            color = self.colors[i % len(self.colors)]
            
            # Skip drawing if no pose was found
            if pose.get('source') == 'no_pose_found':
                continue
            
            if pose['source'] == 'new_api':
                # Draw keypoints manually for new API (skip face landmarks 0-10)
                keypoints = pose['keypoints']
                
                # Draw keypoints
                for idx, kp in enumerate(keypoints):
                    if idx <= 10:  # Skip face landmarks
                        continue
                    if len(kp) >= 3 and kp[2] > 0.5:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 3, color, -1)
                        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
                
                # Draw connections (skip face connections)
                connections = self.mp_pose.POSE_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    # Skip connections involving face landmarks (0-10)
                    if start_idx <= 10 or end_idx <= 10:
                        continue
                    if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                        start_kp = keypoints[start_idx]
                        end_kp = keypoints[end_idx]
                        if (len(start_kp) >= 3 and len(end_kp) >= 3 and 
                            start_kp[2] > 0.5 and end_kp[2] > 0.5):
                            start_point = (int(start_kp[0]), int(start_kp[1]))
                            end_point = (int(end_kp[0]), int(end_kp[1]))
                            cv2.line(frame, start_point, end_point, color, 2)
            
            elif pose['source'] == 'legacy_api':
                # Draw keypoints manually for legacy API (skip face landmarks 0-10)
                keypoints = pose['keypoints']
                
                # Draw keypoints
                for idx, kp in enumerate(keypoints):
                    if idx <= 10:  # Skip face landmarks
                        continue
                    if len(kp) >= 3 and kp[2] > 0.5:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 3, color, -1)
                        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
                
                # Draw connections (skip face connections)
                connections = self.mp_pose.POSE_CONNECTIONS
                for connection in connections:
                    start_idx, end_idx = connection
                    # Skip connections involving face landmarks (0-10)
                    if start_idx <= 10 or end_idx <= 10:
                        continue
                    if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                        start_kp = keypoints[start_idx]
                        end_kp = keypoints[end_idx]
                        if (len(start_kp) >= 3 and len(end_kp) >= 3 and 
                            start_kp[2] > 0.5 and end_kp[2] > 0.5):
                            start_point = (int(start_kp[0]), int(start_kp[1]))
                            end_point = (int(end_kp[0]), int(end_kp[1]))
                            cv2.line(frame, start_point, end_point, color, 2)
    
    def detect_all(self, frame: np.ndarray, timestamp_ms: int, frame_count: int = 0) -> tuple:
        """Detect people, poses, and ball in one call (no tracking, just frame-by-frame detection)"""
        # Detect people using object detection
        people = self.detect_people_objects(frame, timestamp_ms)
        
        # Detect poses for people (no tracking, just current frame)
        poses = self.detect_poses_for_people(frame, people, timestamp_ms)
        
        # Assign simple sequential IDs for coloring (no tracking between frames)
        for i, pose in enumerate(poses):
            pose['stable_id'] = i
        
        # Detect ball
        ball_detections = self.detect_ball(frame)
        
        return people, poses, ball_detections


def test_combined_detection_on_video(video_path: str, output_path: str, ball_model_path: str) -> Dict[str, Any]:
    """Test combined detection on video"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Testing Combined Pose + Ball Detection on: {video_name}")
    
    # Initialize combined detector
    detector = CombinedPoseBallDetector(ball_model_path=ball_model_path, max_people=12)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames:,} frames ({total_frames/fps:.1f}s duration)")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics
    stats = {
        'video_name': video_name,
        'total_frames': total_frames,
        'people_detected': 0,
        'poses_detected': 0,
        'ball_detections': 0,
        'processing_time': 0,
        'frames_processed': 0,
        'max_people_in_frame': 0,
        'max_poses_in_frame': 0,
        'max_balls_in_frame': 0
    }
    
    start_time = time.time()
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate timestamp
            timestamp_ms = int(frame_count * 1000 / fps)
            
            # Detect all objects
            people, poses, ball_detections = detector.detect_all(frame, timestamp_ms, frame_count)
            
            # Draw results
            detector.draw_results(frame, people, poses, ball_detections)
            
            # Update statistics
            people_count = len(people)
            poses_count = len(poses)
            ball_count = len(ball_detections)
            
            stats['people_detected'] += people_count
            stats['poses_detected'] += poses_count
            stats['ball_detections'] += ball_count
            stats['max_people_in_frame'] = max(stats['max_people_in_frame'], people_count)
            stats['max_poses_in_frame'] = max(stats['max_poses_in_frame'], poses_count)
            stats['max_balls_in_frame'] = max(stats['max_balls_in_frame'], ball_count)
            
            # Add info overlay
            trajectory_length = len(detector.ball_tracker.ball_positions)
            
            info_lines = [
                f"Combined Pose + Ball Detection (No Tracking)",
                f"Frame: {frame_count+1}/{total_frames}",
                f"People: {people_count} | Poses: {poses_count} | Ball: {ball_count}",
                f"Ball Trajectory: {trajectory_length} points",
                f"Processing: Frame-by-frame detection (no smoothing)",
                f"Max: {stats['max_people_in_frame']} people, {stats['max_poses_in_frame']} poses, {stats['max_balls_in_frame']} balls"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 25
                # Background for text
                cv2.rectangle(frame, (5, y_pos-20), (700, y_pos+5), (0, 0, 0), -1)
                cv2.putText(frame, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            stats['frames_processed'] += 1
            frame_count += 1
            
            # Progress update
            if frame_count % 10 == 0 or frame_count == total_frames:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                percentage = (frame_count / total_frames) * 100
                
                # Create progress bar
                bar_length = 30
                filled_length = int(bar_length * frame_count // total_frames)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                print(f"\rProgress: [{bar}] {percentage:.1f}% ({frame_count}/{total_frames}) | {fps_current:.1f} FPS | People: {people_count} | Poses: {poses_count} | Ball: {ball_count}", end='', flush=True)
    
    finally:
        cap.release()
        out.release()
        print()  # New line after progress bar
    
    # Final statistics
    stats['processing_time'] = time.time() - start_time
    avg_fps = stats['frames_processed'] / stats['processing_time'] if stats['processing_time'] > 0 else 0
    avg_people = stats['people_detected'] / stats['frames_processed'] if stats['frames_processed'] > 0 else 0
    avg_poses = stats['poses_detected'] / stats['frames_processed'] if stats['frames_processed'] > 0 else 0
    avg_balls = stats['ball_detections'] / stats['frames_processed'] if stats['frames_processed'] > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Combined Pose + Ball Detection Results for {video_name}")
    print(f"{'='*80}")
    print(f"  Frames processed:     {stats['frames_processed']:,}")
    print(f"  People detected:      {stats['people_detected']:,} ({avg_people:.1f} avg per frame)")
    print(f"  Poses detected:       {stats['poses_detected']:,} ({avg_poses:.1f} avg per frame)")
    print(f"  Ball detections:      {stats['ball_detections']:,} ({avg_balls:.1f} avg per frame)")
    print(f"  Max people in frame:  {stats['max_people_in_frame']}")
    print(f"  Max poses in frame:   {stats['max_poses_in_frame']}")
    print(f"  Max balls in frame:   {stats['max_balls_in_frame']}")
    print(f"  Ball trajectory:      {len(detector.ball_tracker.ball_positions)} points")
    print(f"  Processing time:      {stats['processing_time']:.1f}s")
    print(f"  Average FPS:          {avg_fps:.1f}")
    print(f"  Object Detection:     {'✓' if detector.object_detection_available else '✗ (fallback)'}")
    print(f"  Pose Estimation:      {'✓ New API' if detector.pose_estimation_available else '✗ Legacy API'}")
    print(f"  Ball Detection:       ✓ YOLO")
    print(f"  Person Tracking:      ✗ Disabled (frame-by-frame detection)")
    print(f"  Ball Tracking:        ✓ (max trajectory: {detector.ball_tracker.max_trajectory_length} points)")
    print(f"  Pose Smoothing:       ✗ Disabled (raw keypoints)")
    print(f"  Ball Filtering Stats:")
    print(f"    - Total raw detections:     {detector.ball_tracker.total_detections}")
    print(f"    - Valid detections:         {detector.ball_tracker.valid_detections}")
    print(f"    - Rejected (low confidence): {detector.ball_tracker.rejected_low_confidence}")
    print(f"    - Rejected (velocity):      {detector.ball_tracker.rejected_velocity}")
    print(f"    - Rejected (size change):   {detector.ball_tracker.rejected_size_change}")
    if detector.ball_tracker.total_detections > 0:
        filter_rate = ((detector.ball_tracker.total_detections - detector.ball_tracker.valid_detections) / detector.ball_tracker.total_detections) * 100
        print(f"    - Filter rate:              {filter_rate:.1f}%")
    print(f"  Output video:         {os.path.basename(output_path)}")
    print(f"{'='*80}")
    
    return stats


def main():
    """Main function to test combined detection"""
    print("="*80)
    print("   Combined MediaPipe Pose + YOLO Ball Detection Test")
    print("="*80)
    print()
    
    # Paths
    test_dir = "data/test_videos_3n_10d"
    output_dir = "data/pose_estimation_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Model paths
    ball_model_path = "C:/Users/illya/Documents/volleyball_analitics/volleystat/models/yolov8_curated_training/yolov8n_volleyball_curated4/weights/epoch130.pt"
    
    # Test first video
    video_file = "test_video_1_GX010378.mp4"
    video_path = os.path.join(test_dir, video_file)
    output_path = os.path.join(output_dir, "GX010378_combined_pose_ball_detection_smooth_fx_fx.mp4")
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    
    if not os.path.exists(ball_model_path):
        print(f"Error: Ball model not found: {ball_model_path}")
        return
    
    try:
        stats = test_combined_detection_on_video(video_path, output_path, ball_model_path)
        
        # Save results
        results_file = os.path.join(output_dir, "combined_pose_ball_test_results.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=== Combined Pose + Ball Detection Results ===\n\n")
            f.write(f"Video: {stats['video_name']}\n")
            f.write(f"Total frames: {stats['total_frames']}\n")
            f.write(f"Frames processed: {stats['frames_processed']}\n")
            f.write(f"People detected: {stats['people_detected']} ({stats['people_detected']/stats['frames_processed']:.2f} avg per frame)\n")
            f.write(f"Poses detected: {stats['poses_detected']} ({stats['poses_detected']/stats['frames_processed']:.2f} avg per frame)\n")
            f.write(f"Ball detections: {stats['ball_detections']} ({stats['ball_detections']/stats['frames_processed']:.2f} avg per frame)\n")
            f.write(f"Max people in frame: {stats['max_people_in_frame']}\n")
            f.write(f"Max poses in frame: {stats['max_poses_in_frame']}\n")
            f.write(f"Max balls in frame: {stats['max_balls_in_frame']}\n")
            f.write(f"Processing time: {stats['processing_time']:.1f}s\n")
            f.write(f"Average FPS: {stats['frames_processed']/stats['processing_time']:.1f}\n")
        
        print(f"\nResults saved to: {results_file}")
        print("Combined pose + ball detection test completed successfully!")
        
    except Exception as e:
        print(f"Error during combined detection test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 