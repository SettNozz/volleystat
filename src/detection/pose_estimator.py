import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Import TransformersPoseEstimator if available
try:
    from .transformers_pose_estimator import TransformersPoseEstimator
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class KeyPoint:
    """Keypoint representation"""
    x: float
    y: float
    confidence: float


@dataclass
class Pose:
    """Pose representation with keypoints"""
    keypoints: List[KeyPoint]
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


class BasePoseEstimator(ABC):
    """Base class for pose estimation models"""
    
    @abstractmethod
    def detect_poses(self, frame: np.ndarray) -> List[Pose]:
        """Detect poses in frame"""
        pass
    
    @abstractmethod
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get connections between keypoints for drawing skeleton"""
        pass


class MediaPipePoseEstimator(BasePoseEstimator):
    """MediaPipe pose estimation implementation"""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose estimator
        
        Args:
            static_image_mode: Whether to treat input as static images
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe pose connections
        self.connections = self.mp_pose.POSE_CONNECTIONS
        
    def detect_poses(self, frame: np.ndarray) -> List[Pose]:
        """Detect poses using MediaPipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.mp_pose.process(rgb_frame)
        
        poses = []
        if results.pose_landmarks:
            keypoints = []
            h, w, _ = frame.shape
            
            for landmark in results.pose_landmarks.landmark:
                keypoint = KeyPoint(
                    x=landmark.x * w,
                    y=landmark.y * h,
                    confidence=landmark.visibility
                )
                keypoints.append(keypoint)
            
            # Calculate overall confidence
            confidence = np.mean([kp.confidence for kp in keypoints])
            
            # Calculate bounding box
            x_coords = [kp.x for kp in keypoints if kp.confidence > 0.5]
            y_coords = [kp.y for kp in keypoints if kp.confidence > 0.5]
            
            if x_coords and y_coords:
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                bbox = (x1, y1, x2, y2)
            else:
                bbox = None
            
            pose = Pose(keypoints=keypoints, confidence=confidence, bbox=bbox)
            poses.append(pose)
        
        return poses
    
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get MediaPipe pose connections"""
        return [(connection[0], connection[1]) for connection in self.connections]


class YOLOv8PoseEstimator(BasePoseEstimator):
    """YOLOv8 pose estimation implementation"""
    
    def __init__(self, model_path: str = 'yolov8n-pose.pt'):
        """
        Initialize YOLOv8 pose estimator
        
        Args:
            model_path: Path to YOLOv8 pose model
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        # COCO pose keypoint connections
        self.connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    def detect_poses(self, frame: np.ndarray) -> List[Pose]:
        """Detect poses using YOLOv8"""
        results = self.model(frame, verbose=False)
        
        poses = []
        for result in results:
            if result.keypoints is not None:
                for keypoints_data in result.keypoints.data:
                    keypoints = []
                    for kp in keypoints_data:
                        x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                        keypoints.append(KeyPoint(x=x, y=y, confidence=conf))
                    
                    # Calculate overall confidence
                    confidence = np.mean([kp.confidence for kp in keypoints if kp.confidence > 0])
                    
                    # Calculate bounding box
                    valid_kps = [kp for kp in keypoints if kp.confidence > 0.5]
                    if valid_kps:
                        x_coords = [kp.x for kp in valid_kps]
                        y_coords = [kp.y for kp in valid_kps]
                        x1, x2 = int(min(x_coords)), int(max(x_coords))
                        y1, y2 = int(min(y_coords)), int(max(y_coords))
                        bbox = (x1, y1, x2, y2)
                    else:
                        bbox = None
                    
                    pose = Pose(keypoints=keypoints, confidence=confidence, bbox=bbox)
                    poses.append(pose)
        
        return poses
    
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get COCO pose connections"""
        return self.connections


class OpenPosePoseEstimator(BasePoseEstimator):
    """OpenPose estimation implementation (simplified version using OpenCV DNN)"""
    
    def __init__(self, 
                 prototxt_path: Optional[str] = None,
                 weights_path: Optional[str] = None,
                 confidence_threshold: float = 0.1):
        """
        Initialize OpenPose estimator
        
        Args:
            prototxt_path: Path to prototxt file
            weights_path: Path to weights file
            confidence_threshold: Minimum confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        
        # Use default paths if not provided
        if prototxt_path is None or weights_path is None:
            print("Warning: OpenPose model files not provided. Using MediaPipe as fallback.")
            self.fallback_estimator = MediaPipePoseEstimator()
            self.use_fallback = True
        else:
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
                self.use_fallback = False
            except:
                print("Warning: Could not load OpenPose model. Using MediaPipe as fallback.")
                self.fallback_estimator = MediaPipePoseEstimator()
                self.use_fallback = True
        
        # COCO pose keypoint connections
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head to right arm
            (1, 5), (5, 6), (6, 7),  # Head to left arm
            (1, 8), (8, 9), (9, 10),  # Head to right leg
            (1, 11), (11, 12), (12, 13),  # Head to left leg
            (8, 11)  # Torso connection
        ]
    
    def detect_poses(self, frame: np.ndarray) -> List[Pose]:
        """Detect poses using OpenPose or fallback to MediaPipe"""
        if self.use_fallback:
            return self.fallback_estimator.detect_poses(frame)
        
        # OpenPose implementation would go here
        # For now, return empty list as placeholder
        return []
    
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get pose connections"""
        if self.use_fallback:
            return self.fallback_estimator.get_pose_connections()
        return self.connections


class PoseEstimatorFactory:
    """Factory class for creating pose estimators"""
    
    @staticmethod
    def create_estimator(estimator_type: str, **kwargs) -> BasePoseEstimator:
        """
        Create pose estimator
        
        Args:
            estimator_type: Type of estimator ('mediapipe', 'yolov8', 'openpose', 'transformers')
            **kwargs: Additional arguments for estimator
            
        Returns:
            Pose estimator instance
        """
        if estimator_type.lower() == 'mediapipe':
            return MediaPipePoseEstimator(**kwargs)
        elif estimator_type.lower() == 'yolov8':
            return YOLOv8PoseEstimator(**kwargs)
        elif estimator_type.lower() == 'openpose':
            return OpenPosePoseEstimator(**kwargs)
        elif estimator_type.lower() == 'transformers':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library not available. Install with: pip install transformers torch")
            return TransformersPoseEstimator(**kwargs)
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}") 