import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


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


class YOLO11PoseEstimator(BasePoseEstimator):
    """YOLO11 pose estimation implementation"""
    
    def __init__(self, model_path: str = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\models\pretrained\yolo11n-pose.pt'):
        """
        Initialize YOLO11 pose estimator
        
        Args:
            model_path: Path to YOLO11 pose model
        """
        try:
            from ultralytics import YOLO
            
            # Check if specified model exists
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"✅ YOLO11 pose model loaded from: {model_path}")
            else:
                print(f"⚠️  Model not found at {model_path}")
                print("   Trying to download YOLO11n-pose model...")
                self.model = YOLO('yolo11n-pose.pt')  # This will download automatically
                print("✅ Default YOLO11n-pose model downloaded and loaded")
                
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            print(f"⚠️  Could not load YOLO11 model: {e}")
            print("   Trying to download default model...")
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolo11n-pose.pt')  # This will download automatically
                print("✅ Default YOLO11n-pose model downloaded and loaded")
            except Exception as e2:
                raise Exception(f"Could not initialize YOLO11 pose model: {e2}")
        
        # COCO pose keypoint connections (17 keypoints for human pose)
        # Based on https://docs.ultralytics.com/ru/tasks/pose/
        self.connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),     # Head connections
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (5, 11), (6, 12),         # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Keypoint names according to YOLO11 documentation
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def detect_poses(self, frame: np.ndarray) -> List[Pose]:
        """Detect poses using YOLO11"""
        try:
            results = self.model(frame, verbose=False)
            
            poses = []
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    # YOLO11 returns keypoints in format [batch, num_persons, num_keypoints, 3]
                    # where 3 = [x, y, confidence]
                    for person_keypoints in result.keypoints.data:
                        keypoints = []
                        for i, kp in enumerate(person_keypoints):
                            x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
                            keypoints.append(KeyPoint(x=x, y=y, confidence=conf))
                        
                        # Calculate overall confidence (average of visible keypoints)
                        visible_keypoints = [kp for kp in keypoints if kp.confidence > 0.3]
                        confidence = np.mean([kp.confidence for kp in visible_keypoints]) if visible_keypoints else 0.0
                        
                        # Calculate bounding box from valid keypoints
                        valid_kps = [kp for kp in keypoints if kp.confidence > 0.5]
                        if valid_kps:
                            x_coords = [kp.x for kp in valid_kps]
                            y_coords = [kp.y for kp in valid_kps]
                            
                            # Add some padding to the bounding box
                            padding = 20
                            x1 = max(0, int(min(x_coords)) - padding)
                            y1 = max(0, int(min(y_coords)) - padding)
                            x2 = int(max(x_coords)) + padding
                            y2 = int(max(y_coords)) + padding
                            
                            bbox = (x1, y1, x2, y2)
                        else:
                            bbox = None
                        
                        # Only add pose if it has enough visible keypoints
                        if len(visible_keypoints) >= 5:  # At least 5 visible keypoints
                            pose = Pose(keypoints=keypoints, confidence=confidence, bbox=bbox)
                            poses.append(pose)
            
            return poses
            
        except Exception as e:
            print(f"Error in YOLO11 pose detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get COCO pose connections"""
        return self.connections


class DummyPoseEstimator(BasePoseEstimator):
    """Dummy pose estimator for testing when no models available"""
    
    def __init__(self):
        print("⚠️  Using dummy pose estimator - no actual pose detection")
        self.connections = []
    
    def detect_poses(self, frame: np.ndarray) -> List[Pose]:
        """Return empty list - no pose detection"""
        return []
    
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Return empty connections list"""
        return []


class SimplePoseEstimatorFactory:
    """Factory class for creating simple pose estimators"""
    
    @staticmethod
    def create_estimator(estimator_type: str = 'yolo11', **kwargs) -> BasePoseEstimator:
        """
        Create pose estimator
        
        Args:
            estimator_type: Type of estimator ('yolo11', 'yolov8', 'dummy')
            **kwargs: Additional arguments for estimator
            
        Returns:
            Pose estimator instance
        """
        if estimator_type.lower() in ['yolo11', 'yolov8']:
            try:
                return YOLO11PoseEstimator(**kwargs)
            except Exception as e:
                print(f"⚠️  Could not create YOLO11 estimator: {e}")
                print("   Falling back to dummy estimator")
                return DummyPoseEstimator()
        elif estimator_type.lower() == 'dummy':
            return DummyPoseEstimator()
        else:
            print(f"⚠️  Unknown estimator type: {estimator_type}, using YOLO11")
            try:
                return YOLO11PoseEstimator(**kwargs)
            except:
                return DummyPoseEstimator() 