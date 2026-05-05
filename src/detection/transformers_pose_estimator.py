"""
Transformers-based pose estimation using HuggingFace models
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import pipeline
import torch
from PIL import Image

from .simple_pose_estimator import BasePoseEstimator, Pose, KeyPoint


class TransformersPoseEstimator(BasePoseEstimator):
    """Transformers-based pose estimation implementation"""
    
    def __init__(self, 
                 model_name: str = "LucyintheSky/pose-estimation-front-side-back",
                 confidence_threshold: float = 0.5,
                 device: Optional[str] = None):
        """
        Initialize Transformers pose estimator
        
        Args:
            model_name: HuggingFace model name
            confidence_threshold: Minimum confidence threshold
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Initialize the pipeline
            self.pipe = pipeline(
                "image-classification", 
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            self.available = True
            print(f"✅ Transformers pose estimator loaded: {model_name}")
        except Exception as e:
            print(f"❌ Error loading transformers model: {e}")
            self.available = False
            self.pipe = None
        
        # Since this is a classification model, we'll create dummy keypoints
        # based on the classification result
        self.dummy_keypoints = self._create_dummy_keypoints()
        
        # Connections for basic skeleton visualization
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Head to shoulders
            (1, 5), (1, 6),  # Shoulders
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
    
    def _create_dummy_keypoints(self) -> Dict[str, List[Tuple[float, float]]]:
        """Create dummy keypoint positions for different pose types"""
        # Normalized coordinates (0-1) for different pose types
        return {
            'front': [
                (0.5, 0.1),   # nose
                (0.45, 0.12), # left_eye
                (0.55, 0.12), # right_eye
                (0.4, 0.14),  # left_ear
                (0.6, 0.14),  # right_ear
                (0.35, 0.3),  # left_shoulder
                (0.65, 0.3),  # right_shoulder
                (0.25, 0.45), # left_elbow
                (0.75, 0.45), # right_elbow
                (0.2, 0.6),   # left_wrist
                (0.8, 0.6),   # right_wrist
                (0.4, 0.55),  # left_hip
                (0.6, 0.55),  # right_hip
                (0.35, 0.75), # left_knee
                (0.65, 0.75), # right_knee
                (0.3, 0.95),  # left_ankle
                (0.7, 0.95),  # right_ankle
            ],
            'side': [
                (0.5, 0.1),   # nose
                (0.48, 0.12), # left_eye
                (0.52, 0.12), # right_eye
                (0.45, 0.14), # left_ear
                (0.55, 0.14), # right_ear
                (0.4, 0.3),   # left_shoulder
                (0.6, 0.3),   # right_shoulder
                (0.35, 0.45), # left_elbow
                (0.65, 0.45), # right_elbow
                (0.3, 0.6),   # left_wrist
                (0.7, 0.6),   # right_wrist
                (0.45, 0.55), # left_hip
                (0.55, 0.55), # right_hip
                (0.4, 0.75),  # left_knee
                (0.6, 0.75),  # right_knee
                (0.35, 0.95), # left_ankle
                (0.65, 0.95), # right_ankle
            ],
            'back': [
                (0.5, 0.1),   # nose
                (0.52, 0.12), # left_eye
                (0.48, 0.12), # right_eye
                (0.55, 0.14), # left_ear
                (0.45, 0.14), # right_ear
                (0.65, 0.3),  # left_shoulder
                (0.35, 0.3),  # right_shoulder
                (0.75, 0.45), # left_elbow
                (0.25, 0.45), # right_elbow
                (0.8, 0.6),   # left_wrist
                (0.2, 0.6),   # right_wrist
                (0.6, 0.55),  # left_hip
                (0.4, 0.55),  # right_hip
                (0.65, 0.75), # left_knee
                (0.35, 0.75), # right_knee
                (0.7, 0.95),  # left_ankle
                (0.3, 0.95),  # right_ankle
            ]
        }
    
    def _classify_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Classify pose orientation using transformers model"""
        if not self.available or not self.pipe:
            return None
        
        try:
            # Convert frame to PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Get predictions
            results = self.pipe(pil_image)
            
            # Process results
            if results and len(results) > 0:
                top_result = results[0]
                return {
                    'label': top_result['label'].lower(),
                    'confidence': top_result['score']
                }
            
            return None
            
        except Exception as e:
            print(f"Error in pose classification: {e}")
            return None
    
    def detect_poses(self, frame: np.ndarray) -> List[Pose]:
        """Detect poses using transformers classification"""
        if not self.available:
            return []
        
        # Classify the pose
        classification = self._classify_pose(frame)
        
        if not classification or classification['confidence'] < self.confidence_threshold:
            return []
        
        # Get dummy keypoints for the classified pose type
        pose_type = classification['label']
        if pose_type not in self.dummy_keypoints:
            # Try to match partial labels
            if 'front' in pose_type:
                pose_type = 'front'
            elif 'side' in pose_type:
                pose_type = 'side'
            elif 'back' in pose_type:
                pose_type = 'back'
            else:
                pose_type = 'front'  # Default
        
        # Convert normalized coordinates to actual coordinates
        h, w = frame.shape[:2]
        keypoints = []
        
        for norm_x, norm_y in self.dummy_keypoints[pose_type]:
            x = norm_x * w
            y = norm_y * h
            # Use classification confidence as keypoint confidence
            keypoint = KeyPoint(x=x, y=y, confidence=classification['confidence'])
            keypoints.append(keypoint)
        
        # Calculate bounding box
        x_coords = [kp.x for kp in keypoints]
        y_coords = [kp.y for kp in keypoints]
        
        padding = 20
        x1 = max(0, int(min(x_coords)) - padding)
        y1 = max(0, int(min(y_coords)) - padding)
        x2 = min(w, int(max(x_coords)) + padding)
        y2 = min(h, int(max(y_coords)) + padding)
        
        bbox = (x1, y1, x2, y2)
        
        # Create pose object
        pose = Pose(
            keypoints=keypoints,
            confidence=classification['confidence'],
            bbox=bbox
        )
        
        return [pose]
    
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """Get pose connections for skeleton drawing"""
        return self.connections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'available': self.available,
            'type': 'transformers_classification'
        } 