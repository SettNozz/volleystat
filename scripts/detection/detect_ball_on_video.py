#!/usr/bin/env python3
"""
Volleyball ball detection on video
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from ultralytics import YOLO
import time
from datetime import datetime
import torch
import json
import uuid


class BallDetectorVideo:
    """Ball detector for video."""
    
    def __init__(self, model_path: str):
        """
        Initialize detector.
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        print(f"🤖 Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✅ Model loaded successfully!")
        
        # Check model info
        if hasattr(self.model, 'names'):
            print(f"📋 Model classes: {self.model.names}")
            # Check if 'Volleyball Ball' class exists
            ball_classes = [name for name in self.model.names.values() if 'ball' in name.lower() or 'volleyball' in name.lower()]
            if ball_classes:
                print(f"⚽ Found ball-related classes: {ball_classes}")
            else:
                print("⚠️ Warning: No ball-related classes found in model!")
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'yaml'):
            print(f"🏗️ Model architecture: {self.model.model.yaml.get('nc', 'unknown')} classes")
        
        # Configure for maximum GPU utilization
        import torch
        if torch.cuda.is_available():
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            # Configure for maximum speed and throughput
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for A100/RTX 30xx
            torch.backends.cuda.matmul.allow_tf32 = True
            # Pre-allocate memory for stability
            torch.cuda.empty_cache()
            # Set cache allocator for better memory usage
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            print("⚡ GPU optimized for maximum throughput")
        else:
            print("⚠️ GPU not found, using CPU")
        
        # Detection parameters  
        self.confidence_threshold = 0.1  # Lowered threshold for better detection
        self.nms_threshold = 0.4
        # Maximize batch size due to very low GPU utilization (15% SM)
        if torch.cuda.is_available():
            # Calculate maximum batch based on available GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # For 4K video we can allow larger batch
            self.batch_size = min(128, int(gpu_memory_gb * 8))  # ~8 frames per GB
            print(f"🚀 Auto-set batch size: {self.batch_size}")
        else:
            self.batch_size = 8
        
        # Colors for visualization
        self.colors = {
            'ball': (0, 255, 0),      # Green for ball
            'bbox': (0, 255, 0),      # Green for bbox
            'text': (255, 255, 255),  # White for text
            'trajectory': (255, 0, 0) # Red for trajectory
        }
        
        # Ball position history for trajectory
        self.ball_positions: List[Tuple[int, int]] = []
        self.max_trajectory_length = 30
        
        # False positive filtering
        self.last_valid_position: Optional[Tuple[int, int]] = None
        self.consecutive_misses = 0
        self.last_bbox_size: Optional[float] = None
        
        # Filtering parameters
        self.max_velocity = 400  # Maximum pixels per frame the ball can move (higher for video)
        self.min_confidence = 0.2  # Minimum confidence to consider (lower for YOLO)
        self.max_missing_frames = 15  # Max frames before resetting tracking
        self.size_change_threshold = 4.0  # Max ratio change in bbox size
        
        # Statistics
        self.total_detections = 0
        self.valid_detections = 0
        self.rejected_low_confidence = 0
        self.rejected_velocity = 0
        self.rejected_size_change = 0
        
        # Label Studio annotation data
        self.label_studio_annotations: List[Dict[str, Any]] = []
        self.annotation_id_counter = 1
    
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
        
    def detect_ball(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Ball detection on frame.
        
        Args:
            frame: Video frame
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        # Run detection with GPU optimization
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
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
    
    def detect_ball_batch(self, frames: List[np.ndarray]) -> List[List[Tuple[float, float, float, float, float]]]:
        """
        Batch ball detection on multiple frames simultaneously.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of detection lists for each frame
        """
        # Run batch detection for maximum GPU utilization
        results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half=True,  # Use FP16 for speed
            stream=False,  # Disable streaming for batches
            max_det=100,  # Increase maximum detections
            augment=False,  # Disable TTA for speed
            agnostic_nms=False,  # Class-specific NMS
            retina_masks=False  # Disable masks
        )
        
        batch_detections = []
        total_raw_detections = 0
        total_valid_detections = 0
        
        for i, result in enumerate(results):
            detections = []
            if result.boxes is not None:
                total_raw_detections += len(result.boxes)
                for box in result.boxes:
                    # Get coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    if confidence >= self.confidence_threshold:
                        detections.append((x1, y1, x2, y2, confidence))
                        total_valid_detections += 1
            batch_detections.append(detections)
        return batch_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[float, float, float, float, float]]) -> np.ndarray:
        """
        Draw detections on frame with false positive filtering.
        
        Args:
            frame: Video frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        frame_with_detections = frame.copy()
        
        if not detections:
            # No detections - update tracker
            self.update_no_detection()
            return frame_with_detections
        
        valid_detections = 0
        for x1, y1, x2, y2, confidence in detections:
            # Try to validate detection
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            bbox_size = (x2 - x1) * (y2 - y1)
            
            self.total_detections += 1
            is_valid = True
            rejection_reason = ""
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                self.consecutive_misses += 1
                self.rejected_low_confidence += 1
                is_valid = False
                rejection_reason = "low_conf"
            
            # If we have previous position, validate the movement
            elif self.last_valid_position is not None:
                last_x, last_y = self.last_valid_position
                
                # Check velocity constraint
                distance = ((center_x - last_x)**2 + (center_y - last_y)**2)**0.5
                if distance > self.max_velocity:
                    self.consecutive_misses += 1
                    self.rejected_velocity += 1
                    is_valid = False
                    rejection_reason = "velocity"
                
                # Check size consistency
                elif self.last_bbox_size is not None:
                    size_ratio = max(bbox_size, self.last_bbox_size) / min(bbox_size, self.last_bbox_size)
                    if size_ratio > self.size_change_threshold:
                        self.consecutive_misses += 1
                        self.rejected_size_change += 1
                        is_valid = False
                        rejection_reason = "size"
            
            # Draw detection based on validity
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if is_valid:
                # Valid detection
                valid_detections += 1
                self.valid_detections += 1
                
                # Update tracking state
                self.last_valid_position = (center_x, center_y)
                self.last_bbox_size = bbox_size
                self.consecutive_misses = 0
                
                # Add position to trajectory
                self.ball_positions.append((center_x, center_y))
                if len(self.ball_positions) > self.max_trajectory_length:
                    self.ball_positions.pop(0)
                
                # Draw bbox
                cv2.rectangle(
                    frame_with_detections,
                    (x1, y1), (x2, y2),
                    self.colors['bbox'],
                    2
                )
                
                # Draw center
                cv2.circle(
                    frame_with_detections,
                    (center_x, center_y),
                    5,
                    self.colors['ball'],
                    -1
                )
                
                # Confidence text with validation mark
                label = f"Ball: {confidence:.2f} ✓"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Text background
                cv2.rectangle(
                    frame_with_detections,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    self.colors['bbox'],
                    -1
                )
                
                # Text
                cv2.putText(
                    frame_with_detections,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.colors['text'],
                    2
                )
            else:
                # Invalid detection - draw as rejected
                cv2.rectangle(
                    frame_with_detections,
                    (x1, y1), (x2, y2),
                    (0, 0, 255),  # Red for rejected
                    1
                )
                
                # Mark as rejected
                label = f"Rejected ({rejection_reason}): {confidence:.2f}"
                cv2.putText(
                    frame_with_detections,
                    label,
                    (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
        
        # Update tracker if no valid detections
        if valid_detections == 0:
            self.update_no_detection()
        
        return frame_with_detections
    
    def draw_trajectory(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw ball trajectory.
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with drawn trajectory
        """
        if len(self.ball_positions) < 2:
            return frame
        
        frame_with_trajectory = frame.copy()
        
        # Draw trajectory lines
        for i in range(1, len(self.ball_positions)):
            # Transparency decreases for older points
            alpha = i / len(self.ball_positions)
            thickness = max(1, int(3 * alpha))
            
            cv2.line(
                frame_with_trajectory,
                self.ball_positions[i-1],
                self.ball_positions[i],
                self.colors['trajectory'],
                thickness
            )
        
        return frame_with_trajectory
    
    def create_label_studio_annotation(
        self, 
        frame: np.ndarray, 
        detections: List[Tuple[float, float, float, float, float]],
        frame_filename: str,
        frame_width: int,
        frame_height: int
    ) -> Optional[Dict[str, Any]]:
        """
        Create Label Studio annotation format for detected balls.
        
        Args:
            frame: Video frame
            detections: List of detections
            frame_filename: Frame filename
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Label Studio annotation or None if no detections
        """
        if not detections:
            return None
        
        # Create result items for each detection
        result_items = []
        for i, (x1, y1, x2, y2, confidence) in enumerate(detections):
            # Convert absolute coordinates to percentage
            x_percent = (x1 / frame_width) * 100
            y_percent = (y1 / frame_height) * 100
            width_percent = ((x2 - x1) / frame_width) * 100
            height_percent = ((y2 - y1) / frame_height) * 100
            
            result_item = {
                "original_width": frame_width,
                "original_height": frame_height,
                "image_rotation": 0,
                "value": {
                    "x": round(x_percent, 2),
                    "y": round(y_percent, 2),
                    "width": round(width_percent, 2),
                    "height": round(height_percent, 2),
                    "rotation": 0,
                    "rectanglelabels": ["Volleyball Ball"]
                },
                "id": f"ball_{i}",
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "manual"
            }
            result_items.append(result_item)
        
        # Create annotation
        annotation = {
            "id": self.annotation_id_counter,
            "completed_by": 1,
            "result": result_items,
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "draft_created_at": None,
            "lead_time": 0,
            "prediction": {},
            "result_count": len(result_items),
            "unique_id": f"task_{self.annotation_id_counter}",
            "import_id": None,
            "last_action": None,
            "bulk_created": False,
            "task": self.annotation_id_counter,
            "project": 1,
            "updated_by": 1,
            "parent_prediction": None,
            "parent_annotation": None,
            "last_created_by": None
        }
        
        # Create task
        task = {
            "id": self.annotation_id_counter,
            "annotations": [annotation],
            "file_upload": frame_filename,
            "drafts": [],
            "predictions": [],
            "data": {
                "image": f"/data/upload/1/{frame_filename}"
            },
            "meta": {},
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "inner_id": self.annotation_id_counter,
            "total_annotations": 1,
            "cancelled_annotations": 0,
            "total_predictions": 0,
            "comment_count": 0,
            "unresolved_comment_count": 0,
            "last_comment_updated_at": None,
            "project": 1,
            "updated_by": 1,
            "comment_authors": []
        }
        
        self.annotation_id_counter += 1
        return task
    
    def save_frame_with_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Tuple[float, float, float, float, float]], 
        output_dir: Path, 
        frame_num: int
    ) -> Optional[str]:
        """
        Save frame with detections as image file.
        
        Args:
            frame: Video frame
            detections: List of detections
            output_dir: Output directory for images
            frame_num: Frame number
            
        Returns:
            Filename if saved, None otherwise
        """
        if not detections:
            return None
        
        # Create unique filename
        frame_id = str(uuid.uuid4())[:8]
        filename = f"{frame_id}-frame_{frame_num:05d}.jpg"
        filepath = output_dir / filename
        
        # Save frame
        success = cv2.imwrite(str(filepath), frame)
        if not success:
            print(f"Failed to save frame: {filepath}")
            return None
        
        return filename
    
    def save_label_studio_annotations(self, output_path: Path) -> None:
        """
        Save Label Studio annotations to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if not self.label_studio_annotations:
            print("No annotations to save")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.label_studio_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.label_studio_annotations)} annotations to: {output_path}")
    
    def _process_batch(
        self,
        frame_batch: List[np.ndarray],
        frame_data_batch: List[Tuple[int, int]],
        out: cv2.VideoWriter,
        show_trajectory: bool,
        current_detections_count: int,
        processed_frames: int,
        max_frames: int,
        start_time: float,
        labelstudio_images_dir: Optional[Path] = None,
        visualization_frames_dir: Optional[Path] = None
    ) -> int:
        """
        Process batch of frames.
        
        Returns:
            Number of new detections
        """
        import torch
        
        # Batch detection
        batch_detections = self.detect_ball_batch(frame_batch)
        
        new_detections_count = 0
        
        # Process each frame from batch
        for i, (frame, detections) in enumerate(zip(frame_batch, batch_detections)):
            frame_num, _ = frame_data_batch[i]
            
            if detections:
                new_detections_count += len(detections)
                
                # Save frame with detections for Label Studio if directory is provided
                if labelstudio_images_dir is not None:
                    frame_height, frame_width = frame.shape[:2]
                    
                    # Save image
                    filename = self.save_frame_with_detections(
                        frame, detections, labelstudio_images_dir, frame_num
                    )
                    
                    # Create Label Studio annotation
                    if filename:
                        annotation = self.create_label_studio_annotation(
                            frame, detections, filename, frame_width, frame_height
                        )
                        if annotation:
                            self.label_studio_annotations.append(annotation)
            
            # Draw detections
            frame_with_detections = self.draw_detections(frame, detections)
            
            # Draw trajectory
            if show_trajectory:
                frame_with_detections = self.draw_trajectory(frame_with_detections)
            
            # Add frame info
            info_text = f"Frame: {frame_num} | Detections: {len(detections)} | GPU: {torch.cuda.is_available()}"
            cv2.putText(
                frame_with_detections,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Save visualization frame if requested and has detections
            if visualization_frames_dir is not None and detections:
                vis_frame_filename = f"frame_{frame_num:05d}_detections.jpg"
                vis_frame_path = visualization_frames_dir / vis_frame_filename
                cv2.imwrite(str(vis_frame_path), frame_with_detections)
            
            # Write frame
            out.write(frame_with_detections)
        
        # Show progress
        if processed_frames % (self.batch_size * 4) == 0:
            progress = (processed_frames / max_frames) * 100  # Fixed progress calculation
            elapsed = time.time() - start_time
            fps_current = processed_frames / elapsed if elapsed > 0 else 0
            total_detections = current_detections_count + new_detections_count
            print(f"   Progress: {progress:.1f}% | FPS: {fps_current:.1f} | Detections: {total_detections} | Batch: {len(frame_batch)}")
        
        return new_detections_count
    
    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
        show_trajectory: bool = True,
        skip_frames: int = 0,
        create_labelstudio_data: bool = False,
        labelstudio_output_dir: Optional[str] = None,
        max_duration_seconds: Optional[int] = None,
        save_visualization_frames: bool = True
    ) -> None:
        """
        Process video with ball detection.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            show_trajectory: Whether to show trajectory
            skip_frames: Number of frames to skip (for speedup)
            create_labelstudio_data: Whether to create Label Studio annotations
            labelstudio_output_dir: Output directory for Label Studio data
            max_duration_seconds: Maximum duration to process (None for full video)
            save_visualization_frames: Whether to save individual frames with bounding boxes
        """
        input_path = Path(input_video_path)
        output_path = Path(output_video_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_video_path}")
        
        # Create results folder
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup Label Studio directories if needed
        labelstudio_images_dir = None
        labelstudio_annotations_file = None
        if create_labelstudio_data:
            if not labelstudio_output_dir:
                labelstudio_output_dir = f"volleystat/data/labelstudio_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            labelstudio_base_dir = Path(labelstudio_output_dir)
            labelstudio_images_dir = labelstudio_base_dir / "images"
            labelstudio_images_dir.mkdir(parents=True, exist_ok=True)
            
            labelstudio_annotations_file = labelstudio_base_dir / "annotations.json"
            print(f"📝 Label Studio images will be saved to: {labelstudio_images_dir}")
            print(f"📝 Label Studio annotations will be saved to: {labelstudio_annotations_file}")
        
        # Setup visualization frames directory if needed
        visualization_frames_dir = None
        if save_visualization_frames:
            if labelstudio_output_dir:
                visualization_frames_dir = Path(labelstudio_output_dir) / "visualization_frames"
            else:
                visualization_frames_dir = output_path.parent / "visualization_frames"
            visualization_frames_dir.mkdir(parents=True, exist_ok=True)
            print(f"🖼️  Visualization frames will be saved to: {visualization_frames_dir}")
        
        print(f"📹 Processing video: {input_path}")
        print(f"💾 Result will be saved to: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
        
        # Get video parameters
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Apply duration limit if specified
        max_frames = total_frames
        if max_duration_seconds is not None:
            max_frames = min(total_frames, int(fps * max_duration_seconds))
        
        print(f"📊 Video parameters:")
        print(f"   • Size: {width}x{height}")
        print(f"   • FPS: {fps}")
        print(f"   • Total frames: {total_frames}")
        print(f"   • Total duration: {total_frames/fps:.1f} sec")
        if max_duration_seconds is not None:
            print(f"   • Processing frames: {max_frames}")
            print(f"   • Processing duration: {max_frames/fps:.1f} sec (limited to {max_duration_seconds} sec)")
        
        # Create writer for output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        # Statistics
        detections_count = 0
        processed_frames = 0
        start_time = time.time()
        
        print(f"🚀 Starting batch processing (batch size: {self.batch_size})...")
        
        frame_num = 0
        frame_batch = []
        frame_data_batch = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process last batch if exists
                if frame_batch:
                    self._process_batch(frame_batch, frame_data_batch, out, show_trajectory, 
                                      detections_count, processed_frames, max_frames, start_time, 
                                      labelstudio_images_dir, visualization_frames_dir)
                break
            
            frame_num += 1
            
            # Stop if reached max frames limit
            if frame_num > max_frames:
                # Process last batch if exists
                if frame_batch:
                    self._process_batch(frame_batch, frame_data_batch, out, show_trajectory, 
                                      detections_count, processed_frames, max_frames, start_time, 
                                      labelstudio_images_dir, visualization_frames_dir)
                break

            # Skip frames if needed
            if skip_frames > 0 and frame_num % (skip_frames + 1) != 0:
                continue

            processed_frames += 1

            # Add frame to batch
            frame_batch.append(frame.copy())
            frame_data_batch.append((frame_num, processed_frames))

            # When batch is full - process it
            if len(frame_batch) >= self.batch_size:
                detections_count += self._process_batch(
                    frame_batch, frame_data_batch, out, show_trajectory,
                    detections_count, processed_frames, max_frames, start_time, 
                    labelstudio_images_dir, visualization_frames_dir
                )
                frame_batch = []
                frame_data_batch = []
        
        # Close files
        cap.release()
        out.release()
        
        # Save Label Studio annotations if directory exists
        if labelstudio_images_dir is not None and self.label_studio_annotations:
            labelstudio_annotations_file = labelstudio_images_dir.parent / "annotations.json"
            self.save_label_studio_annotations(labelstudio_annotations_file)
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        
        print(f"✅ Processing completed!")
        print(f"📊 Statistics:")
        print(f"   • Processed frames: {processed_frames}")
        print(f"   • Found detections: {detections_count}")
        print(f"   • Label Studio annotations: {len(self.label_studio_annotations)}")
        print(f"   • Processing time: {total_time:.1f} sec")
        print(f"   • Average FPS: {avg_fps:.1f}")
        print(f"   🎾 Ball Filtering Statistics:")
        print(f"      - Total raw detections:     {self.total_detections}")
        print(f"      - Valid detections:         {self.valid_detections}")
        print(f"      - Rejected (low confidence): {self.rejected_low_confidence}")
        print(f"      - Rejected (velocity):      {self.rejected_velocity}")
        print(f"      - Rejected (size change):   {self.rejected_size_change}")
        if self.total_detections > 0:
            filter_rate = ((self.total_detections - self.valid_detections) / self.total_detections) * 100
            print(f"      - Filter rate:              {filter_rate:.1f}%")
        print(f"   • Result saved to: {output_path}")
        if labelstudio_images_dir is not None:
            print(f"   • Label Studio images: {labelstudio_images_dir}")
            print(f"   • Label Studio annotations: {labelstudio_images_dir.parent / 'annotations.json'}")
        if visualization_frames_dir is not None:
            # Count saved visualization frames
            viz_frames_count = len(list(visualization_frames_dir.glob("*.jpg"))) if visualization_frames_dir.exists() else 0
            print(f"   • Visualization frames: {viz_frames_count} saved to {visualization_frames_dir}")


def main():
    """Main function."""
    
    # Parameters
    model_path = "C:/Users/illya/Documents/volleyball_analitics/volleystat/models/yolov8_curated_training/yolov8n_volleyball_curated4/weights/epoch130.pt"
    input_video = r"C:/Users/illya/Videos/video_for_sharing/first_record/GX010378_splits/GX010378_part2.mp4"
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = f"C:/Users/illya/Documents/volleyball_analitics/volleystat/data/yolo_150_epoch_test_video/visualisation_{timestamp}.mp4"
    
    try:
        # Create detector
        detector = BallDetectorVideo(model_path)
        
        # Process video with optimized parameters
        detector.process_video(
            input_video_path=input_video,
            output_video_path=output_video,
            show_trajectory=True,
            skip_frames=0  # Process every frame
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 