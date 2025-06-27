#!/usr/bin/env python3
"""
Volleyball ball detection on video
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
import time
from datetime import datetime
import torch


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
        self.confidence_threshold = 0.3
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
        for result in results:
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append((x1, y1, x2, y2, confidence))
            batch_detections.append(detections)
        
        return batch_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple[float, float, float, float, float]]) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame: Video frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        frame_with_detections = frame.copy()
        
        for x1, y1, x2, y2, confidence in detections:
            # Bbox coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bbox
            cv2.rectangle(
                frame_with_detections,
                (x1, y1), (x2, y2),
                self.colors['bbox'],
                2
            )
            
            # Ball center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Draw center
            cv2.circle(
                frame_with_detections,
                (center_x, center_y),
                5,
                self.colors['ball'],
                -1
            )
            
            # Add position to trajectory
            self.ball_positions.append((center_x, center_y))
            if len(self.ball_positions) > self.max_trajectory_length:
                self.ball_positions.pop(0)
            
            # Confidence text
            label = f"Ball: {confidence:.2f}"
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
    
    def _process_batch(
        self,
        frame_batch: List[np.ndarray],
        frame_data_batch: List[Tuple[int, int]],
        out: cv2.VideoWriter,
        show_trajectory: bool,
        current_detections_count: int,
        processed_frames: int,
        total_frames: int,
        start_time: float
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
            
            # Write frame
            out.write(frame_with_detections)
        
        # Show progress
        if processed_frames % (self.batch_size * 4) == 0:
            progress = (processed_frames / (total_frames // 10)) * 100  # Account for skip_frames
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
        skip_frames: int = 0
    ) -> None:
        """
        Process video with ball detection.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            show_trajectory: Whether to show trajectory
            skip_frames: Number of frames to skip (for speedup)
        """
        input_path = Path(input_video_path)
        output_path = Path(output_video_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_video_path}")
        
        # Create results folder
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        
        print(f"📊 Video parameters:")
        print(f"   • Size: {width}x{height}")
        print(f"   • FPS: {fps}")
        print(f"   • Frames: {total_frames}")
        print(f"   • Duration: {total_frames/fps:.1f} sec")
        
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
                                      detections_count, processed_frames, total_frames, start_time)
                break
            
            frame_num += 1
            
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
                    detections_count, processed_frames, total_frames, start_time
                )
                frame_batch = []
                frame_data_batch = []
        
        # Close files
        cap.release()
        out.release()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        
        print(f"✅ Processing completed!")
        print(f"📊 Statistics:")
        print(f"   • Processed frames: {processed_frames}")
        print(f"   • Found detections: {detections_count}")
        print(f"   • Processing time: {total_time:.1f} sec")
        print(f"   • Average FPS: {avg_fps:.1f}")
        print(f"   • Result saved to: {output_path}")


def main():
    """Main function."""
    
    # Parameters
    model_path = "volleystat/models/yolov8_volleyball_training/yolov8n_volleyball/weights/best.pt"
    input_video = r"C:\Users\illya\Videos\tmp_second_game\GX020380.mp4"
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = f"volleystat/results/video/GX020380_ball_detection_{timestamp}.mp4"
    
    try:
        # Create detector
        detector = BallDetectorVideo(model_path)
        
        # Process video with optimized parameters
        detector.process_video(
            input_video_path=input_video,
            output_video_path=output_video,
            show_trajectory=True,
            skip_frames=9  # Take every 10th frame for better coverage at high speed
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 