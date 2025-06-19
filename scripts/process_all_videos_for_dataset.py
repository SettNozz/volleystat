#!/usr/bin/env python3
"""
Process All Videos for Dataset Preparation
Processes all videos in a folder structure to detect balls and prepare dataset for manual review.

Input folder structure:
C:/Users/illya/Videos/video_for_sharing/
├── folder1/
│   ├── video_part1.mp4
│   ├── video_part2.mp4
│   └── ...
├── folder2/
│   ├── video_part1.mp4
│   └── ...
└── ...

Output structure:
data/pipeline_result/
├── ball_generated_images/
│   ├── ball_images/
│   └── original_images/
├── detection_results/
│   ├── folder1/
│   │   ├── video_part1/
│   │   └── ...
│   └── ...
└── summary/
"""

import os
import cv2
import torch
import numpy as np
import json
import yaml
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse
from typing import List, Dict, Tuple, Optional
import time
import cProfile
import pstats
from functools import wraps

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.video_processor import VideoProcessor, create_support_image_from_video
from src.evaluation.evaluator import mask_to_obb_coordinates
from configs.config import YOLO_MODEL_PATH, MODEL_LOAD_PATH


def profile_function(func):
    """Decorator to profile a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Save stats to file
        stats_file = f"profile_{func.__name__}_{int(time.time())}.prof"
        profiler.dump_stats(stats_file)
        
        # Print top 10 time-consuming functions
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print(f"\n📊 Profiling results for {func.__name__}:")
        stats.print_stats(10)
        
        return result
    return wrapper


class VideoDatasetProcessor:
    """Process all videos in a folder structure for ball detection and dataset preparation."""
    
    def __init__(self, input_folder: str, output_base: str, yolo_model_path: str = None, 
                 unet_model_path: str = None, device: str = 'auto'):
        """
        Initialize the video dataset processor.
        
        Args:
            input_folder: Root folder containing video folders
            output_base: Base output directory
            yolo_model_path: Path to YOLO model for person detection
            unet_model_path: Path to U-Net model for ball segmentation
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.input_folder = Path(input_folder)
        self.output_base = Path(output_base)
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"🔧 Using device: {self.device}")
        
        # Initialize video processor
        self.processor = VideoProcessor(
            yolo_model_path=yolo_model_path or YOLO_MODEL_PATH,
            unet_model_path=unet_model_path or MODEL_LOAD_PATH,
            device=self.device
        )
        
        # Create output directories
        self._create_output_dirs()
        
        # Statistics
        self.stats = {
            'total_folders': 0,
            'total_videos': 0,
            'processed_videos': 0,
            'skipped_videos': 0,
            'total_detections': 0,
            'total_ball_crops': 0,
            'processing_times': [],
            'errors': []
        }
        
        # Performance optimizations
        self.batch_size = 4  # Process multiple frames at once
        self.frame_interval = 10  # Process every 10th frame instead of 5th
        
    def _create_output_dirs(self):
        """Create output directory structure."""
        dirs = [
            self.output_base / "ball_generated_images" / "ball_images",
            self.output_base / "ball_generated_images" / "original_images",
            self.output_base / "detection_results",
            self.output_base / "summary",
            self.output_base / "yolo_dataset"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
    
    def _get_video_files(self) -> Dict[str, List[Path]]:
        """Get all video files organized by folder."""
        video_files = {}
        
        if not self.input_folder.exists():
            raise ValueError(f"Input folder does not exist: {self.input_folder}")
        
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        
        for folder_path in self.input_folder.iterdir():
            if folder_path.is_dir():
                folder_name = folder_path.name
                video_files[folder_name] = []
                
                for video_file in folder_path.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                        video_files[folder_name].append(video_file)
                
                if video_files[folder_name]:
                    print(f"📁 Found {len(video_files[folder_name])} videos in {folder_name}")
                else:
                    print(f"⚠️ No videos found in {folder_name}")
        
        return video_files
    
    def _is_video_already_processed(self, video_path: Path, folder_name: str, video_name: str) -> bool:
        """Check if video has already been processed."""
        # Check if summary file exists
        summary_path = self.output_base / "detection_results" / folder_name / video_name / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    if summary.get('success', False):
                        print(f"⏭️ Skipping {folder_name}/{video_name} - already processed")
                        return True
            except:
                pass
        
        return False
    
    def _crop_ball_from_image(self, frame: np.ndarray, ball_bbox: Tuple[int, int, int, int], 
                             padding: int = 10) -> Optional[np.ndarray]:
        """Crop ball from image using bounding box with padding."""
        x1, y1, x2, y2 = ball_bbox
        h, w = frame.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop the ball
        ball_crop = frame[y1:y2, x1:x2]
        
        if ball_crop.size == 0:
            return None
        
        return ball_crop
    
    @profile_function
    def _process_single_video(self, video_path: Path, folder_name: str, 
                             video_name: str) -> Dict:
        """Process a single video for ball detection."""
        print(f"\n🎬 Processing: {folder_name}/{video_name}")
        
        # Check if already processed
        if self._is_video_already_processed(video_path, folder_name, video_name):
            self.stats['skipped_videos'] += 1
            return {'success': True, 'skipped': True, 'reason': 'already_processed'}
        
        start_time = time.time()
        
        # Create output directories for this video
        video_output_dir = self.output_base / "detection_results" / folder_name / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO dataset directories
        yolo_images_dir = self.output_base / "yolo_dataset" / "images"
        yolo_labels_dir = self.output_base / "yolo_dataset" / "labels"
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ball images directory
        ball_images_dir = self.output_base / "ball_generated_images" / "ball_images"
        original_images_dir = self.output_base / "ball_generated_images" / "original_images"
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            error_msg = f"Failed to open video: {video_path}"
            print(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)
            return {'success': False, 'error': error_msg}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  📐 Resolution: {width}x{height}")
        print(f"  ⏱️ Duration: {duration:.1f}s ({total_frames} frames)")
        
        # Create support image from first 10 seconds
        support_path = video_output_dir / "support_ball.jpg"
        support_image_path = create_support_image_from_video(
            str(video_path), 
            str(support_path),
            start_sec=0,
            duration_sec=10,
            model_path=YOLO_MODEL_PATH
        )
        
        if support_image_path:
            self.processor.set_support_image(support_image_path)
            print(f"  ✅ Created support image: {support_path}")
        else:
            print(f"  ⚠️ Could not create support image")
        
        # Process frames with batch processing
        frame_count = 0
        ball_detections = 0
        saved_images = 0
        saved_ball_crops = 0
        
        print(f"  🔄 Processing frames (every {self.frame_interval}th frame)...")
        
        # Pre-allocate batch tensors for better performance
        batch_frames = []
        batch_indices = []
        
        for frame_idx in tqdm(range(0, total_frames, self.frame_interval), desc=f"  {video_name}"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add to batch
            batch_frames.append(frame)
            batch_indices.append(frame_idx)
            
            # Process batch when full or at end
            if len(batch_frames) >= self.batch_size or frame_idx + self.frame_interval >= total_frames:
                # Process batch
                batch_results = self._process_frame_batch(batch_frames, batch_indices, 
                                                        folder_name, video_name,
                                                        yolo_images_dir, yolo_labels_dir,
                                                        ball_images_dir, original_images_dir)
                
                # Update statistics
                ball_detections += batch_results['detections']
                saved_images += batch_results['saved_images']
                saved_ball_crops += batch_results['saved_crops']
                frame_count += len(batch_frames)
                
                # Clear batch
                batch_frames = []
                batch_indices = []
        
        # Cleanup
        cap.release()
        
        processing_time = time.time() - start_time
        
        # Create video summary
        video_summary = {
            'video_path': str(video_path),
            'folder_name': folder_name,
            'video_name': video_name,
            'resolution': f"{width}x{height}",
            'duration': duration,
            'total_frames': total_frames,
            'processed_frames': frame_count,
            'ball_detections': ball_detections,
            'saved_images': saved_images,
            'saved_ball_crops': saved_ball_crops,
            'processing_time': processing_time,
            'success': True
        }
        
        # Save video summary
        summary_path = video_output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(video_summary, f, indent=2)
        
        self.stats['processing_times'].append(processing_time)
        
        print(f"  ✅ Completed: {ball_detections} ball detections, {saved_images} images saved")
        print(f"  ⏱️ Processing time: {processing_time:.1f}s")
        
        return video_summary
    
    def _process_frame_batch(self, frames: List[np.ndarray], frame_indices: List[int],
                           folder_name: str, video_name: str,
                           yolo_images_dir: Path, yolo_labels_dir: Path,
                           ball_images_dir: Path, original_images_dir: Path) -> Dict:
        """Process a batch of frames for better performance."""
        results = {
            'detections': 0,
            'saved_images': 0,
            'saved_crops': 0
        }
        
        # Process each frame in batch (could be optimized further with true batch processing)
        for frame, frame_idx in zip(frames, frame_indices):
            # Segment ball
            ball_mask, _ = self.processor.segment_ball(frame)
            
            if ball_mask is not None:
                # Find ball bounding box
                ball_bbox = self.processor.find_ball_bounding_box(ball_mask)
                
                if ball_bbox:
                    results['detections'] += 1
                    
                    # Generate unique image name
                    image_name = f"{folder_name}_{video_name}_frame_{frame_idx:06d}"
                    
                    # Save original image
                    image_path = original_images_dir / f"{image_name}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    
                    # Crop and save ball image
                    ball_crop = self._crop_ball_from_image(frame, ball_bbox, padding=10)
                    if ball_crop is not None:
                        ball_crop_name = f"{image_name}_ball.jpg"
                        ball_crop_path = ball_images_dir / ball_crop_name
                        cv2.imwrite(str(ball_crop_path), ball_crop)
                        results['saved_crops'] += 1
                    
                    # Save for YOLO dataset
                    yolo_image_path = yolo_images_dir / f"{image_name}.jpg"
                    cv2.imwrite(str(yolo_image_path), frame)
                    
                    # Generate OBB coordinates from mask
                    obb_coordinates = mask_to_obb_coordinates(ball_mask)
                    
                    # Save YOLO label file
                    label_path = yolo_labels_dir / f"{image_name}.txt"
                    with open(label_path, "w") as f:
                        for coord in obb_coordinates:
                            f.write(" ".join(map(str, coord)) + "\n")
                    
                    results['saved_images'] += 1
        
        return results
    
    def process_all_videos(self, start_sec: int = 0, duration_sec: int = None):
        """Process all videos in the input folder structure."""
        print("🏐 Video Dataset Processor")
        print("=" * 50)
        print(f"📁 Input folder: {self.input_folder}")
        print(f"📁 Output folder: {self.output_base}")
        print(f"⚡ Performance optimizations: batch_size={self.batch_size}, frame_interval={self.frame_interval}")
        
        # Get all video files
        video_files = self._get_video_files()
        
        if not video_files:
            print("❌ No video files found!")
            return
        
        self.stats['total_folders'] = len(video_files)
        self.stats['total_videos'] = sum(len(videos) for videos in video_files.values())
        
        print(f"\n📊 Found {self.stats['total_folders']} folders with {self.stats['total_videos']} videos total")
        
        # Process each folder
        all_summaries = []
        
        for folder_name, videos in video_files.items():
            print(f"\n📁 Processing folder: {folder_name}")
            
            folder_summaries = []
            for video_path in videos:
                video_name = video_path.stem
                
                try:
                    summary = self._process_single_video(video_path, folder_name, video_name)
                    folder_summaries.append(summary)
                    
                    if summary['success'] and not summary.get('skipped', False):
                        self.stats['processed_videos'] += 1
                        self.stats['total_detections'] += summary['ball_detections']
                        self.stats['total_ball_crops'] += summary['saved_ball_crops']
                    
                except Exception as e:
                    error_msg = f"Error processing {video_name}: {str(e)}"
                    print(f"❌ {error_msg}")
                    self.stats['errors'].append(error_msg)
                    folder_summaries.append({
                        'video_path': str(video_path),
                        'folder_name': folder_name,
                        'video_name': video_name,
                        'success': False,
                        'error': str(e)
                    })
            
            all_summaries.extend(folder_summaries)
        
        # Create YOLO dataset config
        self._create_yolo_dataset_config()
        
        # Create overall summary
        self._create_overall_summary(all_summaries)
        
        # Print final statistics
        self._print_final_stats()
    
    def _create_yolo_dataset_config(self):
        """Create YOLO dataset configuration file."""
        yolo_dir = self.output_base / "yolo_dataset"
        
        data_yaml = {
            "path": str(yolo_dir.absolute()),
            "train": "images",
            "val": "images",
            "nc": 1,  # number of classes
            "names": ["ball"]  # class names
        }
        
        yaml_path = yolo_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"📄 Created YOLO dataset config: {yaml_path}")
    
    def _create_overall_summary(self, all_summaries: List[Dict]):
        """Create overall summary of all processing."""
        summary = {
            'processing_date': datetime.now().isoformat(),
            'input_folder': str(self.input_folder),
            'output_folder': str(self.output_base),
            'statistics': self.stats,
            'video_summaries': all_summaries,
            'successful_videos': [s for s in all_summaries if s['success']],
            'failed_videos': [s for s in all_summaries if not s['success']],
            'skipped_videos': [s for s in all_summaries if s.get('skipped', False)]
        }
        
        summary_path = self.output_base / "summary" / "overall_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Created overall summary: {summary_path}")
    
    def _print_final_stats(self):
        """Print final processing statistics."""
        print("\n" + "=" * 50)
        print("📊 FINAL STATISTICS")
        print("=" * 50)
        print(f"📁 Total folders processed: {self.stats['total_folders']}")
        print(f"🎬 Total videos found: {self.stats['total_videos']}")
        print(f"✅ Successfully processed: {self.stats['processed_videos']}")
        print(f"⏭️ Skipped (already processed): {self.stats['skipped_videos']}")
        print(f"🏐 Total ball detections: {self.stats['total_detections']}")
        print(f"🖼️ Total ball crops saved: {self.stats['total_ball_crops']}")
        
        if self.stats['processing_times']:
            avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            print(f"⏱️ Average processing time per video: {avg_time:.1f}s")
        
        if self.stats['errors']:
            print(f"\n❌ Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.stats['errors']) > 5:
                print(f"  ... and {len(self.stats['errors']) - 5} more")
        
        print(f"\n📁 Output saved to: {self.output_base}")
        print(f"🎯 YOLO dataset ready at: {self.output_base / 'yolo_dataset'}")
        print(f"🏀 Ball images saved at: {self.output_base / 'ball_generated_images'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process all videos for ball detection dataset")
    parser.add_argument("--input", "-i", 
                       default=r"C:/Users/illya/Videos/video_for_sharing",
                       help="Input folder containing video folders")
    parser.add_argument("--output", "-o", 
                       default="data/pipeline_result",
                       help="Output base directory")
    parser.add_argument("--yolo-model", 
                       default=YOLO_MODEL_PATH,
                       help="Path to YOLO model")
    parser.add_argument("--unet-model", 
                       default=MODEL_LOAD_PATH,
                       help="Path to U-Net model")
    parser.add_argument("--device", 
                       default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for processing")
    parser.add_argument("--batch-size", 
                       type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--frame-interval", 
                       type=int, default=10,
                       help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = VideoDatasetProcessor(
        input_folder=args.input,
        output_base=args.output,
        yolo_model_path=args.yolo_model,
        unet_model_path=args.unet_model,
        device=args.device
    )
    
    # Set performance parameters
    processor.batch_size = args.batch_size
    processor.frame_interval = args.frame_interval
    
    processor.process_all_videos()


if __name__ == "__main__":
    main() 