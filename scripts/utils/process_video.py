#!/usr/bin/env python3
"""
Simple Video Processing Script
Processes video from 00:00:30 for 1 minute with person detection and ball segmentation
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.video_processor import VideoProcessor, create_support_image_from_video
from configs.config import *


def process_video_simple(input_video_path, output_video_path=None):
    """
    Process video with the exact specifications requested:
    - Start time: 00:00:30 (30 seconds)
    - Duration: 1 minute (60 seconds)
    - Person detection using YOLO
    - Ball segmentation using U-Net
    - Bounding box visualization
    """
    
    # Validate input video
    if not os.path.exists(input_video_path):
        print(f"❌ Input video not found: {input_video_path}")
        return False
    
    # Set output path if not provided
    if output_video_path is None:
        input_path = Path(input_video_path)
        output_video_path = input_path.parent / f"{input_path.stem}_processed.avi"
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    print("🏐 Volleyball Video Processing Pipeline")
    print("=" * 50)
    print(f"📹 Input video: {input_video_path}")
    print(f"📹 Output video: {output_video_path}")
    print(f"⏰ Time range: 00:00:30 - 00:01:30 (30s - 90s)")
    print(f"🎯 Person detection: Enabled (YOLO)")
    print(f"⚽ Ball segmentation: Enabled (U-Net)")
    
    # Initialize video processor
    print("\n🔧 Initializing video processor...")
    processor = VideoProcessor(
        yolo_model_path=YOLO_MODEL_PATH,
        unet_model_path=MODEL_LOAD_PATH if os.path.exists(MODEL_LOAD_PATH) else None
    )
    
    # Create support image automatically from the video
    print("\n🔍 Creating support image for ball segmentation...")
    support_path = Path(output_video_path).parent / "support_ball.jpg"
    support_image_path = create_support_image_from_video(
        input_video_path, 
        str(support_path),
        30,  # Start at 30 seconds
        10,  # Search in first 10 seconds of the segment
        YOLO_MODEL_PATH
    )
    
    if not support_image_path:
        print("⚠️ Could not create support image, ball segmentation will be disabled")
        show_ball = False
    else:
        show_ball = True
    
    # Process video
    print(f"\n🔄 Processing video...")
    try:
        results = processor.process_video(
            input_video_path=input_video_path,
            output_video_path=str(output_video_path),
            start_sec=30,  # Start at 30 seconds
            duration_sec=60,  # Process for 1 minute
            support_image_path=support_image_path,
            show_persons=True,
            show_ball=show_ball
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Final Statistics:")
        print(f"   - Frames processed: {results['frames_processed']}")
        print(f"   - Person detections: {results['person_detections']}")
        print(f"   - Ball detections: {results['ball_detections']}")
        print(f"   - Output video: {results['output_path']}")
        
        if support_image_path:
            print(f"   - Support image: {support_image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False


def main():
    """Main function with command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <input_video_path> [output_video_path]")
        print("\nExample:")
        print("  python process_video.py video.mp4")
        print("  python process_video.py video.mp4 output.avi")
        return
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = process_video_simple(input_video, output_video)
    
    if success:
        print("\n🎉 Video processing completed successfully!")
    else:
        print("\n💥 Video processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 