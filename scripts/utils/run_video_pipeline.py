#!/usr/bin/env python3
"""
Volleyball Video Processing Pipeline
Processes video with person detection and ball segmentation
"""

import os
import sys
import argparse
from pathlib import Path

from src.pipeline.video_processor import VideoProcessor, create_support_image_from_video
from configs.config import *


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description='Volleyball Video Processing Pipeline')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Output video path (default: input_name_processed.avi)')
    parser.add_argument('--start-time', '-s', type=int, default=30, help='Start time in seconds (default: 30)')
    parser.add_argument('--duration', '-d', type=int, default=60, help='Duration in seconds (default: 60)')
    parser.add_argument('--yolo-model', default=YOLO_MODEL_PATH, help='Path to YOLO model')
    parser.add_argument('--unet-model', default=MODEL_LOAD_PATH, help='Path to U-Net model')
    parser.add_argument('--support-image', help='Path to support image for ball segmentation')
    parser.add_argument('--auto-support', action='store_true', help='Automatically create support image from video')
    parser.add_argument('--no-persons', action='store_true', help='Disable person detection')
    parser.add_argument('--no-ball', action='store_true', help='Disable ball segmentation')
    
    args = parser.parse_args()
    
    # Validate input video
    if not os.path.exists(args.input_video):
        print(f"❌ Input video not found: {args.input_video}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_video)
        output_path = input_path.parent / f"{input_path.stem}_processed.avi"
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("🏐 Volleyball Video Processing Pipeline")
    print("=" * 50)
    print(f"📹 Input video: {args.input_video}")
    print(f"📹 Output video: {output_path}")
    print(f"⏰ Time range: {args.start_time}s - {args.start_time + args.duration}s")
    print(f"🎯 Person detection: {'Disabled' if args.no_persons else 'Enabled'}")
    print(f"⚽ Ball segmentation: {'Disabled' if args.no_ball else 'Enabled'}")
    
    # Initialize video processor
    print("\n🔧 Initializing video processor...")
    processor = VideoProcessor(
        yolo_model_path=args.yolo_model,
        unet_model_path=args.unet_model if not args.no_ball else None
    )
    
    # Handle support image for ball segmentation
    support_image_path = None
    if not args.no_ball:
        if args.support_image:
            support_image_path = args.support_image
        elif args.auto_support:
            print("\n🔍 Creating support image automatically...")
            support_path = Path(output_path).parent / "support_ball.jpg"
            support_image_path = create_support_image_from_video(
                args.input_video, 
                str(support_path),
                args.start_time, 
                10,  # Search in first 10 seconds
                args.yolo_model
            )
        else:
            print("⚠️ No support image provided for ball segmentation")
            print("   Use --support-image or --auto-support")
            args.no_ball = True
    
    # Process video
    print(f"\n🔄 Processing video...")
    try:
        results = processor.process_video(
            input_video_path=args.input_video,
            output_video_path=str(output_path),
            start_sec=args.start_time,
            duration_sec=args.duration,
            support_image_path=support_image_path,
            show_persons=not args.no_persons,
            show_ball=not args.no_ball
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Final Statistics:")
        print(f"   - Frames processed: {results['frames_processed']}")
        print(f"   - Person detections: {results['person_detections']}")
        print(f"   - Ball detections: {results['ball_detections']}")
        print(f"   - Output video: {results['output_path']}")
        
        if support_image_path:
            print(f"   - Support image: {support_image_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        sys.exit(1)


def run_example():
    """Run with default settings for testing."""
    print("🏐 Running example pipeline...")
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Example video not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in configs/config.py or provide a video file")
        return
    
    # Set output path
    output_path = os.path.join(VIDEO_OUTPUT_PATH, "example_processed.avi")
    os.makedirs(VIDEO_OUTPUT_PATH, exist_ok=True)
    
    # Initialize processor
    processor = VideoProcessor(
        yolo_model_path=YOLO_MODEL_PATH,
        unet_model_path=MODEL_LOAD_PATH if os.path.exists(MODEL_LOAD_PATH) else None
    )
    
    # Create support image automatically
    support_path = os.path.join(VIDEO_OUTPUT_PATH, "support_ball.jpg")
    support_image_path = create_support_image_from_video(
        VIDEO_PATH, 
        support_path,
        VIDEO_START_SEC, 
        10,
        YOLO_MODEL_PATH
    )
    
    # Process video
    results = processor.process_video(
        input_video_path=VIDEO_PATH,
        output_video_path=output_path,
        start_sec=VIDEO_START_SEC,
        duration_sec=VIDEO_DURATION_SEC,
        support_image_path=support_image_path,
        show_persons=True,
        show_ball=True
    )
    
    print(f"✅ Example completed!")
    print(f"📹 Output: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run example
        run_example()
    else:
        # Parse arguments and run main pipeline
        main() 