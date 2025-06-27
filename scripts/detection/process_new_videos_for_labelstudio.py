#!/usr/bin/env python3
"""
Process new videos with YOLO model and create Label Studio annotations
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import os
import glob

from detect_ball_on_video import BallDetectorVideo


def process_videos_for_labelstudio(
    model_path: str,
    video_paths: List[str],
    output_base_dir: Optional[str] = None,
    confidence_threshold: float = 0.3,
    skip_frames: int = 9,
    create_video_output: bool = False
) -> None:
    """
    Process multiple videos and create Label Studio annotations.
    
    Args:
        model_path: Path to trained YOLO model
        video_paths: List of video file paths
        output_base_dir: Base output directory
        confidence_threshold: Detection confidence threshold
        skip_frames: Number of frames to skip between processing
        create_video_output: Whether to create annotated video output
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not output_base_dir:
        output_base_dir = f"volleystat/data/labelstudio_batch_{timestamp}"
    
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Processing {len(video_paths)} videos for Label Studio")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Model: {model_path}")
    print(f"⚡ Confidence threshold: {confidence_threshold}")
    print(f"⏭️  Skip frames: {skip_frames}")
    
    # Create detector
    detector = BallDetectorVideo(model_path)
    detector.confidence_threshold = confidence_threshold
    
    total_annotations = 0
    processed_videos = 0
    
    for i, video_path in enumerate(video_paths, 1):
        video_file = Path(video_path)
        
        if not video_file.exists():
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        print(f"\n📹 Processing video {i}/{len(video_paths)}: {video_file.name}")
        
        # Create video-specific output directory
        video_output_dir = output_dir / f"video_{i:03d}_{video_file.stem}"
        
        try:
            # Create output video path if needed
            output_video_path = None
            if create_video_output:
                output_video_path = video_output_dir / f"{video_file.stem}_annotated.mp4"
            else:
                # Create a dummy video path (won't be used)
                output_video_path = video_output_dir / "dummy.mp4"
            
            # Process video
            detector.process_video(
                input_video_path=str(video_file),
                output_video_path=str(output_video_path),
                show_trajectory=True,
                skip_frames=skip_frames,
                create_labelstudio_data=True,
                labelstudio_output_dir=str(video_output_dir)
            )
            
            video_annotations = len(detector.label_studio_annotations)
            total_annotations += video_annotations
            processed_videos += 1
            
            print(f"✅ Processed {video_file.name}: {video_annotations} annotations")
            
            # Reset annotations for next video
            detector.label_studio_annotations = []
            detector.annotation_id_counter = 1
            
            # Remove dummy video if not needed
            if not create_video_output and output_video_path.exists():
                output_video_path.unlink()
                
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {e}")
            continue
    
    print(f"\n🎉 Batch processing completed!")
    print(f"📊 Statistics:")
    print(f"   • Processed videos: {processed_videos}/{len(video_paths)}")
    print(f"   • Total annotations: {total_annotations}")
    print(f"   • Output directory: {output_dir}")
    
    # Create summary file
    summary_file = output_dir / "processing_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Video Processing Summary\n")
        f.write(f"========================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n")
        f.write(f"Skip frames: {skip_frames}\n")
        f.write(f"Processed videos: {processed_videos}/{len(video_paths)}\n")
        f.write(f"Total annotations: {total_annotations}\n")
        f.write(f"Output directory: {output_dir}\n")
    
    print(f"📄 Summary saved to: {summary_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process videos for Label Studio annotations")
    
    parser.add_argument(
        "--model", 
        type=str,
        default="volleystat/models/yolov8_volleyball_training/yolov8n_volleyball/weights/best.pt",
        help="Path to YOLO model"
    )
    
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        help="Video file paths or glob patterns"
    )
    
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing videos (alternative to --videos)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for Label Studio data"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)"
    )
    
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=9,
        help="Number of frames to skip between processing (default: 9)"
    )
    
    parser.add_argument(
        "--create-videos",
        action="store_true",
        help="Create annotated video outputs"
    )
    
    args = parser.parse_args()
    
    # Collect video paths
    video_paths = []
    
    if args.videos:
        # From command line arguments
        for video_pattern in args.videos:
            if "*" in video_pattern or "?" in video_pattern:
                # Glob pattern
                video_paths.extend(glob.glob(video_pattern))
            else:
                # Direct path
                video_paths.append(video_pattern)
    elif args.video_dir:
        # From directory
        video_dir = Path(args.video_dir)
        if video_dir.exists():
            # Common video extensions
            extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv"]
            for ext in extensions:
                video_paths.extend(glob.glob(str(video_dir / ext)))
        else:
            print(f"❌ Video directory not found: {args.video_dir}")
            return 1
    else:
        # Default example
        video_paths = [r"C:\Users\illya\Videos\tmp_second_game\GX020380.mp4"]
        print("ℹ️  Using default video path. Use --videos or --video-dir to specify custom paths.")
    
    if not video_paths:
        print("❌ No video files found!")
        return 1
    
    print(f"🎬 Found {len(video_paths)} video files")
    
    try:
        process_videos_for_labelstudio(
            model_path=args.model,
            video_paths=video_paths,
            output_base_dir=args.output_dir,
            confidence_threshold=args.confidence,
            skip_frames=args.skip_frames,
            create_video_output=args.create_videos
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 