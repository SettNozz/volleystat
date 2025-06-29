#!/usr/bin/env python3
"""
Process new videos with YOLO model and create Label Studio annotations
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
import glob
import json

from detect_ball_on_video import BallDetectorVideo


def is_video_already_processed(video_file: Path, output_dir: Path, video_index: int) -> bool:
    """
    Check if video has already been processed.
    
    Args:
        video_file: Path to video file
        output_dir: Base output directory
        video_index: Video index in processing order
        
    Returns:
        True if video was already processed, False otherwise
    """
    # Create expected output directory name
    video_output_dir = output_dir / f"video_{video_index:03d}_{video_file.stem}"
    
    if not video_output_dir.exists():
        return False
    
    # Check for essential files that indicate successful processing
    required_files = [
        video_output_dir / f"{video_file.stem}_ball_detection.mp4",  # Output video
        video_output_dir / "annotations.json"  # Annotations file
    ]
    
    # Check if all required files exist
    for required_file in required_files:
        if not required_file.exists():
            return False
    
    # Additional check: annotations file should not be empty
    annotations_file = video_output_dir / "annotations.json"
    try:
        if annotations_file.stat().st_size < 10:  # Very small file, likely empty or corrupted
            return False
    except:
        return False
    
    return True


def get_processed_video_stats(video_file: Path, output_dir: Path, video_index: int) -> Optional[Dict[str, Any]]:
    """
    Get statistics from already processed video.
    
    Args:
        video_file: Path to video file
        output_dir: Base output directory 
        video_index: Video index in processing order
        
    Returns:
        Dictionary with processing stats or None if not found
    """
    video_output_dir = output_dir / f"video_{video_index:03d}_{video_file.stem}"
    annotations_file = video_output_dir / "annotations.json"
    
    if not annotations_file.exists():
        return None
    
    try:
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        return {
            'annotations_count': len(annotations),
            'output_dir': video_output_dir,
            'annotations_file': annotations_file
        }
    except:
        return None


def process_videos_for_labelstudio(
    model_path: str,
    video_paths: List[str],
    output_base_dir: Optional[str] = None,
    confidence_threshold: float = 0.1,
    skip_frames: int = 0,
    max_duration_seconds: Optional[int] = None,
    save_visualization_frames: bool = True,
    force_reprocess: bool = False
) -> None:
    """
    Process multiple videos and create Label Studio annotations.
    
    Args:
        model_path: Path to trained YOLO model
        video_paths: List of video file paths
        output_base_dir: Base output directory
        confidence_threshold: Detection confidence threshold
        skip_frames: Number of frames to skip between processing
        max_duration_seconds: Maximum duration to process per video (None for full video)
        save_visualization_frames: Whether to save individual frames with bounding boxes
        force_reprocess: Force reprocessing of already processed videos
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
    if max_duration_seconds:
        print(f"⏱️  Max duration per video: {max_duration_seconds} seconds")
    else:
        print(f"⏱️  Processing: Full video (no time limit)")
    print(f"🖼️  Save visualization frames: {'Yes' if save_visualization_frames else 'No'}")
    print(f"🔄 Force reprocessing: {'Yes' if force_reprocess else 'No (skip already processed)'}")
    
    # Create detector
    detector = BallDetectorVideo(model_path)
    detector.confidence_threshold = confidence_threshold
    
    total_annotations = 0
    processed_videos = 0
    skipped_videos = 0
    
    for i, video_path in enumerate(video_paths, 1):
        video_file = Path(video_path)
        
        if not video_file.exists():
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        print(f"\n📹 Processing video {i}/{len(video_paths)}: {video_file.name}")
        
        # Check if video was already processed
        if not force_reprocess and is_video_already_processed(video_file, output_dir, i):
            # Get existing stats
            existing_stats = get_processed_video_stats(video_file, output_dir, i)
            if existing_stats:
                print(f"✅ Video already processed - skipping ({existing_stats['annotations_count']} annotations found)")
                total_annotations += existing_stats['annotations_count']
                skipped_videos += 1
                continue
            else:
                print(f"⚠️  Video marked as processed but stats unavailable - reprocessing...")
        
        # Create video-specific output directory
        video_output_dir = output_dir / f"video_{i:03d}_{video_file.stem}"
        
        try:
            # Create output video path - always create annotated video
            output_video_path = video_output_dir / f"{video_file.stem}_ball_detection.mp4"
            
            # Process video
            detector.process_video(
                input_video_path=str(video_file),
                output_video_path=str(output_video_path),
                show_trajectory=True,
                skip_frames=skip_frames,
                create_labelstudio_data=True,
                labelstudio_output_dir=str(video_output_dir),
                max_duration_seconds=max_duration_seconds,
                save_visualization_frames=save_visualization_frames
            )
            
            video_annotations = len(detector.label_studio_annotations)
            total_annotations += video_annotations
            processed_videos += 1
            
            print(f"✅ Processed {video_file.name}: {video_annotations} annotations")
            
            # Reset annotations for next video
            detector.label_studio_annotations = []
            detector.annotation_id_counter = 1
                
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {e}")
            continue
    
    print(f"\n🎉 Batch processing completed!")
    print(f"📊 Statistics:")
    print(f"   • Processed videos: {processed_videos}/{len(video_paths)}")
    print(f"   • Skipped videos (already processed): {skipped_videos}")
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
        f.write(f"Max duration per video: {'Full video' if max_duration_seconds is None else f'{max_duration_seconds} seconds'}\n")
        f.write(f"Force reprocessing: {'Yes' if force_reprocess else 'No'}\n")
        f.write(f"Processed videos: {processed_videos}/{len(video_paths)}\n")
        f.write(f"Skipped videos: {skipped_videos}\n")
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
        default=0.1,
        help="Detection confidence threshold (default: 0.1)"
    )
    
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Number of frames to skip between processing (default: 0)"
    )
    

    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=None,
        help="Maximum duration to process per video in seconds (default: None - full video)"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip saving individual visualization frames"
    )
    
    parser.add_argument(
        "--test-duration",
        type=int,
        help="Process only first X seconds for testing (overrides --max-duration)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already processed videos"
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
    
    # Determine duration setting
    max_duration = args.test_duration if args.test_duration else args.max_duration
    if args.test_duration:
        print(f"🧪 Test mode: Processing only first {args.test_duration} seconds per video")
    
    try:
        process_videos_for_labelstudio(
            model_path=args.model,
            video_paths=video_paths,
            output_base_dir=args.output_dir,
            confidence_threshold=args.confidence,
            skip_frames=args.skip_frames,
            max_duration_seconds=max_duration,
            save_visualization_frames=not args.no_visualization,
            force_reprocess=args.force
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 