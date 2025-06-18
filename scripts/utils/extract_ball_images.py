#!/usr/bin/env python3
"""
Extract Ball Images Script
Extracts ball images and their corresponding original images from video processing
"""

import os
import cv2
import sys
from pathlib import Path
import shutil

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.video_processor import VideoProcessor
from configs.config import *


def extract_ball_images(input_video_path, output_dir, start_sec=30, duration_sec=60):
    """
    Extract ball images and their corresponding original images from video.
    
    Args:
        input_video_path (str): Path to input video
        output_dir (str): Directory to save extracted images
        start_sec (int): Start time in seconds
        duration_sec (int): Duration to process in seconds
    """
    
    # Create output directories
    ball_dir = os.path.join(output_dir, "ball_images")
    original_dir = os.path.join(output_dir, "original_images")
    
    os.makedirs(ball_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    
    print("🏐 Ball Image Extraction Pipeline")
    print("=" * 50)
    print(f"📹 Input video: {input_video_path}")
    print(f"📁 Ball images: {ball_dir}")
    print(f"📁 Original images: {original_dir}")
    print(f"⏰ Time range: {start_sec}s - {start_sec + duration_sec}s")
    
    # Initialize video processor
    print("\n🔧 Initializing video processor...")
    processor = VideoProcessor(
        yolo_model_path=YOLO_MODEL_PATH,
        unet_model_path=MODEL_LOAD_PATH if os.path.exists(MODEL_LOAD_PATH) else None
    )
    
    # Set hardcoded support image
    hardcoded_support_path = r"C:\Users\illya\Documents\volleyball_analitics\data\train_val_test_prepared_for_training\test\support\c085c7ed-frame_00215_ball.jpg"
    if os.path.exists(hardcoded_support_path):
        processor.set_support_image(hardcoded_support_path)
        print(f"✅ Using hardcoded support image: {hardcoded_support_path}")
    else:
        print(f"❌ Hardcoded support image not found: {hardcoded_support_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    start_frame = int(fps * start_sec)
    end_frame = min(int(fps * (start_sec + duration_sec)), total_frames)
    frames_to_process = end_frame - start_frame
    
    print(f"🎞️ Frames to process: {frames_to_process}")
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    frame_count = 0
    ball_count = 0
    
    print("🔄 Processing frames...")
    
    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save original image
        original_filename = f"frame_{frame_count:05d}_original.jpg"
        original_path = os.path.join(original_dir, original_filename)
        cv2.imwrite(original_path, frame)
        
        # Detect ball using YOLO
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = processor.person_detector.model(img_rgb, verbose=False)[0]
        
        ball_found = False
        for box in results.boxes:
            cls = int(box.cls.cpu().item())
            if cls == 32:  # Ball class
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                conf = float(box.conf.cpu())
                
                if conf > 0.3:  # Ball detection threshold
                    # Extract ball crop
                    ball_crop = frame[y1:y2, x1:x2]
                    if ball_crop.size > 0:
                        # Save ball image
                        ball_filename = f"frame_{frame_count:05d}_ball.jpg"
                        ball_path = os.path.join(ball_dir, ball_filename)
                        cv2.imwrite(ball_path, ball_crop)
                        
                        ball_found = True
                        ball_count += 1
                        print(f"✅ Frame {frame_count}: Ball detected (conf: {conf:.2f})")
                        break
        
        if not ball_found:
            print(f"❌ Frame {frame_count}: No ball detected")
        
        frame_count += 1
        
        # Progress update every 100 frames
        if frame_count % 100 == 0:
            print(f"📊 Processed {frame_count}/{frames_to_process} frames, found {ball_count} balls")
    
    # Cleanup
    cap.release()
    
    # Print final statistics
    print(f"\n✅ Extraction completed!")
    print(f"📊 Final Statistics:")
    print(f"   - Frames processed: {frame_count}")
    print(f"   - Ball images extracted: {ball_count}")
    print(f"   - Original images saved: {frame_count}")
    print(f"   - Ball images: {ball_dir}")
    print(f"   - Original images: {original_dir}")
    
    return True


def cleanup_original_images(ball_dir, original_dir):
    """
    Delete original images that don't have corresponding ball images.
    This should be run after manually cleaning the ball_images folder.
    
    Args:
        ball_dir (str): Directory containing ball images
        original_dir (str): Directory containing original images
    """
    print("🧹 Cleaning up original images...")
    
    # Get all ball image filenames
    ball_files = set()
    for filename in os.listdir(ball_dir):
        if filename.endswith("_ball.jpg"):
            # Extract the frame number
            frame_num = filename.replace("_ball.jpg", "")
            ball_files.add(frame_num)
    
    # Get all original image filenames
    original_files = set()
    for filename in os.listdir(original_dir):
        if filename.endswith("_original.jpg"):
            # Extract the frame number
            frame_num = filename.replace("_original.jpg", "")
            original_files.add(frame_num)
    
    # Find original images without corresponding ball images
    to_delete = original_files - ball_files
    
    print(f"📊 Cleanup Statistics:")
    print(f"   - Ball images found: {len(ball_files)}")
    print(f"   - Original images found: {len(original_files)}")
    print(f"   - Original images to delete: {len(to_delete)}")
    
    # Delete original images without corresponding ball images
    deleted_count = 0
    for frame_num in to_delete:
        original_path = os.path.join(original_dir, f"{frame_num}_original.jpg")
        if os.path.exists(original_path):
            os.remove(original_path)
            deleted_count += 1
            print(f"🗑️ Deleted: {frame_num}_original.jpg")
    
    print(f"✅ Cleanup completed! Deleted {deleted_count} original images.")


def main():
    """Main function with command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python extract_ball_images.py <input_video_path> [output_dir]")
        print("\nExample:")
        print("  python extract_ball_images.py video.mp4")
        print("  python extract_ball_images.py video.mp4 extracted_images")
        print("\nAfter manual cleaning, run cleanup:")
        print("  python extract_ball_images.py --cleanup ball_images original_images")
        return
    
    if sys.argv[1] == "--cleanup":
        if len(sys.argv) != 4:
            print("Usage: python extract_ball_images.py --cleanup <ball_dir> <original_dir>")
            return
        
        ball_dir = sys.argv[2]
        original_dir = sys.argv[3]
        cleanup_original_images(ball_dir, original_dir)
        return
    
    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_ball_images"
    
    success = extract_ball_images(input_video, output_dir)
    
    if success:
        print("\n🎉 Ball image extraction completed successfully!")
        print(f"\n📋 Next steps:")
        print(f"1. Manually clean the ball images in: {output_dir}/ball_images/")
        print(f"2. Run cleanup: python extract_ball_images.py --cleanup {output_dir}/ball_images {output_dir}/original_images")
    else:
        print("\n💥 Ball image extraction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 