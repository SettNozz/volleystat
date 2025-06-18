#!/usr/bin/env python3
"""
Ball Detection and Saving Script
Detects balls in video and saves them in YOLOv8 OBB format for future labeling.
Also crops detected balls and saves them for manual cleaning.
"""

import os
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from src.pipeline.video_processor import VideoProcessor, create_support_image_from_video
from src.evaluation.evaluator import mask_to_obb_coordinates
from configs.config import *


def crop_ball_from_image(frame, ball_bbox, padding=10):
    """
    Crop ball from image using bounding box with padding.
    
    Args:
        frame: Original frame
        ball_bbox: Bounding box (x1, y1, x2, y2)
        padding: Padding around the ball in pixels
    
    Returns:
        Cropped ball image or None if cropping fails
    """
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


def detect_and_save_balls_for_labeling(video_path, output_dir, ball_images_dir, start_sec=30, duration_sec=60):
    """
    Detect balls in video and save in YOLOv8 OBB format for labeling.
    Also crop and save ball images for manual cleaning.
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Directory to save the dataset
        ball_images_dir (str): Directory to save cropped ball images
        start_sec (int): Start time in seconds
        duration_sec (int): Duration to process in seconds
    """
    
    # Validate input video
    if not os.path.exists(video_path):
        print(f"❌ Input video not found: {video_path}")
        return False
    
    # Create output directory structure
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(ball_images_dir, exist_ok=True)
    
    print("🏐 Ball Detection and Saving for Labeling")
    print("=" * 50)
    print(f"📹 Input video: {video_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🏀 Ball images directory: {ball_images_dir}")
    print(f"⏰ Time range: {start_sec}s - {start_sec + duration_sec}s")
    
    # Initialize video processor
    print("\n🔧 Initializing video processor...")
    processor = VideoProcessor(
        yolo_model_path=YOLO_MODEL_PATH,
        unet_model_path=MODEL_LOAD_PATH if os.path.exists(MODEL_LOAD_PATH) else None
    )
    
    # Create support image automatically from the video
    print("\n🔍 Creating support image for ball segmentation...")
    support_path = os.path.join(output_dir, "support_ball.jpg")
    support_image_path = create_support_image_from_video(
        video_path, 
        support_path,
        start_sec,
        10,  # Search in first 10 seconds of the segment
        YOLO_MODEL_PATH
    )
    
    if not support_image_path:
        print("❌ Could not create support image, ball segmentation will not work")
        return False
    
    # Set support image
    processor.set_support_image(support_image_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame range
    start_frame = int(fps * start_sec)
    end_frame = min(int(fps * (start_sec + duration_sec)), total_frames)
    frames_to_process = end_frame - start_frame
    
    print(f"📐 Resolution: {width}x{height}")
    print(f"🎞️ Frames to process: {frames_to_process}")
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    frame_count = 0
    ball_detections = 0
    saved_images = 0
    saved_ball_crops = 0
    
    print("\n🔄 Processing frames and saving ball detections...")
    
    for frame_idx in tqdm(range(frames_to_process)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Segment ball
        ball_mask, _ = processor.segment_ball(frame)
        
        if ball_mask is not None:
            # Find ball bounding box
            ball_bbox = processor.find_ball_bounding_box(ball_mask)
            
            if ball_bbox:
                ball_detections += 1
                
                # Save original image
                image_name = f"frame_{frame_count:05d}"
                image_path = os.path.join(images_dir, f"{image_name}.jpg")
                cv2.imwrite(image_path, frame)
                
                # Crop and save ball image
                ball_crop = crop_ball_from_image(frame, ball_bbox, padding=10)
                if ball_crop is not None:
                    ball_crop_name = f"{image_name}_ball.jpg"
                    ball_crop_path = os.path.join(ball_images_dir, ball_crop_name)
                    cv2.imwrite(ball_crop_path, ball_crop)
                    saved_ball_crops += 1
                    print(f"  💾 Saved {image_name}: {len(ball_bbox)} ball(s), crop: {ball_crop_name}")
                
                # Generate OBB coordinates from mask
                obb_coordinates = mask_to_obb_coordinates(ball_mask)
                
                # Save label file
                label_path = os.path.join(labels_dir, f"{image_name}.txt")
                with open(label_path, "w") as f:
                    for coord in obb_coordinates:
                        f.write(" ".join(map(str, coord)) + "\n")
                
                saved_images += 1
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    
    # Create data.yaml file
    data_yaml = {
        "path": output_dir,
        "train": "images",
        "val": "images",
        "nc": 1,  # number of classes
        "names": ["ball"]  # class names
    }
    
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Create summary file
    summary_path = os.path.join(output_dir, "detection_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Ball Detection Summary\n")
        f.write("=====================\n\n")
        f.write(f"Input video: {video_path}\n")
        f.write(f"Time range: {start_sec}s - {start_sec + duration_sec}s\n")
        f.write(f"Frames processed: {frame_count}\n")
        f.write(f"Ball detections: {ball_detections}\n")
        f.write(f"Images saved: {saved_images}\n")
        f.write(f"Ball crops saved: {saved_ball_crops}\n")
        f.write(f"Support image: {support_image_path}\n")
        f.write(f"Ball images directory: {ball_images_dir}\n")
    
    print(f"\n✅ Ball detection and saving completed!")
    print(f"📊 Summary:")
    print(f"   - Frames processed: {frame_count}")
    print(f"   - Ball detections: {ball_detections}")
    print(f"   - Images saved: {saved_images}")
    print(f"   - Ball crops saved: {saved_ball_crops}")
    print(f"   - Dataset location: {output_dir}")
    print(f"   - Ball images location: {ball_images_dir}")
    print(f"   - Support image: {support_image_path}")
    
    return True


def update_dataset_to_ball_crops_only(output_dir, ball_images_dir):
    """
    After manual cleaning, update the dataset to use only cropped ball images.
    Remove original images and update labels accordingly.
    
    Args:
        output_dir (str): Directory containing the dataset
        ball_images_dir (str): Directory containing cleaned ball crop images
    """
    print("\n🔄 Updating dataset to use only ball crop images...")
    
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("❌ Dataset directories not found")
        return False
    
    # Get list of ball crop images (after manual cleaning)
    ball_crop_files = [f for f in os.listdir(ball_images_dir) if f.endswith('.jpg')]
    print(f"Found {len(ball_crop_files)} ball crop images")
    
    # Create new images directory for ball crops
    ball_crops_dataset_dir = os.path.join(output_dir, "ball_crops_dataset")
    ball_crops_images_dir = os.path.join(ball_crops_dataset_dir, "images")
    ball_crops_labels_dir = os.path.join(ball_crops_dataset_dir, "labels")
    os.makedirs(ball_crops_images_dir, exist_ok=True)
    os.makedirs(ball_crops_labels_dir, exist_ok=True)
    
    # Copy ball crop images and corresponding labels
    copied_count = 0
    for ball_crop_file in ball_crop_files:
        # Extract original frame name from ball crop name
        # Format: frame_XXXXX_ball.jpg -> frame_XXXXX
        original_name = ball_crop_file.replace('_ball.jpg', '')
        
        # Check if corresponding label exists
        label_file = f"{original_name}.txt"
        label_path = os.path.join(labels_dir, label_file)
        
        if os.path.exists(label_path):
            # Copy ball crop image
            src_ball_path = os.path.join(ball_images_dir, ball_crop_file)
            dst_ball_path = os.path.join(ball_crops_images_dir, ball_crop_file)
            
            # Copy label file
            dst_label_path = os.path.join(ball_crops_labels_dir, label_file)
            
            # Use shutil for reliable copying
            import shutil
            shutil.copy2(src_ball_path, dst_ball_path)
            shutil.copy2(label_path, dst_label_path)
            
            copied_count += 1
            print(f"  📋 Copied {ball_crop_file} and {label_file}")
    
    # Create new data.yaml for ball crops dataset
    ball_crops_data_yaml = {
        "path": ball_crops_dataset_dir,
        "train": "images",
        "val": "images",
        "nc": 1,  # number of classes
        "names": ["ball"]  # class names
    }
    
    with open(os.path.join(ball_crops_dataset_dir, "data.yaml"), "w") as f:
        yaml.dump(ball_crops_data_yaml, f, default_flow_style=False)
    
    # Create summary
    summary_path = os.path.join(ball_crops_dataset_dir, "ball_crops_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Ball Crops Dataset Summary\n")
        f.write("=========================\n\n")
        f.write(f"Original dataset: {output_dir}\n")
        f.write(f"Ball images source: {ball_images_dir}\n")
        f.write(f"Ball crops copied: {copied_count}\n")
        f.write(f"Dataset location: {ball_crops_dataset_dir}\n")
    
    print(f"\n✅ Dataset updated successfully!")
    print(f"📊 Ball crops dataset:")
    print(f"   - Location: {ball_crops_dataset_dir}")
    print(f"   - Images copied: {copied_count}")
    print(f"   - Ready for training with ball crop images only")
    
    return True


def main():
    """Main function with command line interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect_and_save_balls.py <input_video_path> [output_dir] [ball_images_dir] [start_sec] [duration_sec]")
        print("\nExample:")
        print("  python detect_and_save_balls.py video.mp4")
        print("  python detect_and_save_balls.py video.mp4 ball_dataset ball_images 30 60")
        print("\nAfter manual cleaning, run:")
        print("  python detect_and_save_balls.py --update-dataset ball_dataset ball_images")
        return
    
    # Check if this is an update operation
    if sys.argv[1] == "--update-dataset":
        if len(sys.argv) < 4:
            print("Usage: python detect_and_save_balls.py --update-dataset <output_dir> <ball_images_dir>")
            return
        
        output_dir = sys.argv[2]
        ball_images_dir = sys.argv[3]
        success = update_dataset_to_ball_crops_only(output_dir, ball_images_dir)
        
        if success:
            print("\n🎉 Dataset update completed successfully!")
        else:
            print("\n💥 Dataset update failed!")
            sys.exit(1)
        return
    
    # Normal detection operation
    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ball_dataset_for_labeling"
    ball_images_dir = sys.argv[3] if len(sys.argv) > 3 else r"C:\Users\illya\Documents\volleyball_analitics\volleystat\ball_extraction_full\ball_images"
    start_sec = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    duration_sec = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    
    success = detect_and_save_balls_for_labeling(input_video, output_dir, ball_images_dir, start_sec, duration_sec)
    
    if success:
        print("\n🎉 Ball detection and saving completed successfully!")
        print(f"📁 Dataset ready for labeling in: {output_dir}")
        print(f"🏀 Ball images saved in: {ball_images_dir}")
        print(f"\n💡 After manual cleaning, run:")
        print(f"   python detect_and_save_balls.py --update-dataset {output_dir} {ball_images_dir}")
    else:
        print("\n💥 Ball detection and saving failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 