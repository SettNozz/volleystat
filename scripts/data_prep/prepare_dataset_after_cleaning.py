#!/usr/bin/env python3
"""
Prepare Dataset After Manual Cleaning
Creates the proper few-shot dataset structure after manual cleaning of ball images.
"""

import os
import shutil
import random
import cv2
import numpy as np
import json
from datetime import datetime


def create_dataset_structure(base_dir, run_name):
    """Create the dataset directory structure."""
    dataset_dir = os.path.join(base_dir, run_name)
    
    # Create splits
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        os.makedirs(os.path.join(split_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'query'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'support'), exist_ok=True)
    
    return dataset_dir


def create_mask_from_bbox(bbox, image_shape):
    """Create mask from bounding box coordinates."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    # Ensure coordinates are within image bounds
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    # Create white bounding box on black background
    mask[y:y+height, x:x+width] = 255
    
    return mask


def prepare_dataset_after_cleaning(
    extraction_dir,
    output_base_dir='few_shot_datasets_cleaned',
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    run_name=None
):
    """Prepare few-shot dataset after manual cleaning."""
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"few_shot_run_{timestamp}"
    
    print(f"🚀 Preparing dataset after cleaning: {run_name}")
    print(f"📁 Extraction directory: {extraction_dir}")
    
    # Load coordinates
    coordinates_file = os.path.join(extraction_dir, 'ball_coordinates.json')
    if not os.path.exists(coordinates_file):
        print(f"❌ Coordinates file not found: {coordinates_file}")
        return
    
    with open(coordinates_file, 'r') as f:
        coordinates_data = json.load(f)
    
    print(f"📊 Found {len(coordinates_data)} ball detections")
    
    # Get cleaned ball images (only files that exist after manual cleaning)
    ball_images_dir = os.path.join(extraction_dir, 'ball_images')
    original_images_dir = os.path.join(extraction_dir, 'original_images')
    
    if not os.path.exists(ball_images_dir) or not os.path.exists(original_images_dir):
        print(f"❌ Ball images or original images directory not found")
        return
    
    # Filter coordinates to only include cleaned ball images
    cleaned_coordinates = []
    for coord in coordinates_data:
        ball_path = os.path.join(ball_images_dir, coord['ball_image'])
        original_path = os.path.join(original_images_dir, coord['original_image'])
        
        if os.path.exists(ball_path) and os.path.exists(original_path):
            cleaned_coordinates.append(coord)
    
    print(f"📊 After cleaning: {len(cleaned_coordinates)} valid samples")
    
    if len(cleaned_coordinates) == 0:
        print(f"❌ No valid samples found after cleaning")
        return
    
    # Shuffle and split coordinates
    random.shuffle(cleaned_coordinates)
    n = len(cleaned_coordinates)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    splits = {
        'train': cleaned_coordinates[:n_train],
        'val': cleaned_coordinates[n_train:n_train+n_val],
        'test': cleaned_coordinates[n_train+n_val:]
    }
    
    print(f"📊 Split sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # Create dataset structure
    dataset_dir = create_dataset_structure(output_base_dir, run_name)
    
    # Process each split
    for split_name, split_coordinates in splits.items():
        print(f"\n📁 Processing {split_name} split...")
        
        for i, coord in enumerate(split_coordinates):
            # Source paths
            ball_src = os.path.join(ball_images_dir, coord['ball_image'])
            original_src = os.path.join(original_images_dir, coord['original_image'])
            
            # Destination paths
            base_name = f"{split_name}_{i+1:03d}"
            ball_dst = os.path.join(dataset_dir, split_name, 'support', f"{base_name}_support.jpg")
            original_dst = os.path.join(dataset_dir, split_name, 'query', f"{base_name}.jpg")
            mask_dst = os.path.join(dataset_dir, split_name, 'masks', f"{base_name}_mask.png")
            
            # Copy ball image (support)
            shutil.copy2(ball_src, ball_dst)
            
            # Copy original image (query)
            shutil.copy2(original_src, original_dst)
            
            # Create mask from bounding box
            original_image = cv2.imread(original_src)
            if original_image is not None:
                mask = create_mask_from_bbox(coord['bbox'], original_image.shape)
                cv2.imwrite(mask_dst, mask)
                
                print(f"  ✅ {coord['ball_image']} -> {split_name}/{base_name}")
            else:
                print(f"  ❌ Failed to load original image: {original_src}")
    
    # Save dataset info
    dataset_info = {
        'run_name': run_name,
        'created_at': datetime.now().isoformat(),
        'extraction_dir': extraction_dir,
        'splits': {split: len(coords) for split, coords in splits.items()},
        'total_samples': len(cleaned_coordinates),
        'structure': 'train/val/test with masks/query/support folders'
    }
    
    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Dataset preparation completed!")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📄 Dataset info: {info_path}")
    
    return dataset_dir, dataset_info


def main():
    """Main function to prepare dataset after cleaning."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prepare_dataset_after_cleaning.py <extraction_dir>")
        print("\nExample:")
        print("  python prepare_dataset_after_cleaning.py ball_extraction_results")
        return
    
    # Configuration
    extraction_dir = sys.argv[1]
    
    # Check if extraction directory exists
    if not os.path.exists(extraction_dir):
        print(f"❌ Extraction directory not found: {extraction_dir}")
        print("💡 First run the ball extraction script:")
        print("   python extract_balls_from_video.py <video_path>")
        return
    
    # Prepare dataset
    dataset_dir, dataset_info = prepare_dataset_after_cleaning(
        extraction_dir=extraction_dir,
        output_base_dir='few_shot_datasets_cleaned',
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    
    print(f"\n🎉 Dataset preparation completed!")
    print(f"📊 Dataset summary:")
    for split, count in dataset_info['splits'].items():
        print(f"   - {split}: {count} samples")
    
    print(f"\n🚀 Ready for training!")
    print(f"Run: python train_few_shot_v2.py {dataset_dir} 100")


if __name__ == "__main__":
    main() 