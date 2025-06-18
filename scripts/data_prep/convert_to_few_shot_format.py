#!/usr/bin/env python3
"""
Convert current dataset to few-shot format with Images/Labels/Masks structure.
"""

import os
import shutil
import random
import cv2
import numpy as np

def create_mask_from_obb(image_path, label_path, target_size=(256, 256)):
    """Create binary mask from YOLOv8 OBB coordinates."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                coords = list(map(float, line.strip().split()))
                if len(coords) == 9:
                    points = coords[1:]
                    pixel_points = []
                    for i in range(0, len(points), 2):
                        x = int(points[i] * w)
                        y = int(points[i + 1] * h)
                        pixel_points.append([x, y])
                    if len(pixel_points) >= 3:
                        pixel_points = np.array(pixel_points, dtype=np.int32)
                        cv2.fillPoly(mask, [pixel_points], 255)
    mask_resized = cv2.resize(mask, target_size)
    return mask_resized

def convert_to_few_shot_format(src_images, src_labels, out_dir, mask_target_size=(256, 256), split=(0.7, 0.2, 0.1)):
    """Convert dataset to few-shot format with train/val/test splits."""
    print(f"🔄 Converting dataset to few-shot format...")
    print(f"📁 Source images: {src_images}")
    print(f"📁 Source labels: {src_labels}")
    print(f"📁 Output directory: {out_dir}")
    
    images = [f for f in os.listdir(src_images) if f.endswith('.jpg')]
    print(f"📊 Found {len(images)} images")
    
    random.shuffle(images)
    n = len(images)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    print(f"📊 Split sizes: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    for split_name, split_imgs in splits.items():
        print(f"\n📁 Processing {split_name} split...")
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(os.path.join(split_dir, 'Images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'Labels'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'Masks'), exist_ok=True)
        
        for img in split_imgs:
            base = os.path.splitext(img)[0]
            src_img_path = os.path.join(src_images, img)
            src_label_path = os.path.join(src_labels, base + '.txt')
            dst_img_path = os.path.join(split_dir, 'Images', img)
            dst_label_path = os.path.join(split_dir, 'Labels', base + '.txt')
            dst_mask_path = os.path.join(split_dir, 'Masks', base + '_mask.png')
            
            # Copy image
            shutil.copy2(src_img_path, dst_img_path)
            
            # Copy label
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
                
                # Create mask from OBB coordinates
                mask = create_mask_from_obb(src_img_path, src_label_path, mask_target_size)
                if mask is not None:
                    cv2.imwrite(dst_mask_path, mask)
    
    print(f"\n✅ Dataset conversion completed!")
    print(f"📁 Output directory: {out_dir}")

if __name__ == "__main__":
    # Convert the ball crops dataset to few-shot format
    convert_to_few_shot_format(
        src_images='ball_dataset_for_labeling/ball_crops_dataset/images',
        src_labels='ball_dataset_for_labeling/ball_crops_dataset/labels',
        out_dir='few_shot_dataset',
        mask_target_size=(256, 256)
    ) 