#!/usr/bin/env python3
"""
Volleyball Ball Segmentation Pipeline
One-shot segmentation for ball detection in volleyball videos
"""

import torch
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.data.dataset import SiameseDataset
from src.utils.data_processing import (
    create_one_shot_dataset,
    split_dataset_consistently,
    check_mask_values,
    create_person_ball_one_shot_dataset,
    process_video_with_person_detection
)
from src.training.trainer import train_model
from src.visualization.visualization import plot_training_losses
from src.evaluation.evaluator import evaluate_model
from configs.config import *


def main():
    """Main pipeline function."""
    print("🏐 Volleyball Ball Segmentation Pipeline")
    print("=" * 50)
    
    # Step 0: Person detection and video processing (optional)
    print("\nStep 0: Person detection and video processing...")
    print("Processing video with person detection...")
    process_video_with_person_detection(
        VIDEO_PATH, 
        OUTPUT_DIRS["person_detection_video"],
        VIDEO_START_SEC, 
        VIDEO_DURATION_SEC, 
        YOLO_MODEL_PATH
    )
    
    # Step 0.5: Create person-ball dataset from video
    print("\nStep 0.5: Creating person-ball dataset from video...")
    create_person_ball_one_shot_dataset(
        VIDEO_PATH,
        OUTPUT_DIRS["person_ball_dataset"],
        VIDEO_START_SEC,
        VIDEO_DURATION_SEC,
        FRAME_INTERVAL,
        CROP_SIZE,
        YOLO_MODEL_PATH
    )
    
    # Step 1: Create one-shot dataset structure
    print("\nStep 1: Creating one-shot dataset structure...")
    create_one_shot_dataset(BASE_PATH, OUTPUT_BASE, CROP_SIZE)
    
    # Step 2: Split dataset into train/val/test
    print("\nStep 2: Splitting dataset...")
    split_dataset_consistently(
        OUTPUT_BASE, 
        PREPARED_DIR, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO
    )
    
    # Step 3: Check mask quality
    print("\nStep 3: Checking mask quality...")
    check_mask_values(f"{PREPARED_DIR}/train/masks")
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SiameseDataset(
        f"{PREPARED_DIR}/train/support",
        f"{PREPARED_DIR}/train/query",
        f"{PREPARED_DIR}/train/masks"
    )
    val_dataset = SiameseDataset(
        f"{PREPARED_DIR}/val/support",
        f"{PREPARED_DIR}/val/query",
        f"{PREPARED_DIR}/val/masks"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )
    
    # Initialize model and train
    model = UNet().to(device)
    train_losses, val_losses = train_model(
        train_loader, 
        val_loader, 
        model, 
        device, 
        NUM_EPOCHS, 
        LEARNING_RATE
    )
    
    # Step 5: Plot training results
    print("\nStep 5: Plotting training results...")
    plot_training_losses(train_losses, val_losses, NUM_EPOCHS)
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating on test set...")
    evaluate_model(
        MODEL_LOAD_PATH,
        f"{PREPARED_DIR}/test/query",
        f"{PREPARED_DIR}/test/masks",
        SUPPORT_IMAGE_PATH,
        OUTPUT_DIRS["test_results"]
    )
    
    print("\n✅ Pipeline completed successfully!")
    print("📊 Results saved to:", OUTPUT_DIRS["test_results"])
    print("🎥 Person detection video saved to:", OUTPUT_DIRS["person_detection_video"])
    print("📁 Person-ball dataset saved to:", OUTPUT_DIRS["person_ball_dataset"])


def person_detection_only():
    """Run only person detection pipeline."""
    print("👥 Person Detection Pipeline")
    print("=" * 50)
    
    print("Processing video with person detection...")
    process_video_with_person_detection(
        VIDEO_PATH, 
        OUTPUT_DIRS["person_detection_video"],
        VIDEO_START_SEC, 
        VIDEO_DURATION_SEC, 
        YOLO_MODEL_PATH
    )
    
    print("Creating person-ball dataset...")
    create_person_ball_one_shot_dataset(
        VIDEO_PATH,
        OUTPUT_DIRS["person_ball_dataset"],
        VIDEO_START_SEC,
        VIDEO_DURATION_SEC,
        FRAME_INTERVAL,
        CROP_SIZE,
        YOLO_MODEL_PATH
    )
    
    print("✅ Person detection pipeline completed!")


def ball_segmentation_only():
    """Run only ball segmentation pipeline."""
    print("🏐 Ball Segmentation Pipeline")
    print("=" * 50)
    
    # Step 1: Create one-shot dataset structure
    print("\nStep 1: Creating one-shot dataset structure...")
    create_one_shot_dataset(BASE_PATH, OUTPUT_BASE, CROP_SIZE)
    
    # Step 2: Split dataset into train/val/test
    print("\nStep 2: Splitting dataset...")
    split_dataset_consistently(
        OUTPUT_BASE, 
        PREPARED_DIR, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO
    )
    
    # Step 3: Check mask quality
    print("\nStep 3: Checking mask quality...")
    check_mask_values(f"{PREPARED_DIR}/train/masks")
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SiameseDataset(
        f"{PREPARED_DIR}/train/support",
        f"{PREPARED_DIR}/train/query",
        f"{PREPARED_DIR}/train/masks"
    )
    val_dataset = SiameseDataset(
        f"{PREPARED_DIR}/val/support",
        f"{PREPARED_DIR}/val/query",
        f"{PREPARED_DIR}/val/masks"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )
    
    # Initialize model and train
    model = UNet().to(device)
    train_losses, val_losses = train_model(
        train_loader, 
        val_loader, 
        model, 
        device, 
        NUM_EPOCHS, 
        LEARNING_RATE
    )
    
    # Step 5: Plot training results
    print("\nStep 5: Plotting training results...")
    plot_training_losses(train_losses, val_losses, NUM_EPOCHS)
    
    # Step 6: Evaluate on test set
    print("\nStep 6: Evaluating on test set...")
    evaluate_model(
        MODEL_LOAD_PATH,
        f"{PREPARED_DIR}/test/query",
        f"{PREPARED_DIR}/test/masks",
        SUPPORT_IMAGE_PATH,
        OUTPUT_DIRS["test_results"]
    )
    
    print("\n✅ Ball segmentation pipeline completed!")
    print("📊 Results saved to:", OUTPUT_DIRS["test_results"])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "person":
            person_detection_only()
        elif mode == "ball":
            ball_segmentation_only()
        else:
            print("Usage: python main.py [person|ball|full]")
            print("  person: Run only person detection")
            print("  ball: Run only ball segmentation")
            print("  full: Run complete pipeline (default)")
    else:
        main() 