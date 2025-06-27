#!/usr/bin/env python3
"""
YOLOv8 Training Script for Volleyball Ball Detection
Trains YOLOv8 model on the created dataset with proper train/val split
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, List
import random
import yaml
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


class YOLOv8VolleyballTrainer:
    """YOLOv8 trainer for volleyball ball detection."""
    
    def __init__(self, dataset_path: str, output_dir: str):
        """
        Initialize the trainer.
        
        Args:
            dataset_path: Path to the YOLO dataset
            output_dir: Directory to save training results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.epochs = 150
        self.batch_size = 64  # Reduce batch for larger images
        self.img_size = 1280  # Increase size for better small object detection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {self.device}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
    
    def prepare_train_val_split(self, val_split: float = 0.2) -> None:
        """
        Split dataset into train and validation sets.
        
        Args:
            val_split: Fraction of data to use for validation
        """
        print(f"\nPreparing train/val split with {val_split*100}% validation data...")
        
        # Get all image files
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"
        
        image_files = list(images_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} images")
        
        # Create train/val directories
        train_images_dir = self.dataset_path / "train" / "images"
        train_labels_dir = self.dataset_path / "train" / "labels"
        val_images_dir = self.dataset_path / "val" / "images"
        val_labels_dir = self.dataset_path / "val" / "labels"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split files
        train_files, val_files = train_test_split(
            image_files, 
            test_size=val_split, 
            random_state=42
        )
        
        print(f"Train set: {len(train_files)} images")
        print(f"Validation set: {len(val_files)} images")
        
        # Copy files to train directory
        for img_file in train_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            shutil.copy2(img_file, train_images_dir / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, train_labels_dir / label_file.name)
        
        # Copy files to val directory
        for img_file in val_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            shutil.copy2(img_file, val_images_dir / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, val_labels_dir / label_file.name)
        
        # Update data.yaml
        self.create_training_config()
        
        print("Train/val split completed successfully!")
    
    def create_training_config(self) -> None:
        """Create updated data.yaml for training."""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,
            'names': ['volleyball_ball']
        }
        
        config_path = self.dataset_path / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Updated training configuration: {config_path}")
    
    def train_model(self, model_size: str = 'n') -> Path:
        """
        Train YOLOv8 model.
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            
        Returns:
            Path to the best trained model
        """
        print(f"\nStarting YOLOv8{model_size} training...")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Device: {self.device}")
        
        # Load YOLOv8 model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Train the model
        results = model.train(
            data=str(self.dataset_path / "data.yaml"),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            device=self.device,
            project=str(self.output_dir),
            name=f'yolov8{model_size}_volleyball',
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            val=True,
            plots=True,
            verbose=True,
            patience=30,  # Increase patience for small objects
            optimizer='AdamW',
            lr0=0.005,  # Reduce learning rate for better stability
            weight_decay=0.0005,
            warmup_epochs=5,  # Increase warmup
            box=10.0,  # Increase box loss weight for small objects
            cls=0.5,  # Classification loss weight
            dfl=1.5,  # Distribution focal loss weight
            mosaic=0.5,  # Enable mosaic for better small object training
            mixup=0.1,  # Light mixup augmentation
            copy_paste=0.1,  # Copy-paste for increased diversity
            degrees=10,  # Small rotations
            translate=0.1,  # Translation fraction
            scale=0.2,  # Light scaling
            shear=5,  # Light shear
            perspective=0.0001,  # Minimal perspective
            flipud=0.0,  # Vertical flip probability
            fliplr=0.5,  # Horizontal flip suitable for volleyball
            hsv_h=0.015,  # HSV hue augmentation
            hsv_s=0.7,  # HSV saturation augmentation
            hsv_v=0.4,  # HSV value augmentation
        )
        
        # Get path to best model
        best_model_path = self.output_dir / f'yolov8{model_size}_volleyball' / 'weights' / 'best.pt'
        
        print(f"\nTraining completed!")
        print(f"Best model saved at: {best_model_path}")
        
        return best_model_path
    
    def evaluate_model(self, model_path: Path) -> None:
        """
        Evaluate trained model on validation set.
        
        Args:
            model_path: Path to the trained model
        """
        print(f"\nEvaluating model: {model_path}")
        
        # Load trained model
        model = YOLO(str(model_path))
        
        # Run validation
        results = model.val(
            data=str(self.dataset_path / "data.yaml"),
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            save_json=True,
            save_hybrid=True,
            conf=0.25,
            iou=0.45,
            plots=True,
            verbose=True
        )
        
        print(f"Validation results:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
    
    def test_inference(self, model_path: Path, test_images: List[str] = None) -> None:
        """
        Test model inference on sample images.
        
        Args:
            model_path: Path to the trained model
            test_images: List of test image paths (optional)
        """
        print(f"\nTesting inference with model: {model_path}")
        
        # Load trained model
        model = YOLO(str(model_path))
        
        # Get test images
        if test_images is None:
            val_images_dir = self.dataset_path / "val" / "images"
            test_images = list(val_images_dir.glob("*.jpg"))[:5]  # Take first 5 validation images
        
        # Create inference results directory
        inference_dir = self.output_dir / "inference_results"
        inference_dir.mkdir(exist_ok=True)
        
        for i, img_path in enumerate(test_images):
            print(f"Processing {img_path.name}...")
            
            # Run inference
            results = model(str(img_path), conf=0.25, iou=0.45)
            
            # Save annotated image
            annotated_img = results[0].plot()
            output_path = inference_dir / f"test_{i+1}_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
            
            # Print detection info
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    conf = box.conf[0].item()
                    print(f"  Detected volleyball with confidence: {conf:.3f}")
            else:
                print("  No volleyball detected")
        
        print(f"Inference results saved to: {inference_dir}")
    
    def export_model(self, model_path: Path, formats: List[str] = None) -> None:
        """
        Export trained model to different formats.
        
        Args:
            model_path: Path to the trained model
            formats: List of export formats ('onnx', 'torchscript', 'tflite', etc.)
        """
        if formats is None:
            formats = ['onnx', 'torchscript']
        
        print(f"\nExporting model to formats: {formats}")
        
        # Load trained model
        model = YOLO(str(model_path))
        
        for format_type in formats:
            try:
                print(f"Exporting to {format_type}...")
                model.export(format=format_type, imgsz=self.img_size)
                print(f"Successfully exported to {format_type}")
            except Exception as e:
                print(f"Failed to export to {format_type}: {e}")


def main():
    """Main training function."""
    print("YOLOv8 Volleyball Ball Detection Training")
    print("=" * 50)
    
    # Paths
    dataset_path = r"C:\Users\illya\Documents\volleyball_analitics\volleystat\data\yolo_dataset_improved"
    output_dir = r"C:\Users\illya\Documents\volleyball_analitics\volleystat\models\yolov8_volleyball_training_no_aug"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    # Create trainer
    trainer = YOLOv8VolleyballTrainer(dataset_path, output_dir)
    
    # Prepare train/val split
    trainer.prepare_train_val_split(val_split=0.2)
    
    # Train model (start with nano model for faster training)
    model_path = trainer.train_model(model_size='n')
    
    # Evaluate model
    trainer.evaluate_model(model_path)
    
    # Test inference
    trainer.test_inference(model_path)
    
    # Export model (commented out to avoid ONNX dependency issues)
    # trainer.export_model(model_path, formats=['onnx'])
    
    print("\nTraining pipeline completed successfully!")
    print(f"Best model: {model_path}")


if __name__ == "__main__":
    main() 