#!/usr/bin/env python3
"""
YOLOv8 Training Script for Volleyball Ball Detection - Curated Dataset
Trains YOLOv8 model on the curated dataset with proper train/val split

Memory optimization notes:
- Reduced batch size based on GPU memory
- Reduced image size to 640x640
- Enabled mixed precision training (AMP)
- Reduced number of workers
- Disabled image caching

If you still experience CUDA OOM errors, try setting:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

# Set memory optimization environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


class YOLOv8CuratedTrainer:
    """YOLOv8 trainer for volleyball ball detection using curated dataset."""
    
    def check_gpu_memory(self) -> None:
        """Print current GPU memory usage information."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            print(f"GPU Memory Status:")
            print(f"  Allocated: {memory_allocated:.2f} GB")
            print(f"  Reserved: {memory_reserved:.2f} GB")
            print(f"  Total: {total_memory:.2f} GB")
            print(f"  Free: {total_memory - memory_reserved:.2f} GB")
        else:
            print("CUDA not available")
    
    def get_optimal_batch_size(self) -> int:
        """
        Determine optimal batch size based on available GPU memory.
        
        Returns:
            Optimal batch size for current GPU configuration
        """
        if not torch.cuda.is_available():
            return 4  # Conservative batch size for CPU
        
        try:
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # Conservative batch size estimation based on GPU memory
            if gpu_memory >= 16:
                return 16  # High-end GPU
            elif gpu_memory >= 12:
                return 12  # Good GPU
            elif gpu_memory >= 8:
                return 8   # Mid-range GPU
            elif gpu_memory >= 6:
                return 6   # Lower mid-range GPU
            else:
                return 4   # Low-end GPU
                
        except Exception as e:
            print(f"Warning: Could not determine GPU memory, using conservative batch size: {e}")
            return 4
    
    def __init__(self, dataset_path: str, output_dir: str):
        """
        Initialize the trainer.
        
        Args:
            dataset_path: Path to the curated YOLO dataset
            output_dir: Directory to save training results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters - optimized for curated dataset and GPU memory
        self.epochs = 100  # Reduced epochs for curated high-quality data
        self.img_size = 1280  # Reduced image size to save GPU memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Determine optimal batch size based on GPU memory
        self.batch_size = self.get_optimal_batch_size()
        
        print(f"Using device: {self.device}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Optimal batch size: {self.batch_size}")
        
        # Check initial GPU memory status
        self.check_gpu_memory()
    
    def prepare_train_val_split(self, val_split: float = 0.2, test_split: float = 0.1) -> None:
        """
        Split curated dataset into train, validation, and test sets.
        
        Args:
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
        """
        print(f"\nPreparing train/val/test split...")
        print(f"Train: {(1-val_split-test_split)*100:.1f}%")
        print(f"Validation: {val_split*100:.1f}%")
        print(f"Test: {test_split*100:.1f}%")
        
        # Get all image files
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"
        
        image_files = list(images_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} curated images")
        
        # Create train/val/test directories
        train_images_dir = self.dataset_path / "train" / "images"
        train_labels_dir = self.dataset_path / "train" / "labels"
        val_images_dir = self.dataset_path / "val" / "images"
        val_labels_dir = self.dataset_path / "val" / "labels"
        test_images_dir = self.dataset_path / "test" / "images"
        test_labels_dir = self.dataset_path / "test" / "labels"
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, 
                         test_images_dir, test_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # First split: separate test set
        train_val_files, test_files = train_test_split(
            image_files, 
            test_size=test_split, 
            random_state=42
        )
        
        # Second split: separate train and validation
        train_files, val_files = train_test_split(
            train_val_files, 
            test_size=val_split/(1-test_split), 
            random_state=42
        )
        
        print(f"Train set: {len(train_files)} images")
        print(f"Validation set: {len(val_files)} images")
        print(f"Test set: {len(test_files)} images")
        
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
        
        # Copy files to test directory
        for img_file in test_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            shutil.copy2(img_file, test_images_dir / img_file.name)
            if label_file.exists():
                shutil.copy2(label_file, test_labels_dir / label_file.name)
        
        # Update data.yaml
        self.create_training_config()
        
        print("Train/val/test split completed successfully!")
    
    def create_training_config(self) -> None:
        """Create updated data.yaml for training."""
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['Volleyball Ball']
        }
        
        config_path = self.dataset_path / "data.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Updated training configuration: {config_path}")
    
    def train_model(self, model_size: str = 'n') -> Path:
        """
        Train YOLOv8 model on curated dataset.
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            
        Returns:
            Path to the best trained model
        """
        print(f"\nStarting YOLOv8{model_size} training on curated dataset...")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}")
        print(f"Device: {self.device}")
        
        # Load YOLOv8 model
        model = YOLO(f'yolov8{model_size}.pt')
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check GPU memory before training
        print("\nGPU memory status before training:")
        self.check_gpu_memory()
        
        # Train the model with optimized parameters for curated data and GPU memory
        results = model.train(
            data=str(self.dataset_path / "data.yaml"),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            device=self.device,
            project=str(self.output_dir),
            name=f'yolov8{model_size}_volleyball_curated',
            save=True,
            save_period=20,  # Save checkpoint every 20 epochs to save disk space
            val=True,
            plots=True,
            verbose=True,
            patience=20,  # Patience for early stopping
            optimizer='AdamW',
            lr0=0.01,  # Higher learning rate for clean curated data
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,  # Box loss weight
            cls=0.5,  # Classification loss weight
            dfl=1.5,  # Distribution focal loss weight
            # Memory optimization parameters
            amp=True,  # Enable Automatic Mixed Precision for memory efficiency
            workers=4,  # Reduce number of data loader workers
            cache=False,  # Disable image caching to save RAM
            # Augmentation parameters - reduced for high-quality curated data
            mosaic=0.3,  # Reduced mosaic
            mixup=0.05,  # Light mixup
            copy_paste=0.0,  # Disabled copy-paste for curated data
            degrees=5,  # Small rotations
            translate=0.05,  # Small translation
            scale=0.1,  # Light scaling
            shear=2,  # Minimal shear
            perspective=0.0,  # No perspective distortion
            flipud=0.0,  # No vertical flip
            fliplr=0.5,  # Horizontal flip for volleyball
            hsv_h=0.01,  # Minimal HSV augmentation
            hsv_s=0.3,
            hsv_v=0.2,
        )
        
        # Get path to best model
        best_model_path = self.output_dir / f'yolov8{model_size}_volleyball_curated' / 'weights' / 'best.pt'
        
        print(f"\nTraining completed!")
        print(f"Best model saved at: {best_model_path}")
        
        return best_model_path
    
    def evaluate_model(self, model_path: Path) -> None:
        """
        Evaluate trained model on test set.
        
        Args:
            model_path: Path to the trained model
        """
        print(f"\nEvaluating model: {model_path}")
        
        # Load trained model
        model = YOLO(str(model_path))
        
        # Run validation on test set
        results = model.val(
            data=str(self.dataset_path / "data.yaml"),
            split='test',  # Use test split for final evaluation
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
        
        print(f"Test results:")
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
            test_images_dir = self.dataset_path / "test" / "images"
            test_images = list(test_images_dir.glob("*.jpg"))[:10]  # Take first 10 test images
        
        # Create inference results directory
        inference_dir = self.output_dir / "inference_results"
        inference_dir.mkdir(exist_ok=True)
        
        detection_count = 0
        total_confidence = 0.0
        
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
                    detection_count += 1
                    total_confidence += conf
            else:
                print("  No volleyball detected")
        
        if detection_count > 0:
            avg_confidence = total_confidence / detection_count
            print(f"\nInference summary:")
            print(f"Total detections: {detection_count}")
            print(f"Average confidence: {avg_confidence:.3f}")
        
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
    """Main training function for curated dataset."""
    print("YOLOv8 Volleyball Ball Detection Training - Curated Dataset")
    print("=" * 60)
    
    # Paths for curated dataset
    dataset_path = r"C:\Users\illya\Documents\volleyball_analitics\volleystat\data\curated_dataset"
    output_dir = r"C:\Users\illya\Documents\volleyball_analitics\volleystat\models\yolov8_curated_training_2"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    # Check if images exist
    images_dir = Path(dataset_path) / "images"
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    image_count = len(list(images_dir.glob("*.jpg")))
    print(f"Found {image_count} curated images")
    
    if image_count == 0:
        print("No images found in dataset!")
        return
    
    # Create trainer
    trainer = YOLOv8CuratedTrainer(dataset_path, output_dir)
    
    # Prepare train/val/test split
    trainer.prepare_train_val_split(val_split=0.2, test_split=0.1)
    
    # Train model (start with nano model)
    model_path = trainer.train_model(model_size='n')
    
    # Evaluate model on test set
    trainer.evaluate_model(model_path)
    
    # Test inference
    trainer.test_inference(model_path)
    
    # Export model (uncomment if needed)
    # trainer.export_model(model_path, formats=['onnx'])
    
    print("\nCurated dataset training pipeline completed successfully!")
    print(f"Best model: {model_path}")
    print(f"Model directory: {output_dir}")


if __name__ == "__main__":
    main() 