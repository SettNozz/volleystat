#!/usr/bin/env python3
"""
Ball Segmentation Training Script
Trains U-Net model on ball crops dataset with train/val/test split.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import random

from src.models.ball_unet import BallUNet
from configs.config import *


def create_mask_from_obb(image_path, label_path, target_size=(256, 256)):
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
    images = [f for f in os.listdir(src_images) if f.endswith('.jpg')]
    random.shuffle(images)
    n = len(images)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    for split_name, split_imgs in splits.items():
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
            shutil.copy2(src_img_path, dst_img_path)
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
                mask = create_mask_from_obb(src_img_path, src_label_path, mask_target_size)
                if mask is not None:
                    cv2.imwrite(dst_mask_path, mask)


def create_ball_crops_dataset(ball_images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Create train/val/test split from ball crops dataset.
    
    Args:
        ball_images_dir (str): Directory containing ball crop images
        labels_dir (str): Directory containing labels
        output_dir (str): Output directory for split dataset
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        test_ratio (float): Test set ratio
    """
    print("🔄 Creating train/val/test split from ball crops dataset...")
    
    # Get list of ball crop images
    ball_images = [f for f in os.listdir(ball_images_dir) if f.endswith('.jpg')]
    print(f"Found {len(ball_images)} ball crop images")
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    
    # Calculate split sizes
    total_size = len(ball_images)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    print(f"Split sizes: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split the data
    splits = random_split(ball_images, [train_size, val_size, test_size])
    
    # Copy files to respective directories
    for split_idx, (split_name, split_data) in enumerate([("train", splits[0]), ("val", splits[1]), ("test", splits[2])]):
        split_dir = os.path.join(output_dir, split_name)
        images_split_dir = os.path.join(split_dir, "images")
        labels_split_dir = os.path.join(split_dir, "labels")
        
        print(f"\n📁 Copying {split_name} data ({len(split_data)} images)...")
        
        for ball_image in tqdm(split_data):
            # Copy ball image
            src_image_path = os.path.join(ball_images_dir, ball_image)
            dst_image_path = os.path.join(images_split_dir, ball_image)
            
            # Get corresponding label
            label_name = ball_image.replace('.jpg', '.txt')
            src_label_path = os.path.join(labels_dir, label_name)
            dst_label_path = os.path.join(labels_split_dir, label_name)
            
            # Copy files
            if os.path.exists(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)
            if os.path.exists(src_label_path):
                shutil.copy2(src_label_path, dst_label_path)
    
    # Create data.yaml for each split
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split_name)
        data_yaml = {
            "path": split_dir,
            "train": "images",
            "val": "images",
            "nc": 1,
            "names": ["ball"]
        }
        
        with open(os.path.join(split_dir, "data.yaml"), "w") as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✅ Dataset split completed!")
    print(f"📁 Output directory: {output_dir}")
    
    return output_dir


class BallCropsDataset(torch.utils.data.Dataset):
    """Dataset for ball crops with segmentation masks."""
    
    def __init__(self, images_dir, labels_dir, transform=None, target_size=(256, 256)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of images
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        label_path = os.path.join(self.labels_dir, image_name.replace('.jpg', '.txt'))
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create mask
        mask = create_mask_from_obb(image_path, label_path, self.target_size)
        if mask is None:
            mask = np.zeros(self.target_size, dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).float() / 255.0
        
        return image, mask


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, device='cuda'):
    """
    Train the U-Net model.
    
    Args:
        model: U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (str): Device to train on
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"🚀 Starting training for {num_epochs} epochs...")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                val_loss += loss.item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ball_segmentation_model.pt')
            print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pt')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.title('Training History (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_losses, val_losses


def main():
    """Main training function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_ball_segmentation.py <ball_crops_dataset_dir> [output_dir] [num_epochs]")
        print("\nExample:")
        print("  python train_ball_segmentation.py ball_dataset_for_labeling/ball_crops_dataset")
        print("  python train_ball_segmentation.py ball_dataset_for_labeling/ball_crops_dataset training_output 100")
        return
    
    # Configuration
    ball_crops_dataset_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "ball_segmentation_training"
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    # Check if dataset exists
    if not os.path.exists(ball_crops_dataset_dir):
        print(f"❌ Ball crops dataset not found: {ball_crops_dataset_dir}")
        print("💡 First run the detection script to create the dataset:")
        print("   python detect_and_save_balls.py --update-dataset <dataset_dir> <ball_images_dir>")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create train/val/test split
    print("🔄 Creating train/val/test split...")
    split_dataset_dir = create_ball_crops_dataset(
        os.path.join(ball_crops_dataset_dir, "images"),
        os.path.join(ball_crops_dataset_dir, "labels"),
        os.path.join(output_dir, "split_dataset")
    )
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = BallCropsDataset(
        os.path.join(split_dataset_dir, "train", "images"),
        os.path.join(split_dataset_dir, "train", "labels"),
        transform=transform
    )
    
    val_dataset = BallCropsDataset(
        os.path.join(split_dataset_dir, "val", "images"),
        os.path.join(split_dataset_dir, "val", "labels"),
        transform=transform
    )
    
    test_dataset = BallCropsDataset(
        os.path.join(split_dataset_dir, "test", "images"),
        os.path.join(split_dataset_dir, "test", "labels"),
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model
    print("🏗️ Initializing U-Net model...")
    model = BallUNet()
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        device=device
    )
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Best model saved as: best_ball_segmentation_model.pt")
    print(f"📈 Training history saved as: training_history.png")


if __name__ == "__main__":
    main() 