#!/usr/bin/env python3
"""
Few-Shot Ball Segmentation Training Script
Trains U-Net model for few-shot segmentation with Images/Labels/Masks format.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.unet import UNet


class FewShotDataset(torch.utils.data.Dataset):
    """Dataset for few-shot segmentation with Images/Labels/Masks format."""
    
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of images
        self.images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_name = image_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(self.target_size, dtype=np.uint8)
        else:
            mask = np.zeros(self.target_size, dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).float() / 255.0
        
        return image, mask


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, device='cuda'):
    """Train the few-shot segmentation model."""
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"🚀 Starting few-shot training for {num_epochs} epochs...")
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
            torch.save(model.state_dict(), 'best_few_shot_model.pt')
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
            }, f'few_shot_checkpoint_epoch_{epoch+1}.pt')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Few-Shot Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.title('Few-Shot Training History (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('few_shot_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_losses, val_losses


def main():
    """Main training function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_few_shot.py <dataset_dir> [num_epochs]")
        print("\nExample:")
        print("  python train_few_shot.py few_shot_dataset")
        print("  python train_few_shot.py few_shot_dataset 100")
        return
    
    # Configuration
    dataset_dir = sys.argv[1]
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset not found: {dataset_dir}")
        print("💡 First run the conversion script:")
        print("   python convert_to_few_shot_format.py")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = FewShotDataset(
        os.path.join(dataset_dir, "train", "Images"),
        os.path.join(dataset_dir, "train", "Masks"),
        transform=transform
    )
    
    val_dataset = FewShotDataset(
        os.path.join(dataset_dir, "val", "Images"),
        os.path.join(dataset_dir, "val", "Masks"),
        transform=transform
    )
    
    test_dataset = FewShotDataset(
        os.path.join(dataset_dir, "test", "Images"),
        os.path.join(dataset_dir, "test", "Masks"),
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model
    print("🏗️ Initializing U-Net model for few-shot segmentation...")
    model = UNet(in_channels=3, out_channels=1)  # 3 channels for RGB images
    
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
    
    print(f"\n🎉 Few-shot training completed!")
    print(f"💾 Best model saved as: best_few_shot_model.pt")
    print(f"📈 Training history saved as: few_shot_training_history.png")


if __name__ == "__main__":
    main() 