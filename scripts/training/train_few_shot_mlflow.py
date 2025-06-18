#!/usr/bin/env python3
"""
Few-Shot Ball Segmentation Training with MLflow Tracking
Trains U-Net model for few-shot segmentation with experiment tracking.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from datetime import datetime


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.bottleneck = conv_block(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))

        up1 = self.up1(bottleneck)
        dec1 = self.decoder1(torch.cat([up1, enc2], dim=1))

        up2 = self.up2(dec1)
        dec2 = self.decoder2(torch.cat([up2, enc1], dim=1))

        return torch.sigmoid(self.final(dec2))


class SiameseDataset(torch.utils.data.Dataset):
    """Dataset for few-shot segmentation with support/query/masks format."""
    
    def __init__(self, support_dir, query_dir, masks_dir, transform=None, target_size=(256, 256)):
        self.support_dir = support_dir
        self.query_dir = query_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of query images
        self.query_images = [f for f in os.listdir(query_dir) if f.endswith('.jpg')]
        self.support_images = [f for f in os.listdir(support_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.query_images)
    
    def __getitem__(self, idx):
        query_name = self.query_images[idx]
        query_path = os.path.join(self.query_dir, query_name)
        mask_name = query_name.replace('.jpg', '_mask.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Randomly select a support image
        support_name = np.random.choice(self.support_images)
        support_path = os.path.join(self.support_dir, support_name)
        
        # Load images
        query_image = Image.open(query_path).convert('RGB')
        support_image = Image.open(support_path).convert('RGB')
        
        # Load mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(self.target_size, dtype=np.uint8)
        else:
            mask = np.zeros(self.target_size, dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            query_image = self.transform(query_image)
            support_image = self.transform(support_image)
        
        # Concatenate query and support images (6 channels)
        input_tensor = torch.cat([query_image, support_image], dim=0)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).float() / 255.0
        
        return input_tensor, mask


def train_model_with_mlflow(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=100, 
    learning_rate=1e-3, 
    device='cuda',
    experiment_name="few_shot_ball_segmentation"
):
    """Train the few-shot segmentation model with MLflow tracking."""
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": train_loader.batch_size,
            "model_type": "Siamese U-Net",
            "input_channels": 6,
            "device": device
        })
        
        model = model.to(device)
        criterion = nn.BCELoss()  # Use BCELoss since model outputs sigmoid
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # Training history
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        print(f"🚀 Starting few-shot training for {num_epochs} epochs...")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        print(f"🔬 MLflow experiment: {experiment_name}")
        
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
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                mlflow.pytorch.log_model(model, "best_model")
                print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'checkpoints/few_shot_checkpoint_epoch_{epoch+1}.pt'
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
        
        # Log final model
        mlflow.pytorch.log_model(model, "final_model")
        
        # Create and log training plots
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
        plot_path = 'few_shot_training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        
        return train_losses, val_losses


def main():
    """Main training function with MLflow tracking."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_few_shot_mlflow.py <dataset_dir> [num_epochs]")
        print("\nExample:")
        print("  python train_few_shot_mlflow.py few_shot_datasets/few_shot_run_20241201_120000")
        print("  python train_few_shot_mlflow.py few_shot_datasets/few_shot_run_20241201_120000 100")
        return
    
    # Configuration
    dataset_dir = sys.argv[1]
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset not found: {dataset_dir}")
        print("💡 First run the dataset preparation script:")
        print("   python prepare_few_shot_dataset.py")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = SiameseDataset(
        os.path.join(dataset_dir, "support"),
        os.path.join(dataset_dir, "query"),
        os.path.join(dataset_dir, "masks"),
        transform=transform
    )
    
    # For few-shot, we'll use the same dataset for train/val split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model
    print("🏗️ Initializing Siamese U-Net model for few-shot segmentation...")
    model = UNet(in_channels=6, out_channels=1)  # 6 channels for Siamese (3 query + 3 support)
    
    # Train model with MLflow tracking
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    
    # Create experiment name based on dataset
    dataset_name = os.path.basename(dataset_dir)
    experiment_name = f"few_shot_ball_segmentation_{dataset_name}"
    
    train_losses, val_losses = train_model_with_mlflow(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        device=device,
        experiment_name=experiment_name
    )
    
    print(f"\n🎉 Few-shot training completed!")
    print(f"🔬 MLflow experiment: {experiment_name}")
    print(f"💾 Models logged to MLflow")
    print(f"📈 Training history logged to MLflow")


if __name__ == "__main__":
    main() 