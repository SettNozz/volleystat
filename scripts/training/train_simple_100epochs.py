#!/usr/bin/env python3
"""
Simple Training Script - 100 Epochs
Trains a U-Net model for few-shot ball segmentation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

# Enable cuDNN benchmark for speed
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

class SiameseDataset(Dataset):
    def __init__(self, dataset_dir, split='train', transform=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        
        # Directories
        self.query_dir = os.path.join(dataset_dir, split, 'query')
        self.support_dir = os.path.join(dataset_dir, split, 'support')
        self.masks_dir = os.path.join(dataset_dir, split, 'masks')
        
        # Get all samples
        self.samples = []
        query_files = [f for f in os.listdir(self.query_dir) if f.endswith('.jpg')]
        
        for query_file in query_files:
            base_name = query_file.replace('.jpg', '')
            support_file = f"{base_name}_support.jpg"
            mask_file = f"{base_name}_mask.png"
            
            support_path = os.path.join(self.support_dir, support_file)
            mask_path = os.path.join(self.masks_dir, mask_file)
            
            if os.path.exists(support_path) and os.path.exists(mask_path):
                self.samples.append({
                    'query_file': query_file,
                    'support_file': support_file,
                    'mask_file': mask_file,
                    'base_name': base_name
                })
        
        print(f"  📊 Found {len(self.samples)} valid samples out of {len(query_files)} query images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        query_path = os.path.join(self.query_dir, sample['query_file'])
        support_path = os.path.join(self.support_dir, sample['support_file'])
        mask_path = os.path.join(self.masks_dir, sample['mask_file'])
        
        query_image = Image.open(query_path).convert('RGB')
        support_image = Image.open(support_path).convert('RGB')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Resize mask to 256x256 to match model output
        mask = cv2.resize(mask, (256, 256))
        
        # Apply transforms
        if self.transform:
            query_tensor = self.transform(query_image)
            support_tensor = self.transform(support_image)
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
        
        return query_tensor, support_tensor, mask_tensor

def main():
    """Main training function."""
    
    if len(sys.argv) < 2:
        print("Usage: python train_simple_100epochs.py <dataset_dir> [num_epochs]")
        print("\nExample:")
        print("  python train_simple_100epochs.py data/datasets/combined_datasets/combined_run_20250618_005343")
        print("  python train_simple_100epochs.py data/datasets/combined_datasets/combined_run_20250618_005343 100")
        return
    
    # Configuration
    dataset_dir = sys.argv[1]
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"📊 Creating datasets...")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = SiameseDataset(dataset_dir, 'train', transform)
    val_dataset = SiameseDataset(dataset_dir, 'val', transform)
    test_dataset = SiameseDataset(dataset_dir, 'test', transform)
    
    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        num_workers=min(8, os.cpu_count()), pin_memory=True, persistent_workers=True
    )
    
    print(f"🏗️ Initializing Siamese U-Net model for few-shot segmentation...")
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=6, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print(f"🚀 Starting few-shot training for {num_epochs} epochs...")
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Test samples: {len(test_dataset)}")
    
    # Create directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("results/viz", exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, batch in enumerate(train_loader):
            query, support, mask = batch
            query, support, mask = query.to(device), support.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            # Concatenate query and support images
            inputs = torch.cat([query, support], dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, mask)
            
            loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss
            loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * query.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                query, support, mask = batch
                query, support, mask = query.to(device), support.to(device), mask.to(device)
                
                inputs = torch.cat([query, support], dim=1)
                outputs = model(inputs)
                loss = criterion(outputs, mask)
                
                val_running_loss += loss.item() * query.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join("models/checkpoints", f"best_few_shot_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join("models/checkpoints", f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Test on test set
    model.eval()
    test_running_loss = 0.0
    test_pbar = tqdm(test_loader, desc="Final Test")
    
    with torch.no_grad():
        for batch in test_pbar:
            query, support, mask = batch
            query, support, mask = query.to(device), support.to(device), mask.to(device)
            
            inputs = torch.cat([query, support], dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, mask)
            
            test_running_loss += loss.item() * query.size(0)
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    test_loss = test_running_loss / len(test_loader.dataset)
    
    # Save training history plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = "results/viz/training_history.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training completed!")
    print(f"📊 Final test loss: {test_loss:.4f}")
    print(f"💾 Models saved to: models/checkpoints")
    print(f"📈 Training history saved to: results/viz/")

if __name__ == "__main__":
    main() 