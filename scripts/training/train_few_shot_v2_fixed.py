#!/usr/bin/env python3
"""
Few-Shot Ball Segmentation Training (Fixed Version) - No MLflow
Trains a U-Net model for few-shot ball segmentation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import json
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
# import mlflow  # Commented out
# import mlflow.pytorch  # Commented out


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


class FewShotDatasetV2Fixed(torch.utils.data.Dataset):
    """Dataset for few-shot segmentation with proper train/val/test structure (Fixed)."""
    
    def __init__(self, split_dir, transform=None, target_size=(256, 256)):
        self.split_dir = split_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of query images
        query_dir = os.path.join(split_dir, 'query')
        self.query_images = [f for f in os.listdir(query_dir) if f.endswith('.jpg')]
        
        # Validate that all corresponding files exist
        self.valid_samples = []
        for query_file in self.query_images:
            base_name = query_file.replace('.jpg', '')
            support_file = f"{base_name}_support.jpg"
            mask_file = f"{base_name}_mask.png"
            
            support_path = os.path.join(split_dir, 'support', support_file)
            mask_path = os.path.join(split_dir, 'masks', mask_file)
            
            if os.path.exists(support_path) and os.path.exists(mask_path):
                self.valid_samples.append({
                    'query_file': query_file,
                    'support_file': support_file,
                    'mask_file': mask_file,
                    'base_name': base_name
                })
        
        print(f"  📊 Found {len(self.valid_samples)} valid samples out of {len(self.query_images)} query images")
        
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        query_path = os.path.join(self.split_dir, 'query', sample['query_file'])
        support_path = os.path.join(self.split_dir, 'support', sample['support_file'])
        mask_path = os.path.join(self.split_dir, 'masks', sample['mask_file'])
        
        # Load images
        query_image = Image.open(query_path).convert('RGB')
        support_image = Image.open(support_path).convert('RGB')
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(self.target_size, dtype=np.uint8)
        
        mask = cv2.resize(mask, (256, 256))
        
        # Apply transforms
        if self.transform:
            query_image = self.transform(query_image)
            support_image = self.transform(support_image)
        
        # Concatenate query and support images (6 channels)
        input_tensor = torch.cat([query_image, support_image], dim=0)
        
        # Convert mask to tensor and resize
        mask = cv2.resize(mask, self.target_size)
        mask = torch.from_numpy(mask).float() / 255.0
        
        return input_tensor, mask


def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """Calculate precision, recall, and F1 score for segmentation."""
    # Convert to binary
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > 0.5).astype(np.uint8)
    
    # Flatten for sklearn metrics
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    
    # Check if we have any positive samples
    if np.sum(true_flat) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Calculate metrics
    try:
        precision = precision_score(true_flat, pred_flat, average='binary', zero_division=0)
        recall = recall_score(true_flat, pred_flat, average='binary', zero_division=0)
        f1 = f1_score(true_flat, pred_flat, average='binary', zero_division=0)
    except Exception as e:
        print(f"⚠️ Error calculating metrics: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_epoch(model, data_loader, criterion, device):
    """Evaluate model for one epoch and return loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            total_loss += loss.item()
            
            # Calculate metrics for each sample
            pred_masks = outputs.squeeze().cpu().numpy()
            true_masks = masks.cpu().numpy()
            
            for pred_mask, true_mask in zip(pred_masks, true_masks):
                metrics = calculate_metrics(pred_mask, true_mask)
                all_metrics.append(metrics)
    
    avg_loss = total_loss / len(data_loader)
    
    # Calculate average metrics
    avg_metrics = {}
    for metric_name in ['precision', 'recall', 'f1_score']:
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    
    return avg_loss, avg_metrics


def train_model_with_mlflow_v2_fixed(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    experiment_name="few_shot_ball_segmentation_v2_fixed",
    save_dir="models/checkpoints"
):
    """Train the few-shot segmentation model with MLflow tracking."""
    
    # Set up MLflow - COMMENTED OUT
    # mlflow.set_experiment(experiment_name)
    
    # with mlflow.start_run():  # COMMENTED OUT
    # mlflow.log_params({  # COMMENTED OUT
    #     "model_type": "UNet",
    #     "in_channels": 6,
    #     "out_channels": 1,
    #     "num_epochs": num_epochs,
    #     "learning_rate": optimizer.param_groups[0]['lr'],
    #     "batch_size": train_loader.batch_size,
    #     "criterion": "BCELoss",
    #     "optimizer": "Adam"
    # })

    model = model.to(device)
    criterion = criterion
    optimizer = optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1_scores = []
    val_f1_scores = []
    best_val_precision = 0.0
    
    print(f"🚀 Starting few-shot training for {num_epochs} epochs...")
    print(f"📊 Training samples: {len(train_loader.dataset)}")
    print(f"📊 Validation samples: {len(val_loader.dataset)}")
    print(f"📊 Test samples: {len(test_loader.dataset)}")
    print(f"🔬 MLflow experiment: {experiment_name}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics_list = []
        
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
            
            # Calculate metrics for this batch
            pred_masks = outputs.squeeze().cpu().numpy()
            true_masks = masks.cpu().numpy()
            
            for pred_mask, true_mask in zip(pred_masks, true_masks):
                metrics = calculate_metrics(pred_mask, true_mask)
                train_metrics_list.append(metrics)
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calculate average training metrics
        train_avg_metrics = {}
        for metric_name in ['precision', 'recall', 'f1_score']:
            train_avg_metrics[metric_name] = np.mean([m[metric_name] for m in train_metrics_list])
        
        train_precisions.append(train_avg_metrics['precision'])
        train_recalls.append(train_avg_metrics['recall'])
        train_f1_scores.append(train_avg_metrics['f1_score'])
        
        # Validation phase using the new evaluation function
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_precisions.append(val_metrics['precision'])
        val_recalls.append(val_metrics['recall'])
        val_f1_scores.append(val_metrics['f1_score'])
        
        # Learning rate scheduling based on validation loss
        scheduler.step(val_loss)
        
        # Log metrics to MLflow - COMMENTED OUT
        # mlflow.log_metrics({
        #     "train_loss": train_loss,
        #     "val_loss": val_loss,
        #     "train_precision": train_avg_metrics['precision'],
        #     "val_precision": val_metrics['precision'],
        #     "train_recall": train_avg_metrics['recall'],
        #     "val_recall": val_metrics['recall'],
        #     "train_f1": train_avg_metrics['f1_score'],
        #     "val_f1": val_metrics['f1_score'],
        #     "learning_rate": optimizer.param_groups[0]['lr']
        # }, step=epoch)
        
        # Save best model based on validation precision
        if val_metrics['precision'] > best_val_precision:
            best_val_precision = val_metrics['precision']
            # mlflow.pytorch.log_model(model, "best_model")  # COMMENTED OUT
            print(f"💾 Saved best model (val_precision: {val_metrics['precision']:.4f})")
            
            # Save the best model checkpoint
            best_model_path = f'{save_dir}/best_few_shot_model.pt'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Precision: {train_avg_metrics['precision']:.4f}, Recall: {train_avg_metrics['recall']:.4f}, F1: {train_avg_metrics['f1_score']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'{save_dir}/few_shot_v2_fixed_checkpoint_epoch_{epoch+1}.pt'
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_precision': train_avg_metrics['precision'],
                'val_precision': val_metrics['precision'],
                'train_recall': train_avg_metrics['recall'],
                'val_recall': val_metrics['recall'],
                'train_f1': train_avg_metrics['f1_score'],
                'val_f1': val_metrics['f1_score'],
            }, checkpoint_path)
            # mlflow.log_artifact(checkpoint_path)  # COMMENTED OUT
        
        # Test phase using the new evaluation function
        test_loss, test_metrics = evaluate_epoch(model, test_loader, criterion, device)
        # mlflow.log_metrics({"test_loss": test_loss})  # COMMENTED OUT
        # mlflow.pytorch.log_model(model, "final_model")  # COMMENTED OUT
        
        # Create and log training plots
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 3, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training History - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Precision plot
        plt.subplot(2, 3, 2)
        plt.plot(train_precisions, label='Train Precision')
        plt.plot(val_precisions, label='Val Precision')
        plt.title('Training History - Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        
        # Recall plot
        plt.subplot(2, 3, 3)
        plt.plot(train_recalls, label='Train Recall')
        plt.plot(val_recalls, label='Val Recall')
        plt.title('Training History - Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        
        # F1 Score plot
        plt.subplot(2, 3, 4)
        plt.plot(train_f1_scores, label='Train F1')
        plt.plot(val_f1_scores, label='Val F1')
        plt.title('Training History - F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        # Loss plot (Log Scale)
        plt.subplot(2, 3, 5)
        plt.plot(train_losses, label='Train Loss', alpha=0.7)
        plt.plot(val_losses, label='Val Loss', alpha=0.7)
        plt.title('Training History - Loss (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        # Metrics comparison
        plt.subplot(2, 3, 6)
        plt.plot(val_precisions, label='Val Precision', marker='o')
        plt.plot(val_recalls, label='Val Recall', marker='s')
        plt.plot(val_f1_scores, label='Val F1', marker='^')
        plt.title('Validation Metrics Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = f'{save_dir}/few_shot_v2_fixed_training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        # mlflow.log_artifact(plot_path)  # COMMENTED OUT
        plt.close()
        
        return train_losses, val_losses, test_loss, {
            'train_precisions': train_precisions,
            'val_precisions': val_precisions,
            'train_recalls': train_recalls,
            'val_recalls': val_recalls,
            'train_f1_scores': train_f1_scores,
            'val_f1_scores': val_f1_scores,
            'test_metrics': test_metrics
        }


def main():
    """Main training function with MLflow tracking."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python train_few_shot_v2_fixed.py <dataset_dir> [num_epochs]")
        print("\nExample:")
        print("  python train_few_shot_v2_fixed.py combined_datasets/combined_run_20250618_005343")
        print("  python train_few_shot_v2_fixed.py combined_datasets/combined_run_20250618_005343 100")
        return
    
    # Configuration
    dataset_dir = sys.argv[1]
    num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset not found: {dataset_dir}")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = FewShotDatasetV2Fixed(
        os.path.join(dataset_dir, "train"),
        transform=transform
    )
    
    val_dataset = FewShotDatasetV2Fixed(
        os.path.join(dataset_dir, "val"),
        transform=transform
    )
    
    test_dataset = FewShotDatasetV2Fixed(
        os.path.join(dataset_dir, "test"),
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize model
    print("🏗️ Initializing Siamese U-Net model for few-shot segmentation...")
    model = UNet(in_channels=6, out_channels=1)  # 6 channels for Siamese (3 query + 3 support)
    
    # Train model with MLflow tracking - COMMENTED OUT
    # train_losses, val_losses, test_loss = train_model_with_mlflow_v2_fixed(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     test_loader=test_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     num_epochs=num_epochs,
    #     device=device,
    #     experiment_name=experiment_name,
    #     save_dir=save_dir
    # )

    # Train model WITHOUT MLflow tracking
    train_losses, val_losses, test_loss, metrics = train_model_with_mlflow_v2_fixed(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.BCELoss(),
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=num_epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        experiment_name=f"few_shot_ball_segmentation_v2_fixed_{os.path.basename(dataset_dir)}",
        save_dir="models/checkpoints"
    )

    print(f"✅ Training completed!")
    # print(f"🔬 MLflow experiment: {experiment_name}")  # COMMENTED OUT
    print(f"📊 Final test loss: {test_loss:.4f}")
    print(f"📊 Final test precision: {metrics['test_metrics']['precision']:.4f}")
    print(f"📊 Final test recall: {metrics['test_metrics']['recall']:.4f}")
    print(f"📊 Final test F1 score: {metrics['test_metrics']['f1_score']:.4f}")
    # print(f"💾 Models logged to MLflow")  # COMMENTED OUT
    # print(f"📈 Training history logged to MLflow")  # COMMENTED OUT
    print(f"💾 Models saved to: models/checkpoints")
    print(f"📈 Training history saved to: results/viz/")
    print(f"🏆 Best validation precision: {max(metrics['val_precisions']):.4f}")


if __name__ == "__main__":
    main() 