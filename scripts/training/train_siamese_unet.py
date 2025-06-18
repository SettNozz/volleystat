#!/usr/bin/env python3
"""
Siamese UNet Training Script
Trains a Siamese UNet model for ball segmentation with comprehensive metrics and testing.
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
    def __init__(self, support_dir, query_dir, mask_dir, transform=None):
        self.support_dir = support_dir
        self.query_dir = query_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.query_images = sorted([f for f in os.listdir(query_dir) if f.endswith('.jpg')])
        
        print(f"Found {len(self.query_images)} samples in {query_dir}")

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, idx):
        query_name = self.query_images[idx]
        support_name = query_name.replace(".jpg", "_ball.jpg")
        mask_name = query_name.replace(".jpg", "_mask.png")

        # Load images
        support_img = Image.open(os.path.join(self.support_dir, support_name)).convert("RGB")
        query_img = Image.open(os.path.join(self.query_dir, query_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

        # Apply transforms
        if self.transform:
            support_img = self.transform(support_img)
            query_img = self.transform(query_img)
            mask = self.transform(mask)

        # Concatenate query and support images (6 channels)
        input_tensor = torch.cat([query_img, support_img], dim=0)
        
        print(f"Query: {query_name} -> Support: {support_name}")
        
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
        print(f"Error calculating metrics: {e}")
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
        for inputs, masks in data_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            outputs = model(inputs)
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


def test_model_with_visualizations(model, test_loader, device, epoch, save_dir="results/test_epochs"):
    """Test model and save visualizations for each sample."""
    model.eval()
    test_loss = 0.0
    all_metrics = []
    
    # Create directory for this epoch's results
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    os.makedirs(os.path.join(epoch_dir, "visualizations"), exist_ok=True)
    
    print(f"Testing model at epoch {epoch}...")
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = nn.BCELoss()(outputs, masks.unsqueeze(1))
            test_loss += loss.item()
            
            # Process each sample in the batch
            pred_masks = outputs.squeeze().cpu().numpy()
            true_masks = masks.cpu().numpy()
            
            # Split the 6-channel input back to query and support
            query_images = images[:, :3, :, :].cpu().numpy()  # First 3 channels
            support_images = images[:, 3:, :, :].cpu().numpy()  # Last 3 channels
            
            for sample_idx in range(images.size(0)):
                pred_mask = pred_masks[sample_idx]
                true_mask = true_masks[sample_idx]
                query_img = query_images[sample_idx]
                support_img = support_images[sample_idx]
                
                # Calculate metrics
                metrics = calculate_metrics(pred_mask, true_mask)
                all_metrics.append(metrics)
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Query image
                query_img_display = np.transpose(query_img, (1, 2, 0))
                query_img_display = np.clip(query_img_display, 0, 1)
                axes[0, 0].imshow(query_img_display)
                axes[0, 0].set_title("Query Image")
                axes[0, 0].axis('off')
                
                # Support image
                support_img_display = np.transpose(support_img, (1, 2, 0))
                support_img_display = np.clip(support_img_display, 0, 1)
                axes[0, 1].imshow(support_img_display)
                axes[0, 1].set_title("Support Image")
                axes[0, 1].axis('off')
                
                # Ground truth mask
                axes[1, 0].imshow(true_mask, cmap='gray')
                axes[1, 0].set_title("Ground Truth Mask")
                axes[1, 0].axis('off')
                
                # Predicted mask
                pred_binary = (pred_mask > 0.1).astype(np.uint8)
                axes[1, 1].imshow(pred_binary, cmap='gray')
                axes[1, 1].set_title(f"Predicted Mask\nPrecision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Save visualization
                viz_path = os.path.join(epoch_dir, "visualizations", f"sample_{batch_idx}_{sample_idx}_epoch_{epoch}.png")
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
    
    test_loss /= len(test_loader)
    
    # Calculate average metrics
    avg_metrics = {}
    for metric_name in ['precision', 'recall', 'f1_score']:
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    
    # Save results
    results = {
        'epoch': epoch,
        'test_loss': test_loss,
        'average_metrics': avg_metrics,
        'individual_metrics': all_metrics,
        'num_samples': len(all_metrics)
    }
    
    results_path = os.path.join(epoch_dir, f"test_results_epoch_{epoch}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Epoch {epoch} Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print(f"  F1 Score: {avg_metrics['f1_score']:.4f}")
    print(f"  Results saved to: {epoch_dir}")
    
    return test_loss, avg_metrics


def train_model(train_loader, val_loader, test_loader, num_epochs, device, save_dir="models/checkpoints"):
    """Train the Siamese UNet model."""
    
    # Initialize model, criterion, optimizer
    model = UNet(in_channels=6, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
    
    # Test results tracking
    test_epochs = []
    test_losses = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    
    print(f"Starting Siamese UNet training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Will test model every 25 epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics_list = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, masks in train_pbar:
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(inputs)
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
        
        # Validation phase
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_precisions.append(val_metrics['precision'])
        val_recalls.append(val_metrics['recall'])
        val_f1_scores.append(val_metrics['f1_score'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model based on validation precision
        if val_metrics['precision'] > best_val_precision:
            best_val_precision = val_metrics['precision']
            print(f"Saved best model (val_precision: {val_metrics['precision']:.4f})")
            
            # Save the best model checkpoint
            best_model_path = f'{save_dir}/best_siamese_model.pt'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Precision: {train_avg_metrics['precision']:.4f}, Recall: {train_avg_metrics['recall']:.4f}, F1: {train_avg_metrics['f1_score']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        
        # Test model every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"\nTesting model at epoch {epoch + 1}...")
            test_loss, test_metrics = test_model_with_visualizations(
                model, test_loader, device, epoch + 1, 
                save_dir=f"{save_dir}/test_epochs"
            )
            
            # Track test results
            test_epochs.append(epoch + 1)
            test_losses.append(test_loss)
            test_precisions.append(test_metrics['precision'])
            test_recalls.append(test_metrics['recall'])
            test_f1_scores.append(test_metrics['f1_score'])
            
            print(f"Test at epoch {epoch + 1}:")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")
            print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'{save_dir}/siamese_checkpoint_epoch_{epoch+1}.pt'
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
    
    # Final test at the end of training
    print(f"\nFinal test at epoch {num_epochs}...")
    final_test_loss, final_test_metrics = test_model_with_visualizations(
        model, test_loader, device, num_epochs, 
        save_dir=f"{save_dir}/test_epochs"
    )
    
    # Track final test results
    test_epochs.append(num_epochs)
    test_losses.append(final_test_loss)
    test_precisions.append(final_test_metrics['precision'])
    test_recalls.append(final_test_metrics['recall'])
    test_f1_scores.append(final_test_metrics['f1_score'])
    
    print(f"Final test results:")
    print(f"  Loss: {final_test_loss:.4f}")
    print(f"  Precision: {final_test_metrics['precision']:.4f}")
    print(f"  Recall: {final_test_metrics['recall']:.4f}")
    print(f"  F1 Score: {final_test_metrics['f1_score']:.4f}")
    
    # Create comprehensive training plots
    plt.figure(figsize=(20, 12))
    
    # Loss plot
    plt.subplot(2, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Precision plot
    plt.subplot(2, 4, 2)
    plt.plot(train_precisions, label='Train Precision')
    plt.plot(val_precisions, label='Val Precision')
    if test_epochs:
        plt.scatter(test_epochs, test_precisions, color='red', s=50, label='Test Precision', zorder=5)
    plt.title('Training History - Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    
    # Recall plot
    plt.subplot(2, 4, 3)
    plt.plot(train_recalls, label='Train Recall')
    plt.plot(val_recalls, label='Val Recall')
    if test_epochs:
        plt.scatter(test_epochs, test_recalls, color='red', s=50, label='Test Recall', zorder=5)
    plt.title('Training History - Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    
    # F1 Score plot
    plt.subplot(2, 4, 4)
    plt.plot(train_f1_scores, label='Train F1')
    plt.plot(val_f1_scores, label='Val F1')
    if test_epochs:
        plt.scatter(test_epochs, test_f1_scores, color='red', s=50, label='Test F1', zorder=5)
    plt.title('Training History - F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Test metrics comparison
    plt.subplot(2, 4, 5)
    if test_epochs:
        plt.plot(test_epochs, test_precisions, 'o-', label='Test Precision', linewidth=2, markersize=8)
        plt.plot(test_epochs, test_recalls, 's-', label='Test Recall', linewidth=2, markersize=8)
        plt.plot(test_epochs, test_f1_scores, '^-', label='Test F1', linewidth=2, markersize=8)
        plt.title('Test Metrics Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
    
    # Test loss over epochs
    plt.subplot(2, 4, 6)
    if test_epochs:
        plt.plot(test_epochs, test_losses, 'o-', label='Test Loss', linewidth=2, markersize=8, color='red')
        plt.title('Test Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    
    # Validation metrics comparison
    plt.subplot(2, 4, 7)
    plt.plot(val_precisions, label='Val Precision', marker='o')
    plt.plot(val_recalls, label='Val Recall', marker='s')
    plt.plot(val_f1_scores, label='Val F1', marker='^')
    plt.title('Validation Metrics Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Loss plot (Log Scale)
    plt.subplot(2, 4, 8)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.title('Training History - Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = f'{save_dir}/siamese_training_history.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save test results summary
    test_summary = {
        'test_epochs': test_epochs,
        'test_losses': test_losses,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_f1_scores': test_f1_scores,
        'best_test_precision': max(test_precisions) if test_precisions else 0,
        'best_test_recall': max(test_recalls) if test_recalls else 0,
        'best_test_f1': max(test_f1_scores) if test_f1_scores else 0,
        'best_test_epoch': test_epochs[np.argmax(test_precisions)] if test_precisions else 0
    }
    
    summary_path = f'{save_dir}/test_results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    print(f"Test Results Summary:")
    print(f"  Best test precision: {test_summary['best_test_precision']:.4f} at epoch {test_summary['best_test_epoch']}")
    print(f"  Best test recall: {test_summary['best_test_recall']:.4f}")
    print(f"  Best test F1: {test_summary['best_test_f1']:.4f}")
    print(f"  Summary saved to: {summary_path}")
    
    return train_losses, val_losses, test_losses, {
        'train_precisions': train_precisions,
        'val_precisions': val_precisions,
        'train_recalls': train_recalls,
        'val_recalls': val_recalls,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'test_losses': test_losses,
        'test_precisions': test_precisions,
        'test_recalls': test_recalls,
        'test_f1_scores': test_f1_scores,
        'test_summary': test_summary
    }


def main():
    """Main training function."""
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    # Dataset paths
    train_support_path = "data/datasets/combined_datasets/combined_run_20250618_005343/train/support/"
    train_query_path = "data/datasets/combined_datasets/combined_run_20250618_005343/train/query/"
    train_mask_path = "data/datasets/combined_datasets/combined_run_20250618_005343/train/masks/"

    val_support_path = "data/datasets/combined_datasets/combined_run_20250618_005343/val/support/"
    val_query_path = "data/datasets/combined_datasets/combined_run_20250618_005343/val/query/"
    val_mask_path = "data/datasets/combined_datasets/combined_run_20250618_005343/val/masks/"

    test_support_path = "data/datasets/combined_datasets/combined_run_20250618_005343/test/support/"
    test_query_path ="data/datasets/combined_datasets/combined_run_20250618_005343/test/query/"
    test_mask_path = "data/datasets/combined_datasets/combined_run_20250618_005343/test/masks/"
    # train_support_path = "data/train_val_test_prepared_for_training/train/support/"
    # train_query_path = "data/train_val_test_prepared_for_training/train/query/"
    # train_mask_path = "data/train_val_test_prepared_for_training/train/masks/"
    #
    # val_support_path = "data/train_val_test_prepared_for_training/val/support/"
    # val_query_path = "data/train_val_test_prepared_for_training/val/query/"
    # val_mask_path = "data/train_val_test_prepared_for_training/val/masks/"
    #
    # test_support_path = "data/train_val_test_prepared_for_training/test/support/"
    # test_query_path = "data/train_val_test_prepared_for_training/test/query/"
    # test_mask_path = "data/train_val_test_prepared_for_training/test/masks/"
    
    # Check if datasets exist
    for path in [train_support_path, train_query_path, train_mask_path,
                 val_support_path, val_query_path, val_mask_path,
                 test_support_path, test_query_path, test_mask_path]:
        if not os.path.exists(path):
            print(f"Dataset path not found: {path}")
            return
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SiameseDataset(train_support_path, train_query_path, train_mask_path, transform)
    val_dataset = SiameseDataset(val_support_path, val_query_path, val_mask_path, transform)
    test_dataset = SiameseDataset(test_support_path, test_query_path, test_mask_path, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 80
    
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train model
    train_losses, val_losses, test_losses, metrics = train_model(
        train_loader, val_loader, test_loader, num_epochs, device
    )
    
    print(f"Training completed!")
    print(f"Final test loss: {test_losses[-1]:.4f}")
    print(f"Final test precision: {metrics['test_precisions'][-1]:.4f}")
    print(f"Final test recall: {metrics['test_recalls'][-1]:.4f}")
    print(f"Final test F1 score: {metrics['test_f1_scores'][-1]:.4f}")
    print(f"Models saved to: models/checkpoints")
    print(f"Best validation precision: {max(metrics['val_precisions']):.4f}")
    print(f"Best test precision: {metrics['test_summary']['best_test_precision']:.4f} at epoch {metrics['test_summary']['best_test_epoch']}")
    print(f"Test visualizations saved to: models/checkpoints/test_epochs/")


if __name__ == "__main__":
    main() 