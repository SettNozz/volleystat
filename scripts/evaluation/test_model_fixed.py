#!/usr/bin/env python3
"""
Test Best Model Script
Evaluates the best trained model on test dataset with comprehensive metrics and visualizations.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, jaccard_score
import json
from datetime import datetime

print("Using Python executable:", sys.executable)

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
    def __init__(self, dataset_dir, split='test', transform=None):
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
        
        print(f"Found {len(self.samples)} test samples")
    
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
        
        return query_tensor, support_tensor, mask_tensor, sample['base_name']

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """Calculate comprehensive metrics for segmentation."""
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
            'f1_score': 0.0,
            'accuracy': 1.0 if np.sum(pred_flat) == 0 else 0.0,
            'iou': 0.0,
            'dice': 0.0
        }
    
    # Calculate metrics
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='binary', zero_division=0)
        accuracy = accuracy_score(true_flat, pred_flat)
        iou = jaccard_score(true_flat, pred_flat, average='binary', zero_division=0)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'iou': 0.0,
            'dice': 0.0
        }
    
    # Dice coefficient
    intersection = np.sum(pred_binary * true_binary)
    dice = (2 * intersection) / (np.sum(pred_binary) + np.sum(true_binary) + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'iou': iou,
        'dice': dice
    }

def find_best_model():
    """Find the best model checkpoint."""
    checkpoints_dir = "models/checkpoints"
    
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Look for best model files
    best_models = []
    for file in os.listdir(checkpoints_dir):
        if file.startswith("best_few_shot_model_epoch_"):
            best_models.append(file)
    
    if not best_models:
        # Look for any checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        if checkpoint_files:
            # Sort by epoch number and take the latest
            checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
            return os.path.join(checkpoints_dir, checkpoint_files[-1])
        else:
            print(f"No model files found in {checkpoints_dir}")
            return None
    
    # Sort by epoch number and take the latest best model
    best_models.sort(key=lambda x: int(x.split('_')[-1].replace('.pt', '')))
    return os.path.join(checkpoints_dir, best_models[-1])

def test_model(model_path, test_dataset_dir, output_dir="results/test_results"):
    """Test the model on test dataset."""
    
    print(f"Testing model: {model_path}")
    print(f"Test dataset: {test_dataset_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=6, out_channels=1).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model.eval()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset and loader
    test_dataset = SiameseDataset(test_dataset_dir, 'test', transform)
    num_workers = 0  # Force single-process loading for Windows compatibility
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    # Test metrics
    all_metrics = []
    visualization_samples = []
    
    print(f"\nTesting {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for i, (query, support, mask, sample_name) in enumerate(tqdm(test_loader, desc="Testing")):
            query, support, mask = query.to(device), support.to(device), mask.to(device)
            
            # Predict
            inputs = torch.cat([query, support], dim=1)  # Concatenate along channel dimension
            pred_output = model(inputs)
            pred_mask = pred_output.squeeze().cpu().numpy()
            
            # Convert tensors to numpy for metrics
            true_mask = mask.squeeze().cpu().numpy()
            
            # Calculate metrics
            metrics = calculate_metrics(pred_mask, true_mask)
            all_metrics.append(metrics)
            
            # Store for visualization (first 10 samples)
            if i < 10:
                visualization_samples.append({
                    'query': query.squeeze().cpu().numpy().transpose(1, 2, 0),
                    'support': support.squeeze().cpu().numpy().transpose(1, 2, 0),
                    'true_mask': true_mask,
                    'pred_mask': pred_mask,
                    'metrics': metrics,
                    'sample_name': sample_name
                })
    
    # Calculate average metrics
    avg_metrics = {}
    for metric_name in ['precision', 'recall', 'f1_score', 'accuracy', 'iou', 'dice']:
        avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
    
    print(f"\nTest Results:")
    for metric_name, value in avg_metrics.items():
        print(f"   {metric_name}: {value:.4f}")
    
    # Save results
    results = {
        'test_date': datetime.now().isoformat(),
        'model_path': model_path,
        'test_dataset': test_dataset_dir,
        'num_samples': len(test_dataset),
        'average_metrics': avg_metrics,
        'individual_metrics': all_metrics
    }
    
    results_path = os.path.join(output_dir, 'test_results.json')
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    try:
        if visualization_samples:
            create_test_visualizations(visualization_samples, output_dir)
        else:
            print("No visualization samples to process")
    except Exception as e:
        print(f"Error creating test visualizations: {e}")
    
    # Create metrics summary plot
    try:
        if all_metrics:
            create_metrics_summary(all_metrics, output_dir)
        else:
            print("No metrics to summarize")
    except Exception as e:
        print(f"Error creating metrics summary: {e}")
    
    print(f"\nTesting completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Metrics: {results_path}")
    
    return results

def create_test_visualizations(samples, output_dir):
    """Create visualizations for test samples."""
    try:
        n_samples = len(samples)
        if n_samples == 0:
            print("No samples to visualize")
            return
            
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(samples):
            try:
                # Query image - denormalize and clip
                query_img = sample['query']
                query_img = query_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                query_img = np.clip(query_img, 0, 1)
                axes[i, 0].imshow(query_img)
                axes[i, 0].set_title(f"Query Image\n{sample['sample_name']}")
                axes[i, 0].axis('off')
                
                # Support image - denormalize and clip
                support_img = sample['support']
                support_img = support_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                support_img = np.clip(support_img, 0, 1)
                axes[i, 1].imshow(support_img)
                axes[i, 1].set_title("Support Image")
                axes[i, 1].axis('off')
                
                # True mask
                axes[i, 2].imshow(sample['true_mask'], cmap='gray')
                axes[i, 2].set_title("True Mask")
                axes[i, 2].axis('off')
                
                # Predicted mask - use lower threshold
                pred_binary = (sample['pred_mask'] > 0.1).astype(np.uint8) * 255
                axes[i, 3].imshow(pred_binary, cmap='gray')
                axes[i, 3].set_title(f"Predicted Mask\nF1: {sample['metrics']['f1_score']:.3f}, IoU: {sample['metrics']['iou']:.3f}")
                axes[i, 3].axis('off')
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, "visualizations", "test_samples.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved: {viz_path}")
    except Exception as e:
        print(f"Error in create_test_visualizations: {e}")

def create_metrics_summary(metrics, output_dir):
    """Create summary plots of metrics."""
    try:
        if not metrics:
            print("No metrics to summarize")
            return
            
        metric_names = ['precision', 'recall', 'f1_score', 'accuracy', 'iou', 'dice']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric_name in enumerate(metric_names):
            try:
                values = [m[metric_name] for m in metrics]
                
                axes[i].hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
                axes[i].set_title(f'{metric_name.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel(metric_name.replace("_", " ").title())
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            except Exception as e:
                print(f"Error plotting metric {metric_name}: {e}")
                continue
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "visualizations", "metrics_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics summary saved: {summary_path}")
    except Exception as e:
        print(f"Error in create_metrics_summary: {e}")

def main():
    """Main function to test the best model."""
    
    # Find the best model
    # model_path = find_best_model()
    model_path = 'C:\Users\illya\Documents\volleyball_analitics\volleystat\models\best\siamese_ball_segment_best_67_epoch.pt'
    if not model_path:
        print("No best model found!")
        return
    
    # Test dataset path
    test_dataset_dir = "data/datasets/combined_datasets/combined_run_20250618_005343"
    
    if not os.path.exists(test_dataset_dir):
        print(f"Test dataset not found: {test_dataset_dir}")
        return
    
    # Test the model
    results = test_model(model_path, test_dataset_dir)
    
    if results:
        print(f"\nTesting completed successfully!")
        print(f"Average F1 Score: {results['average_metrics']['f1_score']:.4f}")
        print(f"Average IoU: {results['average_metrics']['iou']:.4f}")
        print(f"Average Dice: {results['average_metrics']['dice']:.4f}")
    else:
        print(f"\nTesting failed!")

if __name__ == "__main__":
    main()