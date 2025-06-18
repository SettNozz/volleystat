#!/usr/bin/env python3
"""
Simple Siamese UNet Training Script
Based on the provided core code with minimal modifications.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse


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


def calculate_metrics(pred_mask, true_mask, threshold=0.5):
    """Calculate precision, recall, and F1 score for segmentation."""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    true_binary = (true_mask > 0.5).astype(np.uint8)
    
    pred_flat = pred_binary.flatten()
    true_flat = true_binary.flatten()
    
    if np.sum(true_flat) == 0:
        return 0.0, 0.0, 0.0
    
    try:
        precision = precision_score(true_flat, pred_flat, average='binary', zero_division=0)
        recall = recall_score(true_flat, pred_flat, average='binary', zero_division=0)
        f1 = f1_score(true_flat, pred_flat, average='binary', zero_division=0)
        return precision, recall, f1
    except:
        return 0.0, 0.0, 0.0


# === Трансформації ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class SiameseDataset(Dataset):
    def __init__(self, support_dir, query_dir, mask_dir):
        self.support_dir = support_dir
        self.query_dir = query_dir
        self.mask_dir = mask_dir
        self.query_images = sorted([f for f in os.listdir(query_dir) if f.endswith('.jpg')])
        print(f"Found {len(self.query_images)} samples in {query_dir}")

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, idx):
        query_name = self.query_images[idx].strip()
        base_name = query_name[:-4]  # remove .jpg
        support_name_support = base_name + '_support.jpg'
        support_name_ball = base_name + '_ball.jpg'
        support_path_support = os.path.join(self.support_dir, support_name_support)
        support_path_ball = os.path.join(self.support_dir, support_name_ball)
        # Try _support.jpg first, then _ball.jpg
        if os.path.exists(support_path_support):
            support_img = transform(Image.open(support_path_support).convert("RGB"))
        elif os.path.exists(support_path_ball):
            support_img = transform(Image.open(support_path_ball).convert("RGB"))
        else:
            raise FileNotFoundError(f"Support image not found for {query_name}: tried {support_name_support} and {support_name_ball}")
        mask_name = query_name.replace(".jpg", "_mask.png")
        query_img = transform(Image.open(os.path.join(self.query_dir, query_name)).convert("RGB"))
        mask = transform(Image.open(os.path.join(self.mask_dir, mask_name)).convert("L"))
        # Only use the ball class: binarize mask
        mask = (mask > 0.5).float()
        input_tensor = torch.cat([query_img, support_img], dim=0)  # 6 x 256 x 256
        return input_tensor, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    args = parser.parse_args()

    # === Шляхи ===
    support_path = "data/datasets/combined_datasets/combined_run_20250618_005343/train/support/"
    query_path = "data/datasets/combined_datasets/combined_run_20250618_005343/train/query/"
    mask_path = "data/datasets/combined_datasets/combined_run_20250618_005343/train/masks/"

    val_support_path = "data/datasets/combined_datasets/combined_run_20250618_005343/val/support/"
    val_query_path = "data/datasets/combined_datasets/combined_run_20250618_005343/val/query/"
    val_mask_path = "data/datasets/combined_datasets/combined_run_20250618_005343/val/masks/"

    # === Параметри ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    num_epochs = 80
    lr = 1e-3

    # === Датасет і розбиття ===
    train_dataset = SiameseDataset(support_path, query_path, mask_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = SiameseDataset(val_support_path, val_query_path, val_mask_path)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # === Модель, критерій, оптимайзер ===
    model = UNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0

    # Resume training if requested
    checkpoint_path = "models/siamese_ball_segment_best.pt"
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_precision = checkpoint.get('best_val_precision', 0.0)
        best_val_recall = checkpoint.get('best_val_recall', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed from epoch {start_epoch}, best precision {best_val_precision:.4f}, best recall {best_val_recall:.4f}, best loss {best_val_loss:.4f}")

    train_losses, val_losses = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_f1s, val_f1s = [], []

    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for epoch in range(start_epoch, num_epochs):
        # --- Тренування ---
        model.train()
        running_loss = 0.0
        epoch_precisions, epoch_recalls, epoch_f1s = [], [], []
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            pred_masks = outputs.squeeze().detach().cpu().numpy()
            true_masks = targets.squeeze().cpu().numpy()
            for pred_mask, true_mask in zip(pred_masks, true_masks):
                precision, recall, f1 = calculate_metrics(pred_mask, true_mask)
                epoch_precisions.append(precision)
                epoch_recalls.append(recall)
                epoch_f1s.append(f1)
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        train_precisions.append(np.mean(epoch_precisions))
        train_recalls.append(np.mean(epoch_recalls))
        train_f1s.append(np.mean(epoch_f1s))

        # --- Валідація ---
        model.eval()
        running_val_loss = 0.0
        val_precisions_epoch, val_recalls_epoch, val_f1s_epoch = [], [], []
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                pred_masks = outputs.squeeze().detach().cpu().numpy()
                true_masks = targets.squeeze().cpu().numpy()
                for pred_mask, true_mask in zip(pred_masks, true_masks):
                    precision, recall, f1 = calculate_metrics(pred_mask, true_mask)
                    val_precisions_epoch.append(precision)
                    val_recalls_epoch.append(recall)
                    val_f1s_epoch.append(f1)
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        val_precision = np.mean(val_precisions_epoch)
        val_recall = np.mean(val_recalls_epoch)
        val_f1 = np.mean(val_f1s_epoch)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {epoch_train_loss:.4f}, Precision: {train_precisions[-1]:.4f}, Recall: {train_recalls[-1]:.4f}, F1: {train_f1s[-1]:.4f}")
        print(f"  Val   - Loss: {epoch_val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # --- Збереження кращої моделі ---
        save_best = False
        if val_precision > best_val_precision:
            best_val_precision = val_precision
            save_best = True
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            save_best = True
        if save_best:
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_precision': best_val_precision,
                'best_val_recall': best_val_recall,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }, checkpoint_path)
            print(f"  💾 Saved best model (Precision: {best_val_precision:.4f}, Recall: {best_val_recall:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
        # Early stopping
        if patience_counter >= patience:
            print(f"No Early stopping at epoch {epoch+1} due to no improvement in precision/recall.")
            # break

    # Save training history
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(train_precisions, label='Train Precision')
    plt.plot(val_precisions, label='Val Precision')
    plt.title('Training History - Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.title('Training History - F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('models/siamese_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training completed!")
    print(f"Best validation precision: {best_val_precision:.4f}")
    print(f"Best validation recall: {best_val_recall:.4f}")
    print(f"Training history saved to: models/siamese_training_history.png")


if __name__ == "__main__":
    main() 