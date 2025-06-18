import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        enc2 = self.encoder2(nn.functional.max_pool2d(enc1, 2))
        bottleneck = self.bottleneck(nn.functional.max_pool2d(enc2, 2))
        up1 = self.up1(bottleneck)
        dec1 = self.decoder1(torch.cat([up1, enc2], dim=1))
        up2 = self.up2(dec1)
        dec2 = self.decoder2(torch.cat([up2, enc1], dim=1))
        return torch.sigmoid(self.final(dec2))

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
    def __len__(self):
        return len(self.query_images)
    def __getitem__(self, idx):
        query_name = self.query_images[idx]
        support_name = query_name.replace(".jpg", "_ball.jpg")
        mask_name = query_name.replace(".jpg", "_mask.png")
        support_img = transform(Image.open(os.path.join(self.support_dir, support_name)).convert("RGB"))
        query_img = transform(Image.open(os.path.join(self.query_dir, query_name)).convert("RGB"))
        mask = transform(Image.open(os.path.join(self.mask_dir, mask_name)).convert("L"))
        input_tensor = torch.cat([query_img, support_img], dim=0)
        return input_tensor, mask, query_name, support_name, mask_name

def calculate_metrics(pred_mask, true_mask, threshold=0.5):
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

def main():
    support_path = "data/datasets/train_val_test_prepared_for_training/test/support/"
    query_path = "data/datasets/train_val_test_prepared_for_training/test/query/"
    mask_path = "data/datasets/train_val_test_prepared_for_training/test/masks/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    dataset = SiameseDataset(support_path, query_path, mask_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = UNet().to(device)
    model.load_state_dict(torch.load("models/siamese_ball_segment_best.pt", map_location=device))
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0
    all_prec, all_rec, all_f1 = [], [], []
    # Visualization output dir
    vis_dir = "results/test_siamese_unet/"
    os.makedirs(vis_dir, exist_ok=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets, query_names, support_names, mask_names) in enumerate(tqdm(loader, desc="Testing")):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            pred_masks = outputs.squeeze().detach().cpu().numpy()
            true_masks = targets.squeeze().cpu().numpy()
            # Split input into query/support for visualization
            query_imgs = inputs[:, :3, :, :].cpu().numpy()
            support_imgs = inputs[:, 3:, :, :].cpu().numpy()
            if pred_masks.ndim == 2:
                pred_masks = np.expand_dims(pred_masks, 0)
                true_masks = np.expand_dims(true_masks, 0)
                query_imgs = np.expand_dims(query_imgs, 0)
                support_imgs = np.expand_dims(support_imgs, 0)
            for i, (pred_mask, true_mask, query_img, support_img, qn) in enumerate(zip(pred_masks, true_masks, query_imgs, support_imgs, query_names)):
                p, r, f = calculate_metrics(pred_mask, true_mask)
                all_prec.append(p)
                all_rec.append(r)
                all_f1.append(f)
                # Visualization
                fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                axes[0, 0].imshow(np.transpose(query_img, (1, 2, 0)))
                axes[0, 0].set_title("Query Image")
                axes[0, 0].axis('off')
                axes[0, 1].imshow(np.transpose(support_img, (1, 2, 0)))
                axes[0, 1].set_title("Support Image")
                axes[0, 1].axis('off')
                axes[1, 0].imshow(true_mask.squeeze(), cmap='gray')
                axes[1, 0].set_title("Ground Truth Mask")
                axes[1, 0].axis('off')
                axes[1, 1].imshow((pred_mask > 0.5).astype(np.uint8), cmap='gray')
                axes[1, 1].set_title(f"Predicted Mask\nP:{p:.2f} R:{r:.2f} F1:{f:.2f}")
                axes[1, 1].axis('off')
                plt.tight_layout()
                out_path = os.path.join(vis_dir, f"{os.path.splitext(qn)[0]}_viz.png")
                plt.savefig(out_path, dpi=200)
                plt.close()
    avg_loss = total_loss / len(dataset)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Precision: {np.mean(all_prec):.4f}")
    print(f"Test Recall: {np.mean(all_rec):.4f}")
    print(f"Test F1: {np.mean(all_f1):.4f}")
    print(f"Visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    main() 