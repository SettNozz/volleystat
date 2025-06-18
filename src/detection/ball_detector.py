import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# --- Model definition (same as training) ---
import torch.nn as nn
import torch.nn.functional as F

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

# --- Config ---
input_video_path = r'C:\Users\illya\Videos\video_for_sharing\GX010373_splits\GX010373_part2.mp4'
output_video_path = r'results/video/ball_detection_GX010373_part2_out.mp4'
model_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\models\to_test\slotel_siamse.pt'
support_img_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\data\datasets\train_val_test_prepared_for_training\train\support\f51dcd3e-frame_00182_ball.jpg'

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
try:
    checkpoint = torch.load(model_path, map_location=device)
except Exception:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# --- Load support image ---
support_img_pil = Image.open(support_img_path).convert('RGB')
support_img = transform(support_img_pil)  # [3, 256, 256]
support_img = support_img.unsqueeze(0).to(device)  # [1, 3, 256, 256]

# --- Open video ---
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Output video writer ---
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# --- Process video ---
print(f"Processing video: {input_video_path}")
print(f"Total frames: {total_frames}")

for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    query_img = transform(frame_pil)  # [3, 256, 256]
    query_img = query_img.unsqueeze(0).to(device)  # [1, 3, 256, 256]
    # Prepare input for model
    input_tensor = torch.cat([query_img, support_img], dim=1)  # [1, 6, 256, 256]
    with torch.no_grad():
        pred_mask = model(input_tensor)  # [1, 1, 256, 256]
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        pred_mask_bin = (pred_mask_np > 0.5).astype(np.uint8)
    # Find bounding box
    contours, _ = cv2.findContours((pred_mask_bin*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Rescale bbox to original frame size
        x = int(x * width / 256)
        y = int(y * height / 256)
        w = int(w * width / 256)
        h = int(h * height / 256)
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Write frame
    out.write(frame)

cap.release()
out.release()
print(f"Output video saved to: {output_video_path}") 