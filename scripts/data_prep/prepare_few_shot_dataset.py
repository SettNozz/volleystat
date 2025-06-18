#!/usr/bin/env python3
"""
Prepare Few-Shot Dataset using existing trained model
Creates support/query/masks structure with masks generated from siamese_ball_segment_best.pt
"""

import os
import shutil
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
from datetime import datetime
from torchvision import transforms


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


def load_trained_model(model_path):
    """Load the trained siamese ball segmentation model."""
    print(f"🔄 Loading trained model from: {model_path}")
    
    # Initialize model with correct parameters (6 input channels for Siamese)
    model = UNet(in_channels=6, out_channels=1)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def create_fallback_mask(image_path):
    """Create a fallback mask based on ball detection in the original image."""
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros((256, 256), dtype=np.uint8)
    
    # Convert to grayscale for ball detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Hough Circle Transform to detect balls
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    
    # Create mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Scale circles to 256x256
        h, w = image.shape[:2]
        scale_x = 256 / w
        scale_y = 256 / h
        
        for (x, y, r) in circles:
            # Scale coordinates
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            r_scaled = int(r * min(scale_x, scale_y))
            
            # Draw circle on mask
            cv2.circle(mask, (x_scaled, y_scaled), r_scaled, 255, -1)
    
    # If no circles detected, create a bounding box around the center
    if np.sum(mask) == 0:
        # Create a simple bounding box in the center
        center_x, center_y = 128, 128
        box_size = 50
        mask[center_y-box_size//2:center_y+box_size//2, 
             center_x-box_size//2:center_x+box_size//2] = 255
    
    return mask


def generate_mask_from_model(model, query_image_path, support_image_path, device='cpu'):
    """Generate mask using the trained Siamese model."""
    # Load and preprocess images
    query_image = cv2.imread(query_image_path)
    support_image = cv2.imread(support_image_path)
    
    if query_image is None or support_image is None:
        return None
    
    # Convert BGR to RGB
    query_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    support_rgb = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    query_resized = cv2.resize(query_rgb, (256, 256))
    support_resized = cv2.resize(support_rgb, (256, 256))
    
    # Normalize and convert to tensor
    query_tensor = torch.from_numpy(query_resized).float() / 255.0
    support_tensor = torch.from_numpy(support_resized).float() / 255.0
    
    # Permute dimensions and concatenate (6 channels: 3 query + 3 support)
    query_tensor = query_tensor.permute(2, 0, 1)  # HWC -> CHW
    support_tensor = support_tensor.permute(2, 0, 1)  # HWC -> CHW
    input_tensor = torch.cat([query_tensor, support_tensor], dim=0).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    input_tensor = input_tensor.to(device)
    
    # Generate mask
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy()
    
    # Apply threshold to get binary mask
    # The model outputs values between 0 and 1 due to sigmoid
    # We threshold at 0.5 to get binary mask
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # If the mask is completely black, try a lower threshold
    if np.sum(binary_mask) == 0:
        # Try different thresholds
        for threshold in [0.3, 0.2, 0.1, 0.05]:
            binary_mask = (mask > threshold).astype(np.uint8) * 255
            if np.sum(binary_mask) > 0:
                print(f"  ⚠️ Using lower threshold {threshold} for mask generation")
                break
    
    # If still black, create a mask based on the ball position in the original image
    if np.sum(binary_mask) == 0:
        print(f"  ⚠️ Creating fallback mask based on ball detection")
        binary_mask = create_fallback_mask(query_image_path)
    
    return binary_mask


def create_few_shot_dataset_structure(base_dir, run_name):
    """Create the few-shot dataset directory structure."""
    dataset_dir = os.path.join(base_dir, run_name)
    
    # Create main directories
    os.makedirs(os.path.join(dataset_dir, 'support'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'query'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'masks'), exist_ok=True)
    
    return dataset_dir


def prepare_few_shot_dataset(
    ball_images_dir,
    model_path,
    output_base_dir='few_shot_datasets',
    num_support_samples=5,
    num_query_samples=20,
    run_name=None
):
    """Prepare few-shot dataset with support/query/masks structure."""
    
    # Generate run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"few_shot_run_{timestamp}"
    
    print(f"🚀 Preparing few-shot dataset: {run_name}")
    print(f"📁 Ball images directory: {ball_images_dir}")
    print(f"🤖 Model path: {model_path}")
    
    # Load trained model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(model_path)
    model = model.to(device)
    
    # Get list of ball images
    ball_images = [f for f in os.listdir(ball_images_dir) if f.endswith('.jpg')]
    print(f"📊 Found {len(ball_images)} ball images")
    
    if len(ball_images) < num_support_samples + num_query_samples:
        print(f"⚠️ Warning: Not enough images. Using all available images.")
        num_support_samples = min(num_support_samples, len(ball_images) // 3)
        num_query_samples = len(ball_images) - num_support_samples
    
    # Randomly select support and query samples
    random.shuffle(ball_images)
    support_images = ball_images[:num_support_samples]
    query_images = ball_images[num_support_samples:num_support_samples + num_query_samples]
    
    print(f"📊 Support samples: {len(support_images)}")
    print(f"📊 Query samples: {len(query_images)}")
    
    # Create dataset structure
    dataset_dir = create_few_shot_dataset_structure(output_base_dir, run_name)
    
    # Copy support images
    print("\n📁 Copying support images...")
    for i, img_name in enumerate(support_images):
        src_path = os.path.join(ball_images_dir, img_name)
        dst_name = f"ball{i+1}.jpg"
        dst_path = os.path.join(dataset_dir, 'support', dst_name)
        shutil.copy2(src_path, dst_path)
        print(f"  ✅ {img_name} -> support/{dst_name}")
    
    # Process query images and generate masks
    print("\n📁 Processing query images and generating masks...")
    for i, img_name in enumerate(query_images):
        query_path = os.path.join(ball_images_dir, img_name)
        
        # Copy query image
        dst_name = f"frame{i+1}.jpg"
        dst_path = os.path.join(dataset_dir, 'query', dst_name)
        shutil.copy2(query_path, dst_path)
        
        # Select a random support image for mask generation
        support_img_name = random.choice(support_images)
        support_path = os.path.join(ball_images_dir, support_img_name)
        
        # Generate mask using Siamese model
        mask = generate_mask_from_model(model, query_path, support_path, device)
        if mask is not None:
            mask_name = f"frame{i+1}_mask.png"
            mask_path = os.path.join(dataset_dir, 'masks', mask_name)
            cv2.imwrite(mask_path, mask)
            print(f"  ✅ {img_name} -> query/{dst_name} + masks/{mask_name} (support: {support_img_name})")
        else:
            print(f"  ❌ Failed to generate mask for {img_name}")
    
    # Save dataset info
    dataset_info = {
        'run_name': run_name,
        'created_at': datetime.now().isoformat(),
        'num_support_samples': len(support_images),
        'num_query_samples': len(query_images),
        'support_images': support_images,
        'query_images': query_images,
        'model_path': model_path,
        'device_used': device,
        'model_type': 'Siamese UNet (6 channels)'
    }
    
    info_path = os.path.join(dataset_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Few-shot dataset prepared successfully!")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📄 Dataset info: {info_path}")
    
    return dataset_dir, dataset_info


def main():
    """Main function to prepare few-shot dataset."""
    # Configuration
    ball_images_dir = 'ball_dataset_for_labeling/ball_crops_dataset/images'
    model_path = 'models/siamese_ball_segment_best.pt'
    output_base_dir = 'few_shot_datasets'
    
    # Check if required files exist
    if not os.path.exists(ball_images_dir):
        print(f"❌ Ball images directory not found: {ball_images_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Prepare dataset
    dataset_dir, dataset_info = prepare_few_shot_dataset(
        ball_images_dir=ball_images_dir,
        model_path=model_path,
        output_base_dir=output_base_dir,
        num_support_samples=5,
        num_query_samples=20
    )
    
    print(f"\n🎉 Dataset preparation completed!")
    print(f"📊 Dataset summary:")
    print(f"   - Support samples: {dataset_info['num_support_samples']}")
    print(f"   - Query samples: {dataset_info['num_query_samples']}")
    print(f"   - Total files: {dataset_info['num_support_samples'] + dataset_info['num_query_samples'] * 2}")


if __name__ == "__main__":
    main() 