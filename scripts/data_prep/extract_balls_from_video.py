#!/usr/bin/env python3
"""
Extract Ball Images from Video
Extracts ball images from the whole video with coordinates and original images.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
from datetime import datetime
import shutil


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
    
    model = UNet(in_channels=6, out_channels=1)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def detect_ball_in_frame(model, frame, support_image, device='cpu'):
    """Detect ball in a frame using the Siamese model."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    support_rgb = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    frame_resized = cv2.resize(frame_rgb, (256, 256))
    support_resized = cv2.resize(support_rgb, (256, 256))
    
    # Normalize and convert to tensor
    frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
    support_tensor = torch.from_numpy(support_resized).float() / 255.0
    
    # Permute dimensions and concatenate (6 channels: 3 frame + 3 support)
    frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
    support_tensor = support_tensor.permute(2, 0, 1)  # HWC -> CHW
    input_tensor = torch.cat([frame_tensor, support_tensor], dim=0).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    input_tensor = input_tensor.to(device)
    
    # Generate mask
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().cpu().numpy()
    
    # Apply threshold to get binary mask
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # If the mask is completely black, try a lower threshold
    if np.sum(binary_mask) == 0:
        for threshold in [0.3, 0.2, 0.1, 0.05]:
            binary_mask = (mask > threshold).astype(np.uint8) * 255
            if np.sum(binary_mask) > 0:
                break
    
    return binary_mask


def find_ball_bbox(mask, original_h, original_w):
    """Find bounding box of the ball from the mask."""
    # Resize mask to original frame size
    mask_original_size = cv2.resize(mask, (original_w, original_h))
    
    # Find contours
    contours, _ = cv2.findContours(mask_original_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(original_w - x, w + 2 * padding)
        h = min(original_h - y, h + 2 * padding)
        
        return x, y, w, h
    
    return None


def extract_balls_from_video(
    video_path,
    model_path,
    support_image_path,
    output_dir='ball_extraction_results',
    confidence_threshold=0.1,
    min_ball_area=100
):
    """Extract ball images from video with coordinates."""
    
    print(f"🚀 Starting ball extraction from video: {video_path}")
    print(f"🤖 Using model: {model_path}")
    print(f"🎯 Support image: {support_image_path}")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(model_path)
    model = model.to(device)
    
    # Load support image
    support_image = cv2.imread(support_image_path)
    if support_image is None:
        print(f"❌ Could not load support image: {support_image_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output directories
    ball_images_dir = os.path.join(output_dir, 'ball_images')
    original_images_dir = os.path.join(output_dir, 'original_images')
    coordinates_file = os.path.join(output_dir, 'ball_coordinates.json')
    
    os.makedirs(ball_images_dir, exist_ok=True)
    os.makedirs(original_images_dir, exist_ok=True)
    
    # Process frames
    frame_count = 0
    ball_count = 0
    coordinates_data = []
    
    print(f"\n🔄 Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 5th frame to avoid too many similar images
        if frame_count % 5 != 0:
            continue
        
        # Detect ball in frame
        mask = detect_ball_in_frame(model, frame, support_image, device)
        
        # Find ball bounding box
        bbox = find_ball_bbox(mask, height, width)
        
        if bbox is not None:
            x, y, w, h = bbox
            
            # Check if ball area is large enough
            if w * h >= min_ball_area:
                ball_count += 1
                
                # Crop ball image
                ball_crop = frame[y:y+h, x:x+w]
                
                # Save ball image
                ball_filename = f"ball_{ball_count:04d}.jpg"
                ball_path = os.path.join(ball_images_dir, ball_filename)
                cv2.imwrite(ball_path, ball_crop)
                
                # Save original frame
                original_filename = f"original_{ball_count:04d}.jpg"
                original_path = os.path.join(original_images_dir, original_filename)
                cv2.imwrite(original_path, frame)
                
                # Store coordinates
                coordinates_data.append({
                    'ball_image': ball_filename,
                    'original_image': original_filename,
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'confidence': float(np.sum(mask) / (256 * 256))  # Normalized mask sum
                })
                
                if ball_count % 10 == 0:
                    print(f"  ✅ Extracted {ball_count} balls (frame {frame_count}/{total_frames})")
    
    cap.release()
    
    # Save coordinates
    with open(coordinates_file, 'w') as f:
        json.dump(coordinates_data, f, indent=2)
    
    # Create summary
    summary = {
        'video_path': video_path,
        'model_path': model_path,
        'support_image_path': support_image_path,
        'total_frames': total_frames,
        'processed_frames': frame_count,
        'extracted_balls': ball_count,
        'extraction_date': datetime.now().isoformat(),
        'confidence_threshold': confidence_threshold,
        'min_ball_area': min_ball_area
    }
    
    summary_file = os.path.join(output_dir, 'extraction_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Ball extraction completed!")
    print(f"📊 Extracted {ball_count} ball images")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Coordinates file: {coordinates_file}")
    print(f"📄 Summary file: {summary_file}")
    
    return output_dir, coordinates_data


def main():
    """Main function to extract balls from video."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_balls_from_video.py <video_path> [support_image_path]")
        print("\nExample:")
        print("  python extract_balls_from_video.py data/video.mp4")
        print("  python extract_balls_from_video.py data/video.mp4 data/support_ball.jpg")
        return
    
    # Configuration
    video_path = sys.argv[1]
    support_image_path = sys.argv[2] if len(sys.argv) > 2 else 'models/support_ball.jpg'
    model_path = 'models/siamese_ball_segment_best.pt'
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not os.path.exists(support_image_path):
        print(f"❌ Support image not found: {support_image_path}")
        print("💡 Please provide a support image path or place one at models/support_ball.jpg")
        return
    
    # Extract balls
    output_dir, coordinates = extract_balls_from_video(
        video_path=video_path,
        model_path=model_path,
        support_image_path=support_image_path,
        output_dir='ball_extraction_results',
        confidence_threshold=0.1,
        min_ball_area=100
    )
    
    print(f"\n📋 Next steps:")
    print(f"1. Manually clean the ball images in: {output_dir}/ball_images/")
    print(f"2. Run dataset preparation script after cleaning")
    print(f"3. Train the few-shot model")


if __name__ == "__main__":
    main() 