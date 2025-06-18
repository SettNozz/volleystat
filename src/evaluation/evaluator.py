import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from src.models.unet import UNet
from src.visualization.visualization import visualize_test_results


def mask_to_obb_coordinates(mask, min_area=100):
    """
    Convert binary mask to YOLOv8 OBB format coordinates.
    Returns list of [class_id, x1, y1, x2, y2, x3, y3, x4, y4] coordinates.
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obb_coordinates = []
    h, w = mask.shape
    
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < min_area:
            continue
            
        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Convert to YOLO format (normalized coordinates)
        # Sort points to ensure consistent order: top-left, top-right, bottom-right, bottom-left
        box = box.astype(np.float32)
        
        # Normalize coordinates
        box[:, 0] /= w  # x coordinates
        box[:, 1] /= h  # y coordinates
        
        # Flatten and add class_id (0 for ball)
        obb_coord = [0] + box.flatten().tolist()
        obb_coordinates.append(obb_coord)
    
    return obb_coordinates


def evaluate_model(model_path, test_query_dir, test_mask_dir, support_path, output_dir):
    """Evaluate the trained model on test dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load support image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    support_img = transform(Image.open(support_path).convert("RGB")).to(device)

    # Inference
    with torch.no_grad():
        for filename in sorted(os.listdir(test_query_dir)):
            if not filename.endswith(".jpg"):
                continue

            name = os.path.splitext(filename)[0]
            query_file_path = os.path.join(test_query_dir, filename)
            mask_path = os.path.join(test_mask_dir, f"{name}_mask.png")

            query_img = transform(Image.open(query_file_path).convert("RGB")).to(device)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"[!] Skipped {name} — GT mask missing")
                continue
            
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[:, :, 0]

            # Inference
            input_tensor = torch.cat([query_img, support_img], dim=0).unsqueeze(0)
            pred = model(input_tensor).squeeze().cpu().numpy()
            
            # Binarization
            pred_mask = (pred > 0.5).astype(np.uint8) * 255
            pred_mask_resized = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))

            # IoU calculation
            intersection = np.logical_and(pred_mask_resized > 127, gt_mask > 127).sum()
            union = np.logical_or(pred_mask_resized > 127, gt_mask > 127).sum()
            iou = intersection / union if union > 0 else 0

            print(f"[{name}] IoU: {iou:.3f}")

            # Visualization
            visualize_test_results(query_file_path, gt_mask, pred_mask_resized, name, iou, output_dir) 