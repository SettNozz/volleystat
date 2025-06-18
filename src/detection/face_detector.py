import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import time
from typing import List, Tuple
import ultralytics
from ultralytics import YOLO

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


class YOLODetector:
    def __init__(self, model_path: str = None):
        """
        Initialize YOLO detector for persons and faces.
        
        Args:
            model_path: Path to custom YOLO model (optional, uses default if None)
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom YOLO model from: {model_path}")
        else:
            # Use default YOLO model
            self.model = YOLO('models/pretrained/yolov8n.pt')
            print("Loaded default YOLOv8n model")
        
        # COCO dataset class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        print("Note: Using person detection and estimating face regions for blurring")
    
    def detect_objects(self, frame: np.ndarray, confidence_threshold=0.5) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Detect objects in the frame using YOLO.
        
        Returns:
            List of (x, y, width, height, confidence, class_name) tuples
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > confidence_threshold:
                        x, y, width, height = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        detections.append((x, y, width, height, confidence, class_name))
        
        return detections
    
    def detect_persons(self, frame: np.ndarray, confidence_threshold=0.5) -> List[Tuple[int, int, int, int, float]]:
        """Detect persons specifically."""
        detections = self.detect_objects(frame, confidence_threshold)
        persons = [(x, y, w, h, conf) for x, y, w, h, conf, cls in detections if cls == 'person']
        return persons
    
    def detect_faces_from_persons(self, frame: np.ndarray, person_boxes: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int]]:
        """
        Estimate face regions from person bounding boxes.
        This is a simple heuristic - face is typically in the upper portion of person bounding box.
        """
        face_boxes = []
        
        for (x, y, width, height, confidence) in person_boxes:
            # Face is typically in the upper 1/3 of the person bounding box
            face_height = int(height * 0.3)  # 30% of person height
            face_width = int(width * 0.6)    # 60% of person width
            face_x = x + int((width - face_width) / 2)  # Center horizontally
            face_y = y + int(height * 0.1)   # Start 10% from top
            
            face_boxes.append((face_x, face_y, face_width, face_height))
        
        return face_boxes


class IntegratedPersonBallDetector:
    def __init__(self, model_path: str, support_img_path: str, yolo_model_path: str = None):
        """
        Initialize the integrated person and ball detection pipeline.
        
        Args:
            model_path: Path to the trained Siamese UNet model
            support_img_path: Path to the support image for ball detection
            yolo_model_path: Path to custom YOLO model (optional)
        """
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ball detection model
        self.model = UNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # Initialize YOLO detector for persons and faces
        self.yolo_detector = YOLODetector(yolo_model_path)
        
        # Load support image
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        support_img_pil = Image.open(support_img_path).convert('RGB')
        self.support_img = self.transform(support_img_pil)
        self.support_img = self.support_img.unsqueeze(0).to(self.device)
        
        print(f"Initialized integrated detector on {self.device}")
        print(f"Detection method: YOLO (persons + estimated faces)")
    
    def detect_ball(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect ball in the frame using Siamese UNet.
        
        Returns:
            Ball bounding box (x, y, width, height) or None if no ball detected
        """
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        query_img = self.transform(frame_pil)
        query_img = query_img.unsqueeze(0).to(self.device)
        
        # Prepare input for model
        input_tensor = torch.cat([query_img, self.support_img], dim=1)
        
        # Run inference
        with torch.no_grad():
            pred_mask = self.model(input_tensor)
            pred_mask_np = pred_mask.squeeze().cpu().numpy()
            pred_mask_bin = (pred_mask_np > 0.5).astype(np.uint8)
        
        # Find ball bounding box
        contours, _ = cv2.findContours(
            (pred_mask_bin * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Rescale to original frame size
            h_orig, w_orig = frame.shape[:2]
            x = int(x * w_orig / 256)
            y = int(y * h_orig / 256)
            w = int(w * w_orig / 256)
            h = int(h * h_orig / 256)
            
            return (x, y, w, h)
        
        return None
    
    def blur_face(self, frame: np.ndarray, x: int, y: int, width: int, height: int, 
                  blur_strength: int = 15) -> np.ndarray:
        """Blur a specific face region."""
        # Extract face region
        face_region = frame[y:y+height, x:x+width]
        
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
        
        # Replace the face region with blurred version
        frame[y:y+height, x:x+width] = blurred_face
        
        return frame
    
    def process_frame(self, frame: np.ndarray, blur_strength: int = 15) -> np.ndarray:
        """
        Process a single frame: detect persons, estimate faces, blur faces, detect and visualize ball.
        
        Args:
            frame: Input frame
            blur_strength: Face blur strength
        
        Returns:
            Processed frame with faces blurred, person bounding boxes, and ball bounding box drawn
        """
        # Detect ball
        ball_bbox = self.detect_ball(frame)
        
        # Detect persons
        person_boxes = self.yolo_detector.detect_persons(frame)
        
        # Estimate face regions from person bounding boxes
        face_boxes = self.yolo_detector.detect_faces_from_persons(frame, person_boxes)
        
        # Blur estimated face regions
        for (x, y, width, height) in face_boxes:
            frame = self.blur_face(frame, x, y, width, height, blur_strength)
        
        # Draw person bounding boxes
        for (x, y, width, height, confidence) in person_boxes:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            cv2.putText(frame, f'Person {confidence:.2f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw ball bounding box if detected
        if ball_bbox is not None:
            x, y, w, h = ball_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Ball', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame


def main():
    # Configuration
    input_video_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\results\video\segments\segment_01_start_000409.mp4'
    output_dir = r'results/video/segments'
    model_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\models\to_test\slotel_siamse.pt'
    support_img_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\data\datasets\train_val_test_prepared_for_training\train\support\f51dcd3e-frame_00182_ball.jpg'
    
    # Initialize integrated detector
    detector = IntegratedPersonBallDetector(
        model_path=model_path,
        support_img_path=support_img_path
    )
    
    # Test on a single frame first
    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()
    if ret:
        processed_frame = detector.process_frame(frame)
        cv2.imwrite('assets/images/test_frame_yolo_faces.jpg', processed_frame)
        print("Test frame saved as 'assets/images/test_frame_yolo_faces.jpg'")
    cap.release()


if __name__ == "__main__":
    main() 