import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from src.utils.person_detection import PersonDetector
from src.models.unet import UNet


class VideoProcessor:
    """Integrated video processor for person detection and ball segmentation."""
    
    def __init__(self, yolo_model_path='yolo11n.pt', unet_model_path=None, device='cuda'):
        """
        Initialize video processor.
        
        Args:
            yolo_model_path (str): Path to YOLO model for person detection
            unet_model_path (str): Path to trained U-Net model for ball segmentation
            device (str): Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize person detector
        self.person_detector = PersonDetector(yolo_model_path, self.device)
        
        # Initialize ball segmentation model
        self.ball_segmenter = None
        if unet_model_path and os.path.exists(unet_model_path):
            self.ball_segmenter = UNet()
            self.ball_segmenter.load_state_dict(torch.load(unet_model_path, map_location=self.device))
            self.ball_segmenter.eval().to(self.device)
            print(f"✅ Loaded ball segmentation model from {unet_model_path}")
        else:
            print("⚠️ No ball segmentation model provided, will only detect persons")
        
        # Transform for ball segmentation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Support image for ball segmentation (will be set from first ball detection)
        self.support_image = None
        
    def set_support_image(self, support_image_path):
        """Set support image for ball segmentation."""
        if os.path.exists(support_image_path):
            self.support_image = self.transform(Image.open(support_image_path).convert("RGB")).to(self.device)
            print(f"✅ Set support image from {support_image_path}")
        else:
            print(f"⚠️ Support image not found: {support_image_path}")
    
    def detect_persons(self, frame):
        """Detect persons in frame using YOLO."""
        return self.person_detector.detect_persons(frame)
    
    def segment_ball(self, frame):
        """Segment ball in frame using U-Net."""
        if self.ball_segmenter is None or self.support_image is None:
            return None, None
        
        # Prepare input
        query_img = self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).to(self.device)
        input_tensor = torch.cat([query_img, self.support_image], dim=0).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            pred = self.ball_segmenter(input_tensor).squeeze().cpu().numpy()
        
        # Binarize prediction
        pred_mask = (pred > 0.5).astype(np.uint8) * 255
        
        # Resize to original frame size
        h, w = frame.shape[:2]
        pred_mask_resized = cv2.resize(pred_mask, (w, h))
        
        return pred_mask_resized, pred
    
    def find_ball_bounding_box(self, mask, min_area=100):
        """Find bounding box from ball segmentation mask."""
        if mask is None:
            return None
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < min_area:
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, x + w, y + h)
    
    def process_video(self, input_video_path, output_video_path, start_sec=30, duration_sec=60,
                     support_image_path=None, show_persons=True, show_ball=True):
        """
        Process video with person detection and ball segmentation.
        
        Args:
            input_video_path (str): Path to input video
            output_video_path (str): Path to output video
            start_sec (int): Start time in seconds
            duration_sec (int): Duration to process in seconds
            support_image_path (str): Path to support image for ball segmentation
            show_persons (bool): Whether to show person bounding boxes
            show_ball (bool): Whether to show ball bounding box
        """
        # Set support image if provided
        if support_image_path:
            self.set_support_image(support_image_path)
        
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Failed to open video: {input_video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = int(fps * start_sec)
        end_frame = min(int(fps * (start_sec + duration_sec)), total_frames)
        frames_to_process = end_frame - start_frame
        
        print(f"📹 Processing video: {input_video_path}")
        print(f"⏰ Time range: {start_sec}s - {start_sec + duration_sec}s")
        print(f"🎞️ Frames to process: {frames_to_process}")
        print(f"📐 Resolution: {width}x{height}")
        
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError(f"❌ Failed to create output video: {output_video_path}")
        
        # Process frames
        frame_count = 0
        ball_detections = 0
        person_detections = 0
        
        print("🔄 Processing frames...")
        
        for _ in tqdm(range(frames_to_process)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create a copy for drawing
            output_frame = frame.copy()
            
            # Detect persons
            if show_persons:
                person_boxes = self.detect_persons(frame)
                person_detections += len(person_boxes)
                
                # Draw person bounding boxes
                for x1, y1, x2, y2, conf in person_boxes:
                    color = (0, 255, 0)  # Green for persons
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output_frame, f'Person {conf:.2f}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Segment ball
            if show_ball and self.ball_segmenter is not None:
                ball_mask, _ = self.segment_ball(frame)
                ball_bbox = self.find_ball_bounding_box(ball_mask)
                
                if ball_bbox:
                    ball_detections += 1
                    x1, y1, x2, y2 = ball_bbox
                    color = (0, 0, 255)  # Red for ball
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(output_frame, 'Ball', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Write frame
            out.write(output_frame)
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        # Print statistics
        print(f"✅ Processing completed!")
        print(f"📊 Statistics:")
        print(f"   - Frames processed: {frame_count}")
        print(f"   - Person detections: {person_detections}")
        print(f"   - Ball detections: {ball_detections}")
        print(f"   - Output saved to: {output_video_path}")
        
        return {
            'frames_processed': frame_count,
            'person_detections': person_detections,
            'ball_detections': ball_detections,
            'output_path': output_video_path
        }


def create_support_image_from_video(video_path, output_path, start_sec=30, duration_sec=10, 
                                   model_path='yolo11n.pt'):
    """
    Create a support image for ball segmentation from video.
    
    Args:
        video_path (str): Path to input video
        output_path (str): Path to save support image
        start_sec (int): Start time in seconds
        duration_sec (int): Duration to search for ball
        model_path (str): Path to YOLO model
    """
    # Hardcoded support image path
    hardcoded_support_path = r"C:\Users\illya\Documents\volleyball_analitics\data\train_val_test_prepared_for_training\test\support\c085c7ed-frame_00215_ball.jpg"
    
    # Check if the hardcoded file exists
    if os.path.exists(hardcoded_support_path):
        print(f"✅ Using hardcoded support image: {hardcoded_support_path}")
        return hardcoded_support_path
    else:
        print(f"❌ Hardcoded support image not found: {hardcoded_support_path}")
        return None 