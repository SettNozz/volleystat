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
            
            # Load checkpoint
            checkpoint = torch.load(unet_model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                # Full checkpoint with optimizer state, etc.
                self.ball_segmenter.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"✅ Loaded ball segmentation model from checkpoint: {unet_model_path}")
            else:
                # Just model weights
                self.ball_segmenter.load_state_dict(checkpoint, strict=False)
                print(f"✅ Loaded ball segmentation model from {unet_model_path}")
            
            self.ball_segmenter.eval().to(self.device)
            
            # Enable optimizations for inference
            if hasattr(torch, 'jit'):
                try:
                    self.ball_segmenter = torch.jit.optimize_for_inference(self.ball_segmenter)
                    print("✅ Applied JIT optimizations for inference")
                except:
                    pass
        else:
            print("⚠️ No ball segmentation model provided, will only detect persons")
        
        # Optimized transform for ball segmentation (pre-computed)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Support image for ball segmentation (will be set from first ball detection)
        self.support_image = None
        
        # Performance optimizations
        self.cache_support_image = True
        self.support_image_tensor = None
        
    def set_support_image(self, support_image_path):
        """Set support image for ball segmentation."""
        if os.path.exists(support_image_path):
            # Pre-process and cache support image tensor
            support_img = Image.open(support_image_path).convert("RGB")
            self.support_image_tensor = self.transform(support_img).to(self.device)
            print(f"✅ Set support image from {support_image_path}")
        else:
            print(f"⚠️ Support image not found: {support_image_path}")
    
    def detect_persons(self, frame):
        """Detect persons in frame using YOLO."""
        return self.person_detector.detect_persons(frame)
    
    def segment_ball(self, frame):
        """Segment ball in frame using U-Net."""
        if self.ball_segmenter is None or self.support_image_tensor is None:
            return None, None
        
        # Use true batch processing for the model
        batch_tensors = []
        for frame in frames:
            batch_tensors.append(self.transform(frame))
        batch_input = torch.stack(batch_tensors)
        predictions = self.ball_segmenter(batch_input)  # Single forward pass
        
        # Binarize prediction
        pred_mask = (predictions.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        
        # Resize to original frame size
        h, w = frame.shape[:2]
        pred_mask_resized = cv2.resize(pred_mask, (w, h))
        
        return pred_mask_resized, predictions.squeeze().cpu().numpy()
    
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
        for i in range(0, total_frames, 30):
            # Skip frames by reading and discarding
            for _ in range(i):
                cap.read()
            ret, frame = cap.read()  # Read the frame we want
        
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
        
        # Save images in batches instead of one by one
        image_batch = []
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
            
            # Add frame to batch
            image_batch.append(output_frame)
            frame_count += 1
            
            # Save batch if full
            if len(image_batch) >= 10:
                save_image_batch(image_batch)
                image_batch = []
        
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