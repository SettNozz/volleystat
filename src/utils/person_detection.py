import cv2
import torch
from ultralytics import YOLO
from tqdm.notebook import tqdm
import numpy as np
import os


class PersonDetector:
    """Person detection using YOLO model."""
    
    def __init__(self, model_path='yolo11n.pt', device='cuda'):
        """
        Initialize person detector.
        
        Args:
            model_path (str): Path to YOLO model
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        self.model.model.half()  # Use half precision for speed
        self.device = device
        
    def detect_persons(self, frame):
        """
        Detect persons in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            list: List of person bounding boxes [(x1, y1, x2, y2, confidence), ...]
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(img_rgb, verbose=False)[0]
        
        person_boxes = []
        for box in results.boxes:
            cls = int(box.cls.cpu().item())
            if cls == 0:  # Person class in COCO
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                conf = float(box.conf.cpu())
                person_boxes.append((x1, y1, x2, y2, conf))
                
        return person_boxes
    
    def process_video_segment(self, video_path, output_path, start_sec=30, duration_sec=30):
        """
        Process a segment of video and detect persons.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to output video
            start_sec (int): Start time in seconds
            duration_sec (int): Duration to process in seconds
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "❌ Failed to open video file."
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = int(fps * start_sec)
        end_frame = min(int(fps * (start_sec + duration_sec)), total_frames)
        frames_to_process = end_frame - start_frame
        
        # Skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        assert out.isOpened(), "❌ Failed to initialize VideoWriter."
        
        print(f"🔁 Processing frames from {start_sec} to {start_sec + duration_sec} seconds...")
        
        for _ in tqdm(range(frames_to_process)):
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame not read, stopping.")
                break
            
            # Detect persons
            person_boxes = self.detect_persons(frame)
            
            # Draw bounding boxes
            for x1, y1, x2, y2, conf in person_boxes:
                color = (0, 255, 0)  # Green for persons
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"✅ Done! Processed {frames_to_process} frames.")
        print(f"📄 File saved: {output_path}")
        
        return person_boxes


def create_person_ball_dataset(video_path, output_dir, start_sec=30, duration_sec=30, 
                              frame_interval=1, model_path='yolo11n.pt'):
    """
    Create a dataset with both person and ball detections.
    
    Args:
        video_path (str): Path to input video
        output_dir (str): Output directory for dataset
        start_sec (int): Start time in seconds
        duration_sec (int): Duration to process
        frame_interval (int): Extract every Nth frame
        model_path (str): Path to YOLO model
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "person_annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ball_annotations"), exist_ok=True)
    
    detector = PersonDetector(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * start_sec)
    end_frame = min(int(fps * (start_sec + duration_sec)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    saved_count = 0
    
    print(f"🔁 Creating dataset from video segment...")
    
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Detect persons and balls
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.model(img_rgb, verbose=False)[0]
            
            person_boxes = []
            ball_boxes = []
            
            for box in results.boxes:
                cls = int(box.cls.cpu().item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                conf = float(box.conf.cpu())
                
                if cls == 0:  # Person
                    person_boxes.append((x1, y1, x2, y2, conf))
                elif cls == 32:  # Ball (COCO dataset)
                    ball_boxes.append((x1, y1, x2, y2, conf))
            
            # Save frame and annotations if detections found
            if person_boxes or ball_boxes:
                frame_filename = f"frame_{saved_count:05d}.jpg"
                frame_path = os.path.join(output_dir, "frames", frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # Save person annotations
                if person_boxes:
                    person_anno_path = os.path.join(output_dir, "person_annotations", 
                                                   f"frame_{saved_count:05d}.txt")
                    with open(person_anno_path, 'w') as f:
                        for x1, y1, x2, y2, conf in person_boxes:
                            f.write(f"0 {x1} {y1} {x2} {y2} {conf}\n")
                
                # Save ball annotations
                if ball_boxes:
                    ball_anno_path = os.path.join(output_dir, "ball_annotations", 
                                                 f"frame_{saved_count:05d}.txt")
                    with open(ball_anno_path, 'w') as f:
                        for x1, y1, x2, y2, conf in ball_boxes:
                            f.write(f"1 {x1} {y1} {x2} {y2} {conf}\n")
                
                saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Dataset created with {saved_count} frames")
    print(f"📁 Saved to: {output_dir}")


def extract_person_crops(frame, person_boxes, crop_size=128):
    """
    Extract person crops from frame.
    
    Args:
        frame: Input frame
        person_boxes: List of person bounding boxes
        crop_size: Size of square crops
        
    Returns:
        list: List of cropped person images
    """
    crops = []
    for x1, y1, x2, y2, conf in person_boxes:
        # Expand bounding box to square
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        half_size = crop_size // 2
        
        # Ensure crop is within frame bounds
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(frame.shape[1], center_x + half_size)
        crop_y2 = min(frame.shape[0], center_y + half_size)
        
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            # Resize to target size
            crop = cv2.resize(crop, (crop_size, crop_size))
            crops.append(crop)
    
    return crops 