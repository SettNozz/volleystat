import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import ultralytics
from ultralytics import YOLO
from typing import List, Tuple

class YOLOFaceDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize YOLO detector for face detection.
        
        Args:
            model_path: Path to custom YOLO model (optional, uses default if None)
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom YOLO model from: {model_path}")
        else:
            # Use default YOLO model
            self.model = YOLO('yolov8n.pt')
            print("Loaded default YOLOv8n model")
        
        print("Note: Using person detection and estimating face regions for blurring")
    
    def detect_persons(self, frame: np.ndarray, confidence_threshold=0.5) -> List[Tuple[int, int, int, int, float]]:
        """Detect persons specifically."""
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for person class (class 0 in COCO dataset)
                    if class_id == 0 and confidence > confidence_threshold:
                        x, y, width, height = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        persons.append((x, y, width, height, confidence))
        
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
        """Process a single frame: detect persons, estimate faces, and blur them."""
        # Detect persons
        person_boxes = self.detect_persons(frame)
        
        # Estimate face regions from person bounding boxes
        face_boxes = self.detect_faces_from_persons(frame, person_boxes)
        
        # Blur estimated face regions
        for (x, y, width, height) in face_boxes:
            frame = self.blur_face(frame, x, y, width, height, blur_strength)
        
        return frame


def create_gif_from_video_with_face_blur(video_path, output_gif_path, start_time=4, duration=3, fps=None, resize_factor=0.5):
    """
    Create a GIF animation from a video segment with face blurring.
    
    Args:
        video_path: Path to input video
        output_gif_path: Path to output GIF file
        start_time: Start time in seconds
        duration: Duration in seconds
        fps: Frames per second for the GIF (None = use original video FPS)
        resize_factor: Factor to resize frames (0.5 = half size)
    """
    # Initialize YOLO face detector
    face_detector = YOLOFaceDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info: {total_frames} frames, {video_fps:.1f} fps, {width}x{height}")
    
    # Use original video FPS if not specified
    if fps is None:
        fps = video_fps
        print(f"Using original video FPS: {fps:.1f}")
    
    # Calculate frame positions
    start_frame = int(start_time * video_fps)
    end_frame = int((start_time + duration) * video_fps)
    frames_to_extract = end_frame - start_frame
    
    # Calculate new dimensions
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    
    print(f"Extracting frames {start_frame} to {end_frame} ({frames_to_extract} frames)")
    print(f"Resizing to {new_width}x{new_height}")
    print(f"GIF FPS: {fps:.1f}")
    
    # Set video position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Extract frames
    frames = []
    
    # If GIF FPS is same as video FPS, extract all frames
    if abs(fps - video_fps) < 0.1:
        frame_interval = 1
        print("Extracting all frames (same FPS)")
    else:
        # Calculate frame interval to match desired FPS
        frame_interval = max(1, int(video_fps / fps))
        print(f"Extracting every {frame_interval} frame(s) to match {fps:.1f} FPS")
    
    for i in tqdm(range(frames_to_extract), desc="Processing frames with face blur"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only take every nth frame to match desired FPS
        if i % frame_interval == 0:
            # Process frame with face blurring
            processed_frame = face_detector.process_frame(frame, blur_strength=15)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_resized)
            frames.append(pil_image)
    
    cap.release()
    
    if not frames:
        print("Error: No frames extracted!")
        return
    
    print(f"Extracted {len(frames)} frames for GIF")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
    
    # Calculate frame duration in milliseconds
    frame_duration = int(1000 / fps)
    
    # Save as GIF
    print(f"Saving GIF to {output_gif_path}")
    print(f"Frame duration: {frame_duration}ms per frame")
    
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,  # Duration in milliseconds
        loop=0,  # 0 means loop forever
        optimize=True,
        quality=85
    )
    
    print(f"GIF created successfully!")
    print(f"Size: {new_width}x{new_height}")
    print(f"Frames: {len(frames)}")
    print(f"Duration: {len(frames) / fps:.1f} seconds")
    print(f"File: {output_gif_path}")


def main():
    # Configuration
    video_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\results\video\segments\segment_01_start_000409.mp4'
    output_gif_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\results\video\segments\volleyball_face_blur_animation.gif'
    
    # Create GIF from 00:00:04 to 00:00:07 (3 seconds) with face blurring
    create_gif_from_video_with_face_blur(
        video_path=video_path,
        output_gif_path=output_gif_path,
        start_time=4,      # Start at 4 seconds
        duration=3,        # 3 seconds duration
        fps=None,          # Use original video FPS for correct speed
        resize_factor=0.6  # Resize to 60% for smaller file size
    )


if __name__ == "__main__":
    main() 