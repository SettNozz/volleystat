import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple
import time

class FaceDetector:
    def __init__(self, method='mediapipe', confidence_threshold=0.5):
        """
        Initialize face detector with specified method.
        
        Args:
            method: 'mediapipe' (fast) or 'opencv_dnn' (accurate)
            confidence_threshold: Detection confidence threshold
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        
        if method == 'mediapipe':
            # MediaPipe Face Detection (fast, good for real-time)
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=confidence_threshold
            )
            
        elif method == 'opencv_dnn':
            # OpenCV DNN Face Detection (more accurate, slightly slower)
            model_path = "models/face_detection_yunet_2022mar.onnx"
            self.face_detector = cv2.FaceDetectorYN_create(
                model_path,
                "",
                (320, 320),  # Input size
                0.9,  # Confidence threshold
                0.3,  # NMS threshold
                5000  # Top k
            )
            
        else:
            raise ValueError("Method must be 'mediapipe' or 'opencv_dnn'")
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        
        return faces
    
    def detect_faces_opencv_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV DNN."""
        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        
        _, faces = self.face_detector.detect(frame)
        if faces is None:
            return []
        
        # Convert to (x, y, width, height) format
        face_boxes = []
        for face in faces:
            x, y, width, height = face[:4].astype(int)
            face_boxes.append((x, y, width, height))
        
        return face_boxes
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame."""
        if self.method == 'mediapipe':
            return self.detect_faces_mediapipe(frame)
        elif self.method == 'opencv_dnn':
            return self.detect_faces_opencv_dnn(frame)
    
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
        """Process a single frame: detect faces and blur them."""
        faces = self.detect_faces(frame)
        
        for (x, y, width, height) in faces:
            frame = self.blur_face(frame, x, y, width, height, blur_strength)
        
        return frame


def process_video_with_face_blur(input_video_path: str, output_video_path: str, 
                                method: str = 'mediapipe', blur_strength: int = 15):
    """
    Process video with face detection and blurring.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        method: 'mediapipe' (fast) or 'opencv_dnn' (accurate)
        blur_strength: Blur strength (odd number, higher = more blur)
    """
    # Initialize face detector
    face_detector = FaceDetector(method=method)
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Processing video with {method} face detection...")
    print(f"Total frames: {total_frames}")
    
    frame_times = []
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Time the processing
        start_time = time.time()
        
        # Process frame
        processed_frame = face_detector.process_frame(frame, blur_strength)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        frame_times.append(processing_time)
        
        # Write frame
        out.write(processed_frame)
        
        # Print progress every 100 frames
        if frame_idx % 100 == 0:
            avg_time = np.mean(frame_times[-100:]) if len(frame_times) > 0 else 0
            print(f"Frame {frame_idx}/{total_frames}, Avg processing time: {avg_time:.3f}s")
    
    cap.release()
    out.release()
    
    avg_processing_time = np.mean(frame_times)
    print(f"Video processing completed!")
    print(f"Average processing time per frame: {avg_processing_time:.3f}s")
    print(f"Output saved to: {output_video_path}")


def integrate_with_ball_detection(frame: np.ndarray, face_detector: FaceDetector, 
                                 ball_bbox: Tuple[int, int, int, int] = None) -> np.ndarray:
    """
    Integrate face blurring with ball detection visualization.
    
    Args:
        frame: Input frame
        face_detector: Initialized face detector
        ball_bbox: Ball bounding box (x, y, width, height) or None
    
    Returns:
        Processed frame with faces blurred and ball bounding box drawn
    """
    # First, blur faces
    frame = face_detector.process_frame(frame)
    
    # Then, draw ball bounding box if provided
    if ball_bbox is not None:
        x, y, width, height = ball_bbox
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, 'Ball', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


if __name__ == "__main__":
    # Example usage
    input_video = r"C:\Users\illya\Videos\video_for_sharing\GX010373_splits\GX010373_part2.mp4"
    output_video = r"results/video/face_blurred_GX010373_part2.mp4"
    
    # Process video with face blurring
    process_video_with_face_blur(
        input_video_path=input_video,
        output_video_path=output_video,
        method='mediapipe',  # Use 'opencv_dnn' for more accuracy
        blur_strength=15
    ) 