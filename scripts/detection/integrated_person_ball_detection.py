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


class OpenCVDNNFaceDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize OpenCV DNN face detector using pre-trained model.
        
        Args:
            confidence_threshold: Detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        
        # Try to load DNN model, fallback to Haar cascade if it fails
        if self._load_dnn_model():
            print("Initialized OpenCV DNN face detector")
        else:
            print("Using Haar cascade face detector (DNN model failed to load)")
    
    def _load_dnn_model(self):
        """Try to load DNN model, return True if successful."""
        try:
            # Try different model file names and formats
            model_files = [
                ("opencv_face_detector_uint8.pb", "models/configs/opencv_face_detector.pbtxt"),
                ("opencv_face_detector.pb", "models/configs/opencv_face_detector.pbtxt"),
                ("face_detection_yunet_2022mar.onnx", None)  # ONNX format
            ]
            
            for model_path, config_path in model_files:
                if os.path.exists(model_path):
                    try:
                        if config_path:
                            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                        else:
                            # Try ONNX format
                            self.net = cv2.dnn.readNetFromONNX(model_path)
                        
                        # Test the model with a dummy input
                        dummy_blob = cv2.dnn.blobFromImage(np.zeros((300, 300, 3), dtype=np.uint8), 1.0, (300, 300), [104, 117, 123], False, False)
                        self.net.setInput(dummy_blob)
                        self.net.forward()
                        
                        # Set preferred backend and target
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                        return True
                    except Exception as e:
                        print(f"Failed to load model {model_path}: {e}")
                        continue
            
            # If no existing model works, try to download
            return self._download_face_model()
            
        except Exception as e:
            print(f"Error loading DNN model: {e}")
            return False
    
    def _download_face_model(self):
        """Download OpenCV face detection model files."""
        import urllib.request
        
        try:
            # Try downloading the correct model files
            print("Downloading OpenCV face detection model...")
            
            # Download TensorFlow model
            model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            config_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            
            print("Downloading model file...")
            urllib.request.urlretrieve(model_url, "models/pretrained/opencv_face_detector.caffemodel")
            print("Downloading config file...")
            urllib.request.urlretrieve(config_url, "models/configs/opencv_face_detector.prototxt")
            
            # Load the Caffe model
            self.net = cv2.dnn.readNetFromCaffe("models/configs/opencv_face_detector.prototxt", "models/pretrained/opencv_face_detector.caffemodel")
            
            # Test the model
            dummy_blob = cv2.dnn.blobFromImage(np.zeros((300, 300, 3), dtype=np.uint8), 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(dummy_blob)
            self.net.forward()
            
            # Set preferred backend and target
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            print("Model files downloaded and loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    
    def _use_haar_cascade(self):
        """Fallback to Haar cascade if DNN model is not available."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Warning: Could not load Haar cascade either!")
            self.face_cascade = None
        else:
            print("Using Haar cascade as fallback")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV DNN or Haar cascade fallback."""
        if hasattr(self, 'net'):
            try:
                return self._detect_faces_dnn(frame)
            except Exception as e:
                print(f"DNN face detection failed: {e}, falling back to Haar cascade")
                self._use_haar_cascade()
                return self._detect_faces_haar(frame)
        elif hasattr(self, 'face_cascade') and self.face_cascade is not None:
            return self._detect_faces_haar(frame)
        else:
            return []
    
    def _detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV DNN."""
        h, w = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        
        # Run inference
        detections = self.net.forward()
        
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                # Get coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_boxes.append((x1, y1, x2 - x1, y2 - y1))
        
        return face_boxes
    
    def _detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar cascade (fallback)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_boxes = []
        for (x, y, width, height) in faces:
            face_boxes.append((x, y, width, height))
        
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
        """Process a single frame: detect faces and blur them."""
        faces = self.detect_faces(frame)
        
        for (x, y, width, height) in faces:
            frame = self.blur_face(frame, x, y, width, height, blur_strength)
        
        return frame


class YOLOPersonDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize YOLO person detector.
        
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
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in the frame using YOLO.
        
        Returns:
            List of (x, y, width, height, confidence) tuples
        """
        # Run YOLO detection
        results = self.model(frame, verbose=False)
        
        person_boxes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for person class (class 0 in COCO dataset)
                    if class_id == 0 and confidence > 0.5:
                        x, y, width, height = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        person_boxes.append((x, y, width, height, confidence))
        
        return person_boxes


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
        
        # Initialize face detector (OpenCV DNN)
        self.face_detector = OpenCVDNNFaceDetector()
        
        # Initialize YOLO person detector
        self.person_detector = YOLOPersonDetector(yolo_model_path)
        
        # Load support image
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        support_img_pil = Image.open(support_img_path).convert('RGB')
        self.support_img = self.transform(support_img_pil)
        self.support_img = self.support_img.unsqueeze(0).to(self.device)
        
        print(f"Initialized integrated detector on {self.device}")
        print(f"Face detection method: OpenCV DNN")
        print(f"Person detection method: YOLO")
    
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
    
    def process_frame(self, frame: np.ndarray, blur_strength: int = 15) -> np.ndarray:
        """
        Process a single frame: detect and blur faces, detect persons, detect and visualize ball.
        
        Args:
            frame: Input frame
            blur_strength: Face blur strength
        
        Returns:
            Processed frame with faces blurred, person bounding boxes, and ball bounding box drawn
        """
        # Detect ball
        ball_bbox = self.detect_ball(frame)
        
        # Detect persons
        person_boxes = self.person_detector.detect_persons(frame)
        
        # Blur faces
        frame = self.face_detector.process_frame(frame, blur_strength)
        
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
    
    def process_video_segments(self, input_video_path: str, output_dir: str, 
                              segment_duration: int = 30, num_segments: int = 5,
                              blur_strength: int = 15):
        """
        Process video and save random 30-second segments with integrated person and ball detection.
        
        Args:
            input_video_path: Path to input video
            output_dir: Directory to save output video segments
            segment_duration: Duration of each segment in seconds
            num_segments: Number of random segments to extract
            blur_strength: Face blur strength
        """
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        print(f"Video info: {total_duration:.1f}s total, {fps:.1f} fps, {total_frames} frames")
        
        # Calculate possible segment start times
        frames_per_segment = int(segment_duration * fps)
        max_start_frame = total_frames - frames_per_segment
        
        if max_start_frame <= 0:
            print("Video is too short for the specified segment duration!")
            cap.release()
            return
        
        # Generate random start frames for segments
        import random
        random.seed(42)  # For reproducible results
        segment_starts = random.sample(range(0, max_start_frame), min(num_segments, max_start_frame))
        segment_starts.sort()  # Sort for easier processing
        
        print(f"Extracting {len(segment_starts)} segments of {segment_duration}s each")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        for i, start_frame in enumerate(segment_starts):
            print(f"\nProcessing segment {i+1}/{len(segment_starts)} (starting at frame {start_frame})")
            
            # Set video position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Output video writer for this segment
            output_path = os.path.join(output_dir, f"segment_{i+1:02d}_start_{start_frame:06d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_times = []
            ball_detections = 0
            person_detections = 0
            
            # Process frames for this segment
            for frame_idx in tqdm(range(frames_per_segment), desc=f"Segment {i+1}"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Time the processing
                start_time = time.time()
                
                # Process frame
                processed_frame = self.process_frame(frame, blur_strength)
                
                # Check if ball was detected
                ball_bbox = self.detect_ball(frame)
                if ball_bbox is not None:
                    ball_detections += 1
                
                # Check person detections
                person_boxes = self.person_detector.detect_persons(frame)
                person_detections += len(person_boxes)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                frame_times.append(processing_time)
                
                # Write frame
                out.write(processed_frame)
            
            out.release()
            
            # Print segment statistics
            avg_processing_time = np.mean(frame_times)
            ball_detection_rate = ball_detections / frames_per_segment * 100
            person_detection_rate = person_detections / frames_per_segment
            start_time_seconds = start_frame / fps
            
            print(f"Segment {i+1} completed:")
            print(f"  Start time: {start_time_seconds:.1f}s")
            print(f"  Duration: {segment_duration}s")
            print(f"  Avg processing time: {avg_processing_time:.3f}s")
            print(f"  Ball detection rate: {ball_detection_rate:.1f}%")
            print(f"  Avg persons per frame: {person_detection_rate:.1f}")
            print(f"  Saved to: {output_path}")
        
        cap.release()
        
        print(f"\nAll segments processed and saved to: {output_dir}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total segments: {len(segment_starts)}")
        print(f"  Segment duration: {segment_duration}s")
        print(f"  Total processed time: {len(segment_starts) * segment_duration}s")
        print(f"  Output directory: {output_dir}")


def main():
    # Configuration
    input_video_path = r'C:\Users\illya\Videos\video_for_sharing\GX010373_splits\GX010373_part2.mp4'
    output_dir = r'results/video/segments'
    model_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\models\to_test\slotel_siamse.pt'
    support_img_path = r'C:\Users\illya\Documents\volleyball_analitics\volleystat\data\datasets\train_val_test_prepared_for_training\train\support\f51dcd3e-frame_00182_ball.jpg'
    
    # Initialize integrated detector
    detector = IntegratedPersonBallDetector(
        model_path=model_path,
        support_img_path=support_img_path
    )
    
    # Process video segments
    detector.process_video_segments(
        input_video_path=input_video_path,
        output_dir=output_dir,
        segment_duration=8,  # 10 seconds
        num_segments=3,       # 3 random segments
        blur_strength=15
    )


if __name__ == "__main__":
    main() 