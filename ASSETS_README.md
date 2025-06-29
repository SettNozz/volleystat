# Asset Requirements

Download these files to the appropriate directories:

## YOLOv8 Model
- Download to: `models/pretrained/yolov8n.pt`
- URL: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

## OpenCV Face Detector
- Model: `models/pretrained/opencv_face_detector.caffemodel`
- Config: `models/configs/opencv_face_detector.prototxt`
- URLs:
  - https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.caffemodel
  - https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.prototxt

## Pose Estimation + Ball Detection

### MediaPipe Models (Auto-downloaded)
- **Object Detection**: EfficientDet-Lite (automatically downloaded by MediaPipe)
- **Pose Estimation**: MediaPipe Pose Landmarker (automatically downloaded by MediaPipe)

### YOLO Ball Detection Model
- **Custom trained volleyball detection model**
- Path: `models/yolov8_curated_training/yolov8n_volleyball_curated4/weights/epoch130.pt`
- This model is trained specifically for volleyball detection
- Alternative: Use `yolo11n-pose.pt` in root directory for pose-only detection

### Dependencies
Install pose estimation requirements:
```bash
pip install -r requirements_pose.txt
```

### Visual Assets
- **Demo GIF**: `assets/volleyball_pose_ball_detection_demo.gif` (13MB)
- Shows combined pose estimation and ball detection in action
