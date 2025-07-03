# Volleyball Analytics Project

![Volleyball Pose + Ball Detection](assets/volleyball_pose_ball_detection_demo.gif)

## Workflow Overview

This project implements a comprehensive volleyball analytics pipeline using state-of-the-art computer vision techniques:

### Ball Detection Workflow
1. **Custom YOLO Model**: Trained specifically for volleyball detection using curated dataset
2. **Temporal Filtering**: Advanced filtering system with velocity and size constraints
3. **Trajectory Tracking**: Real-time ball trajectory visualization with 50-point history
4. **False Positive Reduction**: Multi-stage validation to ensure accurate detections

### Human Detection & Pose Estimation Workflow  
1. **MediaPipe Object Detection**: EfficientDet-Lite finds people in video frames
2. **MediaPipe Pose Landmarker**: Estimates 33 body keypoints per person
3. **Keypoint Filtering**: Removes face landmarks (0-10), shows only body keypoints (11-32)
4. **Multi-Person Support**: Handles multiple people simultaneously with color-coded visualization

### Tracking & Analytics
- **Ball Trajectory Tracking**: Maintains temporal consistency with filtering
- **Frame-by-Frame Processing**: No inter-frame pose tracking to prevent keypoint drift
- **Real-Time Visualization**: Live statistics and progress monitoring
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support

---

## ⚠️ ВАЖЛИВО: Обов'язкова калібровка поля

**ПЕРЕД АНАЛІЗОМ ВІДЕО НЕОБХІДНО ВИКОНАТИ КАЛІБРОВКУ ПОЛЯ!**

Система вимагає точної калібровки волейбольного поля для коректної трансформації координат м'яча та гравців.

### Швидка калібровка:
```bash
cd volleystat
python scripts/interactive_court_calibration.py -i "your_video.mp4"
```

### Детальна інструкція:
📚 **[COURT_CALIBRATION_REQUIREMENTS.md](COURT_CALIBRATION_REQUIREMENTS.md)** - повний посібник з калібровки

### Що включає калібровка:
- ✅ Інтерактивна розмітка 5 етапів (задня лінія, бокові лінії, сітка, передня лінія)
- ✅ Автоматичний розрахунок перспективної трансформації
- ✅ Перевірка точності калібровки
- ✅ Збереження у форматі JSON для повторного використання

**Без калібровки координати м'яча будуть неточними!**

---

## Project Components

### 🔧 **Dataset Curator**
Web-based tool for curating volleyball ball detection datasets with keyboard shortcuts and real-time visualization.

- **Location**: [`curator/`](curator/)
- **Quick Start**: `cd curator && python run_curator.py`
- **Documentation**: [Curator README](curator/README.md)

---

## Installation

### 1. **Install Dependencies**
```bash
# Core dependencies
pip install -r requirements.txt

# Pose estimation dependencies  
pip install -r requirements_pose.txt
```

### 2. **GPU Setup (Recommended)**
```bash
# For NVIDIA GPUs with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Using the Pipeline

### Ball Detection Only
For volleyball detection and trajectory tracking:

```bash
cd volleystat
python scripts/detection/detect_ball_on_video.py
```

**Configuration (edit script)**:
```python
# Input video path
input_video = "path/to/your/video.mp4"

# Trained ball detection model
model_path = "models/yolov8_curated_training/yolov8n_volleyball_curated4/weights/epoch130.pt"

# Output video path
output_video = "data/results/ball_detection_output.mp4"
```

**Features**:
- Custom YOLO model for volleyball detection
- Temporal filtering (velocity + size constraints)
- Ball trajectory visualization with fading effect
- Batch processing for GPU optimization
- False positive filtering statistics

### Combined Pose + Ball Detection
For complete volleyball analytics with human pose estimation:

```bash
cd volleystat  
python scripts/mediapipe_pose_ball_combined.py
```

**Configuration (edit script)**:
```python
# Input video path
video_path = "data/test_videos_3n_10d/test_video_1_GX010378.mp4"

# Ball detection model path
ball_model_path = "models/yolov8_curated_training/yolov8n_volleyball_curated4/weights/epoch130.pt"

# Output video path  
output_path = "data/pose_estimation_visualization/output_video.mp4"
```

**Features**:
- MediaPipe multi-person pose estimation (33 keypoints)
- Custom YOLO volleyball detection
- Color-coded skeletons for each person
- Ball trajectory tracking with temporal filtering
- Real-time statistics and progress monitoring
- Frame-by-frame processing (no tracking between frames)

## Training Models

### Train YOLO Ball Detection Model
Train a custom volleyball detection model:

```bash
cd volleystat
python scripts/training/train_yolo_curated.py
```

**Configuration**:
- Edit dataset paths in the script
- Adjust training parameters (epochs, batch size, learning rate)
- Monitor training progress with Ultralytics integration

**Training Features**:
- Custom volleyball dataset support
- Data augmentation for volleyball scenarios
- Automatic validation and metrics tracking
- Model checkpointing and best model saving

### Monitor Training Progress
View training metrics and validation results:

```bash
cd volleystat
python scripts/training/view_training_progress.py
```

**Features**:
- Real-time training metrics visualization
- Loss curves and mAP progression
- Validation image previews
- Model performance comparison
- TensorBoard integration for detailed analysis

---

## Pose Estimation + Ball Detection

![Pose Detection Demo](assets/volleyball_pose_ball_detection_demo.gif)

This project includes an advanced pose estimation and ball detection system that combines:
- **MediaPipe Pose Estimation** (human skeleton detection)
- **YOLO Ball Detection** (volleyball detection)
- **Ball Trajectory Tracking** (with false positive filtering)
- **Real-time Visualization**

### Features

- **Multi-person pose detection** with 33 body landmarks
- **Face landmark filtering** (only body keypoints are shown)
- **Ball detection and trajectory visualization** with temporal filtering
- **Frame-by-frame processing** (no inter-frame tracking to avoid keypoint drift)
- **GPU-accelerated inference** for both pose and ball detection
- **Configurable detection parameters** and visualization options

### Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements_pose.txt
   ```

2. **Run pose estimation + ball detection**:
   ```bash
   cd volleystat
   python scripts/mediapipe_pose_ball_combined.py
   ```

3. **Configure paths** (edit the script):
   ```python
   # Test video path
   video_path = "data/test_videos_3n_10d/test_video_1_GX010378.mp4"
   
   # Ball detection model path  
   ball_model_path = "models/yolov8_curated_training/yolov8n_volleyball_curated4/weights/epoch130.pt"
   
   # Output path
   output_path = "data/pose_estimation_visualization/output_video.mp4"
   ```

### Technical Details

**Pose Detection:**
- Uses MediaPipe Object Detection to find people (EfficientDet-Lite)
- Applies MediaPipe Pose Landmarker to detect 33 body keypoints per person
- Filters out face landmarks (indices 0-10) for cleaner visualization
- No inter-frame tracking to prevent keypoint drift

**Ball Detection:**
- Uses custom-trained YOLOv8 model for volleyball detection
- Applies temporal filtering to reduce false positives:
  - Confidence threshold (0.5)
  - Velocity constraint (max 300px/frame movement)
  - Size consistency (max 3x bbox size change)
- Maintains ball trajectory with 50-point history

**Performance:**
- Processes 4K video at ~4.5 FPS on modern GPU
- Real-time visualization with progress indicators
- Comprehensive statistics and filtering metrics

### Output

The system generates:
- **Processed video** with pose keypoints and ball trajectory
- **Statistics file** with detection metrics and performance data
- **Console output** with real-time progress and frame-by-frame stats

---

## Documentation

### Core Documentation
- **README.md** - This file (project overview and general usage)
- **ASSETS_README.md** - Required models and assets download instructions
- **SUCCESSFUL_COMMANDS.md** - Useful commands and snippets

### Feature-Specific Documentation
- **POSE_BALL_DETECTION_README.md** - Comprehensive guide for pose estimation + ball detection
- **POSE_ESTIMATION_README.md** - Pose estimation only (MediaPipe)
- **POSE_SMOOTHING_README.md** - Pose smoothing and temporal filtering
- **POSE_DUPLICATE_ELIMINATION_README.md** - Duplicate pose removal techniques
- **BALL_TRACKING_FILTER_README.md** - Ball detection and trajectory filtering
- **CURATOR_README.md** - Dataset curation and management tool

### Configuration Files
- **local_paths.json** - Local file paths (edit for your system)
- **requirements.txt** - Core dependencies
- **requirements_pose.txt** - Pose estimation dependencies
- **requirements_curator.txt** - Data curation dependencies

---

## Notes
- All large files (data, models, results) are gitignored.
- Update `local_paths.json` with your local paths if needed.
- See individual README files for specific feature documentation.
- MediaPipe models are auto-downloaded on first use.

---

## License
MIT License
