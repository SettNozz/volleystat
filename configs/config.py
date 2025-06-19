# Configuration file for volleyball ball segmentation project
import os
import json

# Helper to load user-specific paths from local_paths.json
def load_local_paths():
    try:
        with open(os.path.join(os.path.dirname(__file__), '../local_paths.json'), 'r') as f:
            return json.load(f)
    except Exception:
        return {}

local_paths = load_local_paths()

# Data paths (relative or from local_paths)
BASE_PATH = local_paths.get('data_paths', {}).get('dataset_root', 'data/')
OUTPUT_BASE = 'data/one_shot_segmentation_data/train_val_test_with_masks/'
PREPARED_DIR = 'data/train_val_test_prepared_for_training/'

# Video processing paths
VIDEO_PATH = local_paths.get('video_paths', {}).get('input_video', 'data/sample_video.mp4')
VIDEO_OUTPUT_PATH = local_paths.get('video_paths', {}).get('output_dir', 'results/video/segments/')
PERSON_BALL_DATASET_PATH = 'data/person_ball_dataset/'

# Model paths
MODEL_SAVE_PATH = local_paths.get('model_paths', {}).get('siamese_model', 'models/checkpoints/siamese_ball_segment_best.pt')
MODEL_LOAD_PATH = MODEL_SAVE_PATH
YOLO_MODEL_PATH = local_paths.get('model_paths', {}).get('yolo_model', 'models/pretrained/yolov8n.pt')

# Training parameters
BATCH_SIZE = 4
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Data processing parameters
CROP_SIZE = 64
IMAGE_SIZE = 256

# Video processing parameters
VIDEO_START_SEC = 30
VIDEO_DURATION_SEC = 30
FRAME_INTERVAL = 1

# Person detection parameters
PERSON_DETECTION_CONFIDENCE = 0.5
BALL_DETECTION_CONFIDENCE = 0.3
PERSON_CROP_SIZE = 128

# Dataset split ratios
DATASET_SPLITS = {
    "train": TRAIN_RATIO,
    "val": VAL_RATIO,
    "test": TEST_RATIO
}

# Output directories
OUTPUT_DIRS = {
    "training_loss_plot": "training_validation_loss.png",
    "test_results": "results/segmentation_result_test_dataset_improved_",
    "person_detection_video": os.path.join(VIDEO_OUTPUT_PATH, "person_detection_output.avi"),
    "person_ball_dataset": PERSON_BALL_DATASET_PATH
}

# Support image for evaluation
SUPPORT_IMAGE_PATH = local_paths.get('model_paths', {}).get('support_image', 'data/train_val_test_prepared_for_training/test/support/example_support_ball.jpg') 