# Configuration file for volleyball ball segmentation project

# Data paths
BASE_PATH = "C:/Users/illya/Documents/volleyball_analitics/data/one_shot_segmentation_data/training_val_test_full_dataset_after_export/project-1-at-2025-06-16-03-31-0d8e2e8a/"
OUTPUT_BASE = "C:/Users/illya/Documents/volleyball_analitics/data/one_shot_segmentation_data/train_val_test_with_masks/"
PREPARED_DIR = "C:/Users/illya/Documents/volleyball_analitics/data/train_val_test_prepared_for_training/"

# Video processing paths
VIDEO_PATH = "C:/Users/illya/Videos/TMP_VOLLEY_GOPRO/GX010373.mp4"
VIDEO_OUTPUT_PATH = "C:/Users/illya/Documents/volleyball_analitics/data/bb_visualization/"
PERSON_BALL_DATASET_PATH = "C:/Users/illya/Documents/volleyball_analitics/data/person_ball_dataset/"

# Model paths
MODEL_SAVE_PATH = "models/siamese_ball_segment_best.pt"
MODEL_LOAD_PATH = "models/siamese_ball_segment_best.pt"
YOLO_MODEL_PATH = "models/yolo11n.pt"

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
    "test_results": "C:/Users/illya/Documents/volleyball_analitics/data/segmentation_result_test_dataset_improved_",
    "person_detection_video": f"{VIDEO_OUTPUT_PATH}GX010373_person_detection_output.avi",
    "person_ball_dataset": PERSON_BALL_DATASET_PATH
}

# Support image for evaluation
SUPPORT_IMAGE_PATH = "C:/Users/illya/Documents/volleyball_analitics/data/train_val_test_prepared_for_training/test/support/278f8b51-frame_00189_ball.jpg" 