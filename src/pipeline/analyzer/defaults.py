"""Default model paths shared by the analyzer mixins."""

from pathlib import Path

# Project root (same expression as the original game_analyzer.py, one level deeper)
_ROOT = Path(__file__).parent.parent.parent.parent

# Measured on 232 human-labeled frames of video 3 (results/ball_eval): this
# model finds the ball on 0.72 of them at imgsz 1280, against 0.59 for the
# curated8 weights that used to be the default. The old "curated8 55% vs
# fullframe 38%" comparison counted frames where the detector fired, not frames
# where it was right.
_BALL_MODEL = "models/best_fullframe/weights/best.pt"
_BALL_IMGSZ = 1280
# Second opinion for the candidate cloud. curated8 sliced into tiles finds balls
# the primary model misses (union recall 0.93 vs 0.72); the trajectory solver
# sorts out which of the extra candidates are real. Costs a second forward pass
# per tile, so it is opt-in.
_BALL_MODEL_2 = "models/yolov8_curated_training/yolov8n_volleyball_curated8/weights/best.pt"
_BALL_TILE_GRID_2 = (3, 2)
_POSE_MODEL = "models/pretrained/yolo11m-pose.pt"
_POSE_FALLBACK = "yolo11n-pose.pt"  # will be auto-downloaded
