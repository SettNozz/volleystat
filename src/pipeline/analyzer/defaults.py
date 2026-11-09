"""Default model paths shared by the analyzer mixins."""

from pathlib import Path

# Project root (same expression as the original game_analyzer.py, one level deeper)
_ROOT = Path(__file__).parent.parent.parent.parent

# The registry entry ball_detector decides the weights and imgsz; these are
# only used when an entry carries no imgsz. best_fullframe fine-tuned at 1920
# finds the ball on 0.967 of the 6_tiPcRYGcBtc window after the trajectory
# solver, against 0.944 for best_fullframe at 1280 (results/ball_eval_20261108_v1).
_BALL_MODEL = "models/ball_fullframe_ft1920_20261108_v1/weights/best.pt"
_BALL_IMGSZ = 1920
# Second opinion for the candidate cloud. curated8 sliced into tiles finds balls
# the primary model misses (union recall 0.93 vs 0.72); the trajectory solver
# sorts out which of the extra candidates are real. Costs a second forward pass
# per tile, so it is opt-in.
_BALL_MODEL_2 = "models/yolov8_curated_training/yolov8n_volleyball_curated8/weights/best.pt"
_BALL_TILE_GRID_2 = (3, 2)
_POSE_MODEL = "models/pretrained/yolo11m-pose.pt"
_POSE_FALLBACK = "yolo11n-pose.pt"  # will be auto-downloaded
