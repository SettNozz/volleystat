# Successful Commands for Volleyball Analytics Project

## Environment Setup
```bash
conda create -n volley_env python=3.8
conda activate volley_env
pip install torch torchvision opencv-python numpy matplotlib pillow tqdm scikit-learn ultralytics pandas PyYAML
```

## Training Commands
```bash
# Train Siamese UNet model
python scripts/training/train_siamese.py --epochs 150 --batch_size 4 --lr 0.001
```

## Testing Commands
```bash
# Test single model
python scripts/evaluation/test_model.py --model_path models/checkpoints/best_model.pt
```

## Video Processing Commands
```bash
# Process video with person and ball detection
python src/detection/video_processor.py --input video.mp4 --output results/
```

## GIF Creation Commands
```bash
# Create GIF with face blurring
python src/visualization/gif_creator.py --input video.mp4 --start 4 --duration 3
```

## Model Evaluation Commands
```bash
# Evaluate all checkpoints
python scripts/evaluation/evaluate_all.py --checkpoint_dir models/checkpoints/
```
