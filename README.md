# Volleyball Analytics Project

![Pipeline Animation](assets/images/volleyball_face_blur_animation.gif)

## Project Overview

This project provides a full computer vision pipeline for volleyball analytics, including:
- **Person Detection** (YOLOv8)
- **Ball Segmentation** (Siamese U-Net)
- **Face Detection & Blurring** (OpenCV DNN)
- **Video Processing & Visualization**
- **GIF/Animation Creation**

The pipeline is modular and can be used for:
- Player tracking
- Ball tracking
- Privacy-preserving video analytics (face blurring)
- Dataset preparation and model training

---

## Directory Structure

```
volleystat/
├── assets/                # Images and visual assets
├── configs/               # Configuration files
├── data/                  # Datasets (add your own, gitignored)
├── docs/                  # Documentation
├── examples/              # Example scripts and notebooks
├── models/                # Model weights and configs (gitignored)
├── results/               # Output videos, GIFs, metrics (gitignored)
├── scripts/               # All scripts (training, detection, utils, etc.)
├── src/                   # Core source code (models, utils, pipeline)
├── tests/                 # Unit and integration tests
├── .gitignore
├── ASSETS_README.md       # How to get required models/assets
├── local_paths.json       # Local paths (gitignored)
├── requirements.txt
├── SUCCESSFUL_COMMANDS.md # Useful commands
└── README.md
```

---

## How to Use

### 1. **Install Requirements**
```bash
pip install -r requirements.txt
```

### 2. **Prepare Data**
- Place your raw videos and images in the `data/` directory.
- Use scripts in `scripts/data_prep/` to convert, split, or format your data as needed.
- Example:
  ```bash
  python scripts/data_prep/combine_datasets.py
  python scripts/data_prep/convert_to_few_shot_format.py
  ```

### 3. **Train the Model**
- Use scripts in `scripts/training/` to train ball segmentation or person detection models.
- Example:
  ```bash
  python scripts/training/train_siamese.py --epochs 150 --batch_size 4
  ```
- Model checkpoints will be saved in `models/checkpoints/` (gitignored).

### 4. **Run the Full Pipeline**
- Use the main pipeline script to process a video, detect people, segment the ball, and blur faces:
  ```bash
  python main.py --input_video path/to/video.mp4 --output_dir results/video/segments/
  ```
- The output will include processed videos and GIFs (see the animation above).

### 5. **Create GIFs/Animations**
- Use scripts in `scripts/visualization/` to create GIFs from processed videos:
  ```bash
  python scripts/visualization/create_gif_from_video.py --input results/video/segments/your_video.avi --output results/video/segments/your_animation.gif
  ```

### 6. **Test and Evaluate**
- Use scripts in `scripts/testing/` and `scripts/evaluation/` to test models and evaluate results.
- Example:
  ```bash
  python scripts/testing/test_best_model.py --model_path models/checkpoints/best_model.pt
  ```

---

## Notes
- All large files (data, models, results) are gitignored.
- Update `local_paths.json` with your local paths if needed.
- See `ASSETS_README.md` for instructions on downloading required model weights.
- See `SUCCESSFUL_COMMANDS.md` for a list of useful commands.

---

## License
MIT License
