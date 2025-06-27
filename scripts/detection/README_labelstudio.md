# YOLO to Label Studio - Instructions

This set of scripts allows applying a trained YOLO model to new videos and automatically create annotations for Label Studio.

## Files

- `detect_ball_on_video.py` - main class for ball detection with additional Label Studio functionality
- `process_new_videos_for_labelstudio.py` - convenient script for processing multiple videos

## Quick Start

### 1. Processing a single video

```bash
cd C:\Users\illya\Documents\volleyball_analitics\volleystat

python scripts\detection\detect_ball_on_video.py
```

### 2. Processing multiple videos

```bash
# Process all videos from directory
python scripts\detection\process_new_videos_for_labelstudio.py --video-dir "C:\Users\illya\Videos\volleyball_new"

# Process specific videos
python scripts\detection\process_new_videos_for_labelstudio.py --videos "C:\path\to\video1.mp4" "C:\path\to\video2.mp4"

# Use patterns
python scripts\detection\process_new_videos_for_labelstudio.py --videos "C:\Videos\*.mp4"
```

## Command Line Parameters

### process_new_videos_for_labelstudio.py

- `--model` - path to YOLO model (default: our best model)
- `--videos` - list of video files or patterns
- `--video-dir` - directory with videos (alternative to --videos)
- `--output-dir` - directory for saving results
- `--confidence` - detection confidence threshold (0.0-1.0, default: 0.3)
- `--skip-frames` - number of frames to skip (default: 9, i.e., every 10th frame)
- `--create-videos` - also create annotated videos

## Usage Examples

### Basic usage
```bash
python scripts\detection\process_new_videos_for_labelstudio.py --video-dir "C:\Users\illya\Videos\new_matches"
```

### With custom settings
```bash
python scripts\detection\process_new_videos_for_labelstudio.py \
    --video-dir "C:\Users\illya\Videos\new_matches" \
    --confidence 0.5 \
    --skip-frames 4 \
    --output-dir "volleystat\data\labelstudio_march_2025" \
    --create-videos
```

### Processing specific files
```bash
python scripts\detection\process_new_videos_for_labelstudio.py \
    --videos "C:\Videos\match1.mp4" "C:\Videos\match2.mp4" \
    --confidence 0.4
```

## Results

After processing videos, the following will be created:

```
volleystat/data/labelstudio_batch_YYYYMMDD_HHMMSS/
├── processing_summary.txt              # General statistics
├── video_001_match1/
│   ├── images/                         # Images with detected balls
│   │   ├── abcd1234-frame_00123.jpg
│   │   ├── efgh5678-frame_00456.jpg
│   │   └── ...
│   ├── annotations.json               # Label Studio annotations for this video
│   └── match1_annotated.mp4          # Annotated video (if --create-videos)
├── video_002_match2/
│   ├── images/
│   ├── annotations.json
│   └── ...
└── ...
```

### Label Studio Annotation Format

Each `annotations.json` file contains annotations in standard Label Studio format:

```json
[
  {
    "id": 1,
    "annotations": [
      {
        "result": [
          {
            "value": {
              "x": 60.72,     // X coordinate in percentage
              "y": 29.4,      // Y coordinate in percentage  
              "width": 1.35,   // Width in percentage
              "height": 2.64,  // Height in percentage
              "rectanglelabels": ["Volleyball Ball"]
            },
            "type": "rectanglelabels"
          }
        ]
      }
    ],
    "data": {
      "image": "/data/upload/1/abcd1234-frame_00123.jpg"
    }
  }
]
```

## Import to Label Studio

1. Create a new project in Label Studio
2. Configure labels for "Volleyball Ball" class
3. Upload images from `images/` directory
4. Import annotations from `annotations.json` file

## Model Configuration

To use a different model:

```bash
python scripts\detection\process_new_videos_for_labelstudio.py \
    --model "path\to\your\model.pt" \
    --video-dir "C:\Videos"
```

## Performance Parameters

- **skip_frames**: Higher value = faster processing, fewer frames
  - `0` - process all frames
  - `9` - every 10th frame (recommended)
  - `29` - every 30th frame (for very fast processing)

- **confidence**: Confidence threshold
  - `0.1` - many detections, possible errors
  - `0.3` - balanced (recommended)
  - `0.5` - high confidence, fewer detections

## Troubleshooting

### GPU not detected
- Make sure CUDA compatible PyTorch version is installed
- Check if GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`

### Low RAM
- Reduce batch_size in code
- Process videos one by one

### Video won't open
- Check video format (supported: mp4, avi, mov, mkv, wmv)
- Make sure OpenCV can open the file

## Contact

If you encounter problems using the scripts, contact the developer. 