# Volleystat — Volleyball Analytics Pipeline

> Computer-vision pipeline that processes volleyball match footage to detect the ball,
> estimate player poses, track game events, and classify ball-contact actions.

---

## What it does

| Stage | Technology | Output |
|---|---|---|
| Ball detection | YOLOv8 (curated8) + temporal filter | Ball position per frame |
| Pose estimation | YOLO11m-pose | Player bboxes → team A / B |
| Court calibration | Interactive perspective tool | `court_calibration.json` (net_y, etc.) |
| FSM game tracking | Finite-state machine on ball + net | Rally state, event triggers |
| Event clip extraction | FSM contact events → per-clip Parquet | Trajectory features, video features |
| Sequence classification | DualStreamProjectedLSTM (trajectory + CNN) | Per-clip multi-label: serve/reception/pass/attack/block/other |
| **Action Spotting** | **SpottingTCN (dilated conv)** | **Per-frame contact detection** |
| Overlay on video | StreamingEventPredictor (Transformer v3) | "Event: attack" text on annotated video |
| Results viewer | Flask + D3.js | Interactive field overlay at localhost:5050 |

---

## Quick Start

```bash
# 1. Activate virtual environment
volley_env_311\Scripts\activate

# 2. (Once) Calibrate court for your camera angle
python scripts/interactive_court_calibration.py -i "your_game.mp4"

# 3. Analyze a video
python analyze_video.py --video "C:/path/to/game.mp4"

# 4. View results in browser
python viewer.py
# → http://localhost:5050/load?video=<name>&results_dir=results/<name>
```

---

## Installation

```bash
# Core (ball detection, pose, pipeline)
pip install -r requirements.txt

# Pose estimation (YOLO11 pose)
pip install -r requirements_yolo.txt

# GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Python 3.11** is required. The local venv is `volley_env_311/`.

---

## Main Pipeline

### Analyze a video

```bash
# Full analysis with event overlay (default)
python analyze_video.py --video game.mp4

# Skip video output (faster, still produces stats + annotations JSON)
python analyze_video.py --video game.mp4 --no-video

# Limit to first 60 seconds
python analyze_video.py --video game.mp4 --duration 60

# Use a different event-recognition checkpoint
python analyze_video.py --video game.mp4 \
    --event-checkpoint action_recognition/models/checkpoints_transformer_v3/best_event_sequence.pt

# Disable event overlay completely
python analyze_video.py --video game.mp4 --no-event-overlay

# Process a frames directory (for testing)
python analyze_video.py --frames data/dataset_30_sec/frames/
```

**Outputs** (in `results/<video_name>/`):
- `<name>_stats.json` — per-frame ball positions, team detections, FSM events, rally stats
- `<name>_annotations.json` — frame-level ball + player + event annotations for the viewer
- `<name>_annotated.mp4` — annotated video (unless `--no-video`)

### Batch: analyze 3.mp4 and open viewer

```bat
run_analyze_3mp4_and_view.bat
```

---

## Court Calibration

Must be done **once per camera angle** before any analysis.

```bash
python scripts/interactive_court_calibration.py -i "your_video.mp4"
```

Interactive 5-stage process:
1. Back line (far side)
2. Side lines
3. Net position
4. Front line (near side)
5. Confirm & save

Saves `court_calibration.json`. Without it, net_y is wrong and the FSM will not split teams correctly.

---

## Action Recognition

### A. Sequence classification (per-clip, trained)

Each FSM contact event produces a ~2-second clip.
The **DualStreamProjectedLSTM** classifies the clip as one or more of:
`serve | reception | pass | attack | block | other`

**Event taxonomy:** multi-label (a block can also be an attack).

#### Prepare clips from analysis results

```bash
# If analysis was run with --no-video, pass --original-video
python -m action_recognition.event_clips.prepare \
    --results-dir results/part_2 \
    --output-dir data/event_clips_part2 \
    --original-video "C:/path/to/video.mp4"
```

#### Merge datasets

```bash
python -m action_recognition.event_clips.merge_manifests \
    --manifest1 data/event_clips_merged/manifest.json \
    --manifest2 data/event_clips_part3/manifest_predicted.json \
    --output-dir data/event_clips_merged_v2
```

#### Train

```bash
# Best model: DualStreamProjectedLSTM (--dual-stream flag)
python -m action_recognition.training.train_fusion_lstm \
    --manifest data/event_clips_merged_v5/manifest.json \
    --features-dir data/event_clips_merged_v5/features \
    --video-features-dir data/event_clips_merged_v5/video_features \
    --checkpoint-dir action_recognition/models/checkpoints_fusion_v4_dual \
    --dual-stream --epochs 60

# Or Transformer / LSTM (trajectory only)
python -m action_recognition.training.train_event_sequence \
    --manifest data/event_clips_merged_v5/manifest.json \
    --model transformer \
    --checkpoint-dir action_recognition/models/checkpoints_transformer_v3 \
    --epochs 50
```

**Best result (merged_v5, 1235 clips):**
`checkpoints_fusion_v4_dual` — macro F1 = **0.491** (optimised thresholds) / 0.481 (default)

#### Pre-label new clips

```bash
python -m action_recognition.inference.predict_event_sequence \
    --manifest data/event_clips_part2/manifest.json \
    --features-dir data/event_clips_part2/features \
    --checkpoint action_recognition/models/checkpoints_fusion_v4_dual/best_event_sequence.pt \
    --output data/event_clips_part2/manifest_predicted.json
```

#### Full pre-label pipeline for a new video

```bash
python -m action_recognition.inference.prelabel_new_video \
    --video "C:/path/to/part_1.mp4" \
    --checkpoint action_recognition/models/checkpoints_fusion_v4_dual/best_event_sequence.pt \
    --no-video
```

---

### B. Action Spotting (frame-level, new)

Instead of classifying a pre-segmented clip, the TCN predicts **which exact frame** is a ball contact
and **what action** it is — no pre-segmentation needed.

#### 1. Prepare spotting dataset

```bash
python -m action_recognition.spotting.prepare_spotting_data \
    --manifest data/event_clips_merged_v5/manifest.json \
    --features-dir data/event_clips_merged_v5/features \
    --output-dir data/spotting_v1 \
    --window 200 --stride 50 --sigma 3
```

Produces sliding windows with Gaussian soft labels around each contact frame.

#### 2. Train SpottingTCN

```bash
python -m action_recognition.spotting.train_spotting \
    --data-dir data/spotting_v1 \
    --checkpoint-dir action_recognition/models/checkpoints_spotting_v1 \
    --epochs 80 --lr 3e-4 --pos-weight 10
```

Evaluation metric: **mAP@5** (Mean Average Precision at 5-frame tolerance).

#### 3. Use in streaming inference

The `StreamingSpotter` class (same API as `StreamingEventPredictor`) processes a continuous frame stream:

```python
from action_recognition.spotting.streaming_spotter import StreamingSpotter

spotter = StreamingSpotter("action_recognition/models/checkpoints_spotting_v1/best_spotting.pt")
# In the frame loop:
label = spotter.add_frame(ball_pos, player_teams, net_y, fps, fsm_state="rally_active")
if label:
    print(f"Spotted: {label}")
```

---

## Event Clip Annotator

Fast Flask UI for labelling clips and correcting model or VLM pre-labels.

```bash
python -m action_recognition.annotator.event_clip_annotator \
  --manifest data/event_clips_part2/manifest_predicted.json
# Open: http://localhost:5056
```

**Workflow:**
1. Load existing manifest (e.g. `data/event_clips_part2/manifest_predicted.json`)
2. Annotator shows the clip video + current label
3. Correct labels → **Save & Next**
4. After labelling, merge into the main dataset and retrain

---

### Remote Ollama VLM pre-labels

A MacBook running Ollama can pre-label clips without moving the videos off the
Windows workstation. Start with `qwen3-vl:8b-instruct`:

```powershell
$env:OLLAMA_BASE_URL = "http://<mac-ip>:11434"
python -m action_recognition.vlm.prelabel_events --check
python -m action_recognition.vlm.prelabel_events `
  --manifest data/event_clips_part2/manifest.json `
  --limit 10
```

The source manifest is untouched; progress is saved after every clip to
`manifest_vlm_predicted.json`. See
[docs/OLLAMA_VLM_MAC.md](docs/OLLAMA_VLM_MAC.md) for Mac setup, benchmarking,
resuming, and troubleshooting.

---

## Results Viewer

Flask app that renders ball trajectory, player positions, FSM state, and events on a 2D field.

```bash
python viewer.py
# Open: http://localhost:5050
# Load results: http://localhost:5050/load?video=<name>&results_dir=results/<name>
```

**Overlays:** Ball position, trajectory trail (~40 frames), net (SAM band + net_y line),
Team A / Team B players, FSM state, detected events.

---

## Dataset Curator

FastAPI web tool for curating ball detection training data.

```bash
cd curator && python run_curator.py
# Open: http://localhost:8000
# Keyboard: A = Accept, S = Skip
```

---

## MLflow Experiment Tracking

```bash
# Windows (use --workers 1 to avoid WinError 10022)
mlflow ui --workers 1
# Open: http://localhost:5000
```

Experiment name for event sequence: `volleyball_event_sequence`

---

## Directory Structure

```
volleystat/
├── analyze_video.py          ← Main entry point
├── viewer.py                 ← Flask results viewer
├── court_calibration.json    ← Saved calibration (committed)
│
├── src/
│   ├── detection/            # ball_detector.py, pose_estimator.py
│   ├── tracking/             # integrated_volleyball_fsm.py, ball_tracker.py
│   ├── visualization/        # pose_ball_visualizer.py
│   └── pipeline/             # game_analyzer.py (main pipeline + event overlay)
│
├── action_recognition/
│   ├── event_clips/          # prepare.py, extract_clips.py, compute_features.py
│   ├── annotator/            # event_clip_annotator.py (Streamlit)
│   ├── data/                 # event_clip_dataset.py
│   ├── models/               # sequence_event_model.py + checkpoints_*/
│   ├── training/             # train_event_sequence.py, train_fusion_lstm.py
│   ├── inference/            # predict_event_sequence.py, streaming_event_predictor.py
│   └── spotting/             # SpottingTCN, prepare_spotting_data.py, train_spotting.py
│
├── scripts/
│   ├── detection/            # detect_ball_on_video.py, pose_estimation.py
│   ├── training/             # YOLO training scripts
│   └── utils/                # video_splitter.py, images_to_pdf.py
│
├── curator/                  # FastAPI dataset curation tool
├── configs/                  # config.py
├── models/                   # Trained YOLO/ball weights (local, not committed)
├── data/                     # Datasets, clips, features (local, not committed)
└── results/                  # Analysis outputs (local, not committed)
```

---

## Key Models

| Model | Path | Purpose |
|---|---|---|
| Ball detector | `models/yolov8_curated_training/yolov8n_volleyball_curated8/weights/best.pt` | Ball detection (55% recall) |
| Pose estimator | `yolo11n-pose.pt` (auto-download) | Player bboxes |
| Event overlay | `action_recognition/models/checkpoints_transformer_v3/best_event_sequence.pt` | Streaming event display |
| Sequence classifier | `action_recognition/models/checkpoints_fusion_v4_dual/best_event_sequence.pt` | Per-clip classification |
| Action spotter | `action_recognition/models/checkpoints_spotting_v1/best_spotting.pt` | Frame-level spotting |

---

## Notes

- **Court calibration is mandatory** before any analysis run
- `volley_env_311/` — local Python 3.11 venv (not committed)
- `data/`, `models/`, `results/` — local only (gitignored)
- GPU (CUDA) strongly recommended; CPU works but is slow
- MLflow on Windows: always run with `--workers 1`
