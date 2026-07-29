# Volleystat — Codex Context

## Project Overview

Volleyball analytics pipeline that processes video footage to extract:
- **Ball detection** — custom YOLOv8 model with temporal filtering
- **Pose estimation** — YOLO11m-pose + ByteTrack: 17 COCO keypoints per player with track IDs (split by net_y); skeleton overlay (team_b/near side highlighted), keypoints saved to annotations JSON, pose-derived features (jump, wrist-above-net, wrist–ball distance) via `src/tracking/pose_feature_extractor.py`
- **Court calibration** — interactive perspective transformation
- **FSM event tracking** — rally detection and game event analysis
- **Event clips** — one clip per contact (serve/reception/pass/attack/block/other), trajectory features, LSTM/Transformer multi-label classifier
- **Action Spotting** — SpottingTCN (dilated 1-D convolutions), per-frame contact detection, mAP@5 metric; replaces sliding-window approach
- **Event overlay in main pipeline** — sliding-window Transformer v3 predicts event type and shows "Event: serve" etc. on the annotated video
- **Event clip annotator** — Streamlit UI to label clips (from analysis results or existing manifest), pre-label with model, Save & Next
- **Dataset curator** — FastAPI web tool for labeling/curation

## Tech Stack

- Python 3.11 (`volley_env_311/` — local venv)
- PyTorch + YOLOv8 (Ultralytics) for ball detection
- MediaPipe for pose estimation
- OpenCV for video processing
- FastAPI + SQLite for dataset curator
- Streamlit for action recognition annotator
- MLflow for experiment tracking

## Key Commands

```bash
# === MAIN PIPELINE (start here) ===
# Default ball model: curated8 (55% detection, best overall)
# Analyze a video file (with event overlay from Transformer v3 by default):
python analyze_video.py --video path/to/game.mp4

# Disable event overlay or use another checkpoint:
python analyze_video.py --video game.mp4 --no-event-overlay
python analyze_video.py --video game.mp4 --event-checkpoint action_recognition/models/checkpoints_transformer_v2/best_event_sequence.pt

# Analyze frames directory (for testing):
python analyze_video.py --frames C:/Users/illya/Documents/volleyball_analitics/data/dataset_30_sec/frames/

# With options:
python analyze_video.py --video game.mp4 --output results/ --duration 60 --skip-frames 1

# Stats only (no annotated video output, faster):
python analyze_video.py --video game.mp4 --no-video

# Analyze 3.mp4 and open viewer (trajectory, net, players, events). In PowerShell: .\run_analyze_3mp4_and_view.bat
# Or manually: analyze then .\run_viewer_and_open_results.bat
# run_analyze_3mp4_and_view.bat — runs SAM net update, full analysis, then viewer at http://localhost:5050/load?video=3&results_dir=results/3

# === EVENT CLIPS (prepare, merge, train, pre-label) ===
# Prepare clips + features from analysis results (use --original-video if analysis was --no-video):
python -m action_recognition.event_clips.prepare --results-dir results/part_2 --output-dir data/event_clips_part2 --original-video "C:/path/to/video.mp4"

# Merge two manifests into one dataset:
python -m action_recognition.event_clips.merge_manifests --manifest1 data/event_clips_merged/manifest.json --manifest2 data/event_clips_part3/manifest_predicted.json --output-dir data/event_clips_merged_v2

# Train LSTM or Transformer (saves to --checkpoint-dir):
python -m action_recognition.training.train_event_sequence --manifest data/event_clips_merged_v2/manifest.json --features-dir data/event_clips_merged_v2/features --model transformer --checkpoint-dir action_recognition/models/checkpoints_transformer_v2 --epochs 50

# Pre-label new clips with trained model:
python -m action_recognition.inference.predict_event_sequence --manifest data/event_clips_part2/manifest.json --features-dir data/event_clips_part2/features --checkpoint action_recognition/models/checkpoints_transformer_v2/best_event_sequence.pt --output data/event_clips_part2/manifest_predicted.json

# Full prelabel pipeline for a new video (analyze + prepare + predict):
python -m action_recognition.inference.prelabel_new_video --video "C:/path/to/part_1.mp4" --checkpoint action_recognition/models/checkpoints_transformer_v2/best_event_sequence.pt --no-video

# After labeling part_1: merge, retrain, compare metrics:
python -m action_recognition.training.evaluate_after_relabel_part1

# === ACTION SPOTTING (frame-level contact detection) ===
# Step 1: build sliding-window dataset from existing clips
python -m action_recognition.spotting.prepare_spotting_data --manifest data/event_clips_merged_v5/manifest.json --features-dir data/event_clips_merged_v5/features --output-dir data/spotting_v1

# Step 2: train SpottingTCN (mAP@5 metric)
python -m action_recognition.spotting.train_spotting --data-dir data/spotting_v1 --checkpoint-dir action_recognition/models/checkpoints_spotting_v1 --epochs 80

# Step 3: streaming inference — use StreamingSpotter (same API as StreamingEventPredictor)
# See action_recognition/spotting/streaming_spotter.py

# === EVENT CLIP ANNOTATOR ===
# Start annotator (then load manifest or prepare from results):
streamlit run action_recognition/annotator/event_clip_annotator.py
# Or run batch that opens browser: action_recognition/annotator/run_event_clip_annotator.bat

# === SUPPORTING TOOLS ===
# Court calibration (run once per camera angle)
python scripts/interactive_court_calibration.py -i "your_video.mp4"

# Dataset curator (web UI at http://localhost:8000)
cd curator && python run_curator.py

# MLflow UI (Windows: use --workers 1)
mlflow ui --workers 1
```

## Pipeline Architecture

```
Video / Frames dir
       ↓
GameAnalyzer (src/pipeline/game_analyzer.py)
  ├── BallDetectorVideo  → YOLO epoch130.pt  (ball bbox in pixels)
  ├── YOLO11m-pose       → player bboxes → split by net_y → {team_a, team_b}
  ├── IntegratedVolleyballFSM  (pixel coords, net_y from calibration.json)
  └── StreamingEventPredictor  (optional) — sliding window 60 frames, Transformer v3 → "Event: serve" etc. on overlay
            ↓
       GameStats → results/<name>_stats.json + annotated video (+ event_clips from FSM events)
```

**Key design choices:**
- Everything in **pixel coordinates** — no world transforms needed
- `net_y` from `court_calibration.json` — only real reference used (scaled to video resolution)
- Players split by net_y: `y < net_y` → team_a (far side), `y > net_y` → team_b (near side)
- FSM tracks net crossings (ball y crossing net_y) to switch team control
- **Event overlay:** last 60 frames of trajectory features (ball_x, ball_y, ball_speed, ball_dir, ball_net_dist, player_dist, team_side, frame_norm) → Transformer v3 → display state smoothed by mode over last 5 predictions

**Test data:** `C:\Users\illya\Documents\volleyball_analitics\data\dataset_30_sec\frames\` (899 frames, ~30sec @ 30fps)

## Directory Structure

```
volleystat/
├── src/
│   ├── detection/        # ball_detector.py, pose_estimator.py, court_detector.py
│   ├── tracking/         # ball_tracker.py, integrated_volleyball_fsm.py, rally_analyzer.py
│   ├── visualization/    # pose_ball_visualizer.py
│   ├── models/           # unet.py, ball_unet.py
│   └── pipeline/         # game_analyzer.py (main pipeline + event overlay)
├── scripts/
│   ├── detection/        # detect_ball_on_video.py, pose_estimation.py
│   ├── training/         # YOLO training scripts
│   ├── data_prep/        # dataset preparation
│   ├── data_curation/    # curation tools
│   ├── tracking/         # tracking utilities
│   └── utils/            # video_splitter.py, images_to_pdf.py
├── action_recognition/
│   ├── event_clips/      # config.py, extract_clips.py, compute_features.py, prepare.py, merge_manifests.py
│   ├── annotator/        # event_clip_annotator.py (Streamlit), run_event_clip_annotator.bat
│   ├── data/             # event_clip_dataset.py (60-frame sequences, Parquet)
│   ├── models/           # sequence_event_model.py (LSTM/Transformer), checkpoints_* / best_event_sequence.pt
│   ├── training/         # train_event_sequence.py, train_fusion_lstm.py, evaluate_after_relabel_part1.py
│   ├── inference/        # predict_event_sequence.py, streaming_event_predictor.py, prelabel_new_video.py
│   └── spotting/         # spotting_model.py (SpottingTCN), prepare_spotting_data.py, train_spotting.py, streaming_spotter.py
├── curator/              # FastAPI web curator (keyboard: A=Accept, S=Skip)
├── configs/              # config.py
├── data/                 # event_clips*, event_clips_merged*, results/
├── models/               # Trained YOLO/ball weights
├── court_calibration.json  # Saved calibration (committed)
├── analyze_video.py      # Main entry: --video, --no-video, --event-checkpoint, --no-event-overlay
├── main.py               # Ball segmentation entry point (UNet one-shot)
└── run_complete_pipeline.py  # Full dataset processing workflow
```

## Architecture Notes

### Ball Detection
- YOLOv8 custom-trained model; **default: `curated8`** (`models/yolov8_curated_training/yolov8n_volleyball_curated8/weights/best.pt`) — 55% detection rate on test clip vs 38% for best_fullframe
- Temporal filter: max velocity 300px/frame, max bbox change 3x
- 50-point trajectory history visualization
- Multi-stage false positive reduction
- `NO_BALL_RALLY_TIMEOUT = 150` frames (5 s) — increased from 45 to avoid premature rally end

### Pose Estimation (keypoints + tracking)
- YOLO11m-pose via `model.track(persist=True, tracker="bytetrack.yaml")` — 17 COCO keypoints + stable track IDs per player; falls back to per-frame inference if tracking deps missing
- `src/detection/player_pose.py` — `PlayerPose` dataclass (team, center, bbox, keypoints (17,3), track_id, `foot_point()`), `extract_player_poses()`, `draw_skeleton()`, `CourtProjector` (image↔world via calibration `perspective_matrix`)
- **On-court filter (main defense vs spectators/benches/referees):** `CourtPlayerGate` (stateful, in `player_pose.py`) — track becomes a *member* the moment its (ankle-confirmed) foot point lands on court (±1.5 m sidelines, ±4 m serve zones), at any frame, not just video start; members stay visible up to 3 m outside for up to 5 s (hysteresis — players chasing balls don't vanish); **occlusion pass:** detections with no visible ankles whose bbox overlaps a kept player (≥0.25 of smaller bbox) and within +3 m are kept visible immediately (players emerging from behind teammates), membership granted once ankles confirm on-court; sticky team per track (locked only on trusted feet); max 6/team by inside-ness (`GameAnalyzer(max_players_per_team=...)`). Spectators never step on court → never members. Falls back to x-range heuristic if calibration has no homography. `GameAnalyzer(pose_imgsz=1280)` improves recall on small/far players
- **Team split by court half** (foot world-y vs line under net, near side auto-detected) — NOT by pixel y vs net tape; team_a (far) / team_b (near = the observed team); centers still feed FSM and StreamingEventPredictor unchanged. Homography is rescaled with calibration in `_scale_calibration`; court outline + net floor line drawn on overlay for visual verification
- **Skeleton overlay:** team_b bright yellow, thick bones, track ID shown; team_a dimmed thin — style in `GameAnalyzer._SKELETON_STYLE`
- **Annotations JSON:** each frame gets `poses` ([{id, team, bb, kp}]) and `pose_f` (values in `pose_feature_names` order, stored in file header)
- **Pose features** (`src/tracking/pose_feature_extractor.py`, `POSE_FEATURE_NAMES`, 8 values/frame): nearest-to-ball player's wrist–ball dist, wrist-above-net, jump (ankle rise vs per-track rolling 80th-percentile baseline), arm raise, team side, dist-to-net + team_b max jump / max wrist-above-net. Wrist-above-net reference = SAM `net_tape_top` when available (physical net top), else `net_y`. Intended as extra channels for the event sequence model at dataset-prep time (concat with trajectory features)

### Net segmentation (SAM)
- `src/detection/net_detector.py` — `NetDetector`: SAM vit_h (checkpoint `~/Downloads/sam_vit_h_4b8939.pth`, `pip install segment-anything`) + HoughLinesP hybrid; samples 5 frames from first 5 s at 720p, point prompt at calibration net_y → tape top; Hough in ROI below → tape bottom; median over frames. Returns 720p coords
- **Auto-runs in `process_video`** (`_maybe_run_sam_net_detection`) when `net_detection_sam` missing from calibration; persists 720p values to `court_calibration.json`, keeps native-resolution values in memory (all `CalibrationData` fields share one pixel space; scaled uniformly in `_scale_calibration`)
- Manual refresh (e.g., new camera setup): `python scripts/run_sam_net_detection.py --video path.mp4 [--validate]`; visual check: `scripts/validate_net_detection.py` (HTML page)
- Overlay draws the tape band (translucent orange) instead of the plain net_y line when available

### Court Calibration
- 5-stage interactive point selection (back line → side lines → net → front line)
- Saves to `court_calibration.json` — **must be done before analysis**
- Provides real-world coordinate mapping

### FSM Event Tracking
- `src/tracking/integrated_volleyball_fsm.py` — main FSM
- Detects: serve, rally, attack, out-of-play states
- `video_fsm_processor.py` — applies FSM to video stream

### Event clips and sequence model
- **Taxonomy:** serve, reception, pass, attack, block, other (multi-label).
- **Per clip:** one segment from previous contact to current contact; trajectory features (ball_x, ball_y, ball_speed, ball_dir, ball_net_dist, player_dist, team_side, frame_norm, ball_vy, ball_above_net) in Parquet; fixed 60-frame sequences at train time (pad/trim).
- **Models:** `SequenceEventLSTM`, `SequenceEventTransformer`, `ProjectedSequenceLSTM`, `DualStreamProjectedLSTM` in `action_recognition/models/sequence_event_model.py`; trained with `train_event_sequence.py` or `train_fusion_lstm.py` (BCEWithLogitsLoss+pos_weight, stratified split, MLflow, early stopping).
- **Best model: `checkpoints_fusion_v4_dual/`** — DualStreamProjectedLSTM trained on merged_v5 (1235 clips): test **macro_f1=0.491 (opt thresholds)** / 0.481 (default). Architecture: separate LN+projection per modality (traj 10->32, CNN 576->96, cat->128->BiLSTM). Uses `FusionClipDataset` + `--dual-stream` flag. Optimised thresholds in `checkpoints_fusion_v4_dual/thresholds.json`.
- **Ablation v5** (6 experiments on merged_v5, `run_ablation_v5.py` / `run_fusion_experiments_v5.bat` → `fusion_experiments_v5_comparison.csv`):
  - `v3_dual` baseline (1235 clips): macro_f1 default=0.405, opt=0.437
  - **`v4_dual` (WINNER): default=0.481, opt=0.491** — dual-stream projection helps most (consistent with v4 ablation)
  - `v4_onecycle`: default=0.441, opt=0.469 — OneCycleLR helpful but less than dual-stream
  - `v4_smooth` (label_smooth=0.05): default=0.439, opt=0.466
  - `v4_aug` (noise=0.10): default=0.410, opt=0.424 — traj noise **hurt** (pixel coords scale-variant, 10% too aggressive)
  - `v4_all` (kitchen sink): default=0.438, opt=0.453 — noise+smooth interact badly
- **Ablation v4** (on merged_v4 / 1211 clips, `run_fusion_experiments.bat` → `fusion_experiments_comparison.csv`):
  - `v1` baseline: macro_f1=0.485 → `v2_dual`: **0.515** ✓ winner. `v2_onecycle`: 0.501. `v2_aug`: 0.441. `v2_smooth`: 0.471 (0.496 opt). `v2_all`: 0.462 (0.491 opt).
- **Legacy checkpoints:** `checkpoints_transformer_v3/`, `checkpoints_lstm_v4/` (1211 clips, macro_f1=0.377). Default for overlay and pre-label: still Transformer v3 (trajectory-only, compatible with streaming predictor).
- **CNN features:** `data/event_clips_merged_v5/video_features/` — MobileNetV3-Small, 16 frames, 576-dim, extracted by `extract_video_features.py` (run once per dataset). Previous: `data/video_features_mobilenet_v1/` (v4 dataset).
- **Merged datasets:** `event_clips_merged` (280), `event_clips_merged_v2` (424), `event_clips_merged_v3` (534), `event_clips_merged_v4` (1211), `event_clips_merged_v5` (1235, current). Merge with `merge_manifests.py`; prepare with `prepare.py` (use `--original-video` if analysis used `--no-video`).

### Action Spotting (frame-level detection)
- **Goal:** detect exact frame of each ball contact and classify it — no pre-segmentation needed.
- **Files:** `action_recognition/spotting/`
  - `prepare_spotting_data.py` — groups clips by source video, concatenates per-clip Parquet into full video timeline, places Gaussian labels (σ=3 frames) at contact_frame for each class, extracts sliding windows (W=200, stride=50) → `data/spotting_v1/{windows,labels}_{train,val}.npy`
  - `spotting_model.py` — `SpottingTCN`: Linear projection → stack of DilatedResBlock (dilations 1,2,4,8,16,8,4,2,1) → dropout → per-frame linear head → (B, T, C) logits. Also `nms_peaks()` for 1-D NMS.
  - `train_spotting.py` — BCEWithLogitsLoss + pos_weight=10, CosineAnnealingLR, early stopping, metric = mAP@5 (Mean AP at 5-frame tolerance).
  - `streaming_spotter.py` — `StreamingSpotter`: same `add_frame(ball_pos, player_teams, net_y, fps, fsm_state)` API as `StreamingEventPredictor`; ring buffer (200 frames), runs TCN every 10 frames, 1-D NMS, FSM gate (no serve during rally_active).
- **Training data:** `data/spotting_v1/` built from `event_clips_merged_v5` (1235 clips, 9 source videos); val stem = `["3"]`.
- **Checkpoint:** `action_recognition/models/checkpoints_spotting_v1/best_spotting.pt`

### Event overlay in main pipeline
- `StreamingEventPredictor` in `action_recognition/inference/streaming_event_predictor.py`: keeps last 60 frames of trajectory features, runs Transformer (or LSTM) every 2 frames, returns display label; smoothing by mode over last 5 predictions; min confidence 0.35.
- FSM-aware: passes `fsm_state` to predictor → excludes "serve" during `rally_active`.
- Serve cooldown: 8 seconds between consecutive serve displays. Sticky overlay: last non-serve label shown for 2 s after rally ends.
- Loaded by `GameAnalyzer` when `event_checkpoint_path` is set (default: Transformer v3). Disable with `--no-event-overlay` or override with `--event-checkpoint PATH`.

### Viewer (trajectory and field)
- Flask app: `python viewer.py` then open http://localhost:5050. Load pre-existing results via `/load?video=3&results_dir=results/3`.
- **On-field overlays:** Ball (current position), Trail (ball trajectory from last ~40 frames), Net (SAM band + net_y line), Players (team A/B by net_y), FSM State, Events (serve, reception, attack, etc.). Data comes from `*_annotations.json` (per-frame ball, players, state, events).
- **Trajectory:** Viewer builds trail from consecutive frame annotations; gaps > 15 frames break the line. Toggles in UI control what is drawn.

### Action recognition (legacy / other)
- Located in `action_recognition/` (annotator, data loaders, models).
- MLflow for experiment tracking (`mlruns/`). For event sequence: experiment name `volleyball_event_sequence`.

## Requirements Files

- `requirements.txt` — core (torch, torchvision, opencv, ultralytics, pandas)
- `requirements_pose.txt` — mediapipe, opencv, torch
- `requirements_yolo.txt` — ultralytics + data science stack

## Important Notes

- **Court calibration is mandatory** before any video analysis — coordinates will be wrong without it
- `volley_env_311/` is the local Python 3.11 virtual environment (not committed)
- `data/`, `models/`, `results/` are local only (not in git)
- GPU (CUDA) strongly recommended for real-time processing
- **Event overlay:** requires `action_recognition/models/checkpoints_transformer_v3/best_event_sequence.pt`; if missing, overlay is disabled and a log message is printed
- **Event clip annotator:** load "Existing manifest" with path to `manifest_predicted.json` after pre-labeling; or "From analysis results" then Prepare (with `--original-video` for prepare if analysis used `--no-video`)
- **Prepare from analysis:** when analysis was run with `--no-video`, pass `--original-video "path/to/video.mp4"` to `prepare` so clips can be cut from the source video
- GIF creation utility: `scripts/example_gif_usage.py`
- MLflow on Windows: run `mlflow ui --workers 1` to avoid WinError 10022

## graphify

This project has a graphify knowledge graph at `graphify-out/`.

Rules:

- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure
- Prefer `graphify query "<question>" --budget 1200` to gather graph context before broad `rg`/Grep searches; also `graphify explain "<Node>"` and `graphify path "<A>" "<B>"`
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files
- The graphify interpreter path is recorded in `.graphify_python` (uv tool env, NOT `volley_env_311`); in Bash use `"$(cat .graphify_python)"` for direct `python -m graphify...` calls
- The `graphify` CLI is on PATH (installed via `uv tool install graphifyy`); the graphify skill is registered in Codex (`~/.Codex/skills/graphify`)
- After modifying code files in a session, run `graphify update .` (or `"$(cat .graphify_python)" -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"`) to keep the graph current — AST-only, no API cost

## Imported Claude Cowork project instructions
