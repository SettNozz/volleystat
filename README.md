# Volleystat

Offline volleyball-video analytics for 4K fixed-camera footage. Pipeline detects and reconstructs ball trajectory, tracks player poses, calibrates court and net geometry, segments rallies, derives match analytics, predicts contact events, and exports viewer-ready annotations.

Current production path is `analyze_video.py` -> `GameAnalyzer` -> JSON/video artifacts -> `viewer.py`.

## Current capabilities

| Stage | Current implementation | Main output |
|---|---|---|
| Ball detection | YOLO primary detector, optional tiled ensemble, temporal gate and global trajectory DP | Ball candidates and selected point per frame |
| Ball cleanup | Static-source suppression and contact-preserving quadratic smoothing | Stable trajectory without parked false positives |
| Gap reconstruction | Conservative or contact-aware offline reconstruction | Inferred 2D coordinates with provenance/confidence |
| Player tracking | YOLO11m-pose + ByteTrack + calibrated court gate | Player boxes, 17 keypoints, track/team IDs |
| Court and net | Perspective calibration, SAM net tape, optional SAM3 posts, Hough fallback | Court homography, tape band, net-floor divider |
| Game logic | Integrated volleyball FSM + clean trajectory replay | Rallies, serves, contacts, attack map |
| Event model | Four-class causal sequence model | `serve`, `reception`, `pass`, `attack` |
| Rally spotting | Dilated TCN over continuous full-video features | Serve/reception/follow-up contacts and rally ends |
| Match analytics | Manual set/team mapping plus event-sourced aggregation | Sets, side changes, points and team statistics |
| Viewer | Flask app with upload, analysis, court editor and overlays | Interactive video and match dashboards |

Pipeline is offline-first. RTX 3080 is supported; correctness takes priority over real-time speed.

## Requirements

- Windows or Linux
- Python 3.11
- NVIDIA GPU recommended for 4K processing
- Local model weights or running MLflow model registry
- FFmpeg-compatible video input readable by OpenCV

```powershell
py -3.11 -m venv volley_env_311
.\volley_env_311\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Viewer, annotators and Parquet feature files
pip install flask flask-cors pyarrow
```

`data/`, `models/`, `results/`, local MLflow storage and `court_calibration*.json` are gitignored. They must be provisioned separately.

## Quick start

### 1. Calibrate camera

Calibration is required once per fixed camera setup:

```powershell
python scripts/interactive_court_calibration.py -i "C:\path\game.webm"
```

This writes a local `court_calibration.json`. Named court setups edited in viewer are stored in `configs/court_setups.json`.

### 2. Analyze video

```powershell
python analyze_video.py `
  --video "C:\path\game.webm" `
  --calibration court_calibration.json `
  --output "results\game"
```

Useful production options:

```powershell
# Higher ball recall, slower
python analyze_video.py --video game.webm --output results\game --ball-ensemble

# Compare conservative and contact-aware ball-gap reconstruction
python analyze_video.py --video game.webm --output results\game_recon `
  --ball-reconstruction compare `
  --reconstruction-contacts data\full_video_gold\full_video_annotations.json

# Stats and annotations only
python analyze_video.py --video game.webm --output results\game --no-video

# Manual set boundaries and stable team identities
python analyze_video.py --video game.webm --output results\game `
  --match-config configs\match_config.example.json
```

See every supported option:

```powershell
python analyze_video.py --help
```

### 3. Open viewer

```powershell
python viewer.py
```

Open `http://localhost:5050`. Viewer can upload a source video, launch analysis, restore saved jobs, edit court geometry, and show ball trail, reconstructed gaps, players, net, FSM events, attacks and match statistics.

Every analysed result under `results/` is listed at:

```text
http://localhost:5050/library
```

Library scans for `<stem>_stats.json` / `<stem>_annotations.json` pairs, shows rallies, events, frames, length, ball rate and available artifacts, and opens a run in the viewer. Results without stats are listed but not openable. `/api/library` returns the same index as JSON.

A single result directory can also be loaded directly:

```text
http://localhost:5050/load?video=<stem>&results_dir=results/<directory>
```

Second instance next to a running viewer:

```powershell
python viewer.py --port 5051
```

## Analysis artifacts

For input `game.webm` and `--output results/game`:

| Artifact | Purpose |
|---|---|
| `results/game/game_stats.json` | Aggregate metrics, rallies, events, reconstruction/model provenance |
| `results/game/game_annotations.json` | Compact frame-level ball, pose, net, FSM and event data |
| `results/game/game_analyzed.mp4` | Annotated video; omitted with `--no-video` |

Annotations distinguish observed and reconstructed ball positions. Reconstruction never becomes detector ground truth. By default it also does not alter FSM events; `--reconstruction-affects-fsm` admits only high-confidence reconstructed points into the clean FSM replay.

## Ball trajectory workflow

Default analysis applies:

1. Primary YOLO detection; optional tiled second detector with `--ball-ensemble`.
2. Motion-aware live gate.
3. Static-source suppression for parked false positives.
4. Global trajectory DP unless `--no-ball-dp` is set.
5. Contact-preserving smoothing (`--ball-smooth moderate` by default).
6. Optional offline gap reconstruction.
7. Clean FSM replay on final accepted trajectory.

Standalone reconstruction and review rendering:

```powershell
python scripts/reconstruct_ball_gaps.py `
  --annotations results\game\game_annotations.json `
  --output results\game\game_annotations_reconstructed.json `
  --mode compare `
  --contacts data\full_video_gold\full_video_annotations.json `
  --video game.webm
```

Input is not overwritten unless output points to same file and `--force` is explicitly supplied.

## Court and net calibration

`GameAnalyzer` scales calibration to native video resolution. Team split uses calibrated court homography and net-floor line, not raw pixel `net_y` alone.

Net workflow:

- SAM detects net tape and caches it in calibration.
- Isolated SAM3 worker can detect post X positions.
- Physical net height projects tape-to-floor geometry.
- Hough post detection is fallback when SAM3 is unavailable.
- Implausible post spacing emits warning and needs visual validation.

Manual refresh:

```powershell
python scripts/run_sam_net_detection.py `
  --video game.webm `
  --calibration court_calibration.json `
  --validate
```

Use `--posts-only` to keep saved tape bounds and refresh only posts/team divider. Use `--no-net-detection` during normal analysis when validated cached calibration must remain unchanged.

## Authoritative 4K event dataset

Current curated clip dataset:

```text
data/event_clips_4k_gold_v1
```

Current local snapshot contains 430 human-labelled clips from three complete 4K videos:

- `j0gRhgFCgIs_4_4k`
- `j3BO9GgUFBI_3_4k`
- `Khe4P8QcJkg_5_4k`

Feature schema: `resolution_normalized_v2`. Split unit is whole source video; clips from one recording must never be divided randomly between train and validation. `Khe4P8QcJkg_5_4k` is validation source in `split_policy.json`.

Train new models only from authoritative gold datasets. Human labels in full-video gold data are authoritative; FSM/model labels are weak labels only.

Available feature variants:

```text
features/                               original published features
features_baseline_raw_v1/               refreshed observed-ball baseline
features_reconstructed_conservative_v1/ conservative reconstructed trajectory
```

Train current four-class causal event model:

```powershell
python -m action_recognition.training.train_event_sequence `
  --manifest data\event_clips_4k_gold_v1\manifest.json `
  --features-dir data\event_clips_4k_gold_v1\features_reconstructed_conservative_v1 `
  --model transformer `
  --sequence-len 60 `
  --val-stems Khe4P8QcJkg_5_4k `
  --checkpoint-dir action_recognition\models\experiments_4k_gold_transformer `
  --epochs 50 `
  --seed 42
```

Run cross-video model selection before `--final-fit`. Core sequence classifier intentionally excludes pure `other`/`free_ball` contacts.

## Full-video human annotation

Clip labeling UI:

```powershell
python -m action_recognition.annotator.event_clip_annotator `
  --manifest data\event_clips_4k_gold_v1\manifest.json
# http://localhost:5056
```

Full-video labeling UI:

```powershell
python -m action_recognition.annotator.full_video_annotator `
  --manifest data\event_clips_4k_gold_v1\manifest.json
# http://localhost:5058
```

Export one completed video into reusable event, rally, dense-target and spotting artifacts:

```powershell
python -m action_recognition.annotator.export_full_video_annotations `
  --project data\event_clips_4k_gold_v1\full_video_annotations.json `
  --video-id <video-id> `
  --output-dir data\full_video_gold\<video-id> `
  --annotations-path results\<video>\<video>_annotations.json `
  --stats-path results\<video>\<video>_stats.json
```

Dataset replacement/build tools fail on missing human labels, mappings or Parquet files:

```powershell
python -m action_recognition.event_clips.build_4k_human_dataset --help
python -m action_recognition.event_clips.replace_source_dataset --help
```

## Rally spotting

Current spotting code lives under `action_recognition/rally_spotting/`. It operates on continuous full-video features with taxonomy:

```text
serve_contact | reception_contact | followup_contact | rally_end
```

Prepare and train:

```powershell
python -m action_recognition.rally_spotting.prepare `
  --manifest data\full_video_gold\rally_spotting_sources.json `
  --output-dir data\rally_spotting_three_full_gold_v1

python -m action_recognition.rally_spotting.train `
  --data-dir data\rally_spotting_three_full_gold_v1 `
  --checkpoint-dir action_recognition\models\rally_spotting_three_full_gold_v1 `
  --epochs 40
```

Apply checkpoint to saved analysis results without overwriting source files:

```powershell
python scripts\inference\predict_rally_spotting_results.py `
  --results-dir results\game `
  --checkpoint action_recognition\models\rally_spotting_three_full_gold_v1\best_rally_spotting.pt `
  --output-dir results\game_spotting
```

## Model registry

`configs/model_registry.json` maps logical model names to MLflow aliases and local fallbacks:

```text
ball_detector
ball_detector_tiled
pose_estimator
net_sam
event_sequence
spotting
```

Start local registry and register configured weights:

```powershell
python scripts/models/run_mlflow_server.py
# second terminal
python scripts/models/register_models.py --all
```

Pipeline records resolved path, registry version and checksum in result provenance. Explicit `--ball-model`, `--pose-model` and `--event-checkpoint` paths override registry. `--no-model-registry` uses configured fallback files directly.

## Repository map

```text
analyze_video.py                         production CLI
viewer.py                                local Flask viewer and analysis launcher
configs/
  model_registry.json                    model aliases and fallbacks
  court_setups.json                      named viewer court setups
  match_config.example.json              set boundaries and stable team IDs
src/
  detection/                              ball, pose, court and net detection
  pipeline/game_analyzer.py               end-to-end orchestration
  tracking/                               trajectory, reconstruction, FSM, analytics
  models/model_registry.py                model resolution and provenance
action_recognition/
  event_clips/                            schemas, features and gold dataset builders
  annotator/                              clip and full-video labeling tools
  training/                               causal event-model training/evaluation
  rally_spotting/                         continuous frame-level spotting
  hierarchical/                           multi-head phase/action/context model code
scripts/
  models/                                 local MLflow registry tools
  inference/                              offline prediction over saved results
  evaluation/                             trajectory and model diagnostics
tests/                                    unit and integration tests
```

## Validation

```powershell
python -m pytest tests -q
```

For architecture navigation, repository contains generated graph under `graphify-out/`.
