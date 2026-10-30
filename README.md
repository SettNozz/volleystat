# Volleystat

Model outputs are cached automatically for every video/frame analysis. New
hypotheses reuse them; `--require-model-cache` forbids accidental inference.
Default outputs are JSON and the original-video viewer. Use `--render-video`
only for an accepted video export. See [cache/replay workflow](docs/model_output_cache.md).

Stable player IDs now run by default in CLI and viewer analyses: frozen OSNet
appearance galleries plus a Kalman motion filter and lost-ID recovery. Occluded
boxes do not update motion; ambiguous overlapping crops stay unknown. Raw tracker IDs remain
separate; uncertain IDs stay unknown. Install the pinned encoder once and see
[player identity workflow](docs/player_identity_workflow.md). Default scope is
six players on the near side. Confirmed substitutes can receive additional
IDs through a per-video roster file; returning players retain their original
IDs. Local tracklet contradictions are rejected rather than relabelled.
Motion cannot assign an ID whose frozen appearance is clearly worse than
another known person's appearance.
Full-match accuracy and personal contact attribution remain unverified.
The viewer's **Player contacts** panel selects a roster ID, highlights only
that skeleton, and lists its reception, dig/pass/set, attack, block and serve
events with click-to-seek. Unverified contact proposals use saved ball/pose
evidence from both sides; ambiguous contacts stay unresolved. Correct an
attribution in the dropdown; corrections are separate, source-bound records.
No video/model rerun is needed. See [per-player contacts](docs/player_contacts.md).
For short skeleton dropouts, the experimental
[offline pose completion](docs/pose_completion.md) uses the same person's past
and future observed joints. Hidden joints appear as cyan dashed estimates,
with an independent viewer toggle. Detector poses and contact counts remain
unchanged until the estimates have been independently validated.
Viewer shows contacts with unknown team as `—` when both teams are selected;
selecting one team includes only contacts attributed to that team.

Offline volleyball-video analytics for 4K fixed-camera footage. Pipeline detects and reconstructs ball trajectory, tracks player poses, calibrates court and net geometry, segments rallies, derives match analytics, predicts contact events, and exports viewer-ready annotations.

Current production path is `analyze_video.py` -> `GameAnalyzer` -> JSON/video artifacts -> `viewer.py`.

## Current capabilities

| Stage | Current implementation | Main output |
|---|---|---|
| Ball detection | YOLO primary detector, optional tiled ensemble, temporal gate and global trajectory DP | Ball candidates and selected point per frame |
| Ball cleanup | Static-source suppression and contact-preserving quadratic smoothing | Stable trajectory without parked false positives |
| Gap reconstruction | Conservative or contact-aware offline reconstruction | Inferred 2D coordinates with provenance/confidence |
| Player tracking | YOLO11m-pose + ByteTrack + calibrated court gate | Player boxes, 17 keypoints, track/team IDs |
| Player identity | Frozen OSNet galleries + Kalman motion | Stable roster IDs; unknowns retained |
| Model output cache | Source/model/input fingerprints | Replay compatible saved outputs without inference |
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

# JSON and original-video viewer (default; --no-video remains compatible)
python analyze_video.py --video game.webm --output results\game --no-video

# Hypothesis replay: fail instead of rerunning models
python analyze_video.py --video game.webm --output results\game_hypothesis --require-model-cache

# Explicit annotated video export
python analyze_video.py --video game.webm --output results\game_export --require-model-cache --render-video

# Disable stable identities
python analyze_video.py --video game.webm --output results\game_tracks --no-player-identity

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
| `results/game/game_analyzed.mp4` | Optional annotated video; only with `--render-video` |

Annotations distinguish observed and reconstructed ball positions. Reconstruction never becomes detector ground truth. By default it also does not alter FSM events; `--reconstruction-affects-fsm` admits only high-confidence reconstructed points into the clean FSM replay.

After trajectory cleanup and clean FSM replay, the pipeline recomputes `pose_f` and event-model predictions from the final exported ball coordinates and saved player poses. Predictor and pose history start fresh for each replay. Features can use accepted reconstruction without changing the separate FSM reconstruction policy. Normalized clip features preserve missing/off-camera observations and reset velocity at gaps and new tracks, matching streaming inference; legacy pixel features retain their compatibility behavior. Existing annotations, feature files and checkpoints are not rewritten automatically. Inline video overlays remain provisional unless the final rendering pass is enabled.

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

## Experimental SAM3 people masks

Compare body masks with saved poses on a short video interval using the isolated
SAM3 environment. Outputs include a comparison video, missing-pose review
candidates, adjacent-frame mask associations and eight versioned mask features
with validity flags. Production inference and existing checkpoints stay unchanged.
See [SAM people-mask experiment](docs/sam_people_masks.md) for commands, provenance,
raw-YOLO diagnostics and the future training contract.

Player detection uses `--pose-imgsz 2560` by default in both CLI and viewer
for small/far players in 4K. Set `--pose-imgsz 1280` explicitly to reduce cost.
Geometrically impossible detected net-floor lines fall back to the calibrated
court centre line in memory, recorded as `calibrated_court_midline_fallback`.
This prevents both teams collapsing into one six-player cap; it does not add
unobserved players or rewrite the source calibration.

Optional `--person-masks results/<completed-sam-run>/masks.jsonl` recovers real
YOLO-pose keypoints from unmatched SAM person crops. It verifies source-video
identity and keeps recovery inference separate from ByteTrack. Recovered poses
retain their model provenance and a separate negative SAM-mask ID namespace;
all poses still pass the existing court gate and six-per-team cap. See the
[SAM workflow](docs/sam_people_masks.md) for the complete commands and limitations.

## Contact spotting in the pipeline

The dense contact spotter replaces the rally state machine's contact events on
any video, with no labels needed:

```powershell
python analyze_video.py --video "C:\path\game.webm" --output "results\game" `
  --contact-spotter models\contact_spotter\spotter_20261028_v1.pt
```

Its typed contacts become `all_events` and per-frame `ev`; the FSM's are kept in
`all_events_fsm` and `ev_fsm`. Rallies and service attempts are rebuilt from
the spotter's contacts, each opening on its serve (a serve the decode missed
is recovered from the spotter's serve probability and marked `inferred`); a
serve nobody touched is a service attempt, not a rally. The FSM's are kept in
`rallies_fsm`/`service_attempts_fsm`. Score and the attack heatmap stay
FSM-derived. `scripts/apply_spotter_rallies.py` rebuilds a saved result into a
fresh directory without rerunning detection. The spotter cannot tell warm-up from play and does
not attribute a contact to a team, so exclude non-game time with
`--match-config`. Train a checkpoint with
`scripts/models/train_contact_spotter.py`. See
[production contact spotter](docs/production_contact_spotter.md).

## Dense contact spotting with counted structure

An experimental replacement for the hand-written rally state machine: a dense
per-frame model over the nine phase/contact labels, decoded with a
constraint-aware Viterbi whose transitions, start/end labels and duration bounds
are counted from the human annotations instead of written by hand.

```powershell
python scripts\evaluation\run_dense_contact_spotting.py --help
```

It uses the audited contact review in `data/contact_spatial_gold_v1`, which
supplies verified contact frames, and needs no new labelling or extraction. On
three whole-video folds the counted grammar beats threshold-tuned peak picking
(core four-class macro F1 at +/-5 frames 0.7528 -> 0.7897 on pose-enabled
analyses) mainly by removing false positives. The same metric scores the shipped
FSM at 0.1561 once the pipeline is rerun with poses, so the dense spotter is
about five times better at putting a contact on the right frame. A matched
control shows the eight pose channels contribute little outside `block`.
Rescoring that run into its two halves puts detection at 0.8605 type-agnostic F1
and typing at 0.9208 accuracy on contacts it located, so the remaining error is
in finding touches, not naming them. See
[detection versus typing](docs/contact_detection_vs_typing.md),
[dense contact spotting](docs/dense_contact_spotting.md) and the
[landscape review](docs/event_detection_research_2026-10.md). Production models
and datasets are unchanged.
