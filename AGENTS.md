# Volleystat - Agent Context

Updated: 2026-08-10. Code, CLI `--help`, tests, and config files override prose when they disagree.

## Mission and priorities

Volleystat is an offline-first volleyball analytics pipeline for fixed-camera 4K video. Accuracy matters more than runtime; target workstation has RTX 3080. Current work stays in 2D image/court coordinates.

Primary outputs:

- selected and reconstructed ball trajectory;
- tracked player poses and stable court-side/team identities;
- calibrated court/net geometry;
- FSM rallies and contacts;
- event and rally-spotting predictions;
- match analytics;
- viewer-ready JSON and annotated video.

## Non-negotiable data rules

- Authoritative event dataset: `data/event_clips_4k_gold_v1`.
- Current snapshot: 430 human-labelled clips from three complete 4K videos:
  - `j0gRhgFCgIs_4_4k`
  - `j3BO9GgUFBI_3_4k`
  - `Khe4P8QcJkg_5_4k`
- `Khe4P8QcJkg_5_4k` is validation source in `split_policy.json`.
- Split by complete `source_video`. Never random-split adjacent clips from one video across train/validation/test.
- Human labels are trusted. FSM/model labels are weak labels and must retain provenance.
- Reconstructed occluded-ball coordinates are derived data, never detector ground truth. Do not add them as YOLO ball boxes or train detector from them.
- Preserve observed/reconstructed distinction, reconstruction method, confidence, and `ball_out_of_frame` state.
- Missing ball outside camera: `ball=None`, `ball_out_of_frame=true`. Do not interpolate through off-camera periods.
- Reconstruct only plausible in-play gaps. Long gaps may be off-camera, dead ball, spare ball, or ball rolling on floor.
- Players, net, posts, and nearby objects can occlude ball. Player bbox/keypoints are sufficient occlusion evidence; contacts can involve hands, forearms, body, or legs.
- Multiple balls may exist between rallies. Track candidates, but keep one active match ball using existing selection logic.
- `data/`, `models/`, `results/`, MLflow storage, and `court_calibration*.json` are local/gitignored. Do not delete, overwrite, or bulk-regenerate them without explicit scope.

## Production entry points

Main pipeline:

```powershell
python analyze_video.py --video "C:\path\game.webm" --output "results\game"
python analyze_video.py --help
```

Viewer:

```powershell
python viewer.py
# http://localhost:5050
```

Calibration:

```powershell
python scripts/interactive_court_calibration.py -i "C:\path\game.webm"
python scripts/run_sam_net_detection.py --video game.webm --calibration court_calibration.json --validate
```

Current high-value analysis options:

```text
--ball-ensemble
--no-ball-dp
--ball-reconstruction off|conservative|contact-aware|compare
--reconstruction-contacts <JSON>
--reconstruction-affects-fsm
--ball-smooth N|light|moderate|strong|off
--no-static-suppression
--no-net-detection
--match-config <JSON>
--no-model-registry
--no-video
```

For any command, verify exact arguments with `--help` before documenting or running a long job.

## Pipeline architecture

```text
Video / frame directory
  -> GameAnalyzer
     -> model registry resolution + provenance
     -> ball YOLO candidate detection
     -> optional tiled ensemble
     -> BallTrackGate live selection
     -> static-source suppression
     -> BallTrajectoryDP global selection
     -> optional BallGapReconstructor
     -> contact-preserving trajectory smoothing
     -> YOLO11m-pose + ByteTrack + CourtPlayerGate
     -> calibrated net/court team divider
     -> clean IntegratedVolleyballFSM replay
     -> event model + match analytics
  -> *_stats.json
  -> *_annotations.json
  -> *_analyzed.mp4
  -> viewer.py
```

Main code:

- `analyze_video.py`: production CLI.
- `src/pipeline/game_analyzer.py`: orchestration and result export.
- `src/detection/ball_detector.py`: ball inference.
- `src/detection/player_pose.py`: poses, court projection, player gate.
- `src/detection/net_detector.py`: tape/post/floor detection.
- `src/tracking/ball_track_gate.py`: live ball choice.
- `src/tracking/ball_static_sources.py`: fixed false-positive removal.
- `src/tracking/ball_trajectory_dp.py`: global trajectory solver.
- `src/tracking/ball_gap_reconstructor.py`: offline gap reconstruction.
- `src/tracking/ball_trajectory_smoother.py`: contact-safe smoothing.
- `src/tracking/integrated_volleyball_fsm.py`: rally/contact FSM.
- `src/tracking/match_analytics.py`: set/team/score derivation.
- `viewer.py`: upload, background analysis, court editor, overlays, dashboards.

## Ball trajectory invariants

- Primary detector comes from logical model `ball_detector`; tiled second opinion from `ball_detector_tiled`.
- Static-source suppression runs before global solver and again on selected trajectory where needed.
- Global DP chooses candidate identity. Smoother refines coordinate noise; it must not smooth across contacts, gaps, restarts, or impossible jumps.
- Default smoothing is `moderate` (7 frames).
- Reconstruction default is `off`.
- `compare` stores conservative/contact-aware variants and uses conservative output.
- Reconstruction must not affect FSM unless `--reconstruction-affects-fsm`; then only accepted high-confidence points enter clean replay.
- Original observed values and derived values must remain recoverable in annotations.
- FSM/events, viewer, Parquet features, and analytics must consume same final accepted trajectory for a run.

Standalone reconstruction review:

```powershell
python scripts/reconstruct_ball_gaps.py `
  --annotations results\game\game_annotations.json `
  --output results\game\game_annotations_reconstructed.json `
  --mode compare `
  --contacts data\full_video_gold\full_video_annotations.json `
  --video game.webm
```

## Pose, court, and net

- Pose model: YOLO11m-pose + ByteTrack, 17 COCO keypoints.
- `CourtPlayerGate` filters spectators/benches and keeps court-member tracks with hysteresis.
- Team split uses foot projection and calibrated net-floor line. Do not revert to raw `y < net_y` logic.
- `team_a`/`team_b` are court sides. Stable real-team identities come from `team_identity_map` and can change across set-side switches.
- Calibration is mandatory per camera placement and scaled to native video resolution.
- Named viewer setups live in `configs/court_setups.json`.
- Match/set mappings live in files shaped like `configs/match_config.example.json`.
- SAM detects net tape; optional isolated SAM3 worker detects post X; Hough is fallback.
- Physical net height projects tape to floor. Implausible post spacing requires visual review.
- Current tested machine previously fell back to Hough because `SAM3_PYTHON` was unset; saved method was `hough_post_x_physical_height`, with 14.08 m warning. Do not claim SAM3 post path is end-to-end validated until rerun succeeds.

## Event data and models

Core sequence taxonomy is single-label, four classes:

```text
serve | reception | pass | attack
```

Raw `other`, `free_ball`, `dig`, `block`, and richer contact semantics remain in annotations. Collapse rules live in `action_recognition/event_clips/config.py`.

Authoritative dataset paths:

```text
data/event_clips_4k_gold_v1/manifest.json
data/event_clips_4k_gold_v1/features/
data/event_clips_4k_gold_v1/features_baseline_raw_v1/
data/event_clips_4k_gold_v1/features_reconstructed_conservative_v1/
```

Feature schema: `resolution_normalized_v2`. Event features contain 10 trajectory + 8 pose channels. Confirm manifest and feature sidecar before training.

Current training command:

```powershell
python -m action_recognition.training.train_event_sequence `
  --manifest data\event_clips_4k_gold_v1\manifest.json `
  --features-dir data\event_clips_4k_gold_v1\features_reconstructed_conservative_v1 `
  --model transformer `
  --sequence-len 60 `
  --val-stems Khe4P8QcJkg_5_4k `
  --checkpoint-dir action_recognition\models\experiments_4k_gold_transformer `
  --epochs 50 --seed 42
```

Run cross-video model selection before `--final-fit`. Do not resurrect metrics or checkpoint claims from merged_v4/merged_v5 experiments as current results.

## Human annotation

Clip annotator is Flask, not Streamlit:

```powershell
python -m action_recognition.annotator.event_clip_annotator --manifest data\event_clips_4k_gold_v1\manifest.json
# http://localhost:5056
```

Full-video annotator:

```powershell
python -m action_recognition.annotator.full_video_annotator --manifest data\event_clips_4k_gold_v1\manifest.json
# http://localhost:5058
```

Exporter produces enriched events, rallies/score, event clips, spotting labels, dense targets, phase segments, QC, and source manifest:

```powershell
python -m action_recognition.annotator.export_full_video_annotations --help
```

Never rewrite human source annotation files implicitly. Derived exact-duplicate collapse belongs only in exported datasets.

## Rally spotting

Current implementation is `action_recognition/rally_spotting/`, not removed `action_recognition/spotting/`.

Taxonomy:

```text
serve_contact | reception_contact | followup_contact | rally_end
```

Split unit is full source video. Trainer accepts human labels by default; weak FSM labels require explicit `--allow-weak-labels`.

```powershell
python -m action_recognition.rally_spotting.prepare --help
python -m action_recognition.rally_spotting.train --help
python scripts/inference/predict_rally_spotting_results.py --help
```

`action_recognition/hierarchical/` contains multi-head phase/action/context model code. Treat as experimental unless a validated training/evaluation artifact says otherwise.

## Model registry

`configs/model_registry.json` defines logical models and local fallbacks:

```text
ball_detector
ball_detector_tiled
pose_estimator
net_sam
event_sequence
spotting
```

MLflow alias `production` is authoritative when registry is reachable. Local fallback is used otherwise. Every run must record resolved source, version/path, and checksum.

```powershell
python scripts/models/run_mlflow_server.py
python scripts/models/register_models.py --all
```

Explicit model CLI paths override registry. Note: logical `spotting` entry still points at older checkpoint naming; verify checkpoint schema against `action_recognition/rally_spotting` before using it.

## Viewer and artifacts

For `game.webm --output results/game`:

```text
results/game/game_stats.json
results/game/game_annotations.json
results/game/game_analyzed.mp4
```

Viewer reads actual annotations, not redrawn guesses. It supports ball/trail, reconstruction provenance, players, net/court, FSM/events, attack heatmap, match analytics, upload jobs, persisted result restore, and court editing.

Existing result URL:

```text
http://localhost:5050/load?video=game&results_dir=results/game
```

Result library (`/library`, JSON at `/api/library`) indexes every `<stem>_stats.json` under `results/`. Entries without stats are listed as blocked, not hidden. Persisted upload jobs open through `/results/<job_id>`; all other directories through `/load`. `python viewer.py --port 5051` runs a second instance.

## Tests and validation

Use standalone Python 3.11 with project packages when venv launcher is broken:

```powershell
$env:PYTHONPATH=(Resolve-Path 'volley_env_311/Lib/site-packages').Path
$env:PYTHONUTF8='1'
$py=(Resolve-Path '.python/cpython-3.11.14-windows-x86_64-none/python.exe').Path
& $py -m pytest tests -q
```

Minimum targeted tests should follow changed subsystem. Useful groups:

```text
tests/test_ball_gap_reconstructor.py
tests/test_ball_static_sources.py
tests/test_ball_trajectory_dp.py
tests/test_ball_trajectory_smoother.py
tests/test_net_detector.py
tests/test_player_pose.py
tests/test_game_analyzer_rally_integration.py
tests/test_event_feature_schema.py
tests/test_full_video_annotator.py
tests/test_rally_spotting.py
tests/test_model_registry.py
tests/test_viewer_upload_workflow.py
```

Warnings about inaccessible `.pytest_cache` do not mean test failure.

## Working rules

- Inspect `git status` before edits. Preserve unrelated user changes.
- Use `apply_patch` for edits.
- Prefer `rg`/`rg --files` for search.
- Do not commit, push, merge, create PRs, retrain, or rewrite datasets unless user asks.
- Before any training or feature refresh, print and verify manifest path, feature path, source videos, label provenance, schema, split policy, and clip counts.
- Treat data/model mutations as material. Keep old artifacts; write versioned output directories.
- Never overwrite source annotations or analysis results by default.
- Stop and ask only for critical ambiguity: destructive target, authoritative dataset conflict, human-label rewrite, external publication, or choice that changes experiment meaning.
- For ordinary implementation details, inspect code/tests and proceed.
- README contains user-facing workflow. Update README and this file together when production commands or architecture change.
- `CLAUDE.md` imports this file and must remain a thin compatibility entry to prevent duplicated stale context.

## graphify

Repository knowledge graph lives in `graphify-out/`.

Rules:

- Before architecture/codebase work, read `graphify-out/GRAPH_REPORT.md` and prefer a focused graphify query before broad searches.
- Interpreter path is `.graphify_python` in repository root, separate from `volley_env_311`.
- Typical PowerShell query:

```powershell
$py=(Get-Content -Raw '.graphify_python').Trim()
& $py -m graphify query "question" --budget 1200
```

- Current graph contains `.runs/` snapshots and third-party nodes. Prefer facts whose `source_file` is current `src/`, `action_recognition/`, `scripts/`, `tests/`, or root config/docs. Ignore duplicate facts from `.runs/` when current source differs.
- Incremental update may refuse when shrink guard sees large corpus mismatch. Do not use `--force` without explicit approval; preserve existing graph when update refuses.
- After changing code, attempt `graphify update .`. Report refusal honestly; graph failure does not invalidate separately passing code tests.

## Repository map

```text
analyze_video.py
viewer.py
configs/
src/detection/
src/pipeline/
src/tracking/
src/models/model_registry.py
action_recognition/event_clips/
action_recognition/annotator/
action_recognition/training/
action_recognition/rally_spotting/
action_recognition/hierarchical/
scripts/models/
scripts/inference/
scripts/evaluation/
tests/
```

Read `README.md` for user-facing setup and examples.
