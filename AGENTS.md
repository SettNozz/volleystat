# Volleystat - Agent Context

Updated: 2026-11-10. Code, CLI `--help`, tests, and config files override prose when they disagree.

## Mission and priorities

Volleystat is an offline-first volleyball analytics pipeline for fixed-camera 4K video. Accuracy matters more than runtime; target workstation has RTX 5070 Ti 16 GB (Blackwell sm_120; needs torch >= 2.7 cu128, tested 2.7.1+cu128), i9-10900F, 32 GB RAM. Current work stays in 2D image/court coordinates.

Primary outputs:

- selected and reconstructed ball trajectory;
- tracked player poses and stable court-side/team identities;
- calibrated court/net geometry;
- FSM rallies and contacts;
- event and rally-spotting predictions;
- match analytics;
- viewer-ready JSON and annotated video.

## Non-negotiable data rules

- Authoritative human labels are the completed sources in `data/full_video_gold`.
- Verified active checkpoint training dataset: `data/event_clips_three_full_gold_causal_120f_v1` (MLflow manifest/features parameters).
- Current active snapshot: 414 human-labelled clips, 410 core four-class examples, from three complete 4K videos:
  - `j0gRhgFCgIs_4_4k`
  - `j3BO9GgUFBI_3_4k`
  - `Khe4P8QcJkg_5_4k`
- `Khe4P8QcJkg_5_4k` is the primary validation source. The preserved `data/event_clips_4k_gold_v1/split_policy.json` specifies the same whole-source split.
- The older 430-clip `data/event_clips_4k_gold_v1` remains unchanged; do not confuse it with the actual current checkpoint training source.
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
--ball-imgsz N
--no-ball-dp
--ball-reconstruction off|conservative|contact-aware|compare
--reconstruction-contacts <JSON>
--reconstruction-affects-fsm
--ball-smooth N|light|moderate|strong|off
--no-static-suppression
--no-net-detection
--match-config <JSON>
--no-model-registry
--contact-spotter <CHECKPOINT>
--no-contact-side
--no-player-identity
--player-identity-side a|b
--player-identity-seed <JSON>
--player-identity-method roster-discovery|seed-frame
--player-reid-assets <DIR>
--player-reid-device auto|cpu|cuda
--no-model-prefetch
--no-video
```

GPU throughput (2026-11-06/07): `src/pipeline/model_prefetch.py` runs decoding
and missing raw ball/pose inference ahead of the ordered loop (1 decoder,
1 ball, `POSE_PREFETCH_WORKERS=2` pose threads with separate model
instances). Main thread alone touches SQLite, ByteTrack, gate and FSM.
Raw pose uses `src/detection/pose_gpu.py` (`GpuPoseModel`: pinned decode,
GPU letterbox, Ultralytics NMS/scaling, CUDA graph captured by `prepare()`
before other CUDA threads start; never capture mid-run, the ball thread's
legacy stream invalidates it). Its source and `PREPROCESS` are in the
`pose_raw` key; the user accepted that pose-cache break on 2026-11-07. Never
edit the other hashed sources (`PoseTracker`, `detect_ball_on_video.py`,
`player_reid_encoder.py`) merely for speed. Ball batch stays 8 (cache key).
`BackgroundEncoder` computes OSNet embeddings for every saved pose box in the
loop (same encoder/crop/key, serialised off-thread, one transaction per drain),
so identity/contact-author passes need no second decode. OSNet batch is 128.
A stopped MLflow server costs one 0.5 s TCP probe (`_require_reachable`).
30 s of `j0gRhgFCgIs_4_4k`, empty cache: 151 s -> 61-64 s wall, loop 15 -> 24 fps,
identity 34 s -> 3 s, steady GPU 47% -> 67%; cache-only replay 14 s. Now ~GPU
bound: pose ~20 ms + OSNet ~10 ms (0.84 ms/crop) + ball/event per frame
(ceiling ~28 fps, actual 23-24); without identity 35.6 fps = VP9 decoder.
Rejected 2026-11-07 (see README table): OpenCV D3D11/NVDEC decode (slower,
different pixels under unchanged ball/OSNet keys), FFmpeg thread counts (no
gain), OSNet resize process pool (24 -> 19.2 fps), OSNet CUDA graph (no gain).
OSNet channels_last/FP16 change embeddings (1.75e-5/3e-4) and would need a new
OSNet key; TensorRT pose needs a new pose key. Ask before either.
GPU reruns are not bit-identical even serially (822/899 identical raw ball
outputs between two serial runs), so judge speed changes against a serial
control run and replay hypotheses through the cache.

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
     -> contact side proposals (ball box size + possession order)
     -> offline OSNet whole-video roster identity (default near side, raw IDs preserved)
     -> final annotation-based video/frame rendering
  -> *_stats.json
  -> *_annotations.json
  -> *_analyzed.mp4
  -> viewer.py
```

Main code:

- `analyze_video.py`: production CLI.
- `src/pipeline/game_analyzer.py`: `GameAnalyzer` frame loop, offline passes
  (`_run_offline_passes`), `process_video`/`process_frames`, result export.
  Re-exports `GameStats`, `CalibrationData` from `src/pipeline/analyzer/`.
- `src/pipeline/analyzer/`: one `GameAnalyzer` mixin per concern: `cache`,
  `calibration` (incl. net detection), `models` (loading/warm-up/FSM),
  `detection` (ball/pose inference, prefetch), `ball_trajectory` (DP, smoothing),
  `events` (clean FSM replay, features, spotter), `identity` (OSNet roster,
  contact authors), `rendering`; plus `defaults.py` and `types.py`.
- `src/detection/player_pose.py`: poses, court projection, player gate.
- `src/detection/net_detector.py`: tape/post/floor detection.
- `src/tracking/ball_track_gate.py`: live ball choice.
- `src/tracking/ball_static_sources.py`: fixed false-positive removal.
- `src/tracking/ball_trajectory_dp.py`: global trajectory solver.
- `src/tracking/ball_gap_reconstructor.py`: offline gap reconstruction.
- `src/tracking/ball_trajectory_smoother.py`: contact-safe smoothing.
- `src/tracking/integrated_volleyball_fsm.py`: rally/contact FSM.
- `src/tracking/match_analytics.py`: set/team/score derivation.
- `viewer.py`: Flask routes, jobs, upload/background analysis, library index.
  State-free helpers live in `src/viewer_app/` (stats, library, calibration
  preview, player-contact inputs, court-correction I/O, media/thumbnails).
- `viewer/templates/viewer.html`: results-page markup and Jinja boot constants;
  its scripts are `viewer/static/viewer_page.js` (state, listeners, init, loaded
  last), `viewer_charts.js`, `viewer_tables.js`, `viewer_overlay.js` (redraw),
  `viewer_set_trajectories.js` (set-to-attack pairs, zone/setter filters,
  clickable overlay lines),
  `viewer_attack_height.js` (approximate near-side contact height), and
  `viewer_page.css`. The feature scripts only declare functions and state.

## Ball trajectory invariants

- Primary detector comes from logical model `ball_detector`; tiled second opinion from `ball_detector_tiled`.
- Since 2026-11-08 `ball_detector` is `best_fullframe` fine-tuned at imgsz 1920 (MLflow `volleystat_ball_detector_1920`); it only runs at 1920. The 1280 model stays as `volleystat_ball_detector` / `models/best_fullframe`; use it with `--ball-imgsz 1280`. Measured 0.967 vs 0.944 after DP on the `6_tiPcRYGcBtc` window and 0.881 vs 0.843 raw on 369 contacts, 1.7x ball cost. Weights and imgsz are both in the ball cache key, so old ball caches are not reused. Registry downloads are cached per MLflow name, and a registry file other than the pin is flagged `sha256_mismatch`.
- Static-source suppression runs before global solver and again on selected trajectory where needed.
- Global DP chooses candidate identity. Smoother refines coordinate noise; it must not smooth across contacts, gaps, restarts, or impossible jumps.
- Default smoothing is `moderate` (7 frames).
- Reconstruction default is `off`.
- `compare` stores conservative/contact-aware variants and uses conservative output.
- Reconstruction must not affect FSM unless `--reconstruction-affects-fsm`; then only accepted high-confidence points enter clean replay.
- Original observed values and derived values must remain recoverable in annotations.
- `ball_size` (longer side of the selected detector box, px) exists only on observed detections. Interpolated, bridged and reconstructed points never carry one; it is a measurement, not a label.
- FSM/events, viewer, Parquet features, and analytics must consume same final accepted trajectory for a run.
- After clean FSM replay, recompute `pose_f` and event-model predictions from final exported ball coordinates and saved poses with fresh temporal history. Feature reconstruction use does not change `--reconstruction-affects-fsm`. Normalized clip features must preserve missing/off-camera observations and reset velocity at gaps/new tracks; legacy pixel compatibility remains unchanged. Existing feature files/checkpoints require explicit versioned refresh, never implicit rewriting.

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
- Before clean FSM replay, poses beyond the sideline tolerance need a nearby court observation on the same continuous tracklet, from past or future frames.
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

Verified active model dataset paths:

```text
data/event_clips_three_full_gold_causal_120f_v1/manifest.json
data/event_clips_three_full_gold_causal_120f_v1/features/
```

Older 430-clip gold data remains at `data/event_clips_4k_gold_v1`. Full-video human annotation sources retain authority over derived model/FSM fields.

Feature schema: `resolution_normalized_v2`. Event features contain 10 trajectory + 8 pose channels. Confirm manifest and feature sidecar before training.

Current training command:

```powershell
python -m action_recognition.training.train_event_sequence `
  --manifest data\event_clips_three_full_gold_causal_120f_v1\manifest.json `
  --features-dir data\event_clips_three_full_gold_causal_120f_v1\features `
  --model multiscale `
  --sequence-len 120 --short-sequence-len 60 `
  --val-stems Khe4P8QcJkg_5_4k `
  --checkpoint-dir action_recognition\models\experiments_4k_gold_transformer `
  --epochs 50 --seed 42
```

Run cross-video model selection before `--final-fit`. Do not resurrect metrics or checkpoint claims from merged_v4/merged_v5 experiments as current results.

## Human annotation

Contact author and movement review share the same confirmed roster in `action_recognition.annotator.contact_author_annotator`. See `docs/contact_author_annotation.md`.

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

`configs/model_registry.json` lists every production model:

- ball_detector, ball_detector_tiled, pose_estimator, player_reid, event_sequence, contact_spotter, contact_author, net_sam;
- the optional legacy `spotting` entry.

Each entry pins `sha256` of its fallback file, and `files` pins companions (OSNet code/licence, spotter sidecar, contact-author deployment.json, event fold checkpoints).

MLflow alias `production` is authoritative when the registry is reachable; otherwise the local fallback is used. Every run records the resolved source, version/path and sha256, and a fallback with other bytes than its pin gets `sha256_mismatch: true`.

```powershell
python scripts/models/run_mlflow_server.py          # mlflow.db + mlartifacts/ in repo root
python scripts/models/register_models.py --all      # backs up main + companion files
python scripts/models/verify_models.py [--restore]  # pins/backup check; restores missing files only
```

Never register bytes that differ from a pin; change `fallback` and `sha256` together when promoting. The viewer resolves `contact_spotter` from the config. Explicit model CLI paths override the registry.

Incident 2026-11-08: `git worktree remove --force` followed Windows junctions and emptied `models/` and `experiments_three_full_gold_multiscale_v2`. Everything was restored and verified byte-identical: by a cache-only replay of a pre-incident run, and the event checkpoints from `shutil.copy2` copies in `results/event_team_pose_offline_20261027_v3/historical_checkpoints`.

Before removing a worktree, `rmdir` its junctions, and never `--force` while a junction exists. Run `verify_models.py` after any model-file change.

## Viewer and artifacts

For `game.webm --output results/game`:

```text
results/game/game_stats.json
results/game/game_annotations.json
results/game/game_analyzed.mp4
```

Viewer reads actual annotations, not redrawn guesses. It supports ball/trail, reconstruction provenance, players, net/court, FSM/events, attack heatmap, match analytics, upload jobs, persisted result restore, and court editing.

Viewer uploads select the ball batch automatically. Prefer compatible cached detections before sizing from available GPU memory. Reduce the batch on CUDA memory exhaustion. Explicit CLI batches remain supported.

New analysis accepts a dropped video file: `/api/upload_video` copies it to
`data/uploads/` (same name + size reuses the copy) because browsers never expose
a dropped file's path. It can preview a source frame and adjust saved court
lines/corners and net tape before processing; a magnifier shows a native-res
crop (`/api/analysis_calibration/frame`) around the active handle. Confirmed geometry is stored as a job-local
`court_calibration.json`; net detection is disabled for that job and its floor
divider follows the edited court centre line. Shared calibrations and saved
camera setups are not modified.

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

- Every video hypothesis must first reuse existing annotations and model-output
  caches. For pipeline replay use `--require-model-cache`; for ID-only work use
  `scripts/apply_player_identity.py apply --require-model-cache` or the frozen
  roster experiment. Never rerun compatible YOLO/OSNet/SAM inference merely to
  change tracking, geometry, smoothing, seeds, roster or decoding logic.
- New source/model inputs are captured automatically by the main workflow in
  `data/inference_cache`; only missing stages run. If a hypothesis genuinely
  needs missing model outputs, identify that stage before a capture. Preserve
  existing caches and never fabricate raw detections from derived annotations.
- Default output is JSON + original-video viewer. Rendering is separate and
  explicit (`--render-video`); do not render a full MP4 for each hypothesis.
- See `docs/model_output_cache.md`. Cache provenance and human-label provenance
  remain separate. Reconstructed ball data is never detector ground truth.
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
- This file is loaded into every agent session: keep it to rules, entry points and
  invariants. Put dated experiment notes, measurements and artifact paths in a
  topic doc or `docs/agent_history.md`; add at most a one-line pointer here.

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

## Production subsystems (summary)

Default-on stages after the core ball/pose/FSM pipeline. Measurements, artifact
paths and experiment history: `docs/agent_history.md` and the linked docs.

- **Model-output cache** (always on): `src/models/inference_cache.py` binds raw
  ball proposals, raw YOLO poses (before court filtering), OSNet embeddings
  (keyed by frame + pixel crop), event logits and contact probabilities to video
  bytes, weights, settings and runtime. ByteTrack replays on CPU from raw poses.
  `--require-model-cache` forbids inference. New weights invalidate only their
  stage; post-processing changes never invalidate raw outputs.
  `scripts/model_cache.py inspect|import-reid`. See `docs/model_output_cache.md`.
- **Contact spotter**: `--contact-spotter <ckpt>`; viewer uploads use
  `models/contact_spotter/spotter_20261028_v1.pt` when present. A
  `contact_spotter_v1` checkpoint (trainer `scripts/models/train_contact_spotter.py`,
  loader `action_recognition/rally_spotting/contact_spotter.py`) must ship its
  members, normalisation and decode constraints. It replaces `all_events`/`ev`,
  `rallies`, `service_attempts`, `rally_lifecycle` (`spotter_rallies_v1`,
  serve-anchored; recovered serves stay `inferred`); FSM versions stay in the
  `*_fsm` keys; `match_analytics` and `attack_heatmap` stay FSM-derived. The
  spotter writes `team: null`. A final-fit checkpoint has no held-out score:
  quote leave-one-video-out experiments. `scripts/apply_spotter_rallies.py`
  rebuilds a saved result. See `docs/production_contact_spotter.md`.
- **Player identity** (default on, side b, initial P1-P6): method
  `osnet_tracklet_roster_v5` (`src/tracking/player_roster_discovery.py`,
  `player_identity_workflow.py`, `src/models/player_reid_encoder.py`). The
  roster is the six people present longest among appearance groups of clean
  tracklets from the whole video; co-visible tracklets never merge; boxes are
  matched to those galleries per frame, and known other people stay unknown.
  `--player-identity-method seed-frame` is the older
  `osnet_frozen_gallery_kalman_roster_v4` (`player_reid.py`, `player_motion.py`,
  `player_identity_consistency.py`); it also runs for short clips and jersey
  constraints. Score identity changes with
  `scripts/evaluation/evaluate_player_identity_movement.py` against a
  `human_player_movement_v1` export before changing defaults.
  `poses[].id` stays raw; `player_id` and
  `identity` are separate; unknown is null / `P?`, never a recycled ID.
  Substitutes come only from explicit `player_identity_roster_v2` anchors
  (`scripts/apply_player_identity.py add-player`); returning players keep their
  IDs. Install assets once with `scripts/models/install_player_reid.py`; no
  runtime downloads. `scripts/apply_player_identity.py export-seed|apply`
  writes fresh paired results. Opt-in jersey OCR (pinned TrOCR,
  `scripts/models/install_jersey_ocr.py`; reads from
  `scripts/evaluation/read_jersey_numbers.py`; `apply --jersey-reads`) adds
  number constraints to `associate()`; it is not in default analysis. See
  `docs/player_identity_workflow.md`.
- **Contact author** (default on): `src/tracking/contact_author_runtime.py` runs
  after identity and final events with logical model `contact_author` (all10
  `expanded_soft_hgb`, SHA256 `4fd55ffaea4a7280a1a87bb358c303fcfeb7a364bed92324389fe018dbf96091`,
  installed by `scripts/models/promote_contact_author.py install`).
  `--no-contact-author` (CLI) / `--legacy` (replay) restore the old heuristic.
  `scripts/apply_player_contacts.py` replays cache-only; `--capture-missing-reid`
  permits missing OSNet crops only. Human attribution reviews append to
  `<stem>_player_contact_reviews.jsonl`, bound to SHA256 of both analysis
  files. No measured attribution score exists for automatic rosters. See
  `docs/contact_author_production.md`, `docs/player_contacts.md`.
- **Contact side** (default on, `--no-contact-side`): `src/tracking/contact_side.py`
  runs after the spotter on the final `all_events`, with no inference. Ball box
  size against the net-plane ball size (tape height over the floor line), summed
  over a possession; weak possessions alternate inside a rally. Adds `side`
  (`a` far / `b` near), `side_score`, `side_source`, `side_confident`,
  `side_ball_px` to events and a `contact_side_v1` report to both exports.
  `team` is never written. Abstains without net geometry, for a camera that is
  not end-on, and when box sizes disagree with the calibration. With contact
  authors it records `author_check` disagreements as review hints only. See
  `docs/team_attribution.md`.
- **Viewer player views**: **Player contacts** and **Player statistics** derive
  per-roster counts/outcomes from saved contacts, reviews, FSM attempts and the
  point ledger without inference; unknowns are excluded from totals. With both
  teams selected, unknown-team contacts are listed as `—`; selecting one team
  excludes them. Never infer a team to fill the list.
- **Pose input and net floor**: default `--pose-imgsz 2560`.
  `GameAnalyzer._validated_net_floor` falls back to the homography world-y=0
  line (`calibrated_court_midline_fallback`) when the detected divider fails the
  post-spacing gate; source calibration stays unchanged. The user draws the
  divider by hand when needed (viewer `manual_court_midline`); do not make the
  court-midline divider the default without asking.
- **Optional SAM pose recovery**: `--person-masks <masks.jsonl>` enables
  `src/detection/sam_pose_recovery.py`. Keypoints come only from YOLO crops;
  output `source=sam_guided_yolo_pose` with negative mask-track IDs (separate
  namespace, not ByteTrack continuity). See `docs/sam_people_masks.md`.
- **Calibration order**: `scripts/interactive_court_calibration.py` writes
  corners in a different order than `GameAnalyzer._load_calibration` expects.
  Always run `scripts/normalize_calibration.py` and check
  `world y=-9 < y=0 < y=+9` in image rows. The 4K gold videos use
  `configs/court_setups.json:setup_1` via
  `scripts/evaluation/build_setup_calibration.py`, never root
  `court_calibration.json`.
- **Contact review of unannotated video**:
  `action_recognition/annotator/contact_review_annotator.py`; records start
  `label_source: "model"`, `verified: false`; nothing is promoted into a dataset
  automatically. See `docs/contact_review_workflow.md`.

## Settled findings (do not re-propose)

- Team side from 2D court geometry tops out at ~0.715 against 406 human labels
  (airborne ball, net congestion). Do not derive side from court geometry again.
  See `docs/team_attribution.md`.
- Ball box size is the one depth cue that works for side (see
  `docs/team_attribution.md`), but on the three development videos the contact
  author's own side decision was right more often. Do not overrule the author
  with the ball-size side, and do not write it into `team`, without a held-out
  comparison on sources the author was not trained on.
- Spotter typing (0.92 accuracy) already beats detection (0.86 F1). Do not
  propose a two-stage detector + clip classifier; the cost is false positives.
  See `docs/contact_detection_vs_typing.md`.
- j3's weak precision is warm-up labelled `non_game`, not weak detection. Do not
  build a phase gate for warm-up; use `--match-config` or more `non_game` data.
- Four-class pose effect on the spotter is zero within seed noise (gain only in
  `block`). Measure pose effects with `--zero-pose-channels` on the same
  analyses, never by comparing two different analyses.
- Fair FSM baseline is 0.1561 core macro F1 (`results/gold_pose_fsm_20261028_v1`);
  never quote the pose-less 0.034 floor as FSM quality.
- Reviewed-queue sparse contact-only targets over-fire and were not promoted;
  review extra events for true negatives before retrying.
- Archived, not usable: STES RGB spotting, offline clip-classifier team/pose
  features, contact-candidate pooling (branch `archive/research-snapshot-2026-10-30`).
- Contact-author and identity numbers come from three development videos and
  sparse development checks: no untouched final test. Never quote them as
  full-match identity, per-person or unseen-camera accuracy.
- Completed poses, reconstructed ball points, latent MIL actors and contact-lab
  peaks are derived data: never detector labels, contact evidence or a
  replacement for `all_events`.

## Research and experiment history

Dated experiment notes (annotation projects, benchmarks, measured numbers,
artifact paths, ports and their caveats) live verbatim in
`docs/agent_history.md`. Read the matching section before touching:
contact-author annotation/benchmark/learning/inference, volleyball contact lab,
P4 case corrections, per-player contacts, pose completion, SAM people masks,
dense contact spotting, expanded contact training, player identity pilot/ReID/
roster, or the model-output cache. Topic docs in `docs/` hold full detail.

## Repository map

```text
analyze_video.py
viewer.py
configs/
src/detection/
src/pipeline/            game_analyzer.py + analyzer/ mixins
src/tracking/
src/viewer_app/          viewer.py helpers
viewer/templates/ viewer/static/
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
