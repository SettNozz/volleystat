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
Full-match production identity accuracy remains unverified. The experimental
[human contact-author benchmark](docs/contact_author_benchmark.md) now measures
person attribution on reviewed contacts from three complete videos: 382 strict
examples, with whole-video held-out training. The OSNet/temporal follow-up's
best fixed `class_fusion` gets 79.35% exact near-person recall and 78.92%
assignment precision; nested algorithm selection gets 75.54% near-person recall.
The 80% person target is not reached. These are known-contact development scores, not
end-to-end contact detection or a production promotion. A read-only error
viewer compares all methods with the human photo roster.
The [paired comparison with Claude's person ranker](docs/contact_author_comparison.md)
uses identical 382 strict contacts: fixed HGB gets 85.87% exact near-person
recall versus `class_fusion` 79.35%; corrected nested selection gets 82.07%
versus 75.54%. These remain three-video development estimates.
The [unlabelled-video author workflow](docs/contact_author_inference.md) applies
three frozen `class_fusion` fold members to an existing saved analysis, with no
training or detector replay. Its read-only viewer shows contact predictions,
original skeleton evidence and the saved automatic roster; accuracy stays
unmeasured until this source receives human author labels.
The [human contact-author annotator](docs/contact_author_annotation.md) collects
person-level reference labels: video on the left, confirmed player photo cards
on the right, plus other-team/unsure/not-contact decisions. It reuses the active
414-clip human event dataset and audited contact times (403 review items after
duplicate grouping and known-error exclusion). Substitutes get new IDs;
progress, revisions and JSON export remain separate from model predictions.
Player cards can be removed with an explicit confirmation and restored. Assigned
contacts return to unlabelled only after confirmation; deleted IDs are reserved.
Save requests have a timeout and reconcile saved state before another decision.
The **Тип контакту** selector separately corrects an event type with autosave
and a reset to the original value. Human type reviews enter the JSON export
with their original predictions; author labels and timing remain independent.
New-source author annotation can reuse existing spotter results with
`scripts/evaluation/prepare_contact_author_expansion.py`. It excludes sources
in the active event-training manifest and completed author snapshot, creates a
fresh empty project, retains model time/type provenance and requires photo-roster
confirmation. Current expansion: seven videos, 759 contact proposals, port 5070.
See [new-source annotation](docs/contact_author_annotation.md#new-sources-outside-the-existing-datasets).
Open **Player statistics** from a match to see each tracked player's attack
attempts, inferred point-winning attacks, and serve-reception quality.
The page uses saved analyses and contact-author corrections. Attack errors
need explicit fault evidence. Older results may show unresolved outcomes.

The viewer's **Player contacts** panel selects a roster ID, highlights only
that skeleton, and lists its reception, dig/pass/set, attack, block and serve
events with click-to-seek. Unverified contact proposals use saved ball/pose
evidence from both sides; ambiguous contacts stay unresolved. Correct an
attribution in the dropdown; corrections are separate, source-bound records.
No video/model rerun is needed. See [per-player contacts](docs/player_contacts.md).
An isolated [contact-author learning experiment](docs/contact_author_learning.md)
compares geometric rules, nine team-side classifiers and weak actor ranking on
frozen data. Review results expose an approach selector, paused contact frames
and uncertain player suggestions. Team-side validation is not individual-author
accuracy; nothing is promoted into production or source annotations.
The [volleyball contact lab](docs/volleyball_contact_lab.md) additionally compares
independent flight/limb detectors, bounded retiming and learned team sequences.
Its read-only viewer exposes new untyped peaks and author changes on the same
frozen JIZ3 video, with whole-video held-out timing/team comparisons.
The [P4 case corrections](docs/P4_contact_case_corrections.md) use observed arm
actions, ground/landing context and the user-confirmed P4 reference to reject
inactive defenders and separate a confused identity. Five source-bound review
buttons compare the exact examples; these development checks are not accuracy.
For short skeleton dropouts, the experimental
[offline pose completion](docs/pose_completion.md) uses the same person's past
and future observed joints. Hidden joints appear as cyan dashed estimates,
with an independent viewer toggle. Detector poses and contact counts remain
unchanged until the estimates have been independently validated.
Viewer shows contacts with unknown team as `—` when both teams are selected;
selecting one team includes only contacts attributed to that team.

The [frozen OSNet follow-up](docs/player_reid_experiment.md) compares learned
appearance and explicit lost-ID recovery on those same detections, with the
existing human checks and a separate simulated detection blackout.

Event predictions suppress serve during an active rally and outside an FSM
serving episode. A restart within 3 seconds produces a warning by default:
FSM end times are estimates, and this heuristic has not been validated against
human boundaries. Optional `enforce_restart_pause=True` restores strict pause
enforcement, including blocking an early episode until a later restart.
Missing balls do not end rallies. Raw predictions, suppression and warning
reasons are retained in `event_model_decision`; configure the pause through
`StreamingEventPredictor(..., serve_restart_seconds=3.0)`.

Offline volleyball-video analytics for 4K fixed-camera footage. Pipeline detects and reconstructs ball trajectory, tracks player poses, calibrates court and net geometry, segments rallies, derives match analytics, predicts contact events, and exports viewer-ready annotations.

Current production path is `analyze_video.py` -> `GameAnalyzer` -> JSON/video artifacts -> `viewer.py`.

## Current capabilities

| Stage | Current implementation | Main output |
|---|---|---|
| Ball detection | YOLO primary detector, optional tiled ensemble, temporal gate and global trajectory DP | Ball candidates and selected point per frame |
| Ball cleanup | Static-source suppression and contact-preserving quadratic smoothing | Stable trajectory without parked false positives |
| Gap reconstruction | Conservative or contact-aware offline reconstruction | Inferred 2D coordinates with provenance/confidence |
| Player tracking | YOLO11m-pose + ByteTrack + calibrated court gate | Player boxes, 17 keypoints, track/team IDs |
| Player identity | Offline frozen OSNet galleries + Kalman motion + lost-ID recovery | Roster IDs, confirmed substitutes and provenance; unknowns preserved |
| Model output cache | Persistent source/model/input fingerprints; raw detections, poses, embeddings and logits | Replay hypotheses without repeating compatible neural inference |
| Court and net | Perspective calibration, SAM net tape, optional SAM3 posts, Hough fallback | Court homography, tape band, net-floor divider |
| Game logic | Integrated volleyball FSM + clean trajectory replay | Rallies, serves, contacts, attack map |
| Event model | Four-class causal sequence model | `serve`, `reception`, `pass`, `attack` |
| Rally spotting | Dilated TCN over continuous full-video features | Serve/reception/follow-up contacts and rally ends |
| Match analytics | Manual set/team mapping plus event-sourced aggregation | Sets, side changes, points and team statistics |
| Viewer | Flask app with upload, analysis, court editor and overlays | Interactive video and match dashboards |

Pipeline is offline-first; correctness takes priority over real-time speed.
Reference workstation: RTX 5070 Ti 16 GB (Blackwell, sm_120), i9-10900F,
32 GB RAM. Older cards (RTX 3080) still work.

## Requirements

- Windows or Linux
- Python 3.11
- NVIDIA GPU recommended for 4K processing. RTX 50xx (Blackwell, sm_120)
  needs PyTorch >= 2.7 built for CUDA 12.8; the default PyPI wheel on Windows
  is CPU-only. Tested: torch 2.7.1+cu128, torchvision 0.22.1+cu128.
- Local model weights or running MLflow model registry
- FFmpeg-compatible video input readable by OpenCV

```powershell
py -3.11 -m venv volley_env_311
.\volley_env_311\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA build first, so requirements.txt does not pull a CPU-only torch.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Viewer, annotators and Parquet feature files
pip install flask flask-cors pyarrow

# Pinned offline player-identity encoder (new directory, no model replacement)
python scripts/models/install_player_reid.py
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

# Hypothesis replay: fail on missing model outputs instead of running inference
python analyze_video.py --video game.webm --output results\game_hypothesis --require-model-cache

# Explicitly encode an annotated video
python analyze_video.py --video game.webm --output results\game_export --require-model-cache --render-video

# Raw tracking only, without the default offline player-identity pass
python analyze_video.py --video game.webm --output results\game_tracks --no-player-identity

# Manual set boundaries and stable team identities
python analyze_video.py --video game.webm --output results\game `
  --match-config configs\match_config.example.json
```

See every supported option:

```powershell
python analyze_video.py --help
```

GPU utilisation. Inference that the cache does not hold yet runs ahead of the
ordered analysis loop on background threads: one decoder thread, one ball
thread and two YOLO-pose threads (one model instance each). ByteTrack, ball
selection, FSM and all cache writes stay on the main thread in frame order.
`--no-model-prefetch` restores the serial loop for debugging.

- YOLO-pose runs through `src/detection/pose_gpu.py`: frames are decoded into
  pinned memory, uploaded once, letterboxed on the GPU and the network replays
  a CUDA graph (eager dispatch cost ~22 ms of GIL-holding Python per 2560
  frame). Boxes match the CPU letterbox path to a median 0.4 px (658/658
  detections matched on a 4K sample). This changed the raw-pose cache key:
  old pose caches recompute once; ball and OSNet caches are untouched.
- OSNet crop embeddings for every saved pose are computed during the
  detection loop while the frame is still in memory, so the identity pass (and
  the contact-author runtime) read them from the cache instead of decoding the
  video again. Same encoder, crop and key (max 2.4e-6 cosine distance against a
  fresh identity-pass encode).
- A stopped MLflow server is detected by a 0.5 s TCP probe instead of ~8 s of
  HTTP retries per logical model.
- `--ball-batch-size` stays 8: 16 was only 8% faster and the batch size is part
  of the ball cache key, so changing it would recompute every cached video.

Measured on the first 30 s (899 frames) of `j0gRhgFCgIs_4_4k`, empty cache,
RTX 5070 Ti, i9-10900F, MLflow server not running:

| Run | Detection loop (steady) | Identity pass | Wall time |
|---|---|---|---|
| Serial, OSNet batch 8 (before, 2026-11-06) | 6.4 fps | 97 s | 314 s |
| Prefetch, CPU letterbox (2026-11-06) | 15.0 fps | 34 s | 151 s |
| + GPU letterbox, pinned decode (2026-11-07) | 25.6 fps | 29 s | 119 s |
| + registry probe, in-loop OSNet, CUDA graph (2026-11-07) | 24 fps | 3 s | 61-64 s |
| Replay of that run, `--require-model-cache` | - | - | 14 s |

In-loop OSNet moves ~11 s of encoding into the loop (35.6 fps without identity,
which is the decoder's ceiling) but removes the 24+ s second decode. GPU
utilisation in the steady loop went 47% -> 67%. Repeated runs are not
bit-identical even serially (FP16 with `cudnn.benchmark`): two serial runs
shared 822 of 899 raw ball outputs exactly and pose boxes moved by up to
15 px, which is why replays should go through the cache.

What limits the loop now (measured 2026-11-07, same clip): GPU work per frame
is ~20 ms YOLO-pose at 2560, ~10 ms OSNet (0.84 ms per crop for the
original+flip pair, ~12 crops per frame) plus ball and event models, about
36 ms, so ~28 fps is the ceiling with identity on; the loop runs at 23-24 fps.
Without identity it reaches 35.6 fps, which is the VP9 software decoder.
Tried and not adopted:

| Idea | Result |
|---|---|
| Hardware decode through OpenCV (`CAP_PROP_HW_ACCELERATION`, D3D11VA/NVDEC) | Slower (12.9 vs 17.5 fps in the same test) and different pixels (75% differ, up to 11 levels: other NV12->BGR conversion). Ball/OSNet keys do not include the decoder, so it would silently mix values. |
| FFmpeg decoder threads (`OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;N`) | 34.2-35.1 fps for 1-16 threads; identical pixels, no gain. |
| OSNet PIL resize in 3 spawned processes | Bit-identical embeddings, loop 24 -> 19.2 fps (pickling crops costs more GIL than it saves). |
| CUDA graph for OSNet (bit-identical to eager) | Loop 23.1 fps, no gain: OSNet is GPU time, not dispatch. |
| OSNet `channels_last` / FP16 | -13% / -43% OSNet GPU time but 1.75e-5 / 3e-4 cosine change under the same cache key; would need a new OSNet key and recomputing cached embeddings. |

The next real gains need a decision: a TensorRT FP16 engine for YOLO-pose (new
pose key) or an FP16 OSNet with a new OSNet key (recomputes saved embeddings,
including the imported JIZ3 cache).

### 3. Open viewer

```powershell
python viewer.py
```

Open `http://localhost:5050`. Viewer can upload a source video, launch analysis, restore saved jobs, edit court geometry, and show ball trail, reconstructed gaps, players, net, FSM events, attacks and match statistics.

In **Video Analysis**, open **Set-to-attack trajectories** to find saved sets followed immediately by an attack in the same rally. The default filter shows pairs with both contacts attributed to the near side. Enable **Overlay all matching paths** to draw them together over the video, or select a row to jump to the set and highlight that path. The overlay uses active ball coordinates from the saved analysis. Dashed lines mark saved derived points, and gaps and track restarts break the line. The list marks partial and missing paths.

The overlay trims the incoming and outgoing legs when the ball changes motion near the saved contacts. The path color comes from the ball's horizontal position at the end of the pass arc, using three equal screen sections across the net. The dot marks that endpoint. Left is blue, center yellow, and right pink. Gray marks an unknown zone.

During a near-side attack, the top-right badge shows the approximate ball contact height above the floor in metres. The viewer projects the saved net height to the estimated depth of the contact and adds the signed offset of the ball above that reference. The attacker's lower ankle positions near the contact approximate the depth. Missing geometry, ball or usable pose evidence gives “Height unavailable.” The Events toggle controls the badge.

On **New analysis**, enter the video path and choose **Preview and edit court & net**. Start from the default calibration or a saved camera setup, then drag court lines, corners, or the gold net tape handles. Change the preview time to find a clearer frame without losing edits. **Analyze Video** uses the edited geometry for that job, with the court centre line as the floor divider. **Use automatic settings** restores the usual net detection workflow.

Viewer selects the ball detector batch automatically. It prefers compatible cached detections and otherwise uses available GPU memory, reducing the batch if CUDA runs out of memory.

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

Before clean FSM replay, the pipeline checks poses beyond the sideline tolerance against past and future court observations from the same continuous tracklet. After replay, it recomputes `pose_f` and event-model predictions from the final exported ball coordinates and saved player poses. Predictor and pose history start fresh for each replay. Features can use accepted reconstruction without changing the separate FSM reconstruction policy. Normalized clip features preserve missing/off-camera observations and reset velocity at gaps and new tracks, matching streaming inference; legacy pixel features retain their compatibility behavior. Existing annotations, feature files and checkpoints are not rewritten automatically. Inline video overlays remain provisional unless the final rendering pass is enabled.

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

Verified active checkpoint dataset:

```text
data/event_clips_three_full_gold_causal_120f_v1
```

The active checkpoint uses `data/event_clips_three_full_gold_causal_120f_v1`: 414 human-labelled clips (410 core examples) from three complete 4K videos. This was verified against its MLflow parameters. The older 430-clip `data/event_clips_4k_gold_v1` remains available unchanged. Sources:

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
  --manifest data\event_clips_three_full_gold_causal_120f_v1\manifest.json `
  --features-dir data\event_clips_three_full_gold_causal_120f_v1\features `
  --model multiscale `
  --sequence-len 120 --short-sequence-len 60 `
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
  pipeline/game_analyzer.py               end-to-end orchestration (frame loop, export)
  pipeline/analyzer/                      GameAnalyzer mixins: cache, calibration, models,
                                          detection, ball trajectory, events, identity, rendering
  tracking/                               trajectory, reconstruction, FSM, analytics
  models/model_registry.py                model resolution and provenance
  viewer_app/                             viewer.py helpers (library, media, court editor I/O)
viewer/
  templates/                              Flask page templates
  static/                                 page scripts and styles (viewer_page.js, viewer_*.js)
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


## Archived research

Finished experiments that were not promoted (offline team/pose and
trajectory-change clip classifiers, STES / RGB event spotting, contact
attribution variants) live on branch `archive/research-snapshot-2026-10-30`,
together with their docs and results summaries. Restore a file with
`git checkout archive/research-snapshot-2026-10-30 -- <path>`.

## Spatial contact review

Run `start_contact_review.cmd` and open http://127.0.0.1:5062.
[Ukrainian annotation guide](docs/contact_spatial_annotation_uk.md).
Reviews land in `data/contact_spatial_gold_v1/reviews.json`; the dense contact
spotter reads its verified contact frames from there, and
`action_recognition/rally_spotting/spatial_labels.py` canonicalizes them.

## Reviewing spotted contacts on new video

After running a video through `--contact-spotter`, review the predictions one
keystroke at a time:

```powershell
python -m action_recognition.annotator.contact_review_annotator --annotations-dir results\batch
# http://localhost:5059
```

Type with `1`-`6`, side with `A`/`B`, `X` rejects a false positive, `+` inserts
one the model missed. Records stay `label_source: "model"`, `verified: false`
until a keystroke makes them human, and the model's own prediction is kept
beside the correction. Calibrate a new camera first and run
`scripts/normalize_calibration.py` on the result. See
[contact review workflow](docs/contact_review_workflow.md).

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
fresh directory without rerunning detection. Viewer uploads run the shipped
checkpoint (`models/contact_spotter/spotter_20261028_v1.pt`) automatically and
fall back to the FSM when it is absent; each rally clip shows the rally's first
frame, decoded from the original video. The spotter cannot tell warm-up from play and does
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

## Expanded human contact experiments

**Default contact authors (2026-11-07):** expanded all-side temporal OSNet +
soft HGB. Install the frozen artifact with
`python scripts/models/promote_contact_author.py install`. New analyses apply it
after roster identity; saved-result replay defaults to the same model and a
strict cache check. `--no-contact-author` / replay `--legacy` retains geometry.
The viewer reads exported model predictions. Event models are unchanged.
See [installation, photo setup and cached replay](docs/contact_author_production.md).

The completed author-review projects can be frozen and evaluated with
`scripts/evaluation/train_expanded_contacts.py` in the isolated experiment
worktree. Whole-video folds compare old-three-source and expanded training for
named authors and contact typing. Unchanged model event types remain unverified;
only explicit type corrections augment the event dataset. Run artifacts retain
SQLite snapshots, source hashes, features, seeds, fold predictions, metrics and
model files. TcUru's confirmed five-person roster is an explicit exception to
the normal six-person court policy. See
[expanded training and history](docs/expanded_contact_training.md).
