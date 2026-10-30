# Offline event-model experiment

## Trajectory-change experiment

`scripts/evaluation/run_trajectory_contact_experiment.py --help` runs a matched
baseline versus an offline trajectory block on seeds 42/43/44 and all three
whole-video folds (18 fits). It retains the original normalized 18-channel,
causal 120/60 input and multiscale architecture, trained from scratch in both
conditions. The future context is supplied only through a separate zero-initialized
linear logit correction (34 inputs, no bias), not through rewritten old features.
No historical checkpoint is fine-tuned or promoted.

Schema `contact_trajectory_before_after_v1` has 17 values plus availability:
pre/post image velocities and speeds, their changes, signed sine/cosine of the
turn, line-fit residuals, ball coverage and reconstructed fractions. Normalized
positions use frame width/height and velocities use seconds. Side fits use
[-12,-2] and [+2,+12] source frames. A fit requires its complete contiguous
window; changes additionally require uninterrupted [-12,+12] context without
track restarts. Missing/off-camera observations remain unavailable. Derived
ball coordinates retain source/confidence and reconstructed fractions; these
are not detector ground truth. There are no pose, team, FSM or human class
inputs to the new block; human contact frames only anchor the candidate window.

The runner verifies the active 414-clip manifest against frozen v3 preflight,
records hashes for baseline files and frozen ball annotations, writes all new
features under its fresh output directory and refuses existing output paths.
Train-only normalization, seed, class weights, optimizer and early stopping are
matched. Checkpoints use a separate experimental model type and are not
compatible with the production loader. Reports include per-seed pooled scores,
per-source metrics and predictions. Best-epoch scores remain model-selection
estimates, not an untouched test. Tests: `tests/test_contact_trajectory.py`.

Completed run: `results/trajectory_contact_20261027_v1/report.json` (18 fits,
414 preserved clips / 410 core examples). Pooled macro F1 baseline versus
trajectory block: seed 42 0.924167 -> 0.925161; seed 43 0.902549 -> 0.908988;
seed 44 0.917486 -> 0.918584. Mean across seeds: 0.914734 -> 0.917578
(+0.284 percentage points). Mean primary Khe validation F1: 0.877485 ->
0.882767; individual video/seed pairs can regress. Seed-42 controls exactly
reproduce all three historical validation scores. Improvement is small relative
to seed variability and is not proof of generalization. No production promotion.

## Contact-candidate follow-up (2026-10-27)

`rebuild_contact_candidates.py` creates schema
`resolution_normalized_team_pose_contact_candidate_v2` from completed v3
skeleton caches without rerunning YOLO. The first 18 values/bits are unchanged
for a matched control. Per-side counts remain counts; the other 11 channels
all describe one candidate, selected by minimum image-pixel distance to an
available anatomical keypoint or forearm segment. This is a proposal, not
verified contact identity. With no ball or usable joints the candidate channels
are unavailable. Track IDs are exported per frame in `candidate_track_ids`.

`--contact-center-pool` makes the multiscale branches classify the encoded
central frame (long index 60, short index 30). Self-attention still sees the
entire offline window, including future frames. Default false and explicit
checkpoint metadata preserve existing model behavior.

`run_contact_candidate_experiment.py --help` documents the follow-up runner.
It refuses existing output paths and performs 12 fits: candidate features with
original attention pooling (3), original trajectory/team features with centre
pooling (6), and candidate features with centre pooling (3). Existing original
attention results supply the matched baseline. Source splits, labels, seed 42,
optimizer and epoch budget are retained. It snapshots code, records parent
feature/skeleton hashes, and writes a combined report. No SAM recall correction
or legacy nearest-player-block change is included: these remain separate
experiments. A `cancel.request` file in the new result directory stops its child.

Completed follow-up: dataset
`data/event_clips_three_gold_contact_candidate_20261027_v1`, combined report
`results/event_contact_candidate_20261027_v1/report.json`. All 12 fits completed.
Pooled out-of-fold macro F1 for team features: original 0.873348,
candidate-only 0.854796, centre-pooling-only 0.869618, both 0.851974 (rounded).
The trajectory control also decreased with centre pooling: 0.895060 to 0.873159.
These interventions did not improve this seed's model-selection performance;
they do not establish a unique cause of the original regression. No model was
promoted. All 414 labels/IDs/frame ranges and original first-18 feature values
and availability bits were verified unchanged against v3.

The active event checkpoint was trained on
`data/event_clips_three_full_gold_causal_120f_v1`: 414 human clips from three
completed full-video annotations, including 410 four-class examples and four
non-core clips. This is verified by the old MLflow `manifest`/`features_dir`
parameters and exact reevaluation of all three saved validation checkpoints.
The older `data/event_clips_4k_gold_v1` (430 clips) remains unchanged, but is not
the dataset used by the current checkpoint.

The user explicitly chose offline analysis: future images are allowed. The new
experiment uses absolute frames `[contact_frame-60, contact_frame+59]` for every
class and source. The contact is at index 60 of the 120-frame input. The short
60-frame branch sees indices 30:90, covering both sides of the contact. Missing
video-edge context stays at its correct position with availability=false; it is
not shifted, copied or interpolated. Human contact times anchor training clips;
no human class/team/contact/FSM context is included as a numeric input feature.
An offline classifier can apply the same window to any candidate centre frame.

## New data and input contract

Current run:

- Dataset: `data/event_clips_three_gold_team_pose_offline_20261027_v3`.
- Queue and logs: `results/event_team_pose_offline_20261027_v3`.
- Expected final report: `training/report.json` and `training/comparison.html`
  below that result directory, generated only after all nine runs complete.
- Source labels, old datasets, old checkpoints and production registry aliases
  are preserved. No new checkpoint is promoted automatically.

Schema: `resolution_normalized_team_pose_offline_v1`, **42 values plus 42
availability bits = 84 model inputs**. The 42 values are:

- Existing 10 trajectory and eight nearest-player/pose summaries.
- For each side, 12 features: player count, ball-to-body/wrist/elbow/forearm/
  head/knee/ankle distance, wrist height above net, arm raise, jump and torso
  confidence. Distances use normalized image geometry or body height as stated
  in the feature code; invalid joints never become observed zero distance.

All values are model-derived, not human keypoint labels. Normalization uses
only valid observations from the training sources. Missing values become
standardized zero and retain availability bits. Full measured COCO skeletons,
confidence, boxes, side and track IDs are also retained in `poses/*.jsonl` for
review and future experiments. IDs reset across disconnected extraction segments;
they are not long-term person identity labels.

Fresh pose extraction uses YOLO11m-pose at 2560 with ByteTrack and the production
court gate. It reuses the existing matching-camera named setup `setup_1`, mapped
to native 4K, and an explicitly inferred calibrated court midline. This geometry
is model/input calibration, not event ground truth. Source video/ball/model
hashes are frozen in `preflight.json`. Ball observations/reconstructions and
out-of-camera states come from the same frozen conservative analyses for every
new condition. They are not exported as detector ground truth. Initial training
isolates team/pose features; it does not rerun SAM segmentation on these three
videos or reuse masks from the unrelated hy2S test video.

The original manifest is copied to `parent_manifest.json`. The new manifest
retains human labels/annotation IDs but records its new feature-window bounds,
original bounds, feature paths and offline policy. The original manifest file is
never rewritten. Non-core clips are still generated and retained, then excluded
from the four-class classifier using the existing taxonomy.

## Comparison protocol

The primary holdout stays `Khe4P8QcJkg_5_4k`: 248 training / 162 validation core
clips. Additionally perform leave-one-complete-source-video-out checks for the
other two sources; adjacent/overlapping clips never cross a fold boundary.

All new conditions share model capacity, seeds, clip centres, future context,
optimizer, schedules and folds:

1. `trajectory`: nine ball/time channels; pose-derived player distance and all
   pose/team channels disabled.
2. `pose`: also player distance and the existing eight pose summaries.
3. `team_pose`: all 42 values and availability bits, including both sides.

Disabled values/bits are zero. Each model still has 84 input channels, so a
capacity change cannot explain the difference between the three new conditions.
Use multiscale transformer, 120/60 centred branches, seed 42, up to 50 epochs,
patience 15, batch 32, AdamW lr .001 / weight decay .0001, cosine scheduling and
label smoothing .05. Normalization/class weights are fitted separately on each
training fold. Report macro F1, per-class F1, confusion matrices, accuracy,
pooled out-of-fold predictions and worst-source performance. These are
model-selection estimates on three related-camera videos and one seed, not an
independent final test or proof of general improvement.

Historical checkpoints keep their original **past-only** window and original
features, and are reevaluated on the exact old source/labels. Their reproduced
macro F1 values are Khe 0.8783258, j0 0.9665106, j3 0.9387117. A separate legacy
primary-fold retrain also reproduced 0.8783258 at epoch 15. These historical
numbers involve a different context/representation. Attribute the benefit of
team features using the matched new-condition comparisons, not the full
historical-to-new delta.

## Run a new version

Use the project environment with CUDA PyTorch, Ultralytics, pandas and NumPy.
If the venv launcher is broken, use the standalone Python 3.11 plus project
`PYTHONPATH`, as in AGENTS.md. Choose fresh output names:

```powershell
python scripts\evaluation\run_offline_event_experiment.py `
  --dataset data\event_clips_three_gold_team_pose_offline_manual_v1 `
  --output results\event_team_pose_offline_manual_v1
```

The queue snapshots code/old checkpoints, extracts all required frames with
90-frame causal pose-history warmup, creates new per-clip NPZs, reevaluates old
checkpoints and trains nine models. Completed extraction segments can be reused
with the extraction worker's `--resume`; incompatible source/weights/settings
are refused. Failed/partial runs cannot be trained as completed datasets.

`queue_status.json` identifies the active stage; `extracting.log` and
`training.log` contain details. Dataset `progress.json` reports frame progress.
Creating `cancel.request` in a queue result directory asks its supervisor to stop
its own child process. Interruption does not modify original datasets or labels.

The superseded v1/v2 draft directories remain incomplete for traceability and
must not be treated as training-ready datasets. Only the v3 run uses both the
user-authorized offline context and the verified active training-label source.
