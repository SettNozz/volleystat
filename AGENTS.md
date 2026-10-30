# Volleystat - Agent Context

Updated: 2026-10-31. Code, CLI `--help`, tests, and config files override prose when they disagree.

## Mission and priorities

Volleystat is an offline-first volleyball analytics pipeline for fixed-camera 4K video. Accuracy matters more than runtime; target workstation has RTX 3080. Current work stays in 2D image/court coordinates.

## Human contact-author annotation (2026-10-31)

- Prepare a fresh project with `scripts/evaluation/prepare_contact_author_annotation.py`
  from the active 414-clip human manifest, `contact_spatial_gold_v1/reviews.json`
  and existing pose-enabled analyses. Current queue: 403 items (seven duplicate
  clips collapsed, four audited wrong events excluded), across three full videos.
- UI: `python -m action_recognition.annotator.contact_author_annotator --project <dir>`
  (localhost port 5066). Core: `contact_author_dataset.py` in the same package.
  No inference, feature refresh, training or source annotation mutation.
- Six initial photo cards are model-derived proposals from a complete near-side
  cached frame, never verified persistent identities. The human confirms/repairs
  them before assigning contacts. Add substitutions explicitly as P7/P8; never
  recycle an ID or replace a confirmed person's original reference. Additional
  views require same-person confirmation. IDs are scoped to source video.
- SQLite autosaves decisions with optimistic revision checks and an append-only
  history. Immutable project/source fingerprints reject changed inputs. Author
  labels start empty; predictions are not prefilled. `unsure` stays unverified.
- Export `human_contact_author_dataset_v1` keeps queue/audit/provenance and all
  decisions; eligible contacts contain only verified player/other-team labels.
  Author verification does not verify type/timing. No automatic training promotion.
  These event-model training videos are author-development data, not an independent
  event-model test. Split future evaluations by complete source video.
- See `docs/contact_author_annotation.md`; tests: `test_contact_author_annotator.py`
  and `test_contact_author_annotation_ui.cjs`.

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
--contact-spotter <CHECKPOINT>
--no-player-identity
--player-identity-side a|b
--player-identity-seed <JSON>
--player-reid-assets <DIR>
--player-reid-device auto|cpu|cuda
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
     -> offline OSNet + Kalman roster identity (default near side, raw IDs preserved)
     -> final annotation-based video/frame rendering
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

## Per-player contacts (2026-10-30)

- `src/tracking/player_contacts.py` derives `player_contacts_v1` from final
  saved events, ball, poses and stable roster IDs. Proximity proposals are
  unverified; both sides and unidentified poses compete. No court projection
  determines the contact side; no missing/off-camera gap is bridged.
- Production export computes this report after identity. For an existing
  result, `scripts/apply_player_contacts.py --annotations ... --output <fresh>`
  writes paired JSON only; no video read, neural inference or MP4 render.
- Viewer **Player contacts** selects all roster IDs (including substitutes),
  highlights the selected skeleton and shows counts plus click-to-seek events.
  Reception, dig/pass/set, attack and block remain separately counted;
  original raw contact types are preserved. Unknowns are excluded from totals.
- Attribution corrections append to `<stem>_player_contact_reviews.jsonl`,
  bound to SHA256 of BOTH analysis files. They never rewrite source JSON,
  human gold, event types, identities or team analytics. Changed sources reject
  old reviews. Human attribution does not verify a model's contact/type.
- JIZ3 demo: `results/JIZ3exHe3hA_player_contacts_20261030_v1`, based on frozen
  `JIZ3exHe3hA_identity_roster_20261030_v4_regret010`. P7 is a substitute;
  initial proposals: 19 assigned, 29 other-side, 55 unresolved, 103 total.
  These are coverage counts, not attribution accuracy or all true contacts.
- See `docs/player_contacts.md`; tests `tests/test_player_contacts.py` and
  `tests/test_player_contacts_ui.cjs`.

## Offline pose completion pilot (2026-10-30)

- `src/tracking/pose_completion.py` produces `temporal_pose_completion_v1`
  overlays from saved same-roster-ID observations before and after a short
  occlusion. No neural inference, raw YOLO cache fabrication or training.
- `scripts/complete_player_poses.py --annotations ... --output <fresh>`
  adds separate `frames[].pose_estimates` and top-level `pose_completion`.
  It never changes observed `poses`, IDs, ball, features, events or analytics.
- Body/leg gaps are bounded to 0.30 s, elbow/wrist gaps to 0.15 s; endpoints
  need adjacent reliable support. Require overlap evidence, plausible motion,
  in-frame geometry and the six-person cap. Existing unidentified/rejected
  poses block duplicate reconstruction; raw track IDs never override ReID.
- Derived keypoints are `[x,y]` or null with per-joint source-frame evidence;
  no detector confidence, raw tracker ID or human verification is invented.
  Viewer shows cyan dashed estimates with an independent **Estimated joints**
  toggle, current-frame guard and existing roster selection.
- Estimates do not feed contact attribution, FSM, feature refresh or detector
  training. Future frames are explicit; this is offline reconstruction, not
  new image-based detection or accurate recovery of a hidden pass swing.
- JIZ3 artifact `results/JIZ3exHe3hA_pose_completion_20261030_v1` retains
  frozen roster-v4-regret010 inputs: 88 missing and 275 partial player-frames,
  1201 estimated joint-frames. These are coverage counts, not accuracy.
- `scripts/evaluation/evaluate_pose_completion.py` masks saved model poses
  in overlap intervals and compares bidirectional estimates, last-pose hold
  and forward velocity on matched accepted joints. Proxy reference is YOLO,
  not human keypoint gold. Report rejected-joint coverage, including the
  out-of-bound 9-frame negative control at 29.97 FPS.
- See `docs/pose_completion.md`; tests `test_pose_completion.py` and
  `test_pose_completion_ui.cjs`. Never enable estimated contact evidence merely
  because the skeleton looks continuous.

## Isolated contact-author learning (2026-10-30)

- User authorized CPU learning in an independent worktree/branch. The active
  experiment is `codex/contact-author-learning`, based on `202bd7a`; never
  switch or stage files in the shared checkout used by another process.
- `scripts/evaluation/run_contact_author_learning.py` creates a fresh manifest,
  frozen features and comparison for 397 human team labels (205 far, 192 near),
  three whole source videos, no person labels. Print/verify manifest, feature
  path, source hashes, provenance, schema, split and counts before fitting.
- `contact_author_side_frozen_v1` uses only saved ball/observed poses, normalized
  body/hand geometry and offline motion. Human ball xy, raw/player IDs, video
  names and absolute match times are excluded from numeric model inputs.
  Fold scalers fit training videos only. Audited event timing/type is given
  in the gold comparison; JIZ3 supplies model-predicted timing/type.
- Fixed logistic/ExtraTrees/HistGradient candidates compare geometry, typed and
  layout feature views. Layout is diagnostic, excluded from automatic choice
  because gold videos share a camera. No random clip split, early-stopping
  holdout, production checkpoint/registry update or image-model inference.
- Best review candidate: ExtraTrees typed, leave-one-video-out team accuracy
  0.84635 / macro F1 0.84596. Score >=0.65 or <=0.35 gives 0.92617 accuracy at
  0.75063 coverage. Scores are uncalibrated. These are selection estimates,
  not an untouched test set, final-fit score or individual-player accuracy.
- `run_contact_author_mil.py` trains a shared latent actor scorer from team
  bags only, seeds 42/43/44, fixed 140 epochs, CPU/two threads. Geometry MIL
  reaches 0.79093 team accuracy; typed MIL 0.77330, so neither is selected.
  Latent actors are never human person labels or detector ground truth.
- Experimental weights/data stay under the worktree's fresh `results/` dirs.
  Completed flat run: `contact_author_learning_20261030_v3`; MIL/review run:
  `contact_author_mil_20261030_v1`. No source annotations/events/IDs/ball/features,
  trained spotter/ReID or another worktree is rewritten.
- `<stem>_player_contact_experiment.json` binds every frozen proposal report to
  SHA256 of both byte-identical source JSON files. Viewer validates event
  indices/frames/types and rejects changed sources or fake human verification.
  Human attribution corrections remain separate and shared across approaches.
- Independent viewer port 5053 offers 11 comparison approaches. Chosen JIZ3
  result: 24 assigned, 21 other-side, 58 unresolved from the SAME 103 events.
  Uncertain suggestions are excluded from totals; human overrides suppress old
  conflicting suggestions. A paused amber box marks the proposed actor only
  at the exact evidence frame and never rewrites source pose identities.
- No block contacts have human side labels in this training set; do not quote
  the measured team accuracy as block or per-person performance. JIZ3 and its
  camera are unlabelled transfer; the user will visually validate it.
- See `docs/contact_author_learning.md`; tests `test_contact_author_learning.py`,
  `test_contact_author_experiments.py`, `test_contact_author_mil.py`, and
  `test_contact_author_ui.cjs`.

## Volleyball contact lab (2026-10-30)

- `scripts/evaluation/run_volleyball_contact_lab.py` compares piecewise 2D
  flight, moving-limb, joint and consensus detection, bounded retiming and
  forward/backward team attribution. Core: `src/tracking/volleyball_contact_lab.py`.
  Research adaptations and limits: `docs/volleyball_contact_lab.md`.
- Keep the independent `codex/contact-author-learning` worktree. Active reviewer
  output: `results/volleyball_contact_lab_20261030_v3/viewer_result`, port 5054.
  v1/v2 remain preserved; v3 reuses v2's hash-verified evidence/metrics and
  isolates sequence side changes from the optional joint actor-ranking ablation.
- Detection targets are 397 distinct audited timestamps, including ambiguous
  raw action types; team data are 397 conclusive records plus nine unsure.
  These sets differ. Derived context groups exact same-frame records with all
  IDs retained; unsure/uncertain timing resets context. Human sources stay intact.
- Fold thresholds and team transitions use only the other two whole videos.
  Cached OOF spotter control must share annotation/phase hashes. Human coverage
  and `non_game` labels are scoring masks only, never inference gates.
- Independent detectors lost: F1@5 flight .4643, limbs .3319, joint .4496,
  consensus .7473 vs spotter .8735. Bounded retiming tied .8735. This is an
  audited time-only protocol, not a new score for the production checkpoint.
- On the spotter's predicted times/types, correct team among 365 matched
  contacts improves 309/365 (.8466) to 341/365 (.9342); joint detection/team F1
  .7375 to .8138. Known-time/type side control is .8463 to .9320, not .9421
  from the superseded duplicate-context preliminary run. All are model-selection
  estimates; zero human person/contact-author labels and no block-team score.
- Use spotter timing + sequence side evidence for review. The selected variant
  preserves local actor ranking and source event times/types; joint ranking is
  a separate unvalidated-person diagnostic. Never claim P1–P7/JIZ3 accuracy.
- `/load?...&contact_lab=1` restores the read-only lab. A separate source-bound
  `*_contact_lab.json` may contain novel **untyped** contacts; do not replace
  `all_events`, rallies/FSM, source poses/IDs or original JSON with these peaks.
  No model registry promotion, image inference, MP4 render or dataset rewrite.
- Tests: `test_volleyball_contact_lab.py`, `test_volleyball_contact_lab_viewer.py`,
  `test_volleyball_contact_lab_ui.cjs`.

## P4 case corrections (2026-10-30)

- Human feedback is in `configs/contact_case_reviews/JIZ3exHe3hA_P4_20261030_v1.json`,
  bound to the exact analysis pair. Three author exclusions (384/1051/11223),
  one positive P4 identity/contact reference (4915), one identity exclusion
  (6376 raw 1262), and opponent-team confirmation at 11223. The latter does not
  name the opponent. The 6376 exclusion does not rule out P4 on another skeleton.
- Replay: `scripts/evaluation/refine_contact_actor_cases.py`; core:
  `src/tracking/contact_actor_context.py`; docs: `docs/P4_contact_case_corrections.md`.
  Current isolated reviewer output is `results/P4_action_context_20261030_v4`.
  Preserve previous runs; v1 failed the positive reception check; v2/v3 still
  showed an insufficiently supported replacement blocker at f384.
- Use same-arm hand contact, shoulder-relative arm motion, two-sided raw-track
  identity through jumps and observed standing/landing anchors. Never infer
  rotational eligibility or the ball's team from court/airborne-foot geometry.
  Backcourt attacks remain allowed. Ground depth is a soft positional cue.
- Only the positive f4915 identity reference enters appearance inference.
  Negative case answers are evaluation checks, not timestamp-specific vetoes.
  Reuse existing OSNet embeddings; validate cache hashes and every key/raw ID/bbox.
  No training, fresh image inference, gallery contamination or source rewrites.
- Source P4 on receiving track 1262 is rejected; blocking track 1267 is a derived
  P4 proposal. f384 author remains unresolved after visual inspection, not P1.
  f1051 proposes P1; f4915 retains P4; f11223 proposes an opponent. Do not claim
  user verified the new track-1267 or P1 labels, or the interpolated ball at4915.
- Read-only viewer context labels are `P4*` (derived correction), `ID?` (source
  P4 rejected). Rejected contact candidates must not draw as accepted authors.
  Five review buttons display exact user frames with evidence times separate.
- Five passing case conditions are development regression checks, not held-out
  person accuracy. Older team metrics do not validate this actor/identity path.
  Tests: `test_contact_actor_context.py`, `test_contact_actor_case_cache.py`, and
  extended `test_volleyball_contact_lab_ui.cjs`.

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

## Experimental people masks

- `scripts/evaluation/sam3_people_masks.py` runs offline SAM3 image segmentation
  in `.sam3_env` against saved final annotations; refuses existing output paths.
- `src/detection/person_masks.py` owns RLE, one-to-one pose matching, causal
  adjacent-frame IoU and `person_mask_features_v1` (8 values + 8 validity bits).
- `scripts/evaluation/compare_sam_yolo_samples.py` independently reruns raw
  YOLO-pose on five frames at 640/1280 for diagnosis, not source annotation edits.
- Unmatched court masks are review candidates, not proven YOLO misses or new
  player identities. Saved poses are already court-filtered. No keypoints/teams
  are fabricated; no ball, FSM, authoritative dataset or checkpoint is rewritten.
- Temporal matching is not native SAM3 video propagation or occlusion recovery.
- See `docs/sam_people_masks.md`; test `tests/test_person_masks.py`.

Player-recall correction: CLI/viewer default pose input is 2560 (CLI override
`--pose-imgsz`). `GameAnalyzer._validated_net_floor` uses the existing post-spacing
plausibility gate; an invalid detected divider falls back to the homography
world-y=0 line before shared runtime calibration is scaled/consumed. Its method
is `calibrated_court_midline_fallback`, not measured/SAM ground truth. Source
calibration remains unchanged; plausible detected floor geometry keeps priority.
Regression tests in `tests/test_player_pose.py` cover both teams surviving the
six-player cap, slope/scaling and preservation of valid geometry.

Optional `--person-masks <masks.jsonl>` enables `SamGuidedPoseRecovery` from
`src/detection/sam_pose_recovery.py`: same-video validated SAM proposals guide
independent YOLO crop inference, never primary ByteTrack callbacks/state.
Recovered keypoints come only from YOLO; duplicates and weak torso evidence are
rejected before the normal court gate. Export `source=sam_guided_yolo_pose`,
`mask_instance_id` and negative mask-track IDs; these are a separate association
namespace, not identity continuity with positive ByteTrack IDs. Record proposal
and accepted person-frame counts in annotation `player_detection` provenance.
SAM feature sidecars tied to old annotations must not silently be reused as
refreshed features for new recovered poses. Tests: `test_sam_pose_recovery.py`.

## Dense contact spotting and constrained decoding (2026-10-27)

`scripts/evaluation/run_dense_contact_spotting.py` is the first step of the plan
in `docs/event_detection_research_2026-10.md`. It trains a dense nine-class
phase/contact spotter over existing per-frame trajectory features and decodes it
with `action_recognition/rally_spotting/constrained_decode.py`, whose transition,
start/end and duration constraints are counted from the training fold only.
Targets come from `action_recognition/rally_spotting/dense_contacts.py`
(`dense_phase_contact_targets_v1`): human phases from the exported
`dense_targets.npz`, verified contact frames from `data/contact_spatial_gold_v1`.

Rules specific to this work:

- Contact frames come from the audited review, not the original annotation
  frame; they differ for 155 of 410 events. Never silently fall back.
- Uncertain timing, uncovered non-core touches and ambiguous raw types are
  masked, never trained as background. `wrong_event` is excluded without masking.
- Constraints must be estimated on the **unguarded** target copy; the guard ring
  separates contacts from phases and would forbid every contact.
- The FSM baseline in `results/gold_v1_ball_reconstruction_conservative` ran
  without poses (`pose_f` all zero, `p` empty) and with `net_y = 1080.0`. Its
  0.034 core macro F1 is a floor, never the shipped FSM's quality.
- Fair FSM baseline, measured 2026-10-28 on the pose-enabled rerun
  `results/gold_pose_fsm_20261028_v1`: **0.1561** core macro F1 at +/-5 frames
  (per fold 0.1912 / 0.1529 / 0.1241), recall 0.26-0.29 at precision 0.15-0.20.
  Quote this number for the FSM, not 0.034. It emits more events than there are
  contacts and still misses 70%; `serve` F1 is 0.00-0.03.
- A pose-enabled rerun also changes the ball trajectory (contact-preserving
  smoothing, median 1.2 px). Never attribute a gain to poses by comparing two
  different analyses; use the matched control `--zero-pose-channels`, which
  blanks the eight pose channels on the same analyses.
- Measured pose effect on the dense spotter: +0.012 core / +0.037 raw macro F1,
  concentrated almost entirely in `block` (raw F1 ~0.1 -> ~0.4). The peak-picking
  control's improvement came from the trajectory, not from poses.
- The counted grammar's margin shrinks as the per-frame model gets more precise
  (+0.068 on frozen features, +0.002 on the matched control, +0.037 with poses)
  and it lost the j3 fold of the control. Do not restate it as a uniform win.
- Calibration for these videos comes from `configs/court_setups.json:setup_1` via
  `scripts/evaluation/build_setup_calibration.py`. The root
  `court_calibration.json` was measured on another camera and must not be used
  for the 4K gold videos. The setup's four corners are the two end lines of the
  full-court world frame (x in [-4.5, 4.5], y in [-9, 9], y = 0 under the net),
  not a half court.
- Detection/typing split measured 2026-10-28 by
  `scripts/evaluation/score_contact_detection.py` (rescoring only, no training):
  detection F1 0.8605 type-agnostic (0.8978 excluding `non_game`), typing on
  located contacts 0.9208 four-class accuracy / 0.9157 macro F1. The halves
  multiply back to the 0.7897 end-to-end score. Typing is already at
  clip-classifier level, so do not propose a two-stage detector plus clip
  classifier; the error is in detection. Detection recall is 0.92-0.96, so the
  cost is false positives, not misses. See `docs/contact_detection_vs_typing.md`.
- j3 is not a weak fold. 36 of its 60 false positives are in the warm-up before
  its first human contact at frame 5012, which the human labelled `non_game`.
  j3 carries 4958 `non_game` frames against 904 and 613 for the others, so its
  training fold contains almost no warm-up. Excluding `non_game`, j3 detection
  goes 0.7612 -> 0.8696. Do not quote j3 as evidence of weak contact detection.
- Same split on the FSM: detection F1 0.4008 (recall ~0.5, precision ~0.33) and
  typing accuracy 0.4889 on what it locates, `serve` typing F1 0.00-0.12. Both
  halves fail independently; threshold tuning cannot fix it.
- Production path (2026-10-28): `scripts/models/train_contact_spotter.py`
  final-fits on all three gold videos and writes a `contact_spotter_v1`
  checkpoint carrying members, normalisation, **decode constraints** and
  provenance; `action_recognition/rally_spotting/contact_spotter.py` loads it;
  `analyze_video.py --contact-spotter <path>` runs it. Library dataset
  assembly lives in `action_recognition/rally_spotting/spotter_dataset.py`.
  See `docs/production_contact_spotter.md`.
- The constraints must ship inside the checkpoint. At inference there is no
  labelled fold to count them from, so a checkpoint without them cannot be
  decoded at all.
- A final-fit checkpoint has **no** held-out score. Quote the leave-one-video-out
  experiment for quality, never a number measured on its own training data.
  There is also nothing to select an epoch on: best epoch ranged 3-15 across
  folds and seeds, so the trainer uses a fixed budget plus a seed ensemble and
  records that as an engineering decision.
- The spotter replaces `all_events`/`ev`, and (2026-10-30) `rallies`,
  `service_attempts` and `rally_lifecycle` via
  `action_recognition/rally_spotting/rally_segmentation.py`
  (`spotter_rallies_v1`). A rally opens on its serve; a missed serve is
  recovered only from the spotter's own `serve` probability (peak >= 0.15,
  0.3-3.5 s before the first contact) and stays marked `inferred`; a serve no
  contact follows is `serve_not_played`, never a rally, and is not split into
  ace/fault. Contacts without serve evidence join no rally. FSM output is in
  `all_events_fsm`/`ev_fsm`/`rallies_fsm`/`service_attempts_fsm`/
  `rally_lifecycle_fsm`; `match_analytics` and `attack_heatmap` stay
  FSM-derived. JIZ3 check: 22 rallies all on a serve (8 inferred, visually
  plausible), not a labelled score. `scripts/apply_spotter_rallies.py`
  rebuilds a saved result into a fresh directory.
- Do not build a phase gate for warm-up. Measured: the decoder labels warm-up
  `rally_live`/`serve_flight`, a surrounding-phase gate catches 1 of 36 and the
  phase probability margin carries no signal (the contact class takes the mass
  at a contact frame). Use `--match-config`, or train on more `non_game`.
- `scripts/evaluation/overlay_calibration_check.py` draws a calibration on a
  frame. `setup_1` was confirmed visually to fit `xZbD8fx3D4k_1_4k`, which is
  the same fixed camera as the gold videos and is not in any training set.
- Calibrations from `scripts/interactive_court_calibration.py` are in a
  DIFFERENT corner order than `GameAnalyzer._load_calibration` expects: the
  tool starts at the line nearest the camera and names it `back_line_points`,
  the loader reads that as the court bottom and expects the far line first. The
  homography is valid either way, so an overlay looks right while
  `court_bottom` lands on the far line and the world y axis is reversed. Always
  run `scripts/normalize_calibration.py` on a freshly clicked calibration, and
  check `world y=-9 < y=0 < y=+9` in image rows. The tool also accepts a still
  image as `-i`, which beats a video whose first frame has players on the lines.
- Contact review for unannotated video:
  `action_recognition/annotator/contact_review_annotator.py` turns a batch of
  `--contact-spotter` analyses into a review queue. Records start
  `label_source: "model"`, `verified: false`; a keystroke makes them
  `"human"`/true while `predicted_type` keeps the model's answer beside
  `review.type`. `unsure` stays unverified. Human-added contacts carry
  `origin: "human"` and no prediction. The queue fingerprints both the
  annotations and the stats file and refuses a changed analysis. Nothing is
  promoted into any dataset; that step does not exist. See
  `docs/contact_review_workflow.md`.
- Pose effect settled deterministically 2026-10-29, paired over seeds 42/43/44:
  live 0.7978 mean vs blanked 0.7839, all three pairs favouring poses, but the
  +0.0139 mean difference is smaller than the 0.0176 spread between seeds and
  the 3-seed ensembles reverse the order (blanked 0.8064 vs live 0.8014). Treat
  the four-class pose effect as zero, not as a small gain.
- Team attribution measured 2026-10-29 against 406 human labels
  (`data/contact_team_gold_v1/labels.json`, collected with
  `action_recognition/annotator/team_attribution_annotator.py`): nearest player
  0.685, its foot side 0.715, ball-vs-divider 0.515 and degenerate,
  net-crossing chain 0.603. Calibration was ruled out (offset sweep peaks at
  0 px on all three videos) and so was a side-switch mix-up (accuracy flat at
  ~70% throughout). The cause is an airborne ball under 2D court geometry, plus
  net congestion: attack is the worst type at 0.558 and restricting to a player
  within 60 px makes it worse. Do not propose deriving the side from court
  geometry again. The spotter writes `team: null`; per-team statistics are not
  available. Untried: a learned binary classifier on those labels, and the
  ball's pixel size as a depth cue (annotations store only its centre). See
  `docs/team_attribution.md`.
- Reviewed-queue expansion measured 2026-10-29
  (`scripts/evaluation/run_reviewed_expansion.py`,
  `results/reviewed_expansion_20261029_v1`): adding the four reviewed
  new-camera videos as sparse contact-only targets
  (`action_recognition/rally_spotting/reviewed_contacts.py`, trainer flag
  `--reviewed`) lifts new-camera recall 0.851 -> 0.946 and serve recall
  0.708 -> 0.938, but gold detection precision drops 0.828 -> 0.806 (limit
  0.02), and the model emits 917 events against 414 human contacts on held-out
  new videos. **Not promoted**; `spotter_20261028_v1` stays. The cause is that
  the targets have no background. The next step is to review the plus models'
  extra events for true negatives, not to relax the rule. `new_base`
  reproduces the shipped checkpoint exactly. See
  `docs/reviewed_queue_training.md`.
- Runs: `results/dense_contact_spotting_20261027_v1` (frozen, ball-only),
  `results/dense_contact_spotting_20261028_v2` (poses live) and
  `..._v2_nopose` (matched control). Model-selection estimates only; the
  experimental checkpoints are not production-loadable and nothing is promoted.
  See `docs/dense_contact_spotting.md`.
