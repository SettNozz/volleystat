# Stable player IDs in the main workflow

Model outputs are now cached automatically. For ID experiments reuse saved
annotations with `apply --require-model-cache`; do not rerun detectors or
render the full video. The default analysis exports JSON for the original-video
viewer; `--render-video` is explicit. See [cache protocol](model_output_cache.md).

OSNet identity is enabled by default in `GameAnalyzer`, `analyze_video.py` and
viewer upload jobs. The offline pass runs after ball/FSM/event processing over
the final saved player boxes. It does not feed identities into ByteTrack, court
membership or pose features. A separate derived
[per-player contact report](player_contacts.md) proposes attribution after
identity; its ambiguous cases remain unknown and can be reviewed in the viewer.

Default: six players on side **b** (near team). P1–P6 are assigned left-to-right
on an early complete frame, then tracked with immutable OSNet galleries and
Kalman motion, including lost-ID recovery. Automatic initialisation is unverified.
IDs belong to a roster within one source video, not names or cross-match IDs.

## Install once

```powershell
python scripts/models/install_player_reid.py --help
python scripts/models/install_player_reid.py
# Or reuse the verified experiment bundle without another download:
python scripts/models/install_player_reid.py --from-dir results/player_identity_reid_hy2S_20261029_v1/assets
```

Use only one installation command: existing destinations are refused. The
installer writes a new `models/player_reid/osnet_msmt17_v1` bundle containing
pinned official OSNet-x1.0/MSMT17 weights, architecture, MIT licence and hashes.
Inference is offline; it never downloads missing assets. Logical model
`player_reid` uses the existing registry resolver and this local fallback.
Registry weights must match the supported pinned checkpoint; architecture
remains in the installed bundle. Setup does not register/promote MLflow aliases.

## Normal analysis

```powershell
python analyze_video.py --video game.webm --output results/game_new
python analyze_video.py --video game.webm --output results/game_json --no-video
# Opt out when only raw tracker IDs are wanted:
python analyze_video.py --video game.webm --output results/game_tracks --no-player-identity
```

`--player-identity-side a|b` selects the court side. `--player-reid-device
auto|cpu|cuda` chooses hardware; auto uses CUDA when available.
`--player-reid-assets DIR` explicitly overrides the pinned bundle/weight path.
Production does not alter process-wide CUDA allocator limits while YOLO is
loaded. The extra video read and encoder inference add runtime. Frame-directory
analysis also assigns automatic IDs and redraws final frame outputs.

Original `poses[].id` remains the raw tracker ID. Additional `player_id` is
P1–P6 (plus explicitly introduced substitutes) or null; nested `identity` records method, reason, cost, source and
`verified: false`. Costs are not probabilities. Both JSON outputs include
`player_identity` seed/gallery/parameter/model provenance, checksums, coverage
and timing. Coverage is not accuracy. `models.player_reid` records resolution.
Viewer/video show P1–P6 or P?; legacy/disabled results show `track #N`.
Final video/frame rendering waits for identity output and uses the same
annotations as the viewer.
Roster IDs are not displayed from a stale previous annotation on unprocessed
frames. Existing analysis artifacts are refused before a new run starts;
choose a fresh output directory instead of overwriting old results.

Missing default assets record `model_unavailable` and P? with an installation
warning. Explicit broken assets/seed fail the run. No complete six-player frame
records `no_complete_roster` and unknown IDs; no roster is fabricated.

## Human initial IDs

```powershell
python scripts/apply_player_identity.py export-seed --review-project results/player_identity_hy2S_20261029_v1 --output results/my_seed_v1.json
python analyze_video.py --video game.webm --output results/game_manual --player-identity-seed results/my_seed_v1.json
```

Export includes initial verified human IDs only, never the independent checks.
The seed binds to video SHA256 and observed boxes. Reruns match boxes one-to-one
by IoU instead of reusing pose indices or tracker IDs. Different video, skipped
seed frame, incomplete roster or ambiguous box matching is refused. Manual
seeds require video input; frame-directory analysis currently uses auto IDs.

## Existing results: fresh copy, no detector rerun

```powershell
python scripts/apply_player_identity.py apply --help
python scripts/apply_player_identity.py apply --annotations results/old/game_annotations.json --output results/game_with_ids_v1 --seed results/my_seed_v1.json
```

Omit `--seed` for automatic initialisation. The same production pass reuses
saved poses and original video, then writes a new paired stats/annotations
result. Existing output directories are refused. No detection, FSM or event
model rerun; no second video is encoded. The normal viewer overlays IDs on the
original video. Existing analysis files are never upgraded in place.

## Scope and proof

User requested adoption after the [OSNet experiment](player_reid_experiment.md).
This remains a single-side method. Substitutes require explicit new-person
confirmation and an observed appearance anchor; discovery is not automatic.
Set-side changes and cross-video identities are not solved. Per-player contact
counts now expose conservative, unverified body-proximity proposals plus
source-bound human corrections; they are not verified personal statistics.

Original EMA v1 dense production regression on all 1348 saved frames, using the exported human
seed and the same 100 development checks: **96 correct / 0 wrong / 4 unknown**.
This is not a held-out score. Original 7545 near-player detections remain;
6765 get an ID (89.66% coverage). Other annotation fields, raw IDs, ball and
events compare equal to source. Auto-initialised IDs have their own initial
ordering. Fresh artifact: `results/player_identity_workflow_hy2S_20261029_v1`.

Tests cover default/disabled operation, missing images, incomplete rosters,
source-video binding, reordered detections, preserved observations/events,
final-render ordering, JSON exports and legacy viewer compatibility. Continue
independent validation before interpreting this as whole-match accuracy.

## Occlusion motion update (2026-10-30)

The Kalman update (`osnet_frozen_gallery_kalman_v2`, retained in old results) keeps OSNet galleries frozen
and replaces the EMA motion estimate with a constant-velocity Kalman filter.
Position/velocity covariance grows over missing observations. Integrated
continuous white acceleration gives the same predicted uncertainty regardless
of how often prediction is requested. Process and
measurement noise scale with person height. Association uses the innovation
distance plus physical displacement and height gates. Reverse processing uses
signed time in seconds, with the same covariance model.

When bounding-box overlap is >=0.6, a crop needs distinct appearance and cannot
update motion. This prevents a truncated/merged box from pulling the predicted
track towards an occluder. Uncertain observations remain P?. Long-loss recovery
still needs three consistent appearance matches. No missing poses, player
positions or contacts are manufactured by Kalman predictions.

Matched saved detections and frozen embeddings, same 100 development checks:
EMA **94 correct / 1 wrong / 5 unknown**; Kalman **94 / 0 / 6**. Correct coverage
did not increase: the old wrong ID became unknown. Dense EMA's 96/0/4 above is a
different sampling interval and cannot be compared directly to this sparse run.
Old experiment plans retain EMA unless they explicitly request Kalman.

```powershell
python scripts/evaluation/compare_player_motion.py --project results/player_identity_hy2S_20261029_v1 --cache results/player_identity_reid_hy2S_20261029_v1/cache_cuda --output results/motion_comparison_new
```

No training or label-driven gallery updates. Tests include same-uniform crossing
after disappearance, covariance growth, occluded-centre drift, one-to-one IDs,
backward time, resolution scaling and existing recovery/annotation invariants.
The same-uniform crossing test also reproduces the old EMA's swapped IDs after
a one-second disappearance; Kalman preserves the two synthetic identities.

Synthetic all-player detection blackouts on those same frozen observations,
scored only after the gap (correct / wrong / unknown):

| Gap | EMA | Kalman |
|---|---|---|
| 0.5 s | 95 / 1 / 4 | 94 / 0 / 6 |
| 1 s | 96 / 1 / 3 | 94 / 0 / 6 |
| 2 s | 87 / 1 / 6 | 87 / 0 / 7 |
| 4 s | 76 / 1 / 8 | 75 / 0 / 10 |

This supports more conservative assignments, not a recall gain. A blackout
removes detections; it does not reproduce the full visual effects of occlusion.
Raw sources and human labels were preserved. Artifacts:
`results/player_identity_kalman_20261030_v1/blackout_stress_continuous_v2`.

After a full run, create a separate diagnostic review without rewriting it:

```powershell
python scripts/evaluation/audit_player_identity.py --annotations results/game_new/game_annotations.json --output results/game_identity_review_new
```

The audit checks ID uniqueness/count consistency, reports coverage per minute,
and renders before/during/after overlap strips spread across the whole video.
The initial roster and every strip remain explicitly model/unverified. No
accuracy is reported without independent human identity checks.

## Confirmed substitutes and local contradictions (v3)

Production method `osnet_frozen_gallery_kalman_roster_v3` distinguishes six
on-court places from the number of known people. A substitute gets a new ID;
the original person's ID remains available when they return. Appearance
galleries remain frozen. An unknown detection never creates a new person.

Example from JIZ3exHe3hA: the user confirmed that the gray-shirt player near
2:27 is a substitute. P7 is anchored to his actual saved box at frame 4397;
P1 remains the original black-shirt woman. His source raw track first appears
at frame 3759, used as a derived earliest-availability boundary. Confirmation
of a new person does not turn that boundary or the automatically selected
anchor box into human-verified labels.

```powershell
python scripts/apply_player_identity.py add-player --help
python scripts/apply_player_identity.py add-player --annotations results/old/game_annotations.json --frame 4397 --from-player-id P1 --new-id P7 --available-from 3759 --confirmation-note "User confirmed a substitute" --output results/game_roster_v2.json
python analyze_video.py --video game.webm --output results/game_roster_new --player-identity-seed results/game_roster_v2.json
```

`--pose-index` can select the observed box instead of `--from-player-id`.
For further additions, supply `--existing-roster`. IDs and anchor observations
must be distinct; the file binds to the source-video checksum. Existing human
v1 seeds still work. v2 roster files preserve automatic initialisation as
unverified while recording explicit human new-person confirmation separately.

The additional rejection pass uses nearby, confident assignments along raw
detector tracks. It rejects a locally bracketed contradiction or a borrowed ID
when its stronger established track is simultaneously visible but ambiguous.
It never copies a majority ID onto a track, invents a corrected assignment or
changes a gallery. A raw track can change people; clean evidence of that change
is retained. Strong appearance can supply evidence even when boxes overlap.
Unknown remains P?. On the original 100 development checks, v3 keeps Kalman's
**94 correct / 0 wrong / 6 unknown**. This is not held-out validation.

For controlled replay, cache exact saved boxes once, then compare fixed-six
versus confirmed roster, each with and without rejection:

```powershell
python scripts/evaluation/cache_player_reid_features.py --annotations results/old/game_annotations.json --output results/game_features_new --device cuda
python scripts/evaluation/run_player_roster_experiment.py --annotations results/old/game_annotations.json --cache results/game_features_new --roster results/game_roster_v2.json --output results/game_roster_comparison_new
python scripts/render_saved_analysis.py --annotations results/game_roster_comparison_new/game_annotations.json --calibration camera.json --output-video results/game_roster_comparison_new/game_analyzed.mp4
```

All outputs must be fresh. The cache contains derived model features, not
training labels. Source fingerprints, exact keys and nonidentity annotation
values are checked. The chosen case uses the same production association and
video renderer as normal analysis. Original detector poses, ball, FSM and
events remain unchanged. Target-video coverage is not identity accuracy.

## Appearance contradiction veto (v4, current default)

`osnet_frozen_gallery_kalman_roster_v4` also checks a local match against all
active immutable galleries. If its cosine distance is more than 0.04 worse
than the best appearance, that candidate cannot win through motion alone.
When the best-looking identity fails the motion gate, the observation can
remain unknown; another ID is not borrowed to fill the gap. The veto runs
before motion updates, preventing incorrect observations from pulling the
filter onto a different person. It never moves a gallery or creates a pose.

On JIZ3exHe3hA frame 5500 (~3:03), the old local matcher assigned white-shirt
P5 to a black-shirt player while the actual white-shirt detection was visible.
The veto rejects the wrong assignment and retains P5 on that visible white
shirt. This is a visual diagnostic, not human-verified identity gold.

Same old 100 development checks: no veto **94/0/6**, 0.04 **94/0/6**, 0.06
**90/0/10**, 0.10 **94/0/6** (correct/wrong/unknown). Motion feedback makes
threshold effects non-monotonic. 0.04 is a conservative engineering choice,
not a held-out optimum or whole-match accuracy estimate. Legacy experiment
parameters still default to no veto. Production selects it explicitly.

The cached runner's optional `--appearance-regret-max 0.04` evaluates one
candidate in a fresh directory against existing frozen controls. No additional
encoder pass is required. Original v2/v3 results remain available for review.
