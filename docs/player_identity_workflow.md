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

Default: six players on side **b** (near team). The roster is found from
tracklets of the whole video (`osnet_tracklet_roster_v5`, see
[Whole-video roster](#whole-video-roster-v5-current-default)), not from one
frame. P1–P6 are named left to right on the first frame that cleanly shows all
six. The automatic roster is unverified. IDs belong to a roster within one
source video, not names or cross-match IDs. The older tracker started from one
complete frame remains as `--player-identity-method seed-frame`; it also runs
when a clip is too short for a roster and for jersey-number constraints.

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

## Whole-video roster (v5, current default)

`src/tracking/player_roster_discovery.py`, method `osnet_tracklet_roster_v5`.
A roster from one frame fails in two ways. A bystander standing on court in
that frame becomes a player for the whole match, and a player missing from it
never gets an ID. Eight crops from half a second also describe a person from
one angle only.

1. Detector tracks are cut into tracklets of at most one second. Each gets the
   mean frozen OSNet embedding of its clean boxes (overlap < 0.3). A tracklet
   whose boxes disagree is dropped. At most 3,000 tracklets are kept, spread
   evenly over the video.
2. Tracklets merge by appearance (average linkage, cosine distance ≤ 0.3).
   Two tracklets cleanly visible in the same three frames are different people
   and never merge. The video supplies this constraint itself.
3. A leftover group joins a larger look-alike when they share at most 2% of
   the smaller group's clean frames. One duplicate detection can otherwise
   split a player in two.
4. The roster is the six people present longest. Each must be present for at
   least 3 s and a quarter of the longest presence. With a human seed or a
   confirmed roster file, the anchors choose and name the people instead.
5. Every saved box is scored against all tracklets of each roster person (mean
   of the three best). Scores are averaged over ±0.5 s of the same detector
   track; overlapped boxes count 0.3. IDs are assigned one-to-one per frame.
   A box needs similarity ≥ 0.6 and a margin of 0.03 over the next person.
6. A non-roster group seen together with every roster person is a known other
   person, such as an opponent at the net. A box that looks more like one of
   them than like any roster person stays unknown (`reason: other_person`).

No motion model is used: ByteTrack continuity enters only through the
tracklets and the per-track average. Nothing is trained and no human check
enters the method. `player_identity.roster_discovery` records the players'
presence, the people left out (`other_person` or `unresolved`) and the
selection rule. `seed` holds one reference box per player.

Substitutes still need a confirmed anchor (`add-player`): a leftover group is
never given an ID on its own. When fewer than six constant players are found,
the seed-frame tracker runs and the report says
`roster_discovery.status: fallback_to_seed_frame`.

```powershell
python scripts/apply_player_identity.py apply --annotations results/old/game_annotations.json --output results/game_ids_v1 --require-model-cache
python scripts/evaluation/render_player_roster_sheet.py --annotations results/game_ids_v1/game_annotations.json --output results/game_ids_v1/roster_sheet.jpg
python scripts/evaluation/evaluate_player_identity_movement.py --movement human_movement.json --annotations results/old/game_annotations.json results/game_ids_v1/game_annotations.json --output results/game_ids_score_v1 --replay
```

The roster sheet shows crops of every ID across the video and the people left
out; check it before trusting an automatic roster. The evaluation script
scores saved results against a `human_player_movement_v1` export
(`src/tracking/player_identity_scoring.py`) and, with `--replay`, runs the
methods from cached embeddings.

### Measured (2026-11-10)

`iKhVYAof2as_4k`, human movement review revision 787: 30,256 reviewed near-side
poses in 0:00–3:13, one pose per player and frame. Model IDs are named by the
one-to-one mapping that agrees best with the reviewer. Artifacts:
`results/player_identity_iKhVYAof2as_4k_r000787_methods_20261110_v2/metrics.json`,
result `results/player_identity_iKhVYAof2as_4k_roster_discovery_20261110_v2`.

| Method | Correct | Wrong ID | Unknown |
|---|---|---|---|
| Seed frame, automatic (v4, previous default) | 67.3% | 4.1% | 28.6% |
| Seed frame, human roster on the earliest complete frame | 73.6% | 3.3% | 23.0% |
| Seed frame, human roster on the median complete frame | 71.8% | 2.6% | 25.6% |
| Seed frame, human roster on the cleanest complete frame | 84.0% | 0.8% | 15.2% |
| Seed-frame tracker with the discovered roster and gallery | 91.6% | 0.4% | 8.0% |
| Whole-video roster (v5) | **97.6%** | **0.2%** | 2.2% |

- The automatic seed frame held a bystander and missed one player. v5 finds
  the six players the reviewer confirmed.
- Per player v5 is 96.0–99.4% correct. Adjacent-frame ID switches fall from 12
  to 4. After a gap of at least 1 s the right ID is on the first frame in 26
  of 30 returns (v4: 2 of 8).
- IDs on poses the reviewer marked "not in the roster": 256 of 438 with v4, 18
  with v5.
- Coverage of all 90,146 near-side poses in the 10-minute video rises from
  71.0% to 94.6%. Coverage is not accuracy.
- On clean boxes v5 is about 99% correct. Most remaining unknowns are runs under
  2 s in heavy overlap.
- The identity pass takes 6 s instead of 24 s on cached embeddings.
- Changing one parameter at a time (tracklet length, merge distance,
  co-visibility, smoothing, margin, thresholds) gives 93.9–98.1% correct and
  at most 0.3% wrong.

Second video, no parameter change: the 100 sparse human checks of
`results/player_identity_hy2S_20261029_v1` (stride 3, 45 s). v4 scores
94 correct / 0 wrong / 6 unknown, v5 99 / 0 / 1.

Limits:

- These are development numbers from two cameras. The defaults were chosen
  while looking at the first video, so it is not a held-out score.
- Missed detections are not scored.
- After the last rally (from about 9:40) people leave the court and others
  walk in. A few boxes there still get roster IDs and there are no labels to
  count them.
- Teammates in identical kit with similar build have not been measured.
- A substitute is listed among the people left out until confirmed.

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

## Appearance contradiction veto (v4, seed-frame method)

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

## Jersey-number constraints (opt-in)

A pinned TrOCR scene-text model (`microsoft/trocr-base-str@26c151f`, MIT)
reads shoulder-to-hip torso crops of saved near-side poses. Only one- or
two-digit strings count. Default analysis does not run OCR.

```powershell
python scripts/models/install_jersey_ocr.py               # or --from-dir <downloaded files>
python scripts/evaluation/read_jersey_numbers.py --annotations results/game/game_annotations.json --output results/game_jersey_v1/reads.json --stride 3
python scripts/apply_player_identity.py apply --annotations results/game/game_annotations.json --output results/game_jersey_v1/result --require-model-cache --jersey-reads results/game_jersey_v1/reads.json
```

`src/tracking/jersey_numbers.py` splits raw track IDs into continuous
tracklets. A tracklet gets a number when at least two reads with probability
≥ 0.8 agree and hold ≥ 75% of its confident reads. A first identity pass
assigns tracklets to roster IDs. A roster ID gets a number when at least two
of its tracklets vote and one number holds ≥ 60% of them. A number claimed by
two IDs is dropped. In the second pass, a detection cannot take an ID when its
tracklet shows another roster player's number or a number other than this
player's. A matching number replaces the appearance margin for a lost return
or a near-net candidate. Distance, motion and recovery-confirmation gates
stay. Reads, tracklet numbers and roster numbers stay `label_source=model`,
unverified.

Measured on `e2F6pef-te0_2160p` (2026-11-10, one match, one camera); artifacts
are in `results/experiments/jersey_ocr_e2f_20261110_v1/`.

- The reader is weak on this footage. It often adds a leading 1 (8 read as
  18, 3 as 13) and reads 7 as 1. A full-video pass at stride 3 read 30,057
  torso crops in 708 s. 11,815 crops read as a number; 1,079 of those had
  probability ≥ 0.8. Front views and small players give no reads.
- The 52 diagnostic ID checks from PR #20 use the #20 final side policy, the
  same windows and cached features only (`scripts/evaluation/compare_jersey_identity.py`).
  The #20 policy alone scores 42 correct / 0 wrong / 10 missing with 4,257
  assigned observations. Read probability ≥ 0.8 (default) leaves the checks
  at 42/0/10 with 4,308 assigned and finds one correct roster number (#8).
  At 0.6 one roster ID got a misread number (7 read as 1) and 17 IDs changed;
  0.4 lost 254 assignments.
- The full-video `apply` replay set `P5=#8`. The replay without reads
  assigned 49,570 of 90,266 observations, and its automatic roster mixes
  people. P5 tracklets read #8 16 times, #9 5 times and #3 once; P4 splits
  between #11 and #9. With the constraints:
  - 1,074 P5 observations of #3 and 205 of #9 lose the ID. Eight random crops
    of each group show players #3 and #9, so these removals are correct.
  - 376 #8 observations gain P5, and the sampled crops show #8.
  - 236 observations gain P4. The sampled crops show player #7 read as "1",
    so these are new errors. The number did not cause them: P4 has no number
    to veto with, and the ID arrives through the motion tracker.
  - Overall 2,292 lost, 1,882 gained and 53 changed. See `jersey_changes.jpg`.

Number evidence finds mixed roster IDs that appearance alone misses. This
generic reader is not reliable enough for default use. Next steps are a
jersey-specific recogniser, such as PARSeq fine-tuned on jersey numbers, and a
check of whether a tracklet faces the camera. None of these numbers are
held-out or human-verified identity accuracy.
