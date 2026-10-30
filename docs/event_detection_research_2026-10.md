# Event-detection approaches: landscape review and migration plan (2026-10-27)

Purpose: decide what replaces the hand-written `IntegratedVolleyballFSM` as the
source of event structure, and quantify the dataset work each option needs.

This document is a research and planning artifact. It changes no data, no
checkpoint and no production default. Numbers taken from papers are cited with
their source; numbers taken from this repository were measured while writing
this file and are marked as local measurements.

## 1. What we actually have (local measurement, 2026-10-27)

Human source of truth: `data/full_video_gold/j3BO9GgUFBI_3_4k/annotations_v1.json`
is cumulative and holds all three videos. The two other `annotations_v1.json`
files are earlier prefixes of the same project; counting all three files
triple-counts `Khe`.

| Quantity | Value |
|---|---|
| Human contact annotations | 430 (Khe 171, j0 137, j3 122) |
| Human `between_rallies` / `non_game` segments | 84 |
| Source frames, all three videos | 61,822 (21,300 + 19,222 + 21,300) |
| FPS / resolution | 29.97 / 3840x2160 |
| Event frames in dense targets | 421 of 61,822 = **0.68%** |
| Per-frame poses already extracted | 18,187 frames = **29.4%** |

Raw human `event_type` distribution (430): `set` 91, `attack` 88, `serve` 80,
`dig` 71, `reception` 69, `block` 26, `free_ball` 3, `other` 2.
Also per contact: `team_side`, `match_phase` (`rally_live` 350 /
`serve_flight` 80), `physical_action` (6 values), `reception_quality`
(6 values), `is_service_reception`. No player/jersey identity.

Already-exported derived artifacts per video, under
`data/full_video_gold/<stem>/derived_7d09206_v2/`:

- `dense_targets.npz`: `event_targets (N, 8) uint8`, `phase_targets (N,) uint8`,
  `reception_quality_targets (N,) int8`, `frame_numbers (N,) int64`.
  Phase vocabulary: `unlabeled, non_game, between_rallies, serve_flight, rally_live`.
- `spotting_labels.json`: 4-class spotting events
  (`serve_contact`, `reception_contact`, `followup_contact`, `rally_end`).
- `rallies_and_score.json`, `phase_segments_raw.json`, `events_enriched.json`, QC.
- `geometry.json` (in the v3 offline dataset) supplies **4 court corners**,
  homography, net line, per-video, at native 4K.

Existing dense model: `action_recognition/rally_spotting/` is already a dilated
residual TCN (`RallySpottingTCN`) over per-frame features
(10 trajectory + 2 ball flags + 8 pose + 2 prior channels,
`resolution_normalized_v2`), trained on continuous full-video spans with human
labels. This is an MS-TCN-shaped dense spotter, not a clip classifier.

So the repository already contains: dense per-frame targets for every frame of
every gold video, a dense per-frame model, court geometry, frozen ball
trajectories, and 29.4% of the per-frame skeletons.

## 2. Landscape

### 2.1 Precise Event Spotting (PES) — the task that matches ours

Lineage, with reported numbers as published:

- **E2E-Spot** (Hong et al., ECCV 2022, arXiv 2207.10213): compact CNN
  (RegNet-Y) + GRU, trained end to end on long windows. Still the baseline
  everyone reports against.
- **T-DEED** (Xarles et al.): winner of the SoccerNet 2024 ball-action spotting
  challenge; temporally discriminative encoder-decoder.
- **ASTRM + SoftIC** (Santra et al., arXiv 2503.00147): replaces GSM with an
  adaptive spatio-temporal refinement module, adds a supervised-contrastive
  memory-bank loss. SoccerNet-V2 test mAP tight/loose **73.74 / 79.11** against
  E2E-Spot 200MF **61.19 / 73.25**; Tennis mAP@0 frames **61.01** vs E2E-Spot
  **45.34**. Clip length 128 frames, 224p, **2 FPS**, 4x A100. No code URL in
  the paper.
- **UMEG-Net** (Liu et al., arXiv 2511.14186, AAAI): drops RGB at the core and
  builds one graph per frame over **player joints + ball + 4 court corners**,
  encoded by a unit GCN with a parameter-free multi-scale temporal shift.
  2.2M parameters. Code: `https://github.com/LZYAndy/UMEG-Net`, paper CC BY 4.0.

UMEG-Net matters to us for one specific reason: it is the only one of these
evaluated in **our data regime**. Its headline table is the *100-labelled-clip*
setting, averaged over five random splits:

| Method | F3Set-Tennis F1/Edit | ShuttleSet | SoccerNet-BAS | Params |
|---|---|---|---|---|
| E2E-Spot 200MF | 2.6 / 11.1 | 35.6 / 50.3 | 15.3 / 35.4 | 4.5M |
| E2E-Spot 800MF | 3.1 / 13.3 | 42.7 / 54.6 | 22.1 / 43.1 | 12.6M |
| T-DEED 200MF | 1.0 / 4.2 | 33.8 / 41.7 | 4.7 / 8.8 | 16.4M |
| T-DEED 800MF | 1.5 / 6.4 | 30.7 / 38.6 | 6.7 / 14.3 | 46.2M |
| BlockGCN (skeleton only) | 6.9 / 18.3 | 47.1 / 59.4 | 19.9 / 43.3 | 2.5M |
| **UMEG-Net** | **9.4 / 31.7** | **49.2 / 64.0** | **27.0 / 44.8** | **2.2M** |
| UMEG-Net + RGB distillation | 12.5 / 40.7 | 59.1 / 69.0 | 27.1 / 50.8 | 67.8M |

Entity ablation at 100 clips (F1/Edit): pose only 5.6/23.9 (F3Set),
pose+court 6.6/26.1, pose+ball 8.6/30.2, pose+ball+court 9.4/31.7. Ball is the
largest single contributor; court helps less and on ShuttleSet pose+court
(46.7) is *below* pose-only (47.4).

Honest counter-evidence: under **full** supervision UMEG-Net loses badly on two
of five datasets (FineGym-BB 59.8 vs E2E-Spot 72.9 F1; SoccerNet-BAS 36.1 vs
46.2). The graph wins when labels are scarce, not in general. Node features are
**2D coordinates only** — no confidence, no visibility channel — and the paper
describes no missing-keypoint policy, which is exactly our hardest case.
Adjacency rules, coordinate normalization and joints-per-person are deferred to
the repo, not stated in the paper.

### 2.2 Replacing the FSM: constrained decoding over learned transitions

**Constraint-Aware Decoding** (Ee et al., arXiv 2605.10149; code
`https://github.com/LUNAProject22/CAD`) is the direct, implementable answer to
"I do not like the FSM". It is pure post-processing on frame-wise logits:

```
Y* = argmax_{Y in Y_R}  sum_t log P[t, y_t] + sum_t log Conf(y_{t-1} -> y_t)
```

subject to hard constraints: `y_1 in S` (valid starts), `y_T in E` (valid ends),
every transition in the observed set `T`, and every segment's normalized
duration within `[d_min[c], d_max[c]]`. Transition confidence is plain counting:
`Conf(A->B) = Count(A->B) / Count(A)`. Viterbi tables carry `(score, segment
duration, backpointer)` per `(t, class)`.

Measured effect on top of ASFormer (F1@10/25/50, Edit, Acc):

- Breakfast: 74.1/68.7/55.5/72.8/72.4 -> **78.8/73.1/58.0/77.9/74.6**
- 50Salads: 83.4/80.8/74.6/75.7/85.0 -> **84.8/83.1/76.0/78.6/82.9** (Acc -2.1)
- On FACT (Breakfast) gains shrink to +0.5 F1@10 and F1@50 *drops* 1.4.
- Classical unconstrained Viterbi is much worse: 45.5/40.2/30.1/39.2/60.1 vs
  55.8/51.4/38.5/48.9/63.4 in the semi-supervised 50Salads setting.
- Decoding cost 2.81 s vs 148.9 s for the grammar baseline (20 Breakfast videos).

Ablation: removing duration bounds costs the most F1; removing transitions costs
edit/accuracy; **soft** constraints consistently underperform **hard** ones.

Known weaknesses, some of them the reviewing agent's analysis rather than the
paper's: duration is not part of the Viterbi state, so the per-cell greedy keep
is not guaranteed optimal; the terminal check enforces only `d_max`; self-
transitions carry no `Conf` penalty; behaviour with no feasible path is
unspecified; and the paper admits the method "can become overly restrictive when
the training data is sparse, noisy, or incomplete" — which describes 430 labels.

Why this is still the right shape for us: our FSM already encodes exactly these
three constraint families (which state may follow which, how long a state may
last, what may start a rally) — but with hand-chosen numbers and no probability.
CAD keeps the structure and takes the numbers from the human annotations.

### 2.3 Volleyball-specific published work

- **VNL-STES** (Nguyen et al., CVPRW 2025, project `hoangqnguyen.github.io/stes`):
  8 VNL matches, **1,028 rallies, 6,137 events, 251,110 frames**, 25 FPS, classes
  serve/receive/set/spike/block/score, with temporal frame *and* normalized
  (x, y) spatial target. Splits 811/102/115 rallies. Reported +9.86 mean temporal
  AP over prior SOTA and 80.21 mAP spatial.
  **Data access is currently blocked**: the project page's data link
  (`bit.ly/vnlvolley1`) redirects to a personal SharePoint URL that returns HTTP
  403. The README describes a 13 GB 398x224 download. No pretrained checkpoint is
  published. Code is BSD-3 (fork of E2E-Spot); `util/dataset.py` hardcodes
  `DATASETS = ["vnl_1.5", "vnl_2.0"]`, so a custom dataset needs a code entry,
  a `class.txt`, `train/val/test.json`, integer-named JPEG frame dirs and an
  undocumented `fps.txt`.
  That code was vendored at `third_party/vnl_stes` (now on branch
  `archive/research-snapshot-2026-10-30`) and the
  local pilot (`results/vnl_stes_author_20261027_v1`) reaches mAP@0/1/2/4 of only
  ~0.030 with ImageNet init — consistent with "no VNL pretraining available".
- **MultiSports** includes volleyball with spatio-temporal action labels
  (CC BY-NC 4.0). The classic **Volleyball group-activity dataset** (Ibrahim et
  al.) is 4,830 annotated frames / 55 videos, group-activity labels, not contact
  spotting.
- Community projects (`VolleyVision`, `volleyball_analytics`, `Volleylitics`)
  are frame-level classifiers or roadmaps, with no benchmark numbers.

No published volleyball method can be adopted wholesale with data we can get
today.

### 2.4 Trajectory-first contact detection

From the table-tennis literature (TT3D, arXiv 2504.10035, and refs therein):
prior work detects bounces/hits either from **heuristics on velocity change** or
with **GRUs over ball positions plus player pose**; TT3D itself segments a rally
into flights by fitting **second-degree polynomials** and choosing the breakpoints
with **dynamic programming**. A table-tennis stroke paper (arXiv 2302.09657)
segments strokes from ball trajectory alone.

This is the principled version of our `contact_trajectory_before_after_v1`
experiment, which gave mean macro F1 0.9147 -> 0.9176 (+0.28 pp) — real but
inside seed noise. The literature's framing (piecewise-parabola DP over the whole
trajectory, not hand-picked [-12,+12] windows) is a strictly stronger formulation
and resolution-independent, which matters because our ball is a few pixels wide.

### 2.5 Audio

Whistle detection is the standard cheap cue for rally boundaries. Evidence is
weak: open projects (`Volleylitics`, `volleyball-stats`) use DSP candidate
detection in a 2-4.5 kHz band plus a CNN filter; the only concrete published
number found is a 2003-era general sports study with a **19.4% failure rate**
(42 FP, 26 FN). No volleyball-specific validated benchmark surfaced.

Still worth testing here, because our rally boundaries currently come from the
FSM, whose `rally_end` time is an estimate — and `ServeGuard`'s 3-second restart
rule is explicitly unvalidated against human boundaries.

### 2.6 Where the field moved (context, not a plan)

SoccerNet 2025 required the acting *team*; SoccerNet 2026 requires the acting
*player* by jersey number, scored on joint (frame, team, jersey, class) matching.
Entries use a two-stage **Track-Aware Action Detector** feeding a **Denoising
Sequence Transduction** autoregressive transformer over game state (FOOTPASS
baselines, arXiv 2606.09679 / 2608.01696; one entry reports 58.94 Macro-F1).
Relevant to us only after player identity exists in our labels — it does not.

## 3. Diagnosis of our own two RGB failures

Both failures are consistent and both are explained by resolution, not by
architecture:

- `results/rgb_event_spotting_20261027_v2`: frozen MobileNetV3-S, 224 letterbox
  full frame. Khe macro F1 **0.084** (8 TP / 154 FN), j0 0.262, j3 0.247.
- `results/vnl_stes_author_20261027_v1`: author RegNet-Y/GSM/BiGRU, ImageNet
  init, 64-frame clips. mAP@{0,1,2,4} ~**0.030** (Khe), ~0.027 (j0); third fold
  never finished.

A 4K ball letterboxed to 224 px height is sub-pixel. This is the same finding as
the stored notes `4k-beats-720p-for-ball` (0.94 vs 0.68 recall at identical
imgsz) and `vlm-not-for-volleyball-labeling` (ball pixel size is the limit, not
model size; ROI crops closed the gap). The literature agrees that resolution is
the untested lever: Santra et al. state plainly that running at 2 FPS and 224p
"is likely to impact the results".

Conclusion: **stop feeding full-frame low-resolution RGB.** Either use entity
coordinates (which are extracted at 2560 and 4K and therefore keep the
resolution), or use crops around the ball/player at native scale.

## 4. Recommended tracks, ranked

Ranking criterion: expected information gain per unit of new data work, given
430 labels and 61,822 frames.

### T1 — Multi-entity graph dense spotter (UMEG-Net-shaped). Highest value.

Build one graph per frame: 17 COCO joints x N court players + 1 ball node +
4 court-corner nodes; intra-person skeletal edges, wrist/forearm-to-ball edges,
foot-to-court-corner edges, corner-to-corner rectangle. Unit GCN + multi-scale
temporal shift (offsets 1/2/4), dense per-frame head over our 8 raw classes plus
background. Replaces the clip classifier *and* removes the need for the FSM to
type contacts.

Why it fits: every input already exists at full resolution; 2.2M parameters suits
430 labels; it is the only method with published numbers at ~100 labelled clips;
and it directly subsumes the per-side anatomical distances that regressed as
hand-made scalars (`a_wrist_ball_dist` and friends) — as graph edges the model
learns the aggregation instead of us hand-picking a nearest player.

Deviations we must make from the paper: add a **validity channel per node**
(the paper uses x,y only and states no missing-keypoint policy; our ball is
missing or off-camera for long stretches and our schema convention already
carries availability bits), and keep `ball_out_of_frame` distinct from
`ball_missing`.

### T2 — Constraint-aware decoding instead of the FSM. Cheapest structural win.

Take frame-wise class probabilities from the dense model (T3 today, T1 later),
estimate `Conf(A->B)`, `S`, `E`, `d_min`, `d_max` **from the 430 human contacts
and 84 human segments on the training fold only**, and decode with constrained
Viterbi. Deliverables: rally segmentation, contact typing and serve eligibility
all come out of one decode, with every constraint traceable to counted human
data instead of a hand-written branch.

This is where `ServeGuard` should end up: `serve_during_active_rally` and
`serve_start_not_confirmed` are transition constraints, and the 3-second restart
rule is a `d_min[between_rallies]` duration bound that should be *measured*, not
assumed.

Keep the FSM as a frozen baseline for comparison; do not delete it in the same
change.

### T3 — Strengthen the dense spotter we already have. Immediate, no new data.

`action_recognition/rally_spotting/` already produces dense per-frame logits over
continuous spans with human labels. Three changes, all cheap:

1. Train at **8 raw classes** (plus background), not the 4-class collapse. The
   collapse currently merges `set` (91) + `dig` (71) into `pass` and `block` (26)
   into `attack` — 188 of 430 labels lose their distinction before the model sees
   them. Keep the 4-class view as an evaluation projection.
2. Fix the foreground weighting for 0.68% positives (UMEG-Net used 5x at <3%),
   and widen targets to a tolerance window around each contact frame.
3. Add the T2 decoder on top and report event F1 at native-frame tolerances
   against the current FSM pipeline.

This is the honest baseline that T1 must beat, and it needs **zero** new labels
and **zero** new extraction.

### T4 — Audio whistle for rally boundaries. Cheap, complementary.

Band-limited (2-4.5 kHz) candidate detection plus a small classifier on log-mel
or PANNs/YAMNet embeddings. Labels already exist: 84 human `between_rallies`
segments give both positives and negatives for free. Use it as an extra input
channel to T2/T3, not as a separate decision maker. Expect it to be noisy — the
only published number is a 19.4% failure rate.

### T5 — Player attribution. Defer.

The SoccerNet-2026 direction (per-player events) needs a jersey/track identity
per contact. Our 430 annotations have `team_side` but **no player identity**, so
this requires new human labelling. Do not start before T1-T3 settle.

### Not recommended

- More full-frame low-resolution RGB (two local failures, explained above).
- Another round of hand-engineered pose summary scalars: regressed twice
  (0.44 -> 0.37 macro F1 in July; 0.895 -> 0.873 pooled in the v3 team-pose run).
  The graph in T1 is the structural alternative, not another scalar block.
- Waiting on VNL pretraining. The data link is 403 and no checkpoint is
  published. Worth one email to the authors, not a blocked plan.

## 5. Dataset change matrix

"New human labelling" is the column that matters; everything else is compute.

| Track | New human labels | New extraction | New feature schema | Split policy |
|---|---|---|---|---|
| T3 dense 8-class + tolerance | **none** | none | reuse `resolution_normalized_v2` + widened targets | existing whole-video folds |
| T2 constrained decoding | **none** | none | none (decoder only) | constraints fitted on train fold only |
| T1 multi-entity graph | **none** | poses for the remaining **43,635 frames** (70.6%) at 2560 | new: node-coordinate tensor + validity, not a flat vector | existing whole-video folds |
| T4 audio whistle | **none** (84 segments suffice) | audio track extraction, cheap | small log-mel/embedding schema | existing |
| T5 player attribution | **430 contacts re-labelled** with player identity | none beyond T1 | +identity target | existing |

### 5.1 What has to change concretely

**Labels: nothing new for T1-T4.** This is the main finding. `dense_targets.npz`
already carries per-frame 8-class event targets, 5-class phase targets and
reception-quality targets for every frame of all three videos. The 4-class
collapse is a *training-time choice* in `action_recognition/event_clips/config.py`
(`LEGACY_TO_V4`), not a property of the data.

**Change the target granularity, not the annotations.** Train on the 8 raw
classes; project to 4 only for comparison against existing checkpoints. Reception
quality and `physical_action` become auxiliary heads, free supervision we are
currently discarding.

**Pose coverage is the real cost.** 18,187 of 61,822 frames (29.4%) have
per-frame skeletons: Khe 1,171, j0 8,698, j3 8,318. A continuous dense graph
model needs all frames, so ~43.6k more frames of YOLO11m-pose at 2560. The
worker exists on branch `archive/research-snapshot-2026-10-30`
(`scripts/evaluation/refresh_team_pose_dataset.py`, segment-based
with `--resume`), the geometry is frozen in `geometry.json`, and the output
format (`poses/*_sNNN.jsonl`, one record per frame with `kp`, `bb`, `team`, `id`)
is already what a graph loader wants. This is a long single GPU job and the only
material resource commitment in the plan.

**Ball is already done.** Frozen conservative full-video analyses exist and are
hashed in `preflight.json`; reconstructed points keep source/confidence and must
stay flagged as derived, never as detector ground truth.

**Court is already done.** 4 corners + homography + net line per video in
`geometry.json`, native 4K. Provenance stays
`model_geometry_not_event_ground_truth`.

**New schema needed only for T1.** Everything so far has been a flat
`(frames, channels)` tensor. A graph needs `(frames, nodes, 2 + validity)` with a
fixed node ordering and a padding convention for fewer than N players. Propose
`entity_graph_v1`, versioned under a fresh dataset directory, with the existing
18/42-channel schemas left untouched.

**Splits unchanged.** `data/event_clips_4k_gold_v1/split_policy.json` already
fixes whole-source folds with `Khe4P8QcJkg_5_4k` primary; reuse it verbatim so
the new numbers are comparable to every experiment run so far.

### 5.2 Class imbalance is the sharpest practical risk

421 event frames in 61,822 = 0.68% positives. UMEG-Net weighted foreground 5x at
<3%. At 0.68%, weighting alone will not do it; the standard remedy in the
spotting literature is to make the target a tolerance window (label a small
neighbourhood of each contact frame) and evaluate with explicit frame
tolerances. Our evaluation must therefore report F1 at several native-frame
tolerances, and those numbers are **not** comparable with the contact-centred
clip-classification macro F1 of ~0.92 that the current checkpoint reports — the
clip task is given the contact frame, the spotting task has to find it.

## 6. Suggested order of work

1. **T3 + T2 on existing data** — dense 8-class spotter with tolerance targets
   and foreground weighting, then constrained decoding on top, measured against
   the frozen FSM pipeline on the three whole-video folds. No new data. This
   establishes whether the FSM is actually the weak link.
2. **T4 audio** as an extra channel into the same decoder.
3. **T1 graph** once the pose extraction job for the remaining 70.6% of frames
   has run, compared against T3 under identical folds, tolerances and decoder.
4. **T5** only if T1 is clearly better and player identity is wanted as output.

Every step keeps the current production path intact, writes to a fresh versioned
output directory, and promotes nothing automatically.

## 7. Open questions and risks

- UMEG-Net's full-supervision regressions (FineGym, SoccerNet-BAS) mean the graph
  advantage may be a small-data effect that shrinks if we ever label more.
- CAD's hard constraints come from counted transitions; with 430 contacts, a
  legal-but-unobserved sequence will be forbidden outright. Need a check for
  unseen transitions on the validation fold before trusting hard constraints.
- Adjacency rules, coordinate normalization and the missing-keypoint policy are
  not in the UMEG-Net paper; they must be read from its repository or chosen by
  us and recorded.
- The three gold videos share a camera placement. All of the above remains
  model-selection estimation on related footage, not an independent test.

## Sources

- [VNL-STES (CVPRW 2025) paper](https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Nguyen_VNL-STES_A_Benchmark_Dataset_and_Model_for_Spatiotemporal_Event_Spotting_CVPRW_2025_paper.pdf) and [project page](https://hoangqnguyen.github.io/stes/), [code](https://github.com/hoangqnguyen/spot)
- [Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation (UMEG-Net)](https://arxiv.org/pdf/2511.14186), [code](https://github.com/LZYAndy/UMEG-Net)
- [Improving Temporal Action Segmentation via Constraint-Aware Decoding](https://arxiv.org/pdf/2605.10149), [code](https://github.com/LUNAProject22/CAD)
- [Precise Event Spotting in Sports Videos: Solving Long-Range Dependency and Class Imbalance](https://arxiv.org/html/2503.00147)
- [Spotting Temporally Precise, Fine-Grained Events in Video (E2E-Spot)](https://arxiv.org/pdf/2207.10213)
- [Entity-Aware Sequence Transduction for Player-Centric Ball Action Spotting](https://arxiv.org/pdf/2608.01696), [SoccerNet 2026 player-centric baselines](https://arxiv.org/pdf/2606.09679), [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS)
- [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182), [team spotting devkit](https://github.com/SoccerNet/sn-teamspotting)
- [TT3D: Table Tennis 3D Reconstruction](https://arxiv.org/html/2504.10035), [Table Tennis Stroke Detection Using Ball Trajectory Data](https://arxiv.org/pdf/2302.09657)
- [Deep Learning for Sports Video Event Detection: survey](https://arxiv.org/pdf/2505.03991), [Temporal Action Segmentation: analysis of modern approaches](https://arxiv.org/pdf/2210.10352)
- [Volleylitics](https://github.com/GYdevy/Volleylitics), [volleyball_analytics](https://github.com/masouduut94/volleyball_analytics), [VolleyVision](https://github.com/shukkkur/VolleyVision), [Automated Referee Whistle Sound Detection](https://www.researchgate.net/publication/49607355_Automated_Referee_Whistle_Sound_Detection_for_Extraction_of_Highlights_from_Sports_Video)
