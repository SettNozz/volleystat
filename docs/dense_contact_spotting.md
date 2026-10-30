# Dense contact spotting with constraint-aware decoding (T3 + T2)

First experiment from the plan in
[`docs/event_detection_research_2026-10.md`](event_detection_research_2026-10.md):
replace the hand-written rally state machine with a dense per-frame model plus a
decoder whose structure is **counted from human labels** instead of written by
hand. No new labelling and no new extraction were needed.

Runs: `results/dense_contact_spotting_20261027_v1` (first pass, frozen
ball-only analyses), then `results/dense_contact_spotting_20261028_v2` on
pose-enabled analyses with the matched control
`results/dense_contact_spotting_20261028_v2_nopose`. Nothing was promoted; the
production path, datasets and existing checkpoints are untouched.

The numbers in **Results** below are the first pass. The pose-enabled rerun and
the fair FSM baseline are further down and supersede its FSM comparison.

```powershell
python scripts\evaluation\run_dense_contact_spotting.py --help
python scripts\evaluation\run_dense_contact_spotting.py `
  --output results\dense_contact_spotting_new_version --epochs 40 --patience 10
```

The runner refuses an existing output path, hashes every frozen input into
`preflight.json`, and writes per-fold probabilities, decoded labels, events,
metrics and an experimental checkpoint.

## Label space

Single-label, every frame covered, nine classes
(`action_recognition/rally_spotting/dense_contacts.py`,
schema `dense_phase_contact_targets_v1`):

```text
0 between       1 serve_flight   2 rally_live
3 serve  4 reception  5 set  6 dig  7 attack  8 block
```

Phases come from the human segment labels already exported in
`data/full_video_gold/<stem>/derived_7d09206_v2/dense_targets.npz`; the
exporter's `unlabeled` frames stay masked.

Contacts come from the audited review `data/contact_spatial_gold_v1/reviews.json`,
which supplies a **verified** contact frame. That matters: the original
full-video annotation frame is only a proposal and differs for 155 of the 410
reviewed events (median absolute drift 1 frame, maximum 62). Review audit as
consumed by this experiment:

| | |
|---|---:|
| Verified contacts kept | 397 |
| Merged duplicate frames | 7 |
| Authorized four-class conflict resolved | 1 |
| Raw type ambiguous, contact masked | 7 |
| Excluded `wrong_event` | 4 |
| Excluded uncertain timing (occluded / out of frame) | 2 |

Per source after masking: Khe 148, j0 121, j3 119 verified contacts.

Three masking rules keep weak evidence out of the loss:

* a contact whose time the reviewer left uncertain is masked, never background;
* a human touch the review set does not cover (non-core `free_ball`, `other`)
  is masked, never background;
* a guard ring of 3 frames around each widened contact segment is masked, so a
  one-frame timing disagreement is not trained against.

The one four-class conflict (`Khe4P8QcJkg_5_4k:3761`) applies the resolution
already authorized in `action_recognition/rally_spotting/spatial_labels.py`
verbatim rather than inventing a new rule.

**Tolerance targets.** Each verified contact is widened to +/-2 frames. Without
this, contacts are 421 frames in 61,822 (0.68%); after widening they are about
1% and learnable. **Foreground weighting** then scales class weights against the
largest class, capped at 25, which lifts every contact class to 25 against
`between` at 1.0 — the phase classes otherwise swallow the loss.

## Model and protocol

* Features: 22 per-frame channels from `rally_spotting.prepare`
  (`resolution_normalized_v2`), built from the frozen conservative ball analyses
  in `results/gold_v1_ball_reconstruction_conservative`. Reconstructed ball
  points stay derived inputs with their provenance; they are never detector
  ground truth.
* Model: the existing `RallySpottingTCN` (dilated residual TCN) with 9 softmax
  classes, hidden 64, dilations 1-64, cross entropy with `ignore_index`.
* Folds: leave-one-complete-source-video-out, three folds, seed 42, up to 40
  epochs, patience 10, AdamW 1e-3 / 1e-4, 240-frame windows at stride 30 with
  contact-bearing windows repeated three times.
* Normalization statistics are fitted on the training fold only.

## Decoder

`action_recognition/rally_spotting/constrained_decode.py`
(schema `constraint_aware_decode_v1`) follows
[arXiv 2605.10149](https://arxiv.org/pdf/2605.10149):

```
Y* = argmax  sum_t log P[t, y_t] + w * sum_t log Conf(y_{t-1} -> y_t)
```

with hard constraints on valid start labels, valid end labels, observed
transitions and per-class segment duration. `Conf(A->B) = Count(A->B)/Count(A)`,
counted on the **training fold only**.

Three deliberate deviations from the paper, each with a reason:

1. Duration bounds are in **frames**, not as a fraction of video length. Our
   classes have physical durations — a contact is a few frames at 29.97 fps
   whatever the match length — so normalising by length would couple them to how
   long a video happens to be.
2. The terminal minimum-duration check is available but **off by default**: the
   last segment is cut off by where the file ends.
3. `between` and `rally_live` have **no upper duration bound**: they can
   legitimately run longer than anything observed in two training videos.

Constraints are counted on an **unguarded** copy of the targets. The guard ring
that protects the loss also separates every contact from its phase, so with
guarded targets `rally_live -> attack` is never observed and the decoder can
emit no contact at all. This was a real bug found in the first run.

As in the paper the per-cell duration is carried greedily rather than being part
of the Viterbi state, so the result is optimal over paths reachable under that
bookkeeping, not provably the constrained optimum.

## Results, first pass (frozen ball-only analyses)

Three conditions over the same per-frame probabilities. `peaks` is per-class NMS
peak picking with **its threshold tuned on the held-out video** so the control is
not handicapped; the constrained decoder has no threshold to tune. `fsm` is the
frozen `IntegratedVolleyballFSM` events already stored in the baseline analyses.

Mean macro F1 across the three folds, by native-frame tolerance:

| View | Condition | @1 | @2 | @4 | @5 | @10 | @15 |
|---|---|---:|---:|---:|---:|---:|---:|
| raw six-class | peaks | 0.3392 | 0.5208 | 0.6190 | 0.6190 | 0.6190 | 0.6190 |
| raw six-class | **constrained** | 0.3606 | 0.5451 | 0.6587 | **0.6654** | 0.6654 | 0.6654 |
| core four-class | peaks | 0.3964 | 0.6096 | 0.7177 | 0.7177 | 0.7177 | 0.7177 |
| core four-class | **constrained** | 0.4418 | 0.6588 | 0.7761 | **0.7856** | 0.7856 | 0.7856 |
| core four-class | fsm | 0.0055 | 0.0138 | 0.0339 | 0.0339 | 0.0595 | 0.0633 |

Per fold at +/-5 frames (macro F1), and the micro counts that explain it:

| Held out | best epoch | peaks thr | truth | raw: peaks -> constrained | core: peaks -> constrained |
|---|---:|---:|---:|---|---|
| Khe4P8QcJkg_5_4k | 8 | 0.55 | 148 | 0.6219 -> 0.6612 | 0.7329 -> 0.8039 |
| j0gRhgFCgIs_4_4k | 5 | 0.60 | 121 | 0.6714 -> 0.7265 | 0.7657 -> 0.8282 |
| j3BO9GgUFBI_3_4k | 17 | 0.50 | 119 | 0.5637 -> 0.6084 | 0.6546 -> 0.7246 |

| Held out | condition | TP | FP | FN | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| Khe | peaks | 117 | 57 | 31 | 0.672 | 0.791 |
| Khe | constrained | 122 | 32 | 26 | 0.792 | 0.824 |
| Khe | fsm | 5 | 79 | 143 | 0.060 | 0.034 |
| j0 | peaks | 103 | 46 | 18 | 0.691 | 0.851 |
| j0 | constrained | 103 | 27 | 18 | 0.792 | 0.851 |
| j0 | fsm | 5 | 97 | 116 | 0.049 | 0.041 |
| j3 | peaks | 106 | 117 | 13 | 0.475 | 0.891 |
| j3 | constrained | 103 | 79 | 16 | 0.566 | 0.866 |
| j3 | fsm | 2 | 70 | 117 | 0.028 | 0.017 |

The mechanism is precision, not recall: the counted grammar removes 44%, 41% and
32% of the control's false positives while recall stays within 2.5 points. That
is the behaviour the paper reports, reproduced here on volleyball.

Per-class raw F1 at +/-5 frames, constrained, best and worst fold: `serve`
0.954 / 0.818, `reception` 0.857 / 0.800, `set` 0.893 / 0.622, `dig` 0.722 /
0.571, `attack` 0.764 / 0.676, `block` 0.182 / 0.000. `block` fails everywhere:
26 instances in the whole dataset, 2-7 per fold, and the review marks several as
ambiguous between block and attack.

## Pose-enabled rerun and the fair FSM baseline (2026-10-28)

The first run had to use the only full-video analyses that existed, which were
ball-only: `p` empty, all eight `pose_f` channels zero, `net_y = 1080.0` from
before calibration. The pipeline was therefore rerun on the three gold videos
with poses, and the experiment repeated.

```powershell
python analyze_video.py --video "data\labeling_intake\2026-07-31_youtube_4k\<stem>.webm" `
  --output results\gold_pose_fsm_20261028_v1 `
  --calibration configs\court_calibration_youtube_4k_setup_1.json `
  --no-net-detection --no-video --no-model-registry --ball-reconstruction conservative
```

`--reconstruction-affects-fsm` was deliberately not passed, so the FSM still
sees observed ball only, exactly as in the frozen baseline. The calibration is
built from the named viewer setup by
`scripts/evaluation/build_setup_calibration.py`; the root `court_calibration.json`
was measured on a different camera and does not describe these videos.

What the rerun produced, per source: poses on every frame (21,300 / 19,215 /
21,300), `net_y = 849.0`, `net_floor_method = calibrated_court_midline_fallback`,
and 213 / 176 / 235 FSM contact events against 84 / 102 / 72 before.

**The rerun changed the ball trajectory as well.** Against the frozen analyses:
median displacement 1.1-1.3 px, 90th percentile 3.7-4.7 px (contact-preserving
smoothing, which the frozen run did not have), a few hundred frames of coverage
difference per video, and a handful of frames where a different candidate was
selected (max displacement ~1300 px). So a pose-enabled run cannot be compared
with the frozen one to attribute anything to poses. A matched control was added
for that: `--zero-pose-channels` blanks the eight pose channels on the *same*
analyses.

### Three runs, mean macro F1 at +/-5 frames

| Run | raw peaks | raw constrained | core peaks | core constrained | core FSM |
|---|---:|---:|---:|---:|---:|
| frozen trajectory, no poses (`20261027_v1`) | 0.6190 | 0.6654 | 0.7177 | 0.7856 | 0.0339 |
| new trajectory, poses blanked (`20261028_v2_nopose`) | 0.6488 | 0.6643 | 0.7749 | 0.7773 | **0.1561** |
| new trajectory, poses live (`20261028_v2`) | **0.6887** | **0.7013** | 0.7528 | **0.7897** | **0.1561** |

Three conclusions, and each contradicts something the first run suggested.

**1. The FSM, measured fairly, scores 0.1561 — not 0.0339.** With poses it finds
4.6x more of the human contacts. It is still about five times worse than the
dense model on the same videos and the same metric:

| Held out | F1 | TP | FP | FN | precision | recall | events | truth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Khe4P8QcJkg_5_4k | 0.1912 | 43 | 170 | 105 | 0.202 | 0.291 | 213 | 148 |
| j0gRhgFCgIs_4_4k | 0.1529 | 32 | 144 | 89 | 0.182 | 0.264 | 176 | 121 |
| j3BO9GgUFBI_3_4k | 0.1241 | 35 | 200 | 84 | 0.149 | 0.294 | 235 | 119 |

Its failure is not a missing-detections failure: it emits *more* events than
there are contacts (213 against 148) and still recovers under 30% of them. Per
class it is worst exactly where the structure is clearest — `serve` F1 is 0.03,
0.00, 0.00, while `pass` reaches 0.28-0.32. The state machine fires rally
touches liberally and almost never puts a serve on the right frame.

**2. The jump in the peak-picking control came from the trajectory, not from
poses.** Blanking the pose channels on the new analyses still gives core peaks
0.7749 against the frozen run's 0.7177. Contact-preserving smoothing is doing
that work.

**3. The eight pose channels add little, and only in the raw view.** Against the
matched control, poses move core four-class constrained from 0.7773 to 0.7897
(+0.012) and raw six-class constrained from 0.6643 to 0.7013 (+0.037). The whole
difference is concentrated in `block`, whose raw F1 per fold goes 0.25 / 0.00 /
0.29 without poses to 0.40 / 0.40 / 0.46 with them — blocks need player evidence
and nothing else does. `block` collapses into `attack` in the four-class view,
which is why the gain nearly vanishes there. `dig` moved the other way on one
fold (0.67 -> 0.60). This is consistent with the earlier finding that adding
pose channels to the clip classifier hurt it.

### The decoder's advantage shrinks as the features improve

Core four-class at +/-5 frames, per fold, peaks -> constrained:

| Run | Khe | j0 | j3 |
|---|---|---|---|
| frozen trajectory, no poses | 0.7329 -> 0.8039 | 0.7657 -> 0.8282 | 0.6546 -> 0.7246 |
| new trajectory, poses blanked | 0.7604 -> 0.8041 | 0.8143 -> 0.8293 | 0.7500 -> **0.6985** |
| new trajectory, poses live | 0.8084 -> 0.8378 | 0.7874 -> 0.8099 | 0.6627 -> 0.7215 |

The mean margin was +0.068 on the frozen features, +0.002 on the matched control
and +0.037 with poses. The counted grammar works by removing false positives, so
the more precise the per-frame model already is, the less there is for it to
remove; on the control's j3 fold it lost outright, the first fold it has lost.
The tuned threshold tells the same story — it rose from 0.50-0.60 to 0.45-0.85,
meaning an aggressive threshold now does much of what the grammar did.

The decoder still wins every fold in the run that matters (new trajectory, poses
live) and still wins on the mean in all three views. But "constrained decoding
beats peak picking" is weaker than the first run implied: it beats a *weak*
control reliably and a *strong* control narrowly.

### Limits specific to the rerun

* Best epoch and the control's threshold are still chosen on the held-out video.
  Model-selection estimates, not an independent test.
* One seed. The per-fold swings here (j3 control losing by 0.05) are within what
  a seed sweep might produce, and no seed sweep was run.
* The FSM number is now fair with respect to poses and calibration. It is still
  one FSM configuration on three videos from one camera.
* Experimental checkpoints in both v2 directories are not production-loadable
  and nothing was promoted.

## Honest limits that apply to every run

* Best epoch and the control's threshold are both chosen on the held-out video.
  Every number above is a model-selection estimate, not an independent test.
* Spotting F1 is **not** comparable with the ~0.92 contact-centred clip macro F1
  of the current production checkpoint: the clip task is handed the contact
  frame, this task has to find it.
* Hard constraints really do forbid things. On the Khe fold the held-out labels
  contain 9 `between -> serve_flight` and 2 `rally_live -> serve` transitions
  that the two training videos never showed, so the decoder could not produce
  them. It still won that fold, but with 430 labels this is the failure mode to
  watch, and `unseen_transitions` is reported per fold for exactly that reason.
* Three videos, one camera placement, one seed. No seed sweep was run.
* `block` has too few examples to learn well: 26 instances in the whole dataset.
  Pose channels lift it from roughly 0.1 to roughly 0.4 raw F1, which is a real
  gain over nothing but still the worst class by a wide margin.

## Next

* The run was since split into its detection and typing halves --
  [`docs/contact_detection_vs_typing.md`](contact_detection_vs_typing.md).
  Detection is the weak half at 0.8605 against typing's 0.9208, and the j3 fold
  turned out to be a warm-up coverage gap rather than a weak fold. Read that
  before picking the next experiment.
* Add the audio whistle channel (T4) to the same decoder. On this evidence the
  per-frame model, not the decoder, is where the remaining error is: the
  grammar's margin is already small against a precise model.
* A seed sweep, so the per-fold swings can be told apart from noise.
* `dig` is the weakest core class everywhere (0.47-0.62) and poses did not fix
  it; it is the next thing worth a targeted look.
* The multi-entity graph (T1) replaces only the feature side; the decoder,
  targets and metrics here are reusable as-is.

Tests: `tests/test_dense_contacts.py`, `tests/test_constrained_decode.py`,
`tests/test_setup_calibration.py`.
