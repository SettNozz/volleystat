# Volleyball contact detection and actor experiments

This experiment compares **when a touch happens**, **which team touched**, and
unverified **which observed player is the candidate**. Those are separate
questions. It does not report personal touch accuracy: there are no human
contact-author/P1–P7 labels. All work stays in the independent
`codex/contact-author-learning` worktree. No production model is promoted.

## Volleyball research behind the comparison

- [Gomez et al., PLOS ONE 2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0111730)
  locate beach-volleyball contacts at intersections of fitted flight segments.
  They also report that serve contact cannot be recovered by that intersection
  mechanism. Here, two short quadratic image-space fits provide a change score.
  This is an adaptation, not their tracker or a claim of 3D reconstruction.
- [Maksai et al., CVPR 2016](https://arxiv.org/abs/1511.06181)
  jointly model players, ball trajectories and interaction states with physical
  constraints, including volleyball sequences. Their 3D/MIP system is not
  reproduced here. The transferable idea is to score motion and the possible
  actor together, rather than accept the nearest body at one instant.
- [VNL-STES, CVPR Workshops 2025 — authors' project](https://hoangqnguyen.github.io/stes/)
  learns volleyball event time and image location jointly. Its image backbone,
  bidirectional GRU and spatial head need inputs/training different from our
  frozen caches. We compare a lightweight joint time/actor hypothesis instead;
  we do not download their dataset/weights or claim to run STES.
- [Cheng, Automatic Data Volley](https://link.springer.com/article/10.1007/s40747-022-00752-3)
  combines temporal event relationships with spatial team information. The
  experiment here learns a soft team-transition prior from training videos.
  It does not substitute airborne-ball court projection for team evidence.
- [FIVB indoor rules 2025–2028](https://www.fivb.com/wp-content/uploads/2025/01/FIVB-Volleyball_Rules2025_2028-EN-v05.pdf),
  rules 9.2 and 14.4, allow contacts with different body parts and exclude a
  block from the team's three hits. Beach and indoor block counting differ.
  Therefore neither “hands only” nor a hard three-contact chain is imposed.
  The sequence model makes soft proposals, not referee decisions.

## Frozen inputs and fair comparisons

- Three pose-enabled analyses in `results/gold_pose_fsm_20261028_v1`;
  exactly the same annotation hashes as the cached spotter control.
- Audited **397 distinct contact timestamps** from
  `data/contact_spatial_gold_v1/reviews.json`, including contacts whose action
  type is ambiguous. Human phase coverage and uncertain/uncovered touches are
  used only for evaluation masks, never as an inference gate.
- `data/contact_team_gold_v1/labels.json` contains 406 records: 397 conclusive
  side labels and nine unsure answers. These are not the same set as the 397
  distinct timing targets: duplicates and timing uncertainty differ.
- Exact same-frame team records become one derived context node, retaining all
  source IDs. Unsure sides, uncertain timing, serves and gaps over four seconds
  break context. Human sources are unchanged. The known-time control is scored
  on the original 397 side-labelled examples for comparison with the old model.
- Existing 366D frozen team features and OOF ExtraTrees scores from
  `results/contact_author_learning_20261030_v3`. Human contact xy, person IDs,
  source names and absolute match times are not numeric inputs.
- Detection reference: cached `constrained` events from
  `results/dense_contact_spotting_20261028_v2`, not a new inference or a newly
  measured production checkpoint. These folds share the frozen pose/ball input.

Every fold holds out one complete video. Flight/interaction thresholds and
transition counts use the other two videos only. NMS radii are 2 or 5 frames;
fixed threshold grid and sequence strength 0.7 are recorded in code/results.
No random adjacent-clip split or held-out early stopping is used. Final models
fit all available side labels only for transfer to unlabelled JIZ3; final fits
have no held-out score. These are model-selection estimates on three videos,
not an untouched test or a per-player/JIZ3 accuracy measurement.

## What the methods actually do

1. **Piecewise flight:** fit quadratic image paths before/after each frame;
   score velocity change, fit improvement and continuity. Every support frame
   must have an observed-backed ball. Off-camera/missing/interpolated or
   reconstructed points and track restarts invalidate support. It cannot
   prove a hidden touch, gravity in world coordinates or distinguish floor/net
   impacts from player touches by itself.
2. **Moving-limb interaction:** compare ball/body distances before and after
   the same raw player track, with a swept relative hand/ball segment and a
   proximity minimum. Observed wrists/forearms are preferred; body evidence
   remains a penalised fallback. No missing limb or ball is generated.
3. **Joint flight/limb:** geometric mean of those two dense scores. It can
   propose a new timestamp independently of the spotter.
4. **Spotter + joint timing/actor:** shift an existing contact at most three
   frames and re-rank observed candidate actors with flight/interaction
   evidence. A neighbour bound preserves close attack/block proposals.
5. **Consensus:** retain every source event and add nonoverlapping joint peaks.
   It tests whether the independent detector recovers more than it falsely adds.
6. **Sequence side attribution:** forward/backward inference over two team
   states, using the local learned score and training-fold probabilities of
   same/opposite team given adjacent action types and a coarse time-gap bin.
   The selected `spotter_sequence` ablation changes side evidence only;
   candidate ranking, source contact frames/types and source identities remain
   the same as the local-author control. Joint actor re-ranking is separate.

## Measured results

Type-agnostic contact detection, one-to-one timestamp matching at ±5 frames:

| Method | Pooled detection F1 |
|---|---:|
| Existing cached spotter | **87.35%** |
| Piecewise flight | 46.43% |
| Moving-limb interaction | 33.19% |
| Joint flight/limb | 44.96% |
| Spotter + bounded joint retiming | 87.35% |
| Spotter + extra joint peaks | 74.73% |

The 87.35% is for this audited time-only target/mask protocol, including
ambiguous action types. It is not a replacement for previously reported typed
spotting scores or a claim that the production checkpoint was re-evaluated.
Results at ±2/±10 frames, precision/recall and the separate exclusion of human
`non_game` intervals are in `comparison_final.json`. No human `non_game` gate
is applied to JIZ3 or to the proposed detections.

Team comparison using **the spotter's predicted times and predicted types**:

| Metric | Local ExtraTrees | With sequence |
|---|---:|---:|
| Correct team on 365 matched contacts | 84.66% (309/365) | **93.42% (341/365)** |
| Joint contact detection + correct team F1 | 73.75% | **81.38%** |
| Joint F1 excluding human `non_game` (diagnostic only) | 77.15% | **85.14%** |

Here each held-out video's side classifier is fitted on the other two videos;
only those training videos supply transition counts. Matching happens by time
before checking team answers. Matched contacts with unknown human team are
excluded from the joint-team denominator, not counted as correct. A wrong team
costs both an FP and FN in joint F1. Per-fold matched-team accuracy changes:
92.0→92.8%, 80.3→96.6%, 80.0→90.9%.

The additional known-human-time/type control improves 84.63→93.20% team
accuracy. A preliminary v1 treated duplicate timing records as repeated context
events; v2 groups them and normalizes unsure/timing-uncertain nodes. v3 keeps
v2's verified numerical features/metrics and exports the isolated sequence-side
ablation. Old runs remain preserved and are not the selected reviewer output.

Conclusion for review: retain the spotter's timing; try sequence side evidence.
The independent geometric detectors add too many false positives. They remain
available for inspecting failure modes and missed-contact candidates, not as an
automatic replacement. Sequence/geometry still cannot verify an occluded person
or fix weak/missing ReID. Block team accuracy is not measured by these side labels.

## Review JIZ3

Completed output in this worktree:
`results/volleyball_contact_lab_20261030_v3/viewer_result`.

```text
http://localhost:5054/load?video=JIZ3exHe3hA_4k&results_dir=results/volleyball_contact_lab_20261030_v3/viewer_result&contact_lab=1
```

The read-only lab offers seven variants, P1–P7 filters, **Author changes only**,
**New detections only**, frame controls and source-bound actor evidence. Clicking
pauses at the actual observed candidate frame; detector/evidence/source frames
are displayed separately. Untyped new peaks say `contact`, with no invented
reception/dig/attack/block label. Unknowns and uncertain suggestions do not enter
assigned-player totals. The amber bbox is a proposal, not relabelled source pose.

Selected sequence variant: same 103 source events, 28 assigned near-player
proposals, 30 other-side, 45 unresolved. The local control has 24/21/58.
Consensus contains 156 proposals, including 53 extras; more proposals do not
mean more true touches. Individual-author correctness needs human review.

The JSON pair is byte-identical to the source analysis. A separate
`<stem>_contact_lab.json` is SHA256-bound to that pair. The API refuses changed
sources, changed source type/frame provenance, unknown roster IDs or invented
human verification. No review is silently saved or promoted as training labels.
The original viewer, event lists/FSM/rallies, ball trajectory and pose IDs stay
unchanged. No full MP4 is rendered.

## Reproduction and verification

```powershell
$sharedRoot = 'C:/Users/illya/Documents/volleyball_analitics/volleystat'
python scripts/evaluation/run_volleyball_contact_lab.py --help
python scripts/evaluation/run_volleyball_contact_lab.py `
  --shared-root $sharedRoot `
  --base-run results/contact_author_learning_20261030_v3 `
  --annotations "$sharedRoot/results/JIZ3exHe3hA_pose_completion_20261030_v1/JIZ3exHe3hA_4k_annotations.json" `
  --output results/volleyball_contact_lab_NEW
python viewer.py --port 5054
```

The runner refuses an existing output. Before feature computation or fitting it
prints the manifest, feature paths, schemas, source hashes/videos, provenance,
counts and whole-video split. Small ExtraTrees fits run on CPU with two threads;
there is no YOLO, OSNet, SAM, VNL-STES or contact-spotter inference. New numerical
features/checkpoints/results remain local/gitignored and versioned. v3 reused
v2 evidence/metrics after hash validation; its manifest records the reuse source.

Targeted tests cover clean-flight vs impulse, missing/derived/off-camera ball,
track restarts, missing/changed player tracks, duplicate/unsure context, held-out
label non-leakage, sequence resets, close attack/block preservation, evaluation
masks, source binding, viewer links, video selection, player counts and UI filters.

Validation: 138 Python regressions passed before the final viewer-only fixes;
the final 43-test core/viewer subset and 14 JavaScript checks passed afterward.
Live Chrome checks confirmed original-video playback, seven methods, 53 novel
consensus rows, P4/change filtering and exact fractional-fps evidence seeking,
with no console errors. SHA256 checks preserved source analyses/labels and the
v2 metric reuse; selected side-only proposals retain the original candidate
costs. The worktree's graph was updated without touching the shared graph.
