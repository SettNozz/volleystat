# Human contact-author benchmark

The completed author annotation project provides real person targets. This
experiment compares algorithms for **who performed a known contact**. It keeps
contact detection/type models, source ball/poses, authoritative annotations,
the live human database and production defaults unchanged.

## Frozen data and scope

Human project revision **421** contains 403 decisions: 188 named near-side
players, 206 other-team contacts, eight unsure and one not-contact. Confirmed
rosters have 7 / 6 / 6 people; IDs are local to each source video.
`results/contact_author_benchmark_20261031_v1/human_labels.json` freezes the
complete export, decision provenance, rosters, photo references and source hashes.
Its SHA-256 is `fc2313f5fb580c02ae52c7ffd159fbfbe42636dd29d25d449e96ffbda33cfb51`.

394 verified author-or-other-team rows enter the broad comparison. The primary
score uses **382** contacts: exclude eight within ±60 source frames of any photo
reference and four with unverified timing. This prevents directly scoring a
roster anchor such as P7's contact-centred photograph. Primary targets: **184
near-person, 198 other-team**. Unsure/false-event decisions remain in the frozen
audit, not eligible author gold. Unknown algorithm answers count as errors.

All three sources were used to train the contact-type model. This is an actor
development benchmark conditional on human contact timing/type, not independent
contact detection or typing quality. Whole-source actor folds do not establish
unseen-camera/person generalization. Only three sources are available; no
untouched final test exists after these comparisons.

## Inputs and methods

The same `gold_pose_fsm_20261028_v1` annotations supply saved ball/pose observations.
No YOLO, OSNet, SAM, spotter inference, feature-dataset refresh or full video
rendering runs. Compatible OSNet embeddings were not available for these three
sources. Handcrafted HSV descriptors decode 824 requested source frames, once,
and are retained in per-video colour NPZ caches.

Box HSV uses the existing three-band descriptor. Pose HSV masks observed torso
and upper-short regions to reject arms/background; missing joints leave that
descriptor unavailable. Clean views are pooled within continuous local tracks,
split at gaps/teleports. Confirmed photo galleries stay immutable. Photo gallery
setup and unlabeled future views from the same test video are allowed offline
inputs; held-out **contact-author targets** are not used in model fitting.
Raw tracker IDs do not become person truth, and numeric features contain no
roster numbers, raw IDs, video names or absolute contact times.

22 comparisons cover geometric controls, independent photo matching, local
one-to-one Hungarian assignment with unknown slots, action-specific hand evidence,
logistic/ExtraTrees team classifiers, shared linear/MLP person-pair rankers, soft
volleyball contact-sequence priors, and central-time candidate selection. Learned
side models reuse the existing normalized 366-channel feature contract without
image layout channels. Shared rankers see relative photo/pose evidence, not a
numeric P1/P2 class feature. Their three seeds use fixed training budgets.

Each outer fold holds out one complete video. Inner whole-video folds on the
other two choose the method by balanced author accuracy (mean near exact recall
and other-team recall). Normalization, model weights and sequence transition
counts use training sources only. Sequence inference reads held-out event
times/types and model scores, never held-out author values. Weak transitions
reset at serves, unknown timing and long gaps; they do not hard-force three hits.

The central-time ablation chooses each continuous track's closest observed pose
to the audited contact frame, with neighbouring observations as missing-pose/ball
fallbacks. It preserves observed coordinates and original/reconstructed ball
provenance. It cannot bridge off-camera/restarted ball tracks. Global learned
features and saved model weights remain byte-identical; separate context JSON
is bound to their manifest. This fixes the inspected Khe f2134 example, where
window-minimum geometry selected a pass-by near another player at f2131.

## Results, 2026-10-31

Primary strict set; all values count abstentions as errors. The baseline below
is a defined geometric/HSV control, **not measured accuracy of the shipped OSNet
pipeline**, which lacks compatible embeddings for these benchmark inputs.

| Method | Team accuracy | Exact near-person recall | Exact author or other team |
|---|---:|---:|---:|
| Nearest limbs + box HSV | 67.5% | 38.6% | 49.0% |
| Nearest limbs + track HSV / photo anchors | 67.5% | 52.2% | 55.5% |
| ExtraTrees team + one-to-one HSV | 83.2% | 56.0% | 70.4% |
| Joint learned MLP | 79.3% | 47.8% | 69.4% |
| ExtraTrees + sequence + one-to-one HSV | 92.4% | 59.8% | 77.0% |
| Same, central observed actor time | **92.4%** | **60.3%** | **77.2%** |
| Inner-selected algorithm, outer held-out score | 92.4% | 59.8% | 77.0% |

Best fixed comparison `trees_sequence_central`: **295/382** exact outcomes,
**111/184** exact near-person outcomes. Near-person assignment precision is
**61.7%** (111 correct among 180 person predictions), near-author F1 **0.610**.
Overall 77.2% must never be described as individual player accuracy.

| Held-out video | Near targets | Baseline near recall | Best fixed near recall |
|---|---:|---:|---:|
| Khe4P8QcJkg_5_4k | 74 | 54.1% | 63.5% |
| j0gRhgFCgIs_4_4k | 57 | 40.4% | 43.9% |
| j3BO9GgUFBI_3_4k | 53 | 62.3% | 73.6% |

The central-time change adds only **one** net correct contact; its advantage is
not a robust finding. Inner selection picks the central action variant for Khe
and the plain central variant for the other sources, returning 294/382 overall
and 110/184 near-person matches. Fixed-method selection and method design used
this development benchmark, so neither number is an untouched final-test score.

Remaining near errors for the best fixed method: **56 wrong people, 15 wrong
teams, two unknowns**. Without human author boxes, wrong-person errors cannot be
conclusively split into identity mistakes versus selecting another pose. P7 has
only three strict positive examples; no reliable substitution-performance claim
is possible. References do not provide verified active intervals, so all known
people remain available as offline identity hypotheses.

Score ≥0.7 retains 71.5% of all contacts at 86.1% overall exact accuracy, but
near-person precision is only **64.6%**, recall **33.7%** at that cutoff. Scores
are not calibrated probabilities. This is not a high-precision person counter.

## Reproduce and review

Verify command `--help` first. Each output must be fresh; prior versions stay
intact. Print source/feature paths, provenance, schema, counts and split contract
before capture/fitting. From a checkout with access to the local source paths:

```powershell
python scripts/evaluation/evaluate_contact_authors.py prepare `
  --project results/contact_author_annotation_20261031_v1 `
  --output results/contact_author_benchmark_NEW

python scripts/evaluation/evaluate_contact_authors.py central `
  --dataset results/contact_author_benchmark_NEW `
  --output results/contact_author_central_NEW/actor_context.json

python scripts/evaluation/evaluate_contact_authors.py evaluate `
  --dataset results/contact_author_benchmark_NEW `
  --actor-context results/contact_author_central_NEW/actor_context.json `
  --output results/contact_author_evaluation_NEW

# Rule/sequence comparisons over the same frozen inputs can reuse all fold weights.
# Add --reuse-models results/contact_author_evaluation_PRIOR/fold_models.

python -m action_recognition.annotator.contact_author_benchmark_viewer `
  --evaluation results/contact_author_evaluation_20261031_v5 --port 5067
# http://127.0.0.1:5067/
```

Current artifacts: `contact_author_benchmark_20261031_v1`,
`contact_author_central_20261031_v1/actor_context.json`, and
`contact_author_evaluation_20261031_v5` under the isolated worktree's `results/`.
The evaluation contains `REPORT.md`, `comparison.json`, source-bound held-out
`predictions.json` and experimental fold checkpoints. Earlier v1–v4 comparisons
remain local. No final-fit checkpoint was promoted or scored on its own training
set. Source/label preservation was separately checked against revision 421.

The viewer defaults to wrong-person examples. Switch algorithms, video or error
filter; compare human green photo cards with orange predicted cards and observed
skeletons. Contact and candidate frame buttons distinguish event time from
evidence time. Reference photos show their own exact source frame. It has no
label-writing endpoint; original human annotation remains on port 5066.

Tests cover photo descriptors, track discontinuities, abstentions, one-to-one
constraints, source-scoped metrics, training-only normalization/sequence priors,
target-independent central-time selection, source/feature preservation and the
read-only viewer's provenance/range-serving behavior.
