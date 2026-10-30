# Isolated contact-author experiments

This branch experiments with the team/player responsible for each **existing**
spotter event. It preserves the 103 JIZ3 source events, their types, detector
poses, stable identities, ball and original analysis JSON. It does not detect
previously missed contacts, retrain the contact spotter/ReID, or establish a
hidden hand movement as measured evidence.

The user explicitly authorized learning, with an independent checkout so a
second process could continue committing. Branch `codex/contact-author-learning`
uses its own worktree/index, based on `202bd7a`. All edits, CPU training, model
artifacts and the port-5053 viewer live there. Shared data/results are read
only; production/default branches and the other viewer are unchanged.

## What was compared

Six frozen geometric controls compare body proximity, hands/forearms,
track deduplication, ball-motion changes, two-sided identity evidence,
action-specific anatomy and estimated-joint evidence. One discovered issue was
a raw track competing against itself when its P-ID became unknown on a nearby
frame. Grouping this evidence avoids a false rival without relabelling a pose.
Image geometry alone still chose the correct team only about 65–66% of the time.

The learned comparison uses 397 human side labels from
`data/contact_team_gold_v1/labels.json`: 205 far (`team_a`), 192 near (`team_b`),
over `Khe4P8QcJkg_5_4k`, `j0gRhgFCgIs_4_4k`, `j3BO9GgUFBI_3_4k`. Nine unsure
answers are excluded. There are **zero human player/contact-author labels**.
Human labels cover serve, reception, pass and attack, not block.

Feature schema `contact_author_side_frozen_v1` has 366 values/availability
channels: ranked observed candidates on both sides, hands/forearms/body
distances, pose configuration and confidence, body-relative ball position,
short-window player/ball motion, contact type and diagnostic image position.
Human contact xy is never used. Missing/off-camera/restarted ball observations
break motion support. No ball or skeleton coordinate is interpolated here.
Existing interpolated-ball provenance is retained in candidate evidence.

Raw track/player IDs, source names and absolute match times do not enter the
numeric model input. They are metadata for candidate continuity and review.
Known and unknown observations of one raw track are grouped locally; rejected
identity conflicts remain unknown. A same-ID two-sided proposal additionally
needs consistent observed motion and strong existing appearance anchors.
No gallery, source ID or source pose is changed.

Three fixed model families (logistic, ExtraTrees, histogram gradient boosting)
compare geometry-only, typed and layout feature views. Every fold holds out
one entire video; normalization fits the other two only. There is no random
clip split, test-fold early stopping or dataset/checkpoint refresh.

Team accuracy in this matched development comparison:

| Approach | Team accuracy |
|---|---:|
| Body baseline over frozen pipeline ball | 64.9% |
| Hands/forearms + local track deduplication | 65.8% |
| Hands + ball motion / temporal identity | 65.1% |
| Action-specific geometry (given human type) | 66.2% |
| Logistic, typed | 82.6% |
| ExtraTrees, typed — selected | **84.6%** |
| Histogram gradient boosting, typed | 83.1% |
| Weak actor learner, geometry | 79.1% |
| Weak actor learner, typed | 77.3% |

Full nine-model metrics are in the run's `comparison.json`. Layout-sensitive
models are diagnostic, excluded from automatic selection because the three
gold sources share a camera. These figures use pipeline ball coordinates,
so they are not the earlier nearest-player number measured with human ball xy.

The selected ExtraTrees model has macro F1 0.84596. Per held-out video accuracy:
89.4%, 81.0%, 82.5%. Recomputing test features at contact times shifted by -2/+2
frames gives 82.4%/83.6% accuracy. At the fixed score cutoffs <=0.35 or >=0.65,
accuracy is 92.6% with 75.1% coverage. These scores are **not calibrated touch
probabilities**. Audited contact timing/type is supplied in this comparison;
JIZ3 uses model-predicted timing/type and a different, unlabelled camera.

These are model-selection estimates on the same three-video corpus, not an
untouched final test. Final fits on all 397 labels have **no held-out score**.
None of these metrics measures the correct P1–P7 author or block attribution.

## Weak actor learning

The additional `contact_author_team_mil_v1` model shares one small actor scorer
across both teams. It pools candidate scores by team and trains only against
the human team label. The positive actor is latent: no nearest-player guess
is stored or treated as a human person target. Geometry and typed views use
three seeds (42/43/44), fixed 140-epoch budgets, train-only normalization and
whole-video held-out folds. All operations are on CPU with two threads.

Its best team result was 79.1%, below ExtraTrees. A predeclared review tradeoff
would prefer learned actor ranking only within 0.02 team macro F1 of the best
flat model; it failed that condition. Both weak variants remain available for
comparison, with explicit unverified individual-author status.

## Reproduce in a separate worktree

The completed flat run is `results/contact_author_learning_20261030_v3`.
The combined reviewer result is under
`results/contact_author_mil_20261030_v1/viewer_result` in this worktree.
Choose fresh versioned outputs for new runs; old runs/models are refused.

```powershell
# Set this to the shared checkout containing the frozen local input artifacts.
$sharedRoot = 'C:/Users/illya/Documents/volleyball_analitics/volleystat'
python scripts/evaluation/run_contact_author_learning.py --help
python scripts/evaluation/run_contact_author_learning.py `
  --annotations "$sharedRoot/results/JIZ3exHe3hA_pose_completion_20261030_v1/JIZ3exHe3hA_4k_annotations.json" `
  --team-labels "$sharedRoot/data/contact_team_gold_v1/labels.json" `
  --heuristic-bundle "$sharedRoot/results/JIZ3exHe3hA_contact_methods_20261030_v1/JIZ3exHe3hA_4k_player_contact_experiment.json" `
  --output results/contact_author_learning_NEW
python scripts/evaluation/run_contact_author_mil.py --help
python scripts/evaluation/run_contact_author_mil.py `
  --base-run results/contact_author_learning_NEW --output results/contact_author_mil_NEW
python viewer.py --port 5053
```

Before fitting, each runner prints and verifies manifest/features paths,
source hashes/videos, 397 labels, provenance, schema and split policy. Frozen
features, source-code contract hash, per-fold predictions/metrics, package
versions and experimental models are stored in the fresh output. MIL verifies
its recomputed features against the frozen flat dataset before fitting.
The validated local environment uses scikit-learn 1.7.2 and NumPy 2.4.3;
existing PyTorch runs the CPU actor experiment. No package/model download or
environment change is part of the workflow.

## Visual review

Open:

```text
http://localhost:5053/load?video=JIZ3exHe3hA_4k&results_dir=results/contact_author_mil_20261030_v1/viewer_result
```

**Review approach** switches between 11 frozen reports: six geometric controls,
the best view from each of three flat model families and two weak actor learners.
Default is ExtraTrees typed. Its JIZ3 output has 24 assigned near-player
proposals, 21 other-side proposals and 58 unresolved, from the same 103 events.
This is coverage, not accuracy or a complete count of true touches.

Select P1–P7 to inspect that player's list. **Include uncertain suggestions**
also shows likely authors that failed a score/margin/identity check; these rows
are amber and excluded from totals. Clicking a contact pauses at its proposed
evidence frame. An amber bbox identifies the candidate at exactly that frame,
including a separately marked temporal identity proposal; source IDs/poses are
not relabelled. Use Play/frame controls to inspect the surrounding movement.

Every report is in `<stem>_player_contact_experiment.json`, bound to SHA256 of
the two byte-identical analysis files. The viewer rejects a changed pair,
mismatched event index/frame/type or invented human verification. Without the
sidecar it retains the old baseline. No model is loaded to change approaches.

Correct the author through the existing dropdown. Corrections append to the
source-bound review log and apply across approaches to the same source event.
Human overrides suppress old conflicting suggestions. Reset preserves history
and restores the currently selected approach's proposal. No review is silently
promoted to training data; event type/timing remains model-derived.

The remaining uncertainty is real: no individual-author labels exist, weak
ReID can remain unknown, occluded hands may be unsupported, and source spotter
events can be missed/false/mistyped. Visual human feedback is the next evidence
needed before claiming accurate personal totals or promoting any model.

## Validation

The targeted Python regression suite passed 176 tests across these experiments,
player contacts/identity/pose/cache handling, spotter/replay, registry and viewer
workflows. Ten JavaScript tests passed for selection, method changes, candidate
focus, human overrides and derived-pose display. A live port-5053 browser check
confirmed method switching, P4 filtering and the paused evidence-frame overlay
without console errors or creating human reviews. SHA256 checks confirmed the
three gold analyses, human side labels and original/exported JIZ3 pair unchanged.
`graphify update .` rebuilt this worktree's local code graph; the shared graph
was not modified.
