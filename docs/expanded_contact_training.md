# Expanded contact training, 2026-11-06

2026-11-07: the winning author model was promoted at the user's request; see
[production workflow](contact_author_production.md). The experiment records
below remain the immutable pre-promotion history. Event models were not changed.

The completed contact-author projects provide 1,137 usable human author labels
over ten videos: 585 named near-side people and 552 opposite-team contacts.
The immutable snapshots are author project revision 421 and expansion revision
862. The expansion has 743 eligible author labels, 15 rejected contacts and one
unsure author. Annotation source databases and prior datasets remain unchanged.

## Separate label scopes

An author decision confirms only the author. The user explicitly confirmed that
**unchanged model event types have not been verified**. Consequently only explicit
type reviews contribute new event targets. Of 29 type overrides, 28 are eligible
core examples; the rejected contact is excluded. The active causal event dataset
contributes its original 410 core examples, giving 438 examples. Source frames of
new corrections remain model-proposed timing, not audited contact-frame gold.

The six-class experiment retains `serve`, `reception`, `set`, `dig`, `attack`,
`block`. Multiple raw types are ambiguous and excluded. Plain `pass` is valid for
the four-class task but cannot establish whether the raw type was set or dig.
Neither author decisions nor type corrections create exhaustive temporal or
background labels. There is no new continuous spotting precision/recall claim.

## Source and roster policy

Every fold holds out one entire source video; train-only normalization, fixed
budgets, and three seeds (42/43/44) prevent per-contact splitting or test-epoch
selection. Old-three-source controls and expanded-training folds use identical
test cases and features. The new correction-only event subset is selected for
errors and cannot represent average new-video typing accuracy.

Author evaluation reports all cases and the subset farther than 60 frames from
every identity reference photo. The latter also supplies training author labels.
Roster photos are permitted setup inputs on each test video; contact labels are
not. Source names, P numbers, absolute contact time and raw track numbers are not
numeric model features. Numeric candidate ordering is not a person feature.

`TcUru_ou4CQ_4k` is the user-confirmed five-person training exception. Its five
actual roster IDs compete; no P6 or sixth person is fabricated. Other videos use
six-person court policy with separate roster entries for substitutes. Roster
length can exceed six and is never used as the on-court cap.

## Cache and related commit

`9b0b581` accelerates raw inference and decoding, not training or target semantics.
These experiments read saved ball/pose/spotter results. Their only potentially
missing inference stage is local OSNet photo/pose crops. They explicitly use the
commit's batch size 128. Existing valid crops are read before any source decode.

JIZ3's imported OSNet stage used an encoder file with LF endings. The current
file has CRLF endings. The experimental read-through requires exact equality
after newline normalization **and** identical runtime, weights and settings.
It retains the original stage/import provenance, never copies its outputs into
a different signature, and computes genuinely missing crops only. The strict
cache audit and subsequent missing-stage counts remain in run logs.

## Run and history

All commands run in the isolated contact-author-learning worktree. Verify
`python scripts/evaluation/train_expanded_contacts.py --help` first.

```powershell
python scripts/evaluation/train_expanded_contacts.py snapshot `
  --run results/contact_expanded_training_NEW `
  --projects results/contact_author_annotation_20261031_v1 `
             results/contact_author_expansion_20261101_v2
python scripts/evaluation/train_expanded_contacts.py prepare --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py authors --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py prepare-refined --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py authors-refined --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py prepare-all-sides --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py authors-all-sides --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py authors-listwise --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py authors-routed --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py authors-soft-control --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py events --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py events-raw --run results/contact_expanded_training_NEW
python scripts/evaluation/rescore_expanded_events.py --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py final-author --run results/contact_expanded_training_NEW
python scripts/evaluation/train_expanded_contacts.py report --run results/contact_expanded_training_NEW
```

Preparation resumes only sealed per-source features, checking their hashes.
Training stages require fresh output directories. History retains the frozen
human JSON, online SQLite backups, source/project hashes and revisions, feature
matrices, fold membership, seeds, predictions, metrics, training losses and
experimental model files. Source analyses and original human labels are read
only. No experiment updates the production registry, checkpoint or viewer.

Author models compare geometry, hard gallery-assigned person HGB/ExtraTrees and
soft appearance/pose HGB. A training-source-only team HMM is shared across
rankers. This uses local OSNet photo assignment, not the former full-video
production-identity pipeline; historic 85.87% numbers are not a matched control.
Four-class typing compares pooled ExtraTrees and the production multiscale
Transformer architecture (50 fixed epochs); six-class typing adds a small
temporal CNN (35 fixed epochs). All use probability averaging over three seeds.

Scores from these folds are development comparisons, not an untouched final test
or the quality of a final model fit on its own training labels.

## Identity and court-side diagnostics

The original variant deliberately preserves a strict saved-side candidate gate.
The refinement variant pools up to eight clean observed embeddings from the same
gap/teleport-split local track, within 45 frames, and conservatively expands photo
galleries from high-margin unlabelled matches. It requires a warm model cache.

The all-side variant addresses a diagnosed input issue: human-confirmed near-side
photo boxes in LdOgh (P7/P8/P12) overlap saved poses labelled side `a`. Discarding
them makes a correct author impossible. Both sides therefore compete against the
confirmed near-side photo roster. Associations and near/far groupings exist only
in derived feature views; source poses and team labels are unchanged. Its new
capture is limited to the additional candidates' missing OSNet crops. Original,
refined and all-side feature/model directories remain separate in history.

The listwise diagnostic fits shared linear/MLP48 person scorers on near-person
contacts only, with fixed 140 epochs and train-only normalization. It keeps
the original source-held-out team decisions, isolating the person objective.

Setup routing chooses all-side features if a confirmed near-side photo uniquely
matches a saved opposite-side pose (IoU >= .35, margin >= .1), otherwise the
refined side-gated view. The router never reads contact-author decisions. Equal
votes over hard HGB, hard ExtraTrees and soft HGB are additional fixed controls;
ties prefer hard HGB. No voting weights are fitted on test cases. All candidate
methods remain development comparisons, not untouched final-test scores.

`final-author` fits the best observed fixed pointwise/routed-vote candidate on
all ten sources and writes a versioned experimental model with explicit head,
feature, roster, team-HMM and routing contracts. This final fit has no held-out
score. It never replaces the production checkpoint or registry entry.

## Event inference contract and selected-example bias

The old active clip features have zero player-distance and pose channels.
Checkpoint `feature_active_mask` must therefore also mask new live inputs for
an old-only fit. The initial experimental scorer omitted that inference mask;
its unmasked predictions remain diagnostic history. The validated score is a
rescore of the exact same weights, using only training-source channel variance.
ExtraTrees does not split on a constant feature, so its predictions are unchanged.
The rescorer never trains, rewrites a checkpoint, or refreshes source features.

Eighteen of the 28 new four-class corrections are attack. A constant attack
answer gets 64.29% on this selected subset. Matching that value does not show
useful typing discrimination or general new-video accuracy. The checked-in
history reports this context and keeps both original and validated runs.

The additional old-three-source soft control matches the winning all-side soft
HGB's features, fixed ranker parameters and seeds. Its side fits and transitions
also use the old sources only. This separates adding human targets from changing
the candidate/identity features or the person-scoring architecture.
