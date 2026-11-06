# Paired comparison: class_fusion vs Claude's person ranker

Audit dated 2026-11-01. `0a1e7ed` contains the OSNet/temporal `class_fusion`
approach; `f7a86eb` adds Claude's person-level ranker. Compare complete approaches:
Claude also uses a full-video production identity pass, whereas `class_fusion`
uses short-window OSNet gallery matching. The gain is not an isolated comparison
of HGB against an MLP with identical identity inputs.

## Identical test membership

Frozen author snapshot revision 421, SHA-256
`fc2313f5fb580c02ae52c7ffd159fbfbe42636dd29d25d449e96ffbda33cfb51`.
Both methods are rescored by the same metric implementation on the same **382**
strict contacts: **184 near-person / 198 other-team**. IDs, video, frame, type,
truth and strict membership are checked one by one. Missing/unknown predictions
cannot disappear from the denominator. X6 has no author gold and is excluded.

Outer folds hold out an entire source. Strict training uses only the other
sources' eligible strict labels. Human contact time/type and each video's photo
roster are given setup inputs. There is no untouched final test after prior
development on these three videos, and no measured end-to-end spotting score.

## Results

| Method | Exact near person | Near precision | Near F1 | Team accuracy | Exact author or other |
|---|---:|---:|---:|---:|---:|
| Codex `class_fusion`, fixed | 146/184 = 79.35% | 78.92% | 79.13% | 93.46% | 331/382 = 86.65% |
| Claude HGB, matched strict training | 158/184 = 85.87% | 84.04% | 84.95% | 94.76% | 344/382 = 90.05% |
| Claude ExtraTrees, matched strict training | 150/184 = 81.52% | 79.79% | 80.65% | 94.76% | 336/382 = 87.96% |
| Codex, nested method selection | 139/184 = 75.54% | 75.14% | 75.34% | 92.41% | 322/382 = 84.29% |
| Claude, corrected nested method selection | 151/184 = 82.07% | 80.32% | 81.18% | 94.76% | 337/382 = 88.22% |

Fixed HGB adds **12** correct near people (**+6.52 percentage points**) over
fixed `class_fusion`. Paired outcomes: **136** both correct, **22** Claude only,
**10** Codex only, **16** both wrong. Near errors change from 12 wrong-team + 26
wrong-person to 8 wrong-team + 18 wrong-person. No abstentions occur here.

| Held-out source | Targets | Codex fixed | Claude HGB strict | Claude nested strict |
|---|---:|---:|---:|---:|
| Khe4P8QcJkg_5_4k | 74 | 60/74 = 81.08% | 64/74 = 86.49% | 59/74 = 79.73% |
| j0gRhgFCgIs_4_4k | 57 | 40/57 = 70.18% | 45/57 = 78.95% | 45/57 = 78.95% |
| j3BO9GgUFBI_3_4k | 53 | 46/53 = 86.79% | 49/53 = 92.45% | 47/53 = 88.68% |

## Difference from the original Claude report

The original saved report reproduces **156/184 = 84.78%** near recall,
85.71% precision, 85.25% F1, 94.76% team accuracy and 90.31% overall. It fits
on all 394 eligible labels, then scores only 382 strict examples. Codex fits
only strict labels. Those extra twelve are not outer-test label leakage, but
they make the training protocols different.

Original nested selection also reuses global leave-one-video-out team
predictions inside the inner loop. For an inner validation source A with outer
test source C, that team model was trained on B+C, including C's author labels.
The inner ranker is trained on B only. Therefore the **nested algorithm choice**
is not independent of the outer test source. The fixed HGB outer prediction
still excludes its own test video's labels; its 84.78% is not an in-sample score.

The matched run excludes non-strict labels from both person/team fitting and
transition counts. Excluded nodes cannot be bridged when counting transitions.
Inner team models and transitions are fitted on the same inner-training source
as the person ranker. Corrected inner choices are trees / HGB / trees for
Khe / j0 / j3, yielding **82.07%**. Original HGB choices are retained only as
the historical result, without overwriting their report.

## Reproduce and audit

Verify `--help` first. Run in the isolated `codex/contact-author-learning`
worktree, with the project Python environment available:

```powershell
python scripts/evaluation/compare_contact_author_methods.py `
  --benchmark results/contact_author_benchmark_20261031_v1 `
  --identity-dir results/contact_author_identity_20261031_v1 `
  --codex-evaluation results/contact_author_rich_eval_20261031_v5 `
  --claude-evaluation results/contact_author_person_ranker_20261031_v1 `
  --output results/contact_author_paired_comparison_NEW
```

Current output: `results/contact_author_paired_comparison_20261101_v1`.
`contract.json` prints input manifests, feature paths, schemas, source videos,
label provenance, clip counts and split policy before work. Person seed 42 and
team seeds 42/43/44 retain the committed fixed budgets. Only small CPU
classifiers are refitted; no YOLO/OSNet/SAM/spotter inference, video processing,
source overwrite or production promotion occurs. Saved Codex strict-fold
predictions are reused and both original reports are independently rescored.

`comparison.json` and `REPORT.md` contain the results. `predictions.json`
retains paired per-contact outcomes; `fit_audit` enumerates every training
contact ID and inner source assignment. `candidate_features.npz` is a new
derived comparison cache, not a rewrite of training data. Confidence/selective
fields from the common scorer are not used: this comparison replays labels,
not calibrated confidence. `source_preservation.json` verifies unchanged
human snapshot, benchmark features, source poses/balls and identity artifacts.

Tests: `tests/test_contact_author_comparison.py` checks strict transition masking,
outer-label independence, exact queue alignment and paired unknown accounting.
