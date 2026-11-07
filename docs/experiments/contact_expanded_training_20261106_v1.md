# Expanded contact experiments — 2026-11-06

Artifacts: `C:\Users\illya\Documents\volleyball_analitics\volleystat\results\worktrees\contact-author-learning\results\contact_expanded_training_20261106_v1`

Frozen annotation revisions 421 and 862. Ten complete source videos. 1,137 human author labels (585 near-person, 552 other-team). Event types: 410 active human core clips + 28 explicit type reviews = 438. Unchanged model types are not human targets. TcUru has five roster people.

## Author attribution

Primary view excludes contacts within 60 frames of any roster photo. Offline candidate timing/type is supplied; new source timing remains model-derived. All controls share local OSNet photo assignment and saved input features. The old-three-source HGB control is not a reproduction of the historic full-video-identity Claude result.

| Method | N | Near correct / N | Near recall | Near precision | Team | Exact author or other |
|---|---:|---:|---:|---:|---:|---:|
| geometry | 1105 | 241/565 | 42.65% | 41.27% | 60.45% | 50.05% |
| old3_hgb | 1105 | 366/565 | 64.78% | 64.66% | 92.85% | 78.37% |
| expanded_hgb | 1105 | 383/565 | 67.79% | 68.27% | 94.03% | 80.72% |
| expanded_trees | 1105 | 369/565 | 65.31% | 65.78% | 94.03% | 79.46% |
| expanded_soft_hgb | 1105 | 380/565 | 67.26% | 67.74% | 94.03% | 80.45% |
| refined_geometry | 1105 | 270/565 | 47.79% | 46.23% | 60.45% | 52.67% |
| refined_old3_hgb | 1105 | 399/565 | 70.62% | 70.49% | 92.85% | 81.36% |
| refined_expanded_hgb | 1105 | 418/565 | 73.98% | 74.51% | 94.03% | 83.89% |
| refined_expanded_trees | 1105 | 406/565 | 71.86% | 72.37% | 94.03% | 82.81% |
| refined_expanded_soft_hgb | 1105 | 409/565 | 72.39% | 72.91% | 94.03% | 83.08% |
| all_sides_geometry | 1105 | 405/565 | 71.68% | 41.71% | 58.73% | 46.52% |
| all_sides_old3_hgb | 1105 | 406/565 | 71.86% | 71.73% | 92.85% | 81.99% |
| all_sides_expanded_hgb | 1105 | 449/565 | 79.47% | 80.04% | 94.03% | 86.70% |
| all_sides_expanded_trees | 1105 | 456/565 | 80.71% | 81.28% | 94.03% | 87.33% |
| all_sides_expanded_soft_hgb | 1105 | 470/565 | 83.19% | 83.78% | 94.03% | 88.60% |
| listwise_old3_linear | 1105 | 323/565 | 57.17% | 57.07% | 92.85% | 74.48% |
| listwise_old3_mlp | 1105 | 359/565 | 63.54% | 63.43% | 92.85% | 77.74% |
| listwise_expanded_linear | 1105 | 361/565 | 63.89% | 64.35% | 94.03% | 78.73% |
| listwise_expanded_mlp | 1105 | 405/565 | 71.68% | 72.19% | 94.03% | 82.71% |
| routed_old3_routed_hgb | 1105 | 397/565 | 70.27% | 70.14% | 92.85% | 81.18% |
| routed_expanded_routed_hgb | 1105 | 444/565 | 78.58% | 79.14% | 94.03% | 86.24% |
| routed_expanded_routed_vote | 1105 | 450/565 | 79.65% | 80.21% | 94.03% | 86.79% |
| routed_expanded_all_sides_vote | 1105 | 459/565 | 81.24% | 81.82% | 94.03% | 87.60% |
| routed_expanded_refined_vote | 1105 | 420/565 | 74.34% | 74.87% | 94.03% | 84.07% |
| soft_control_old3_soft_hgb | 1105 | 432/565 | 76.46% | 76.33% | 92.85% | 84.34% |

New seven-video subset (same cases for both fits):

| Method | N | Near correct / N | Near recall | Exact author or other |
|---|---:|---:|---:|---:|
| old3_hgb | 719 | 227/381 | 59.58% | 74.83% |
| soft_control_old3_soft_hgb | 719 | 282/381 | 74.02% | 82.48% |
| all_sides_expanded_soft_hgb | 719 | 307/381 | 80.58% | 86.65% |

Best observed expanded method: `all_sides_expanded_soft_hgb`. This is selection on development folds, not an untouched final test.

| Held-out source | Near N | Old-three near recall | Expanded near recall | Expanded team |
|---|---:|---:|---:|---:|
| JIZ3exHe3hA_4k | 47 | 72.34% | 74.47% | 96.15% |
| Khe4P8QcJkg_5_4k | 74 | 78.38% | 85.14% | 93.33% |
| LdOghv6rt_4_4k | 67 | 70.15% | 74.63% | 88.39% |
| QhqMwaCf8iw_4k | 48 | 64.58% | 72.92% | 93.51% |
| TcUru_ou4CQ_4k | 55 | 83.64% | 85.45% | 89.80% |
| X6JdbC8SM_U_4k | 39 | 48.72% | 71.79% | 94.81% |
| j0gRhgFCgIs_4_4k | 57 | 78.95% | 89.47% | 96.61% |
| j3BO9GgUFBI_3_4k | 53 | 88.68% | 92.45% | 97.46% |
| qDzHtxhNkBA_4k | 35 | 80.00% | 88.57% | 96.67% |
| xZbD8fx3D4k_1_4k | 90 | 85.56% | 90.00% | 94.24% |

| Contact type | Near correct / N | Near recall |
|---|---:|---:|
| attack | 94/116 | 81.03% |
| attack / block | 1/3 | 33.33% |
| attack / block / dig | 0/0 | 0.00% |
| block | 11/17 | 64.71% |
| dig | 76/98 | 77.55% |
| dig / reception | 0/1 | 0.00% |
| dig / set | 0/0 | 0.00% |
| free_ball | 0/0 | 0.00% |
| free_ball / set | 0/0 | 0.00% |
| other | 0/0 | 0.00% |
| pass | 1/2 | 50.00% |
| reception | 83/92 | 90.22% |
| serve | 91/97 | 93.81% |
| set | 113/139 | 81.29% |

## Four-class causal event typing

Old three videos are the main matched typing evaluation. New types are explicit corrections only, a deliberately selected hard subset; their accuracy is not average new-video accuracy. Core collapse groups set/dig as pass and block as attack; the six-class view preserves those distinctions.

| Method | Old-3 N | Old-3 accuracy | Old-3 macro F1 | New corrections N | New corrections accuracy |
|---|---:|---:|---:|---:|---:|
| old3_trees | 410 | 89.51% | 90.00% | 28 | 25.00% |
| old3_multiscale | 410 | 90.00% | 90.37% | 28 | 25.00% |
| expanded_trees | 410 | 90.24% | 90.71% | 28 | 57.14% |
| expanded_multiscale | 410 | 89.27% | 90.00% | 28 | 64.29% |

Per-class old-three-source metrics for `expanded_trees`:

| Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| serve | 92.77% | 98.72% | 95.65% | 78 |
| reception | 88.73% | 91.30% | 90.00% | 69 |
| pass | 88.46% | 89.03% | 88.75% | 155 |
| attack | 92.00% | 85.19% | 88.46% | 108 |

## Six-class causal event typing

Old three videos are the main matched typing evaluation. New types are explicit corrections only, a deliberately selected hard subset; their accuracy is not average new-video accuracy. Core collapse groups set/dig as pass and block as attack; the six-class view preserves those distinctions.

| Method | Old-3 N | Old-3 accuracy | Old-3 macro F1 | New corrections N | New corrections accuracy |
|---|---:|---:|---:|---:|---:|
| old3_trees | 403 | 80.89% | 71.68% | 25 | 12.00% |
| old3_tcn | 403 | 79.16% | 74.11% | 25 | 12.00% |
| expanded_trees | 403 | 81.64% | 73.46% | 25 | 72.00% |
| expanded_tcn | 403 | 77.92% | 73.23% | 25 | 60.00% |

Per-class old-three-source metrics for `expanded_trees`:

| Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| serve | 91.76% | 100.00% | 95.71% | 78 |
| reception | 86.30% | 91.30% | 88.73% | 69 |
| set | 79.31% | 79.31% | 79.31% | 87 |
| dig | 74.19% | 70.77% | 72.44% | 65 |
| attack | 76.09% | 83.33% | 79.55% | 84 |
| block | 75.00% | 15.00% | 25.00% | 20 |

## Limits and promotion

Main event scores apply the production inactive-channel mask computed from training sources only. Old active clip features have zero pose/player-distance channels; feeding live values into those untrained channels gave an artificial failure on new sources. Original unmasked runs are retained as diagnostics, not the matched main control. Rescoring changes no model weights. The new four-class correction subset is 18/28 attack (64.29% constant-attack accuracy), so a comparable model score is not evidence of useful discrimination.

Variants preserve the original side-gated feature experiment, clean-track/gallery refinement, and an all-side candidate experiment. Human near-side reference photos sometimes match saved poses marked as the opposite side (e.g. LdOgh P7/P8/P12). All-side variants compare both sides against the confirmed photo roster and use derived feature-only associations; no saved pose/team is edited. Additional inference is limited to missing OSNet crops of those additional candidates.

No dense spotting score is available from the new review: it does not label missed contacts, background or exact contact timing. Training-source normalization, transitions and model fits exclude the full held-out source. Epoch budgets are fixed (multiscale 50, raw TCN 35), with seeds 42/43/44. No held-out early stopping or contact-level random split is used.

All checkpoints remain experimental. Production models, registry and source annotations are unchanged. A final fit on every source would have no held-out score. The result does not validate unseen camera placements beyond the reviewed source-held-out comparisons.

Commit 9b0b581 is accounted for by explicit OSNet batch128 and cache reuse; raw inference prefetch does not affect these saved-input training experiments. JIZ3 LF/CRLF-only legacy cache reuse preserves original stage provenance after exact normalized-code/runtime/weight/settings checks.

## Final experimental author fit

`final_author/model.joblib`: authors_all_sides/expanded_soft_hgb, 1105 contacts from all ten sources, three seeds. This all-source fit has no held-out score; the fold reports above evaluate the method. It is not promoted or loaded by production.
