# Contact-author method comparison — 2026-11-14

Question: which method names the near-side player who touched the ball best on
a new video, using only the automatic roster?

## Protocol

- Inputs: the ten fresh automatic analyses of
  `results/contact_system_benchmark_20261114_v1/pipeline/` (automatic roster
  discovery, substitutions, spotter). Cache-only: zero new detector, pose or
  OSNet inference, zero captured crops.
- Targets: the frozen 1,137 human author decisions (585 named near-side
  people, 552 other team) from that benchmark's `ground_truth.json`.
- Known-contact control: human contact frames/types are supplied, as in the
  benchmark, so attribution is separated from detection.
- Automatic IDs are named to human IDs with the benchmark's target-independent
  photo naming. Unnamed IDs, abstentions and other-team answers on named
  contacts are errors ("strict").
- Held-out: `lovo` uses the stored fold models of
  `contact_expanded_training_20261106_v1/authors_all_sides`; each video is scored
  by rankers/team models that never saw it. `final_fit` is the promoted model,
  which saw all ten sources (optimistic, shown only for reference).
- "Best naming" is an optimistic diagnostic: one-to-one automatic-to-human
  naming chosen to maximise agreement. It shows whether an automatic ID
  consistently means one person, which is what a user sees after naming P-IDs
  once in the viewer. It is not the headline number.

Script: `scripts/evaluation/compare_contact_author_identity_methods.py`
(`features`, `score`, `runtime`). Output:
`results/contact_author_methods_20261114_v1/` (`summary.json`, `rows/`,
`runtime_summary.json`, `precision_coverage_auto_gallery_lovo_raw.json`).

## Methods

Roster for the author model:

- `auto`: production default before this change, one automatic seed photo per person.
- `auto_gallery`: plus up to 12 identity-pass views per person (`reason=associated`,
  best appearance first, at least 90 frames apart).
- `human_photos`: the human-confirmed photo roster (needs a person to confirm
  one photo per player; namespace needs no naming).

Decision policy on the same HGB scores:

- `production`: 2026-11-13 hard rules: substitution on-court mask, 0.35 body-height
  distance, missing ball, roster conflict, unconfirmed replacement all abstain/exclude.
- `prod_no_subs`, `guards_dist_no_subs`, `raw_hgb_subs`: one rule family removed at a time.
- `raw_hgb`: argmax of the person scores; team HMM unchanged.
- `raw_fused`, `dist_track_override`: identity-pass track IDs fused with / overriding the model.
- `geometry_identity_pass`: nearest hand/arm to the ball, ID from the identity pass, no person model.
- `lovo_retrain_*`: rankers refit per fold on automatic-roster features (labels via photo naming only).

## Results (585 named contacts, held-out unless noted)

| Roster | Model | Policy | Exact person | Precision | F1 | Author or other team | Best naming | Wrong/unmapped | Unresolved | Wrong team |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| auto | lovo | production | 352 (60.2%) | 69.2% | 64.4% | 76.6% | 385 | 129 | 68 | 36 |
| auto | lovo | prod_no_subs | 386 (66.0%) | 73.9% | 69.7% | 79.6% | 418 | 109 | 54 | 36 |
| auto | lovo | guards_dist_no_subs | 391 (66.8%) | 72.3% | 69.4% | 80.0% | 425 | 119 | 39 | 36 |
| auto | lovo | raw_hgb_subs | 398 (68.0%) | 68.4% | 68.2% | 80.7% | 438 | 151 | 0 | 36 |
| auto | lovo | raw_hgb | 433 (74.0%) | 74.4% | 74.2% | 83.7% | 471 | 116 | 0 | 36 |
| auto | lovo | geometry_identity_pass | 309 (52.8%) | 72.0% | 60.9% | 72.8% | 325 | 101 | 139 | 36 |
| auto_gallery | lovo | production | 350 (59.8%) | 68.8% | 64.0% | 76.4% | 385 | 132 | 67 | 36 |
| auto_gallery | lovo | dist_track_override | 379 (64.8%) | 69.9% | 67.3% | 79.0% | 415 | 132 | 38 | 36 |
| auto_gallery | lovo | raw_fused | 415 (70.9%) | 71.3% | 71.1% | 82.1% | 444 | 134 | 0 | 36 |
| **auto_gallery** | **lovo** | **raw_hgb** | **437 (74.7%)** | **75.1%** | **74.9%** | **84.1%** | **478** | 112 | 0 | 36 |
| auto_gallery | lovo_retrain_auto | raw_hgb | 429 (73.3%) | 73.7% | 73.5% | 83.4% | 472 | 120 | 0 | 36 |
| auto_gallery | lovo_retrain_mix | raw_hgb | 436 (74.5%) | 74.9% | 74.7% | 84.0% | 478 | 113 | 0 | 36 |
| auto_gallery | lovo_retrain_mix_all | raw_hgb | 438 (74.9%) | 75.3% | 75.1% | 84.2% | 480 | 111 | 0 | 36 |
| human_photos | lovo | production rules | 437 (74.7%) | 80.6% | 77.6% | 84.1% | — | 74 | 38 | 36 |
| human_photos | lovo | raw_hgb | 483 (82.6%) | 83.0% | 82.8% | 88.1% | — | 66 | 0 | 36 |
| auto | final_fit | production (seen data) | 366 (62.6%) | 72.0% | 67.0% | 79.2% | 401 | 129 | 74 | 16 |
| auto_gallery | final_fit | raw_hgb (seen data) | 468 (80.0%) | 79.7% | 79.9% | 88.1% | 508 | 101 | 0 | 16 |

Per video, held-out exact person (production → selected → human photos):

| Video | Named | auto/production | auto_gallery/raw_hgb | human_photos/raw_hgb |
|---|---:|---:|---:|---:|
| JIZ3exHe3hA_4k | 49 | 32 (65%) | 38 (78%) | 37 (76%) |
| Khe4P8QcJkg_5_4k | 76 | 60 (79%) | 63 (83%) | 64 (84%) |
| LdOghv6rt_4_4k | 69 | 21 (30%) | 42 (61%) | 55 (80%) |
| QhqMwaCf8iw_4k | 53 | 29 (55%) | 40 (75%) | 44 (83%) |
| TcUru_ou4CQ_4k | 55 | 27 (49%) | 29 (53%) | 47 (85%) |
| X6JdbC8SM_U_4k | 45 | 17 (38%) | 16 (36%) | 27 (60%) |
| j0gRhgFCgIs_4_4k | 59 | 41 (69%) | 53 (90%) | 52 (88%) |
| j3BO9GgUFBI_3_4k | 53 | 43 (81%) | 49 (92%) | 47 (89%) |
| qDzHtxhNkBA_4k | 36 | 15 (42%) | 29 (81%) | 33 (92%) |
| xZbD8fx3D4k_1_4k | 90 | 67 (74%) | 78 (87%) | 77 (86%) |

Precision/coverage of the selected method (held-out; abstain below a
confidence = near-side posterior x top person score):

| Kept | Assigned | Correct | Precision | Recall |
|---:|---:|---:|---:|---:|
| 100% | 582 | 437 | 75.1% | 74.7% |
| 90% | 524 | 417 | 79.6% | 71.3% |
| 80% | 466 | 386 | 82.8% | 66.0% |
| 70% | 407 | 351 | 86.2% | 60.0% |
| 50% | 291 | 262 | 90.0% | 44.8% |

Production runtime check (`runtime` stage, promoted final-fit model, which has
seen all ten sources, so optimistic):

| Runtime | Known contact frames | End-to-end detected contacts +/-5 frames |
|---|---:|---:|
| `--hard-guards` (2026-11-13 behaviour) | 363/585 (62.1%), precision 72.6% | 345/585 (59.0%), precision 70.0% |
| new default | 464/585 (79.3%), precision 79.5% | 436/585 (74.5%), precision 77.7% |

`--hard-guards` reproduces the benchmark (362). The new default differs from the
experiment's 468 by exactly the 4 contacts whose type the runtime does not
attribute. End-to-end keeps 94% of known-frame hits, so a held-out end-to-end
estimate is about 70% (an estimate, not measured).

## Findings

1. The hard rules were the largest loss. The substitution on-court mask alone
   removes 35 correct held-out answers (433 → 398); the distance/roster/replacement
   rules remove another 40+ and leave 54–68 unresolved. Precision did not improve
   either (69.2% with rules vs 74.4% without). With human photos the same rules
   cost 46 correct answers. Substitution detection mistakes (e.g. LdOgh pairs
   P2/P4/P7/P3 several times) excluded the real author.
2. The richer identity-pass gallery adds +4 held-out (+10 with the final fit) and
   fixed the 2026-11-13 7YnJ F13289 P5/P6 confusion without any guard: the set
   stays P5 and the P3 attack/serve duplicates keep their authors.
3. Identity-pass track IDs are worse than the model's own appearance matching
   (override 379, fusion 415, geometry + identity pass 309).
4. Refitting the ranker on automatic-roster features gives no gain (429–438):
   the model is not the bottleneck.
5. The remaining gap to human photos (437 → 483) is roster quality: TcUru, X6J
   and LdOgh. About 41 strict errors are naming only (best naming 478): an
   automatic ID consistently follows one person but photo naming cannot prove it
   (LdOgh human P1 → automatic P5 in 16 contacts; TcUru P8 → P6 in 12).
6. 36 named contacts are given to the other team by the held-out team HMM
   (16 with the final fit).

## Decision

Production default is now `auto_gallery` + unchanged promoted checkpoint +
`raw_hgb` (`osnet_temporal_all_sides_soft_hgb_v2`). Former rules are reported as
`review_flags` on the chosen person, with `confidence` and `score_margin`, but do
not change the answer. `scripts/apply_player_contacts.py --hard-guards` and
`attribute_contacts(..., guards=True)` restore the 2026-11-13 behaviour.

Expected quality on a new video with the automatic roster: about 75% exact
person (held-out mean; 36–92% per video), 84% author-or-other-team. A
human-confirmed photo roster raises this to about 83%. All ten sources are
development data used for method selection: this is not an untouched test.

## Follow-up 2026-11-15: side cue and occlusion localization

Script: `scripts/evaluation/contact_author_side_localization.py` (`features`,
`score`); output `extended_summary.json`, `extended_rows/`.

- Side: ball-box size around the contact, before/after it and summed per
  possession (alternating inside a rally), normalised by the video's median
  contact ball size, so the four sources without net geometry are included;
  plus ball image height relative to `net_y`. Added to the team model.
- Localization: per candidate, hand distance to the contact point
  reconstructed from the ball path before/after, wrist speed, jump height,
  hands above head on the candidate's own continuous track segment.

Every arm is refit LOVO (3 seeds, rankers on human-photo + automatic-gallery
rows, team ExtraTrees, fold transitions); `control` refits the unchanged
features with the same budget.

| Roster | Arm | Exact person | Precision | Author or other | Team accuracy (1,137) | Wrong team | Paired vs control (named) |
|---|---|---:|---:|---:|---:|---:|---|
| automatic + gallery | control | 432 (73.8%) | 74.4% | 83.6% | 93.7% | 38 | — |
| automatic + gallery | side | 439 (75.0%) | 74.2% | 83.9% | 94.1% | 30 | +11/−4, p=0.12 |
| automatic + gallery | localization | 433 (74.0%) | 74.5% | 83.6% | 93.7% | 38 | +6/−5, p=1.0 |
| automatic + gallery | both | 440 (75.2%) | 74.3% | 84.0% | 94.1% | 30 | +17/−9, p=0.17 |
| human photos | control | 474 (81.0%) | 81.6% | 87.2% | 93.7% | 38 | — |
| human photos | side | 481 (82.2%) | 81.2% | 87.6% | 94.1% | 30 | +11/−4, p=0.12 |
| human photos | localization | 480 (82.1%) | 82.6% | 87.8% | 93.7% | 38 | +11/−5, p=0.21 |
| human photos | both | 487 (83.2%) | 82.3% | 88.1% | 94.1% | 30 | +22/−9, p=0.03 |

Findings:

- The side cue moves 8 named contacts from "other team" to the right side but
  costs other-team contacts: over all 1,137 contacts +18/−14 (p=0.60). Team
  accuracy rises only 93.7% → 94.1%.
- The occlusion hypothesis was wrong: 583/585 named contacts have an observed
  ball within ±3 frames, and all 111 human-photo errors have one. Errors come
  from choosing between nearby people with a visible ball (set 111/147,
  dig 73/99, block 11/17), not from a hidden ball.
- Only "both" with human photos reaches p<0.05 on named contacts, and not over
  all contacts. With the automatic roster no arm is distinguishable from noise.

Decision: not promoted. The production checkpoint is unchanged; a new one
would need a held-out source the selection did not see.
