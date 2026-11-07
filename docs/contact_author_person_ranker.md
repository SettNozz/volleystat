# Person-level contact-author ranker

Answers **which near-side person touched a known contact**, or `other_team`.
Same frozen human benchmark as [contact_author_benchmark.md](contact_author_benchmark.md)
(revision 421, strict set 382 contacts: 184 near-person, 198 other-team).
Modules: `src/tracking/contact_author_person_ranker.py` (features, roster seed,
rally-chain team HMM) and `src/tracking/contact_author_person_evaluation.py`
(source-held-out evaluation). CLI: `scripts/evaluation/run_contact_author_person_ranker.py`.

## Why a new formulation

Diagnosis on the frozen OSNet benchmark showed that per-contact appearance was
the ceiling: in only 139/184 near contacts was the true person the best OSNet
match of *any* nearby detected pose. The toucher is often not detected at the
exact contact frame: only 3–5 of 6 near players are detected in most frames,
and contacts happen in clusters. Inside ±15 frames the true person is present
in 166/184 contacts.

So candidates are **people, not poses**:

1. The production identity pass (`osnet_frozen_gallery_kalman_roster_v4`) runs
   over every frame of each source, seeded from the human-confirmed photo
   roster (`seeds` subcommand; Khe P7 enters as a confirmed substitute). This is
   the one model stage that was missing: OSNet crops were captured into
   `data/inference_cache`; no YOLO/SAM/spotter inference and no source rewrite.
2. A near pose takes its `player_id`, or the dominant (≥60%) time-weighted ID of
   its raw track within ±5 s. Raw track IDs are weak evidence, never gold.
3. Each roster person collects its poses in ±15 frames. Geometry is measured
   only on frames whose saved annotation has a ball: hand/forearm/upper-arm/body
   distance in body heights at several window widths, closest-in-time distance,
   presence, ball-relative body position, hand height, relative rank/gap to the
   other candidates, "missing person" flags. All far poses form one
   `other_team` candidate.
4. **Court-row cue** (user's rule: front row attacks/blocks, back row defends;
   nearer the fixed camera = back row): image foot height and depth rank among
   visible near players. Contact type is one-hot.
5. A HistGradientBoosting ranker scores every candidate; argmax over near people.
6. Near/far decision: ExtraTrees on the existing 366 global side features,
   then a two-state forward–backward HMM over each rally chain. Transitions
   `P(same team | previous type, type)` are counted from training sources only;
   chains reset at serves and gaps >150 frames. It reads held-out contact
   times/types, never held-out authors.

No contact-author target, video name, raw ID or absolute time enters numeric inputs.

## Results, 2026-10-31

Outer: one complete source held out. Inner whole-source folds choose the ranker
(HGB in all three folds). Abstentions count as errors.

| Method | Team | Exact near-person recall | Near precision | Exact author or other team |
|---|---:|---:|---:|---:|
| Previous best fixed (`trees_sequence_central`, HSV) | 92.4% | 60.3% | 61.7% | 77.2% |
| Previous OSNet `class_fusion` (codex rich eval v5) | — | 79.35% | 78.9% | — |
| **Person ranker, nested selected** | **94.8%** | **84.8% (156/184)** | **85.7%** | **90.3%** |
| Person ranker, ExtraTrees fixed | 94.8% | 81.0% | 81.9% | 88.5% |

Per held-out video near recall: Khe 83.8%, j0 78.9%, j3 92.5%.
By type (near, strict): set 94%, serve 89%, reception 89%, dig 82%, attack 79%,
block 56% (n=9, too few for a claim).

Development ablations (same folds, HGB; selection on these numbers would be
biased, so they are diagnostics only):

| Change | Near recall |
|---|---:|
| Full | 84.8% |
| No team HMM (person-ranker argmax decides team) | 65.2% |
| Team emission only, no chain HMM: team accuracy | 83.8% (vs 94.8%) |
| No court-row / depth features | 81.5% |
| No raw-track ID vote (production IDs only) | 82.1% |
| Window ±10 / ±20 frames | 85.3% / 83.7% |

Remaining 28 near errors: 11 wrong team, 11 true person never identified
within ±3/±15 frames (undetected or ID unknown), 6 ranked a nearby teammate
higher.

## Limits

**2026-11-01 paired audit:** the original 84.8% above is reproduced, but its
inner team predictions include the outer test source and fitting includes 12
non-strict labels. Under matched strict-only training, fixed HGB gets 85.87%
(158/184); properly nested selection gets 82.07% (151/184), versus Codex's
79.35% fixed / 75.54% nested. See the [paired comparison](contact_author_comparison.md)
for source-by-source metrics, exact protocol differences and reproducible commands.

- Three development videos from one fixed camera; no untouched final test.
  Not an unseen-camera or unseen-team number.
- Contact time and type are the audited human values. With spotter timing/types
  (typing ≈0.92) both the chain HMM and the type features will degrade.
- Identity is seeded from a human photo roster per video. A new match needs the
  same one-time roster confirmation; automatic rosters are not evaluated here.
- One court side (near, `b`); set-side switches are out of scope.
- Scores are uncalibrated. Nothing is promoted to production or the registry;
  there is no final-fit checkpoint.

## Unlabelled video: X6JdbC8SM_U_4k (2026-10-31)

`predict` fits the HGB ranker, the ExtraTrees team emission (seeds 42/43/44) and
the chain transitions on **all** labelled sources, then ranks the unlabelled
source's spotter events. It reuses the existing unlabelled author inputs
(`results/X6JdbC8SM_U_author_inputs_20261031_v1`: 87 spotter events, 366 global
features, targets -1) and the saved X6 analysis, whose production identity pass
already exists. That roster is **automatic initialisation, unverified**
(coverage 66%), not a human photo roster. No inference, no source rewrite.

X6 is a **different camera** (low gym view behind an end line) from the three
gold videos, and its event times/types come from the spotter. There is no human
author label for X6, so it has **no measured accuracy**; the 84.8% applies only
to the gold benchmark. Output `results/X6JdbC8SM_U_person_ranker_20261031_v1`:
44 near-person proposals (P1 16, P2 10, P5 10, P6 4, P3 2, P4 2) and 43
other-team. It agrees with the earlier `class_fusion_ensemble` on 60/87.

Visual review (read-only, port 5069): orange = this ranker's person at the
evidence frame (closest measured ball contact), blue = `class_fusion` when it
differs, grey = other poses labelled with their automatic IDs, yellow = ball.
Filters: player card, disagreements with `class_fusion`.

```powershell
python scripts/evaluation/run_contact_author_person_ranker.py predict `
  --benchmark results/contact_author_benchmark_20261031_v1 `
  --identity-dir results/contact_author_identity_20261031_v1 `
  --inputs results/X6JdbC8SM_U_author_inputs_20261031_v1 `
  --comparison results/X6JdbC8SM_U_author_predictions_20261031_v1/predictions.json `
  --output results/X6JdbC8SM_U_person_ranker_NEW

python -m action_recognition.annotator.contact_author_person_viewer `
  --predictions results/X6JdbC8SM_U_person_ranker_NEW --port 5069
# http://127.0.0.1:5069/
```

## Reproduce

Verify `--help` first; every output must be fresh.

```powershell
python scripts/evaluation/run_contact_author_person_ranker.py seeds `
  --benchmark results/contact_author_benchmark_20261031_v1 `
  --output results/contact_author_identity_NEW/seeds

# Per video; captures only missing OSNet crops into data/inference_cache.
python scripts/apply_player_identity.py apply `
  --annotations <shared>/results/gold_pose_fsm_20261028_v1/<video>_annotations.json `
  --output results/contact_author_identity_NEW/<video> `
  --seed results/contact_author_identity_NEW/seeds/<video>_roster.json `
  --side b --assets <shared>/models/player_reid/osnet_msmt17_v1 --device cuda `
  --no-model-registry --model-cache-dir <shared>/data/inference_cache
# Re-runs: add --require-model-cache.

python scripts/evaluation/run_contact_author_person_ranker.py evaluate `
  --benchmark results/contact_author_benchmark_20261031_v1 `
  --identity-dir results/contact_author_identity_NEW `
  --output results/contact_author_person_ranker_NEW
```

Artifacts: `results/contact_author_identity_20261031_v1` (seeds + identity
results, ~35 min per video on RTX 3080, three in parallel) and
`results/contact_author_person_ranker_20261031_v1` (`REPORT.md`,
`comparison.json`, `predictions.json` with per-person scores).
Tests: `tests/test_contact_author_person_ranker.py`.
