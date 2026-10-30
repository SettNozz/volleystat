# Frozen OSNet and lost-ID recovery experiment

2026-10-29. Follow-up to the [player identity pilot](player_identity_pilot.md).
Same 450 sampled frames / 2528 saved detections from `hy2S-f7zj9Q_4k`.
No detector rerun, fine-tuning, source annotation change or production promotion.

## Fixed design

Cross two factors using the same new association implementation:

- Descriptor: old three-band HSV versus pretrained OSNet-x1.0 (MSMT17 combineall).
- Recovery: local association limited to a three-second unmatched gap versus
  explicit lost-state recovery after that gap.

Parameters and gallery observation keys were saved in `plan.json` before new
scores were computed. No threshold sweep was performed on the 100 human checks.
The checks informed the preceding diagnosis, so these are **development results**,
not a new held-out estimate.

Galleries are immutable: one verified human seed at frame 72 plus at most seven
clean nearby views associated by the seed raw track ID within +/-15 source frames.
The actual gallery spans source frames **57–87**. These nearby associations are
derived tracking evidence, not additional human identity labels. Gallery keys
are checked for disjointness from evaluation keys. Predictions never update
galleries, preventing one wrong match from redefining someone's appearance.

Input: full clipped player bbox, RGB 256x128, ImageNet mean/std. Normalise each
OSNet descriptor, average original/flipped view, normalise again. Frozen
512-dimensional embeddings are cached with original frame/pose keys.

Hungarian assignment uses appearance plus bounded image motion, explicit
unknown slots and ambiguity rejection. Lost-ID recovery uses a stricter
appearance distance/margin and requires three consecutive, spatially consistent
detections. Recovery backfills those confirmations offline. A large gap no
longer permanently destroys roster identity. Same-frame IDs remain one-to-one.

This is not BoT-SORT or a reproduction of a published sports tracker. There
is no jersey OCR, pose-part embedding or global tracklet optimiser yet.

## Results on the existing 100 checks

19 source frames, 4.70–23.52 s of the ~45-second clip. Counts are detected
person instances, not independent sequences or whole-match IDF1/HOTA.

| Method | Correct | Wrong person | Unknown | Correct among assigned |
|---|---:|---:|---:|---:|
| Old motion | 60 | 6 | 34 | 90.9% |
| Old motion + adaptive HSV | 64 | 16 | 20 | 80.0% |
| Frozen HSV, local only | 74 | 7 | 19 | 91.4% |
| Frozen HSV, recovery enabled | 79 | 5 | 16 | 94.0% |
| Frozen OSNet, local only | **94** | **1** | **5** | **98.9%** |
| Frozen OSNet, recovery enabled | **94** | **1** | **5** | **98.9%** |

Per-person OSNet: P1 19/19, P2 17/18 (one unknown), P3 13/13, P4 19/19,
P5 **14/19** (one wrong, four unknown), P6 12/12. The old pilot scored P5 0/19.
The remaining wrong assignment is P5 -> P3 at frame **531 / 17.72 s**, with
box-overlap flag 0.83. Unknowns: P5 at 141, 171, 501 and 561; P2 at 351.

Interpretation:

- Frozen-gallery/new-association controls improve HSV, but OSNet adds another
  20 correct / six fewer wrong versus matched frozen-HSV local association.
- Recovery helps the HSV control. OSNet local/recovery predictions are identical
  on the entire original clip: no long-loss recovery activated. Do not attribute
  the OSNet gain on this clip to the recovery mechanism.
- OSNet retains 2275/2528 detections (90.0%) over the whole clip. This is coverage,
  not measured accuracy beyond the checked interval.

## Separate diagnostic: synthetic four-second blackout

After the main comparison, a fixed diagnostic removes **all** player detections
from source frames **90–209** in an in-memory copy. Source files and cached
features stay unchanged. Initial galleries at 57–87 are preserved. Evaluate
only the **85 existing checks after frame 210**; exclude 15 masked checks for
both controls. This simulates detection loss, not a genuinely filmed occlusion.

| Same OSNet features | Correct / 85 | Wrong person | Unknown |
|---|---:|---:|---:|
| Three-second expiry, local only | 0 | 0 | 85 |
| Lost-ID recovery enabled | **76** | **1** | **8** |

This exercises re-entry without new manual seeds or evaluation-label feedback.
It does not validate substitutions, side changes, real long exits or unseen cameras.

## Reproduce

Verify `--help`. Use fresh directories/files; existing artifacts are refused.

```powershell
python scripts/evaluation/run_player_reid_experiment.py fetch-assets --output results/reid_new/assets
python scripts/evaluation/run_player_reid_experiment.py embed `
  --project results/player_identity_hy2S_20261029_v1 `
  --assets results/reid_new/assets --output results/reid_new/cache `
  --device cuda --batch-size 8 --threads 2
python scripts/evaluation/run_player_reid_experiment.py evaluate `
  --project results/player_identity_hy2S_20261029_v1 `
  --cache results/reid_new/cache --output results/reid_new/comparison_0000.json
python scripts/evaluation/run_player_reid_experiment.py stress-gap `
  --project results/player_identity_hy2S_20261029_v1 `
  --cache results/reid_new/cache --output results/reid_new/stress_gap_0000.json
```

The completed run is `results/player_identity_reid_hy2S_20261029_v1/`:
`assets/`, `cache_cuda/`, `comparison_0000.json`, `stress_gap_0000.json`.
`cache/plan.json` belongs to an interrupted CPU attempt and contains no complete
embedding cache; it was preserved, not overwritten. Completed inference used
float32, GPU batch 8, a per-process 10% allocator limit and ~124.5 MiB peak
tensor allocation (CUDA context/workspaces are additional). Other GPU work
was not stopped. `--device cpu` is available but considerably slower here.

Show all seven methods in the existing review tool:

```powershell
python -m action_recognition.annotator.player_identity_annotator `
  --project results/player_identity_hy2S_20261029_v1 --port 5063 `
  --extra-comparison results/player_identity_reid_hy2S_20261029_v1/comparison_0000.json
```

The tool verifies observation/seed consistency and keeps the original human
labels. New checks rescore frozen experimental predictions; they do not retrain
or regenerate them. The synthetic-blackout result must not be attached here:
its observation set intentionally differs from the real clip.

## Model provenance and sources

Official author [OSNet weights](https://huggingface.co/kaiyangzhou/osnet) and
[Torchreid source](https://github.com/KaiyangZhou/deep-person-reid), MIT licence
copied alongside assets. [Model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)
documents the MSMT17-pretrained models. No model was trained on this video.

- Code revision: `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`.
- HF revision: `a5c5cc037c24235cda3b21085b93ad77c9616224`.
- Source SHA256: `c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae`.
- Weight SHA256: `48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727`.

All weights load strictly with `torch.load(..., weights_only=True)`; no missing
layers or random-backbone fallback. Embedding/source/label fingerprints and
parameters are retained in the run artifacts. This remains a fixed-roster,
single-side experiment. Evaluate the unlabelled 24–45 s interval and another
video before production use or personal event statistics.
