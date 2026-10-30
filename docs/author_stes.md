# Author STES temporal experiment

Source: https://github.com/hoangqnguyen/spot
Pinned revision: bbaa249c285c6bc133b091e50e6da2bf4ed42a39.

ImageNet initialization: timm/regnety_008.pycls_in1k on Hugging Face, revision
ad27a8e42099466890afc4dbcc24cce1340776b9, model.safetensors SHA256
f7f68bb275562d8e8b41891b0af5ca5dc2ab6ac1e511f7f86e00509d9bdca887.
The local weights metadata is saved alongside the safetensors file. Optional
dependencies timm 1.0.20 and tabulate 0.9.0 are isolated in .runtime/vnl_stes_deps.
Original project packages are not upgraded and no remote telemetry is used.

The adapter uses the authors' RegNet-Y 800MF, GSM and three-layer bidirectional
GRU implementation. All parameters receive gradients. GSM zero-padding uses
the input tensor device and dtype; learned operations are unchanged.

This is temporal adaptation, not a reproduction of paper scores: initialization
is ImageNet, classes are four local contacts plus background, and the spatial
head is disabled without human XY. No derived ball coordinates become targets.

Run: results/vnl_stes_author_20261027_v1. It preserves 410 human events and
whole-video folds. Pilot: seed 42, 20 epochs, 64 steps/epoch, two clips per
microbatch, accumulation four, 64-frame clips. RGB caches preserve native frame
indices. The backbone and temporal learning rates are 1e-4 and 1e-3. Validation
runs every two epochs. Selection uses author AP averaged at 0/1/2/4 frames.
Additional metrics are event F1 and false events per minute. These remain
validation model-selection estimates, not an independent test.

Original data and production are preserved. Dependencies are isolated under
.runtime/vnl_stes_deps. Each epoch atomically saves full optimizer and RNG state.
The runner supports --resume with frozen-input checks; an interrupted partial
epoch is repeated. See scripts/evaluation/run_author_stes.py --help.

## Verified spatial training

`scripts/evaluation/run_spatial_stes.py` enables the unchanged authors' MLP XY
head and trains all backbone, temporal and spatial parameters jointly. ImageNet
initializes the backbone; temporal and spatial heads start from random weights.
The older temporal checkpoints are not reused.

Human review: `data/contact_spatial_gold_v1/reviews.json`, preserved unchanged.
The derived snapshot in `results/vnl_stes_spatial_20261027_v1/canonical_labels.json`
contains 397 distinct temporal contacts and 369 spatial targets. Seven redundant
rows are merged across five same-frame groups, with every original review and
annotation ID retained. Duplicate XY uses the median of verified points; large
disagreements stop preparation. The explicit reception/pass conflict at Khe
frame 3761 resolves to reception: first touch after serve, before a separate set.
Four rejected contacts and two uncertain-time contacts are excluded; unknown
contact-time regions are masked rather than treated as background negatives.

The primary pilot holds out all Khe frames: j0/j3 supply 244 temporal contacts
and 230 spatial targets; Khe supplies 153 contacts and 139 verified XY points.
Training uses weighted CE plus equally weighted masked L1 on sigmoid XY, only
at verified contact frames. Full-image resize and horizontal flip update XY
consistently. The native-frame cache is reused and hash checked.

Run with the isolated dependencies on PYTHONPATH:

```powershell
python -m scripts.evaluation.run_spatial_stes --output results/vnl_stes_spatial_20261027_v1
# Interrupted epoch: same command and --resume
```

Pilot budget: 20 epochs, 64 microbatches per epoch, two 64-frame clips per batch,
gradient accumulation four. Validation runs every two epochs. `last.pt` saves
optimizer/scaler/scheduler/RNG state at each completed epoch. `best.pt` minimizes
validation mean coordinate distance; all run inputs and implementation hashes
are frozen for resume. `cancel.request` requests a stop at the next train step.

Spatial errors are measured at **human-known contact frames**, in normalized
224x224-image pixels and native-image pixels. These are conditional localization
metrics, not end-to-end spatial AP and not directly the paper's reported scores.
The train-only constant mean-position baseline uses the same evaluation points.
Temporal AP and event F1 are also reported. Because the validation video selects
the checkpoint, these are model-selection estimates, not an independent test.
Improvement over earlier runs cannot isolate the effect of the spatial head:
the human timestamps and seven event classes were also corrected in review.
Production checkpoints and source annotations remain unchanged.

### Primary pilot result

Completed all 20 epochs. Best coordinate-error checkpoint: epoch 8. Validation
uses 139 verified contact positions from the complete held-out Khe video.

| Conditional localization metric | Train-mean position | STES best |
|---|---:|---:|
| Mean distance, 224x224 pixels | 28.65 | 26.60 |
| Median distance, 224x224 pixels | 21.53 | 20.87 |
| Within 6 pixels | 8.63% | 4.32% |

Event macro F1 at +/-10 native frames is zero with the selected decoder;
there are no accepted detections at its selected threshold. Mean temporal AP
at 0/1/2/4 frames is 0.76%. This pilot does not demonstrate a usable spatial
event spotter. Lower average coordinate error is insufficient evidence of
precise localization; the stricter hit rate is worse than the constant baseline.
All 320 optimizer updates completed without AMP overflow skips.

Artifacts: report.json, canonical_labels.json (all merge decisions), best.pt,
last.pt, best_predictions.npz, and spatial_contact_review.jpg under the run
directory. The gallery samples three evenly spaced contacts per class, rather
than the easiest predictions. Green is human XY, orange is model XY. These
images use known contact frames and must not be presented as detection successes.

## High-resolution input: 1248x702

RegNet/GSM accepts rectangular spatial dimensions; there is no learned 224-pixel
input-size restriction. The training runner now accepts --input-width,
--input-height and --clip-frames. Non-224 inputs get a separate hash-checked RGB
cache decoded from the original 4K sources. Enlarging the old 398x224 cache is
explicitly rejected by the reader. At 1248x702 the original 16:9 aspect ratio is
preserved, so normalized human coordinates require no crop/padding adjustment.

The RTX 3080 10GB cannot practically process a complete 64-frame high-resolution
backbone batch in the tested setup: even block checkpointing saturated GPU
memory and became very slow. That probe was cancelled; it was not a captured
out-of-memory exception.

The working architecture keeps the full 64-frame sequence for the original
three-layer BiGRU, while processing the image backbone in 16-frame chunks.
GSM operates inside each 16-frame chunk, so this deliberately changes its local
temporal boundary conditions. No frames are skipped. Block and whole-chunk
activation checkpointing recompute features during backward. BatchNorm buffers
are cloned for recomputation and restored afterward to avoid updating running
statistics twice. Tests compare outputs, parameter gradients and BatchNorm
buffers against an uncheckpointed model with the same chunk layout.

Measured on this RTX 3080 with AMP and batch one:

| Probe at 1248x702 | Peak allocated VRAM | Result |
|---|---:|---|
| 8 frames, normal backbone | 3.92 GiB | Forward/backward passed |
| 16 frames, block checkpointing | 3.00 GiB | Forward/backward passed |
| 32 frames, block checkpointing | 5.71 GiB | Forward/backward passed |
| 64 frames, chunk 16 plus checkpointing | 3.14 GiB | Forward/backward passed |
| 64 real training-source frames, chunk 16, AdamW step | 3.45 GiB | Parameters updated |

The real-frame check uses j0 frames 544-607, including two verified contacts,
ImageNet initialization, both losses, AMP and a real optimizer step. Its reserved
GPU memory peak was 4.78 GiB. Probe weights were not saved. This verifies memory
and training compatibility, not improved accuracy. There is no completed
high-resolution training/evaluation run yet.

Example full pilot command (new output directory):

```powershell
$env:PYTHONPATH = ((Resolve-Path '.runtime/vnl_stes_deps').Path + ';' + (Resolve-Path 'volley_env_311/Lib/site-packages').Path)
& .python/cpython-3.11.14-windows-x86_64-none/python.exe -m scripts.evaluation.run_spatial_stes --output results/vnl_stes_spatial_1248_20261027_v1 --input-width 1248 --input-height 702 --clip-frames 64 --backbone-chunk-frames 16 --activation-checkpointing --batch 1 --accumulation 8 --steps 128
```

128 microbatches of one clip match the old 64 microbatches of two clips per
epoch; accumulation eight preserves the effective eight-clip batch size.
High-resolution runs disable cuDNN benchmark autotuning to avoid large temporary
workspaces. Metrics include actual input-pixel distances alongside the fixed
224-reference and native-pixel metrics. Checkpoints record input dimensions,
temporal clip length and backbone chunk size, which inference must restore.
An accuracy comparison must account for the changed GSM chunks and BatchNorm
microbatch statistics as well as resolution. Existing 224-run artifacts remain
unchanged; their frozen code snapshots preserve the original implementation.
