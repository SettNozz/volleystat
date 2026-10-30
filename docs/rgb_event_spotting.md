# Continuous RGB event spotting

This is an **STES-inspired baseline**, not a reproduction of VNL-STES (CVPR
Workshops 2025): frozen ImageNet MobileNetV3-Small replaces RegNet-Y/GSM, and
there is no spatial regression head. A three-layer bidirectional GRU predicts
background, serve, reception, pass and attack on continuous native-rate video.
Future images are allowed. There are no ball/pose/FSM inputs or FSM labels.

Reference: https://hoangqnguyen.github.io/stes/

The active experiment is `results/rgb_event_spotting_20261027_v2`.
The CPU extraction draft v1 was explicitly cancelled; its partial arrays are
not training-ready. V2 uses verified CUDA decoding/scaling through FFmpeg,
with native-frame passthrough and no FPS resampling. Container frame-count
rounding of at most two trailing frames is recorded by retaining all actual
decoded frames; larger decode discrepancies are refused.
`preflight.json` records video, human source, backbone and parent manifest hashes.
`features/` holds frozen full-frame RGB features; `*_labels.json` retains event
times and all annotation IDs. All 414 original clip records remain in the
parent manifest. Unsupported raw contacts have ignored neighbourhoods rather
than becoming negative examples (five timestamps in Khe, including one with a
coincident core label). Nearby/coincident core labels remain authoritative.
Only exact frame/class duplicates collapse in the derived event stream.

Frames use 224-pixel letterboxing to preserve the entire scene. Every decoded
source frame is retained; there is no temporal subsampling. Features are cached
as float16 and normalized using training-source statistics only. The small
input resolution and frozen backbone can limit visibility of the ball and
distant hands; improvements are not assumed.

Training uses 120-frame windows, 60-frame stride, background-aware weighted
cross entropy, AdamW and early stopping. Splits hold out a whole source video,
with Khe primary, then j0 and j3. All three folds are model-selection estimates,
not an independent test. Epoch and one global detection threshold are selected
on the held-out video; results must not be presented as unbiased test scores.
Native-frame NMS and one-to-one event matching produce precision, recall and F1
at 1/2/4/5/10/15-frame tolerance and false events per minute. These F1 values
cannot be compared directly with contact-centred clip-classification F1.

```powershell
python scripts/evaluation/run_rgb_event_spotting.py --help
python scripts/evaluation/run_rgb_event_spotting.py --output results/rgb_event_spotting_new_version
```

The runner refuses existing outputs unless `--resume` is explicit. Resume checks
frozen inputs, reuses completed per-video feature caches and completed folds;
an interrupted extraction restarts that video. Partial training folds are
refused rather than silently treated as completed. Originals and production
models are never overwritten or promoted. `progress.json` reports the stage;
`report.json` is written only after all three folds finish. The runner needs an
active local process until completion; it is not a background scheduler.

Tests: `tests/test_rgb_event_spotter.py` and `tests/test_rally_spotting.py`.

## First results

V2 completed all three folds on 61,822 native frames and 410 unique core events.
Validation macro F1 at +/-10 native frames (about 0.33 seconds):

| Held-out source | Macro F1 | TP | FP | FN |
|---|---:|---:|---:|---:|
| Khe4P8QcJkg_5_4k | 0.0841 | 8 | 12 | 154 |
| j0gRhgFCgIs_4_4k | 0.2615 | 31 | 103 | 97 |
| j3BO9GgUFBI_3_4k | 0.2474 | 27 | 55 | 93 |

These model-selection results are poor, especially primary-source recall
(8/162). The frozen low-resolution RGB baseline is not production-ready and
does not establish the performance of original end-to-end VNL-STES. Source
annotations, the previous datasets, old checkpoints and production remain
unchanged. The report contains tighter temporal tolerances and per-class scores.
