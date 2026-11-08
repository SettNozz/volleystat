# Model outputs first; hypotheses replay them

All normal video/frame analyses automatically save compatible model outputs in
`data/inference_cache`. This directory is local and gitignored. A fresh result
directory does **not** mean fresh inference. Existing analysis files are never
overwritten.

The default output is JSON plus the viewer over the original video. MP4/frame
rendering is explicit: `--render-video` or `scripts/render_saved_analysis.py`.
`--no-video` remains accepted for compatibility.

## Normal capture and repeat

```powershell
# First analysis: compute missing model outputs and persist them.
python analyze_video.py --video game.webm --output results/game_v1 --calibration camera.json

# New post-processing hypothesis: inference is forbidden.
python analyze_video.py --video game.webm --output results/game_smooth_v2 --calibration camera.json --ball-smooth light --require-model-cache

# Inspect saved stage coverage without starting a model.
python scripts/model_cache.py inspect --video game.webm

# Render only when the final accepted annotations need a video export.
python scripts/render_saved_analysis.py --annotations results/game_smooth_v2/game_annotations.json --calibration camera.json --output-video results/game_smooth_v2/game_analyzed.mp4
```

`--model-cache-dir DIR` changes cache location. `--require-model-cache` raises
`CacheMiss` if a required model output is absent; it never silently falls back
to GPU inference. Without that flag, only missing/incompatible stages execute.
Configured checkpoint files must remain available for checksum validation.

## What is saved

| Stage | Frozen output | Replay behaviour |
|---|---|---|
| Ball detector(s) | All detector boxes/confidences before selection, static suppression, DP and smoothing | Reapply selection/trajectory policies; each detector has its own key |
| YOLO-pose | Native boxes, scores/classes and all 17 keypoints **before ByteTrack and court filtering** | Replay the installed ByteTrack callback on CPU; rerun current court/team rules |
| ByteTrack | Complete ordered derived track sequence | Reuse compatible sequence; rebuild from cached raw poses for new scope/tracker code |
| OSNet | Frozen crop embeddings, keyed by source frame and actual clamped/rounded pixel crop | New seed, roster, Kalman or ID policy reuses embeddings; only new crops need inference |
| Net | Resolved tape/floor/calibration result and model provenance | Reuse compatible geometry; different geometry/height invalidates this stage only |
| Event overlay | Raw neural logits keyed by checkpoint, backend and exact input tensor | Replay phase gates, smoothing and display rules |
| Contact spotter, when enabled | Per-frame probabilities before constrained decoding | Decode/segment again without neural forward passes when features are unchanged |
| SAM-guided recovery, when enabled | Raw crop-pose outputs and recovered proposals with provenance | Reuse compatible crops/proposals; geometry-dependent filtering remains separate |

No cached model output is detector or human ground truth. Cached reconstructed
ball coordinates are never relabelled as observed detector boxes. Human checks
never enter cache keys or association hints.

Keys bind source **bytes**, checkpoint checksum, inference/preprocessing
settings, backend version and relevant library versions. They exclude roster
names, identity checks, Kalman parameters and downstream trajectory settings.
Raw detector caches exclude court calibration. Neural feature-input changes
invalidate the affected prediction, not unrelated detector outputs.

Stateless outputs are committed as they are produced. Interrupted track
sequences stay incomplete and cannot be reused as completed tracks. Their raw
YOLO outputs survive: the next run rebuilds tracking from those outputs and
infers only missing frames. Corrupt or conflicting entries fail rather than
being silently replaced. Native empty frames and track-box clipping follow
the installed Ultralytics callback; parity is tested against that callback.

A complete compatible detection replay with no requested rendering skips video
decoding. Cached ReID also skips its image iterator and does not load OSNet.
Image-dependent SAM recovery and explicit rendering can still require source
pixels. Optional event checkpoints may load their architecture even when all
forward outputs are cached. This avoids repeated inference, not CPU work on
new hypotheses.

`model_cache` in both JSON exports records source, directory, hits/writes,
computed stages and OSNet batch counts. Identity provenance also records
`encoder.model_loaded_this_run`.

## Background inference on a cold or partial cache

`src/pipeline/model_prefetch.py` computes missing stateless outputs (raw ball
proposals, tiled ball proposals, raw YOLO-pose) ahead of the ordered loop:
one decoder thread, one thread per ball detector and `POSE_PREFETCH_WORKERS`
pose threads, each with its own model instance. Before the loop the main
thread reads which frames each stage already stores; workers infer only the
others. Workers never touch the SQLite connection, ByteTrack, gate or FSM; the
main thread stores each value exactly as the serial path would. Model calls
keep their batch size, input size and precision.

Raw pose (`pose_raw`) is computed by `src/detection/pose_gpu.py`: GPU
letterbox, the Ultralytics NMS/scaling, and a CUDA graph for the network.
Its key includes the source of `_infer_pose_model` and of `pose_gpu.py` plus
`preprocess=gpu_letterbox_bilinear_round_v1`, so pose caches written by the
earlier CPU-letterbox path are not reused (they stay on disk untouched; a new
analysis recomputes pose once). Bilinear resampling in float32 differs from
cv2's fixed-point by at most one grey level; boxes moved by a median 0.4 px
on a 4K sample. Ball and OSNet keys are unchanged.

OSNet embeddings for every saved pose box are encoded during the loop by a
`BackgroundEncoder` thread from the exact `ann['poses'][i]['bb']` the identity
pass looks up, serialised off the main thread and stored in one transaction
per drain. A cold run therefore needs no second decode for identity or contact
authors; anything missing still falls back to decoding. `FrozenOSNet.encode`
prepares crops with `src/models/osnet_fast.py`, whose network input is
bit-identical to `OSNetEncoder.transform`, so the OSNet key (which hashes
`player_reid_encoder.py`) stays valid. Stored values still vary ~1e-5 cosine
between full runs because `cudnn.benchmark` picks convolution algorithms by
timing.
`tests/test_inference_cache.py::test_model_prefetch_matches_serial_pass_and_cache`
checks identical annotations and identical stored checksums against
`--no-model-prefetch`.

Prefetch is off with `--require-model-cache` (nothing to compute) and when no
cache is used. Even serial runs are not bit-identical on a GPU (FP16 with
`cudnn.benchmark`): measure prefetch effects against a serial control run,
never against a single earlier run. OSNet now encodes 128 crops per forward
(`OSNET_BATCH_SIZE`); against batch 8 that moves embeddings by at most
1.3e-6 cosine distance, and embeddings are keyed by crop, not batch.

## Existing analyses and the JIZ3exHe3hA experiment

Old final annotations do not contain discarded ball candidates or unfiltered
YOLO detections. Never pretend that a smoothed trajectory/court-filtered pose
list is a raw detector cache. Reuse those annotations for hypotheses that need
only those frozen inputs; new raw-detector hypotheses need missing outputs to
be captured once.

The previous frozen full-video OSNet artifact can be verified and imported
without any model invocation:

```powershell
python scripts/model_cache.py import-reid --cache results/player_identity_kalman_20261030_v1/target_features_v4
python scripts/apply_player_identity.py apply --annotations results/JIZ3exHe3hA_identity_roster_20261030_v4/JIZ3exHe3hA_4k_annotations.json --output results/game_id_hypothesis_new --seed results/player_identity_kalman_20261030_v1/roster_with_P7_v1.json --require-model-cache --no-model-registry
```

The import checks video, annotation, observation, embedding, encoder-source and
pinned-model fingerprints. It preserves the original producer provenance and
does not claim a previously unrecorded runtime version. The explicit frozen
artifact is the baseline; it is not silently re-encoded.

Measured replay on the full 17592-frame JIZ3exHe3hA analysis: **93155 cached
crops, OSNet not loaded, zero new OSNet batches; identity stage about 47 s**.
This is runtime evidence, not identity accuracy. Cached association outputs
are compared with the accepted v4 run; original artifacts and viewer court
corrections remain intact.

## Export while the viewer is being edited

The standalone renderer saves an immutable `.render_input.json` first and
records its checksum. A later viewer court/heatmap save remains in the result.
That metadata-only edit does not invalidate the same frame inputs; a change
to poses, IDs, ball, FPS or other render inputs does fail verification. The
render provenance records both the frozen input and current viewer checksum.
