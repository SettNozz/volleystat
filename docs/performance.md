# GPU throughput notes — 2026-11-06/07

Historical measurements moved from the README. These runs used the model/settings available on those dates, before the 2026-11-08 promotion of the 1920 ball detector. They are not a benchmark of the current full pipeline.

New analysis reuses compatible cached outputs. Changes to tracking or downstream logic should be evaluated from those outputs; see [cache/replay workflow](model_output_cache.md).

## Implementation and measurements

GPU utilisation. Inference that the cache does not hold yet runs ahead of the
ordered analysis loop on background threads: one decoder thread, one ball
thread and two YOLO-pose threads (one model instance each). ByteTrack, ball
selection, FSM and all cache writes stay on the main thread in frame order.
`--no-model-prefetch` restores the serial loop for debugging.

- YOLO-pose runs through `src/detection/pose_gpu.py`: frames are decoded into
  pinned memory, uploaded once, letterboxed on the GPU and the network replays
  a CUDA graph (eager dispatch cost ~22 ms of GIL-holding Python per 2560
  frame). Boxes match the CPU letterbox path to a median 0.4 px (658/658
  detections matched on a 4K sample). This changed the raw-pose cache key:
  old pose caches recompute once; ball and OSNet caches are untouched.
- OSNet crop embeddings for every saved pose are computed during the
  detection loop while the frame is still in memory, so the identity pass (and
  the contact-author runtime) read them from the cache instead of decoding the
  video again. Same encoder, crop and key (max 2.4e-6 cosine distance against a
  fresh identity-pass encode).
- A stopped MLflow server is detected by a 0.5 s TCP probe instead of ~8 s of
  HTTP retries per logical model.
- `--ball-batch-size` stays 8: 16 was only 8% faster and the batch size is part
  of the ball cache key, so changing it would recompute every cached video.

Measured on the first 30 s (899 frames) of `j0gRhgFCgIs_4_4k`, empty cache,
RTX 5070 Ti, i9-10900F, MLflow server not running:

| Run | Detection loop (steady) | Identity pass | Wall time |
|---|---|---|---|
| Serial, OSNet batch 8 (before, 2026-11-06) | 6.4 fps | 97 s | 314 s |
| Prefetch, CPU letterbox (2026-11-06) | 15.0 fps | 34 s | 151 s |
| + GPU letterbox, pinned decode (2026-11-07) | 25.6 fps | 29 s | 119 s |
| + registry probe, in-loop OSNet, CUDA graph (2026-11-07) | 24 fps | 3 s | 61-64 s |
| Replay of that run, `--require-model-cache` | - | - | 14 s |

In-loop OSNet moves ~11 s of encoding into the loop (35.6 fps without identity,
which is the decoder's ceiling) but removes the 24+ s second decode. GPU
utilisation in the steady loop went 47% -> 67%. Repeated runs are not
bit-identical even serially (FP16 with `cudnn.benchmark`): two serial runs
shared 822 of 899 raw ball outputs exactly and pose boxes moved by up to
15 px, which is why replays should go through the cache.

What limits the loop now (measured 2026-11-07, same clip): GPU work per frame
is ~20 ms YOLO-pose at 2560, ~10 ms OSNet (0.84 ms per crop for the
original+flip pair, ~12 crops per frame) plus ball and event models, about
36 ms, so ~28 fps is the ceiling with identity on; the loop runs at 23-24 fps.
Without identity it reaches 35.6 fps, which is the VP9 software decoder.
Tried and not adopted:

| Idea | Result |
|---|---|
| Hardware decode through OpenCV (`CAP_PROP_HW_ACCELERATION`, D3D11VA/NVDEC) | Slower (12.9 vs 17.5 fps in the same test) and different pixels (75% differ, up to 11 levels: other NV12->BGR conversion). Ball/OSNet keys do not include the decoder, so it would silently mix values. |
| FFmpeg decoder threads (`OPENCV_FFMPEG_CAPTURE_OPTIONS=threads;N`) | 34.2-35.1 fps for 1-16 threads; identical pixels, no gain. |
| OSNet PIL resize in 3 spawned processes | Bit-identical embeddings, loop 24 -> 19.2 fps (pickling crops costs more GIL than it saves). |
| CUDA graph for OSNet (bit-identical to eager) | Loop 23.1 fps, no gain: OSNet is GPU time, not dispatch. |
| OSNet `channels_last` / FP16 | -13% / -43% OSNet GPU time but 1.75e-5 / 3e-4 cosine change under the same cache key; would need a new OSNet key and recomputing cached embeddings. |

Remaining options considered at that point were a TensorRT FP16 engine for
YOLO-pose (new pose key) or FP16 OSNet (new OSNet key, requiring recomputation
of saved embeddings, including the imported JIZ3 cache). Neither is adopted
by these measurements.
