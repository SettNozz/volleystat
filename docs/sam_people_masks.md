# SAM3 people-mask experiment

This worker adds model-derived body masks alongside existing final annotations.
It does not change production inference, ByteTrack IDs, teams, FSM contacts,
ball coordinates, human labels, existing features or checkpoints.

## Run locally

Use the existing isolated `.sam3_env` (Python 3.12, CUDA PyTorch, SAM3, OpenCV,
Pillow, NumPy); the production Python 3.11 environment stays separate.
SAM3 must already be installed and its **image** checkpoint available locally.
No download is performed by this worker.

```powershell
$samPython = '.\.sam3_env\Scripts\python.exe'
$checkpoint = (Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--facebook--sam3\snapshots\*\sam3.pt" | Select-Object -First 1).FullName
$settings = @(
  '--video', 'data\labeling_intake\2026-08-10_youtube\hy2S-f7zj9Q_4k.webm',
  '--annotations', 'results\hy2S_feature_replay_20261026_v1\new\hy2S-f7zj9Q_4k_annotations.json',
  '--checkpoint', $checkpoint,
  '--output', 'results\hy2S_people_sam3_manual_v1',
  '--start', '15', '--duration', '30', '--max-width', '1280',
  '--precision', 'bfloat16'
)
& $samPython scripts\evaluation\sam3_people_masks.py @settings --check
if ($LASTEXITCODE -eq 0) {
  & $samPython -u scripts\evaluation\sam3_people_masks.py @settings
}
```

An existing output directory is refused, including incomplete runs. Pick a new
version when retrying. `--check` validates geometry, FPS, interval coverage,
checkpoint and CUDA, hashes inputs and code, and prints settings. It creates
no output and does not guarantee that a model will fit in GPU memory.
The tested RTX 3080 supports bfloat16. Other explicit precision choices are
float32 and float16; there is no automatic fallback.

All source frames in the requested interval are inferred. At 29.97002997 FPS,
00:15-00:45 selects absolute frames 450 through 1348 (899 frames). The video
is decoded from the beginning to avoid WebM random-seek frame ambiguity.
The source is 3840x2160, masks/preview use a 1280x720 grid by default. Native
geometry and court coordinates are mapped explicitly to that grid.

## Outputs and meaning

- `manifest.json`: settings, input/checkpoint/code hashes, geometry, source,
  schema, temporal method and completion status. Only `status=complete` is a
  usable run; `failure.json` identifies a failed partial run.
- `comparison.mp4`: saved court-filtered YOLO poses at left; SAM masks at right.
  Green means matched saved pose, orange means an unmatched court candidate,
  gray means outside court. The same final exported ball is drawn in both.
- `preview_f*.jpg`: first, middle, last comparisons.
- `masks.jsonl`: one row per absolute source frame; masks, scores, experimental
  SAM instance IDs, optional existing YOLO ID/team, temporal IoU, provenance,
  features and validity. RLE is uncompressed **row-major**, not COCO encoding.
- `review_candidates.jsonl`: candidate person-frame records without RLE.
- `mask_features.npz`: frames `(T,)`, float32 values `(T,8)`, boolean validity
  `(T,8)`, ordered names, schema, full source stem and ball flags. Loads safely
  with `allow_pickle=False`.
- `report.json`: throughput, GPU memory, matched/unmatched person-frame counts,
  temporal IoU and input preservation. Counts are observations across frames,
  not distinct people or confirmed detector misses. No human accuracy score.

The worker prompts SAM3 with `person`, then matches mask bounding boxes to saved
`poses[*].bb` using maximum-total-IoU one-to-one assignment (minimum IoU .15).
Accepted existing court members retain their original IDs and teams, even when
briefly outside the polygon. An unmatched mask's bottom-center inside the
calibrated polygon makes it a **review candidate**, not a new official player.
This simple candidate filter does not replace CourtPlayerGate or support the
full serving zone/hysteresis logic. Arms extending outside the body can also
shift bbox-bottom-center. Spectators are stored but excluded from court features.

Temporal association is causal mask IoU between adjacent source frames only
(minimum .15). Conflicting known YOLO IDs cannot merge. Missing/empty frames
break association. This is not native SAM3 video propagation, optical-flow
warping, ReID or full-occlusion recovery. An absent current YOLO pose never
receives fabricated keypoints, a team, or a carried YOLO identity.

## Is it a YOLO miss or downstream filtering?

Saved `poses` already passed tracking and court/player filters. To inspect five
source frames with raw YOLO-pose detections at 640 and 1280, run in the production
environment containing Ultralytics:

```powershell
python scripts\evaluation\compare_sam_yolo_samples.py `
  --experiment results\hy2S_people_sam3_20261026_v1 `
  --pose-model models\pretrained\yolo11m-pose.pt --device cpu
```

This writes a new `pose_diagnostic/` folder and refuses to overwrite it. It
records boxes, confidences and real re-inferred keypoints, candidate matches,
checkpoint hash and three-panel JPEGs. CPU float32 per-frame predictions are
not an exact replay of the original half-precision ByteTrack history. A raw
match suggests tracking/filtering loss; an absent raw match still requires
visual verification. Diagnostic keypoints are not merged into saved results.

## Training contract

`person_mask_features_v1` has these eight values, always accompanied by eight
boolean validity channels:

| Value | Meaning |
|---|---|
| ball_to_person_mask_distance | Minimum distance to eligible visible mask cells; x and y normalized by native width/height |
| ball_inside_person_mask | Final ball pixel is within any eligible mask |
| closest_person_mask_area_ratio | Visible closest mask area / full mask grid area |
| closest_person_mask_score | SAM confidence, not a calibrated correctness probability |
| closest_person_mask_previous_iou | Raw IoU with the adjacent previous mask, without motion compensation |
| yolo_pose_mask_match_fraction | Fraction of saved poses matched to a SAM mask |
| sam_unmatched_in_court_count | Unmatched masks accepted as court review candidates |
| court_person_mask_count | Matched court members plus court review candidates |

Missing values are numeric zero **and valid=false**. They must not be treated as
observed zero. A completed zero-person detection has valid zero counts. A missing
ball, nonfinite/out-of-frame coordinate, or `ball_out_of_frame=true` invalidates
the first five channels; `ball_raw` is never substituted. No previous ball is
carried into a gap. Previous-mask IoU is independently unavailable on a new mask.

To use these in training, join on exact full `source_video` identity AND absolute
frame index, verify manifest completion/schema/feature order/input hashes, and
introduce a new checkpoint schema. Concatenating the existing 18 values with
8 mask values and 8 validity bits yields **34**, not 26 channels. Fit normalization
only on valid training observations, fill missing numeric values using those
training statistics and retain validity bits. Split by complete source video.
Masks are weak model-derived features, never human labels or ball ground truth.
Do a cross-video ablation before changing default features/checkpoints. This
experiment does not retrain or refresh the authoritative dataset.

## Tests

```powershell
.\.sam3_env\Scripts\python.exe -m unittest discover -s tests -p test_person_masks.py -v
```

These cover RLE, Hungarian assignment against exhaustive matchings, court
membership, spectator exclusion, immutable source poses, temporal gaps/conflicting
IDs, final-ball validity, resolution scaling and exact interval indexing.
Real GPU inference and output validation are separate checks, recorded with the
local experiment results.

## Player recall and court-side correction

The production CLI and viewer now use YOLO-pose `imgsz=2560` by default. This
improves small/far player recall on native 4K footage at additional inference
cost. The CLI accepts `--pose-imgsz` for an explicit lower/higher setting.
Six per side remains an upper bound on actual detections, never a target filled
with synthetic players.

Detected net-floor geometry is checked against the existing post-spacing
plausibility test. An impossible detected line (including the 14.08m Hough
result in the hy2S comparison) otherwise puts both sides on one team and lets
the six-per-team cap discard real players. When a valid court homography is
available, runtime geometry uses its projected world-y=0 centre line at the
stored post X coordinates, preserving the calibrated slope. Results mark this
as `calibrated_court_midline_fallback`; this line is inferred, not measured
post-base ground truth. The source calibration JSON is not rewritten. Existing
plausible detected dividers retain precedence. Runtime validation also checks
fresh net detection before scaling/using the shared calibration.

This correction is independent of SAM people masks. SAM currently provides the
separate diagnostic/feature sidecar documented above; it does not fabricate
missing poses or automatically force a six-player count.

## SAM-guided recovery of missing poses

`analyze_video.py --person-masks <completed-experiment/masks.jsonl>` optionally
uses existing SAM person masks as **crop proposals**, not as poses. The sidecar's
source video bytes, dimensions, FPS, completion and full interval coverage are
validated before use. Frames outside its recorded interval use normal inference.

For each current tracked pose, a mask is matched one-to-one. For each unmatched
eligible person mask, an independent YOLO-pose instance examines a padded crop
at input size 640. Only actual YOLO bounding boxes/keypoints with torso evidence
and spatial agreement are accepted; overlapping duplicate poses are rejected.
All resulting poses still pass the original CourtPlayerGate and six-per-side cap.
No blank player, inferred skeleton, copied keypoint or forced count is added.

Crop inference uses a separate model instance so ByteTrack never receives cropped
images or advances its state on them. Primary YOLO tracks keep positive IDs.
Recovered tracks have `source=sam_guided_yolo_pose`, `mask_instance_id`, and a
negative SAM-mask ID namespace. This identifies a mask association, not a resolved
person identity across handovers/occlusion. Switching between primary and recovery
can change the displayed ID. The annotation `player_detection.sam_guided_recovery`
records proposal/accepted person-frame counts and input provenance.

Example, after the SAM people-mask experiment above is complete:

```powershell
python analyze_video.py `
  --video data\labeling_intake\2026-08-10_youtube\hy2S-f7zj9Q_4k.webm `
  --calibration results\hy2S_feature_replay_20261026_v1\frozen_inputs\court_calibration.json `
  --output results\hy2S_players_sam_manual_v1 `
  --pose-imgsz 2560 `
  --person-masks results\hy2S_people_sam3_20261026_v1\masks.jsonl `
  --duration 45 --no-video --no-net-detection
```

Use a fresh output path. As before, this does not refresh human datasets or train
a model. The optional sidecar currently comes from the separate SAM worker;
viewer analyses without that option benefit from higher pose resolution and
validated court geometry. SAM-guided recovery is not automatically enabled for
unrelated uploads.

The mask-feature NPZ from the original SAM run remains tied to its original
annotation hash. It cannot silently be treated as recomputed features for the
new recovered poses/ball trajectory. Recovery consumes only spatial proposals;
the main workflow recomputes its standard pose/event features from its own final
annotations. A future combined training export must recompute mask features
against those same final annotations and introduce the new checkpoint schema.

Render a saved comparison with:

```powershell
python scripts\evaluation\compare_player_teams.py `
  --before results\hy2S_feature_replay_20261026_v1\new\hy2S-f7zj9Q_4k_annotations.json `
  --after results\hy2S_player_teams_sam_20261026_v2\hy2S-f7zj9Q_4k_annotations.json `
  --video data\labeling_intake\2026-08-10_youtube\hy2S-f7zj9Q_4k.webm `
  --output results\hy2S_player_teams_sam_20261026_v2\comparison_15_45
```

This produces H.264 video, samples, counts and a local review HTML. Detection
counts are not verified player/team accuracy; review spectator/occlusion cases.
