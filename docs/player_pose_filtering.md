# Perspective-aware player pose filtering

## Reported failure

The saved `job_net_geometry_review_20261110_v1` analysis of
`7YnJlvChSzw_4k.webm` contains a skeleton on empty foreground floor at frames
4752–4753 (2:38.56–2:38.59, 29.97003 fps). The reported `F45752` corresponds
to f4752 at the supplied timestamp; this analysis has 21,300 frames.

Raw cached YOLO output confirms the false detection: confidence 0.355, box
approximately `[1344, 1784, 1425, 2026]`, with confident shoulders, hips,
knees and ankles. ByteTrack assigns track 851. This is not a stale canvas
overlay or a keypoint-index mismatch. Its feet project inside the legal
serve area, so the previous court-membership gate admitted it.

## Filter

`filter_perspective_pose_outliers` in `src/detection/player_pose.py` runs in
`CourtPlayerGate.filter` before membership, occlusion recovery and team caps.

For complete torso-and-leg observations, it divides the longest box dimension
by the image length of a one-metre court span across the projected ankles.
With at least three complete on-court observations, it rejects extents below
30% of their median. The reported ghost has normalized extent about 0.32,
versus approximately 2.0 for real players in that frame.

This is a relative perspective check, not a measured human height. It has no
fixed foreground mask, video-specific coordinates, roster restriction or
minimum track lifetime. Missing torso/leg evidence, too few references or
unusable projection scale cause abstention. Using the longest dimension
preserves wide diving poses; the low threshold allows compact players.
Without a court projector the existing fallback remains in use.

The gate changes downstream accepted poses and player centres together before
feature extraction, identity and exports. Raw YOLO and ByteTrack caches are
unchanged and remain reusable. Previously saved results need an explicit
repair or cache replay; changing pipeline code does not rewrite old JSON.

## Validation scope

Regression tests cover the reported native coordinates, down/upscaled input,
homogeneous matrix scaling, existing membership grace, partial observations,
insufficient/on-court references and compact/diving players.

A read-only pass over all 21,300 saved frames rejects 35 observations across
nine raw tracks, including both track-851 observations. All 35 have unknown
roster identity and foreground-floor boxes; visual inspection of one frame
from every rejected track confirms empty floor. A fresh sequential extraction
with and without those observations produces identical `pose_f` for all
21,300 frames. This is a development-video
regression check, not a held-out pose-quality benchmark.

## Split near-camera skeletons

`src/detection/pose_fragments.py` runs after extraction of tracked poses and
before court membership, identity, player caps and feature extraction. It
addresses duplicate upper/lower-body detections of the same person during
near-camera serves. Box overlap alone is insufficient: merging requires
matching hips, at least three confident named torso/body anchors, comparable
hip widths and small joint displacement relative to that width. Every member
of a merged group must agree with every other member. Distinct known roster
IDs never merge. Court-side predictions do not decide whether two observations
describe the same person.

The retained track prefers visible ankles. Each arm/leg chain comes from one
observed donor; coordinates are never averaged across incompatible arms.
The exported box covers the contributing observations. `pose_repair` records
the original IDs, boxes, keypoints, prior repair provenance, per-joint donor
indices, `label_source: model` and `verified: false`. Raw YOLO and ByteTrack
cache entries remain unchanged.

A court candidate or established court member with a box at least one quarter
of image height, confident hips and one visible body half can request a
targeted crop of its missing half. A separate YOLO-pose instance uses the
same pinned weights with `imgsz=1280`, `conf=0.15`; it never touches the primary
tracker. The crop includes the hip anchors. Only one matching candidate may
contribute, and it must add confident joints. For a crop whose knees are cut
off and a source whose shoulders are absent, a precise high-confidence hip
pair is sufficient; this exception never merges independent full-frame
people. Occlusions with no trustworthy body half, ambiguous crop matches,
small observations and failed recovery remain partial.

New crop outputs live in `pose_fragment_crop_raw`, bound to source bytes,
frame, native crop rectangle, weight hash, inference implementation, settings
and runtime. Post-processing changes do not invalidate existing raw crops.
A warm replay loads neither crop pixels nor YOLO. `--require-model-cache`
raises on a missing crop rather than silently skipping recovery. Changed
boxes/features may also require new OSNet embeddings, event logits or contact
probabilities; capture those missing inputs explicitly before strict replay.
Previously saved results require a fresh output directory and replay.

The reported `XNCS5BLBwig_00-00-00_00-02-30` f3376 has two saved tracks, 530
and 593, on the same server. Reconciliation retains 530 with the observed
raised arm from 593 and legs from 530. At f3366 the raw full-frame model
contains only the lower-body detection; a targeted same-frame crop supplies
the shoulders/arms. Hidden or low-confidence face joints remain missing.
These are development-video checks, not a held-out pose-accuracy score.

Validation on this 4,496-frame source produced a fresh paired result in
`results/viewer/job_pose_fragments_20261112_v1`. After explicitly capturing
136 missing pose crops, 236 new OSNet crop embeddings and one contact
probability output, strict replay wrote zero new model outputs. All final
`pose_f` values match fresh sequential extraction, roster IDs remain unique
per frame and each side respects the six-player cap. The earlier P6
f520–560 and P2 f3001 team fixes remain intact. Exported ball coordinates
match the previous result on every frame. Both the result page and annotation
endpoint return HTTP 200. Targeted subsystem validation: 117 passing tests.
