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
