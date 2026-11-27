# Wide and fisheye video correction

GoPro Wide recordings can curve court lines enough to invalidate a planar court
homography. Straighten the image before court calibration, player projection,
ball detection and identity inference. This is an opt-in source preparation step;
the ordinary uncorrected workflow stays unchanged.

## Viewer

1. Select the original recording on **New analysis** and enable **Straighten Wide / fisheye footage**.
2. Adjust **Wide correction** until straight court lines and wall seams look straight.
   The displayed angle is a model fitting parameter, not a measured factory lens FOV.
   Adjust **Framing**: zero crops black borders; one keeps more of the source view.
   Higher framing can leave black borders. Court corners may remain outside the image;
   use the court-line handles when needed.
3. Check several preview times. Then align the court and net on the corrected image.
   Changing lens settings discards the old handles. A calibration from raw pixels
   cannot be submitted for the corrected projection.
4. Start **Analyze Video**. Preparation runs first and reports progress. The result
   uses the corrected recording, including player portraits and original-video
   playback with annotation overlays. No detectors run during lens previews or
   video preparation.

**Use automatic settings** turns correction off and closes the manual calibration.
The viewer refuses corrected analysis without a matching manual preview calibration.

The correction uses an equidistant fisheye model with an adjustable focal scale.
It is a practical manual fit, not a factory-calibrated HERO10 profile. HyperSmooth,
digital zoom and exported video crops can change the effective projection. A static
map cannot undo time-varying stabilization warps or camera motion. Recheck court
alignment across the recording when stabilization settings are unknown.

## CLI preparation

FFmpeg and ffprobe must be on PATH. The project Python environment supplies OpenCV.
Copy `configs/lens_correction_wide.example.json` and tune its settings for the source:

```json
{
  "enabled": true,
  "horizontal_fov_deg": 130.0,
  "balance": 0.5
}
```

Preview one frame without encoding the complete video:

```powershell
python scripts/rectify_video.py --help
python scripts/rectify_video.py --video "C:/path/gopro.mp4" `
  --profile configs/lens_correction_wide.example.json `
  --preview wide_preview_v1.png --seconds 10
```

Prepare the complete video after checking the preview:

```powershell
python scripts/rectify_video.py --video "C:/path/gopro.mp4" `
  --profile configs/lens_correction_wide.example.json
```

The command prints the corrected video path and provenance. Calibrate **that file**
using the existing [CLI calibration workflow](usage.md#cli-calibration), then pass
it to `analyze_video.py --video`. Do not pass a calibration or identity seed drawn
on the distorted source. CLI preparation does not run analysis or rewrite labels.

## Files, timing and cache

Corrected recordings live in `data/rectified_videos/<key>/<source-stem>.mp4`.
The sibling `lens_correction.json` records the original path and SHA256, correction
parameters, OpenCV version, output SHA256, encoder, frame count and coordinate space.
Viewer jobs also save `video_preprocessing.json` beside their result JSON files.
Keep the corrected recording while results reference it: overlays are calibrated
to that file and will not align with playback of the distorted original.

The output retains native image dimensions and a constant frame rate. Preparation
checks decoded frame count and duration, and refuses disagreement in reported frame
rates or duration rather than silently accepting truncated decodes. Use CFR sources;
these checks do not prove that every source timestamp is evenly spaced. Audio is copied for compatible
MP4 codecs, otherwise encoded to AAC. Image correction requires resampling and a
new encode: H.264 NVENC at CQ 16 when available, otherwise libx264 at CRF 16.
`--encoder libx264` or `--encoder h264_nvenc` selects one explicitly.

Identical source bytes/settings reuse a verified corrected file. A changed profile
creates a separate artifact; a changed/incomplete existing output is refused.
Model-output caching uses the bytes of the corrected video. Detections from the raw
video never become detections of rectified pixels. Changing this implementation's
pixel transform or encoder policy requires a new preparation version.

## Local source check: en9RD3w5Kw8, 2026-11-27

The downloaded 25:00–48:00 excerpt was checked at clip times 10, 600 and 1200 s.
Fitting the known straight front orange baseline gave model angles 112.6264,
112.6260 and 112.6462 degrees. Trimmed line residual fell from 51.98–55.76 pixels
to 0.48–0.54 pixels in the fit projection. This checks that line's curvature;
it is not a full intrinsic calibration or an analytics accuracy measurement.

The source-specific manual profile uses `horizontal_fov_deg: 112.6`, `balance: 0.5`.
It does not replace the generic starting profile. Source video and human labels
remain unchanged; models have not been retrained or evaluated on corrected frames.
