# Short occlusions: offline pose completion pilot

The viewer can now display bounded estimates of a known player's hidden joints
using observed poses before **and after** a short gap. This is a separate
offline reconstruction layer. It does not improve the underlying YOLO detector
or turn an invisible hand into measured evidence of a pass.

## JIZ3 result

```powershell
python scripts/complete_player_poses.py --help
python scripts/complete_player_poses.py `
  --annotations results/JIZ3exHe3hA_player_contacts_20261030_v1/JIZ3exHe3hA_4k_annotations.json `
  --output results/JIZ3exHe3hA_pose_completion_20261030_v1
python viewer.py --port 5052
# http://localhost:5052/load?video=JIZ3exHe3hA_4k&results_dir=results/JIZ3exHe3hA_pose_completion_20261030_v1
```

Use a fresh output directory. The command reads only the saved analysis pair;
it never opens video or loads models. It adds `pose_completion` provenance to
both JSON files and `pose_estimates` only to affected annotation frames.
Every original field remains unchanged. Existing original results and review
logs remain in their source directory; source-bound contact corrections are
not silently transferred to the new analysis fingerprint.

Select a roster player as usual. Measured skeletons retain their original
style; estimated anatomy uses cyan dashed lines and square joints. Other
players' estimates stay gray. **Estimated joints** toggles only the derived
layer. A completely missing skeleton gets an explicit `P1 · estimated` label.
Legacy results disable this toggle. Estimates never persist onto a different
source frame through the viewer's last-annotation fallback.

The frozen JIZ3 roster-v4-regret010 result has 17,592 frames. The pilot adds
88 missing-player frames and 275 partial-player frames, comprising 1,201
estimated joint-frames. For example, frame 295 shows the derived P1 skeleton
between observed anchors around a short overlap. These counts and the visual
example are not independently labelled accuracy measurements.

## Why disappearance has more than one cause

An absent stable ID can mean a genuinely missing pose, or an existing pose
whose ReID assignment was rejected. The JIZ3 cache currently contains frozen
OSNet embeddings; it does not contain the discarded raw YOLO pose outputs.
Final court-filtered annotations must not be re-labelled as a raw cache.

The pilot does not bypass an appearance veto with a tracker ID or a majority
label. An existing unassigned pose on the endpoint raw track, or a geometrically
matching pose, blocks a duplicate full-skeleton estimate. This conservative
choice leaves many identity gaps unresolved rather than inventing a second
person. Contact errors from identity rejection, ball gaps, event timing and
two-dimensional congestion remain separate problems.

## Reconstruction contract

- Same stable roster ID and court side on both ends; no extrapolation before
  first observation, after last observation or through a missing source frame.
- Both endpoints require a second adjacent supporting pose with reliable torso
  and sufficient confident joints. Each joint has its own supporting endpoints.
- Up to 0.30 seconds of missing body/leg observations and 0.15 seconds for
  elbows/wrists. At 29.97 FPS this allows at most 8 and 4 missing frames,
  respectively; nine missing frames take 0.3003 seconds and are rejected.
- Plausible bbox centre/scale motion, bounded body-relative joint displacement,
  in-frame anchors and an unchanged six-person on-court cap.
- A missing full skeleton needs another observed person's bbox to overlap the
  expected body during the gap. A partial missing joint must lie behind another
  observed bbox. Bbox overlap supports occlusion but does not prove depth order.
- Interpolate body motion and body-relative shape separately. For partial
  poses, align to the reliable observed torso without moving any measured joint.
- Fast-changing or unsupported joints stay unknown. The algorithm cannot
  recover the acceleration or peak hand position of a fully hidden pass swing.

`frames[].poses` remains exactly as produced by the saved pipeline. Each
`pose_estimates[]` entry has roster ID, side, derived bbox, observed pose index
or null, 17 `[x,y]`/null points, `estimated_joints`, and per-joint endpoint
frames/pose indices, gap length and body-relative change. There is no invented
detector confidence or raw tracker ID. `label_source: derived`, `verified:
false`, source-file SHA256 values and the offline/future-frame contract are
explicit. Thresholds are conservative engineering choices, not calibrated
uncertainty or selected held-out optima.

## Matched synthetic dropout check

```powershell
python scripts/evaluation/evaluate_pose_completion.py --help
python scripts/evaluation/evaluate_pose_completion.py `
  --annotations results/JIZ3exHe3hA_player_contacts_20261030_v1/JIZ3exHe3hA_4k_annotations.json `
  --output results/JIZ3exHe3hA_pose_completion_eval_20261030_v2 `
  --samples-per-gap 40
```

The script hides complete observed skeletons in real overlap intervals and
compares reconstructed points with the hidden saved model points. It uses 40
deterministic intervals for each gap length. Baselines hold the last observed
pose or extrapolate its last one-frame joint velocity. All methods are scored
on exactly the same accepted joint-frames; rejected joints remain in the
coverage denominator. There is no training or tuning on human keypoint labels.

Measured JIZ3 v2 results (error divided by the observed player's bbox height):

| Missing frames | Joint coverage | Bidirectional median | Last-pose median | Forward-velocity median |
|---|---:|---:|---:|---:|
| 1 | 86.2% | 0.00427 | 0.00877 | 0.00898 |
| 3 | 79.8% | 0.00628 | 0.01472 | 0.01547 |
| 6 | 49.3% | 0.00686 | 0.01942 | 0.01882 |
| 9 | 0% | rejected | rejected | rejected |

Six-frame intervals do not estimate elbows/wrists. Nine-frame intervals are an
out-of-bound negative control at the actual FPS. The reduced median error is
evidence for short-gap consistency relative to the saved model; it is **not**
ground-truth hidden-pose accuracy, identity accuracy or improved contact
attribution. Correlated errors in neighbouring detector poses can survive
interpolation. Real occlusions still need independent visual/human checks.

## Contact statistics and production boundaries

Player contacts still consume only observed `poses`. FSM, dense contact types,
ball trajectory, identities, analytics and cached feature/checkpoint files are
not refreshed. A new pose estimate never proves that a hand touched the ball.
This pilot is opt-in saved-result post-processing; normal detector/pipeline
defaults remain unchanged. A future image-based recovery experiment needs
actual missing model-stage outputs, not fabricated detections from this layer.

Tests cover short full/partial occlusion, both temporal anchors, missing frames,
long exits, identity rejection, duplicate IDs, off-frame geometry, impossible
motion, arm limits, the six-person cap, scaling, source preservation, unchanged
contacts, matched evaluation references and viewer toggle/frame guards.
