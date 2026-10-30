# Persistent player identity: first offline pilot

The goal is one roster ID per person through a match. Existing `poses[].id`
is a local ByteTrack ID, or a negative ID in the separate SAM-recovery namespace.
It is not a match-long player identity. The production pipeline remains unchanged.

## Methods worth testing

| Method | Useful evidence | Limit |
|---|---|---|
| ByteTrack / longer track buffer | Position, box overlap, low-confidence detections | No appearance recognition; extending memory can also extend a wrong association |
| Motion + one-to-one roster assignment | Smooth movement, simultaneous detections cannot share a player ID | Crossings and long occlusion remain ambiguous |
| Motion + clothing appearance | Shirt/shorts colours distinguish this recreational team | Similar uniforms, turns, lighting and contaminated crops can confuse it |
| Learned person ReID + offline tracklet stitching | Multiple clear views of each player; future frames available | Generic pedestrian/soccer models need validation on this camera and team |
| Jersey number recognition over a tracklet | Several readable number views provide a strong identity anchor | Many players in this sample have no readable number; never require OCR on every frame |
| Human initialisation and corrections | Reliable roster names and checkpoints around crossings | Initialisation alone does not prove continued identity |

Primary sources:

- [BoT-SORT paper](https://arxiv.org/abs/2206.14651): motion and appearance association.
- [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/):
  ByteTrack has no appearance branch; BoT-SORT ReID is optional and disabled by default.
- [SoccerNet Game State Reconstruction, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf):
  sports-specific ReID, jersey recognition, and aggregation across tracklets.

The installed Ultralytics is **8.3.160**. New online documentation includes
trackers/models not present in that installation. A config change alone is not
proof that native detector embeddings work with this YOLO-pose version. No
Ultralytics upgrade, model download or training is part of this pilot.

## Implemented experiment

`src/tracking/player_identity.py` compares three methods on **identical saved
detections**, sampled every three source frames by default:

1. `raw_track`: retain a roster identity only while its seed raw track ID is seen.
2. `motion`: Hungarian one-to-one assignment with box-centre velocity, scale
   gating, an ambiguity margin, and explicit unknown slots.
3. `motion_appearance`: same assignment plus root-normalised HSV histograms in
   three body bands. This is a colour control, **not a learned ReID model**.

Both directions are processed independently from one seed frame. This uses
future frames and is strictly offline. IDs expire after a three-second
unmatched gap. Motion extrapolation is capped at half a second. Ambiguous
matches remain `player_id: null`; overlapping crops do not update appearance
galleries. Costs are association costs, not calibrated probabilities.

The six-person roster is fixed **for this pilot** on one court side. It does
not handle substitutions, set-side changes or long off-camera re-entry. Never
recycle P1 for a substitute. For a full match, add explicit roster changes,
side segments, clean multi-view galleries and offline tracklet stitching.
Court position/rotation is a soft motion cue, not a player's identity.

No source pose, ByteTrack state, calibration, ball, event, FSM result, human
source annotation or checkpoint is rewritten. Predictions and human labels
are separate files under a new versioned result directory. Persistent identity
alone does not identify who touched the ball: event-to-player attribution is
a separate task, especially near the net.

## Run

Verify each command's `--help`. If the venv launcher is broken, use the
standalone Python and project packages documented in AGENTS.md.

```powershell
python scripts/evaluation/run_player_identity_pilot.py prepare --help
python scripts/evaluation/run_player_identity_pilot.py prepare `
  --annotations results/hy2S_player_teams_sam_20261026_v2/hy2S-f7zj9Q_4k_annotations.json `
  --output results/player_identity_hy2S_20261029_v1 --side b --stride 3

python -m action_recognition.annotator.player_identity_annotator `
  --project results/player_identity_hy2S_20261029_v1 --port 5063
# http://127.0.0.1:5063

python scripts/evaluation/run_player_identity_pilot.py compare `
  --project results/player_identity_hy2S_20261029_v1
```

`prepare` refuses an existing output directory. Outputs:

- `observations.json`: original frame/pose indices, raw IDs, boxes, crop
  descriptors, occlusion flags, annotation SHA256, video path/size/mtime.
- `comparison_0000.json`, `comparison_0001.json`, ...: immutable experiment
  snapshots, parameters, seed provenance, predictions and diagnostics.
- `labels.json`: human seed and independent sparse checks; saved only after
  user action. Human marks have `label_source: human`, `verified: true`.

The automatic seed assigns P1–P6 left-to-right on an early complete frame;
`automatic_initialisation`, `verified: false` explicitly distinguishes it
from human truth. Review startup rejects changed source annotations/video;
cache SHA256 prevents human labels being reused with changed observations.

## Human review

1. **Початкові ID**: inspect the six reference photos. Click a box or its
   `dN` button, then press 1–6. The order is arbitrary; preserve its meaning
   afterwards. Confirm with **Зберегти початкові ID**.
2. **Контрольні кадри**: choose times across the clip, especially immediately
   after overlap. Predictions are hidden. Label visible players you recognise;
   `U` marks uncertainty. Unlabelled/uncertain detections are excluded from scoring.
3. **Наступний складний кадр** proposes overlap/disagreement frames. Also sample
   regular timestamps: an error-focused subset alone is not representative.
   Aim for 15–20 frames spread across the 45-second clip, plus difficult cases.
   Use the original video panel to follow movement through crossings.
4. **Перерахувати й оцінити** reruns all methods using only the seed and
   scores separate human checks. No test label enters association. Once checks
   exist, changing the seed roster meaning requires a new project. Test marks
   are editable; save them and rerun to refresh scores.

Metrics: correctness on all checked detections (unknown counts as failure),
correctness among assigned detections, and coverage. These are **sparse human
check metrics**, not dense-video IDF1/HOTA or an ID-switch rate. Missing detector
boxes are not evaluated here. Do not tune on these checks then call their
score held-out: reserve another interval/video for final evaluation.

## First run, 2026-10-29

Source: `hy2S-f7zj9Q_4k`, 1348 analysed frames (~45 s), same SAM-assisted
pose analysis for every method. Side `b` has six detections in 877 frames;
164 distinct raw IDs occur over the whole analysis, including negative SAM
IDs. This is fragmentation evidence, not 164 players or a measured switch count.

450 sampled frames, 2528 player detections, automatic seed at source frame 72:

| Method | Assigned detections | Coverage |
|---|---:|---:|
| Seed raw track IDs | 1074 | 42.48% |
| Motion | 1384 | 54.75% |
| Motion + HSV appearance | 1661 | 65.70% |

No human identity checks existed at this run. **Accuracy unknown.** Coverage
alone does not select a winner. The output is a runnable comparison and review
queue, not validated match-long identity or personal contact statistics.

Next experiment after review: inspect failure crops, then compare a learned
ReID encoder / BoT-SORT with explicit embeddings against these frozen detections.
Use separate human-labelled calibration and evaluation intervals. Number OCR
is worthwhile only if enough roster members expose readable numbers.

Validation: `tests/test_player_identity.py` covers crossing identities,
ambiguity, long exits, backward association, seed uniqueness, independent
test labels, source/cache fingerprints, and append-only comparisons.

## Human check result: 100 labels, 2026-10-29

Verified human seed and 100 independent-of-association checks on 19 frames,
4.70–23.52 s of the approximately 45-second clip. `comparison_0009.json`:

| Method | Correct / 100 | Wrong person | Unknown | Correct among assigned |
|---|---:|---:|---:|---:|
| Seed raw track IDs | 46 | 7 | 47 | 86.8% |
| Motion | 60 | 6 | 34 | 90.9% |
| Motion + HSV | 64 | 16 | 20 | 80.0% |

HSV resolves 14 motion-unknown cases: four correctly, ten incorrectly. Prefer
motion as the conservative baseline; neither is ready for personal statistics.
P1/P4 score 19/19 each and P6 12/12 with both association methods, while P5
scores 0/19. HSV labels P5 as P3 on 13 checks. Three P3 checks are labelled P2.

The pilot last emits P5 at 3.50 s; its hard three-second lost-gap gate then
prevents re-identification forever from about 6.5 s onward. This is a known
prototype limit, not evidence that P5 vanished. Current-frame occlusion flags
also cannot remove persistent wrong assignments: seven HSV errors occur with
overlap below 0.3.

Visual review shows P3/P5 have similar red/black shirts, but different sleeves
and readable back numbers 1/8 in some frames. Next candidates: explicit lost-ID
recovery, clean multi-view galleries, part-aware learned ReID, offline tracklet
consistency, and occasional jersey-number confirmation. No such method was
trained or evaluated in this audit. Keep a separate interval/video for final
assessment after development; these 100 labels cover only part of one clip
and are correlated across time. Detailed audit and crop montage live under
`results/player_identity_hy2S_20261029_v1/audit_100_checks_v1/`.
