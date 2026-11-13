# Player tracker comparison

`scripts/evaluation/compare_player_trackers.py` compares the installed production
Ultralytics ByteTrack with BoxMOT 26.0.0 BoT-SORT and Deep OC-SORT. It also
reruns the current automatic OSNet roster stage on the saved original track IDs.
Production tracking remains ByteTrack.

## Protocol

All methods receive the same saved pose boxes on the selected court side, in
source-frame order, over the entire saved recording. Existing cached OSNet
embeddings supply appearance to BoT-SORT, Deep OC-SORT and the roster stage.
No detector, encoder, training, video decoding or video rendering runs.

Saved poses have already passed the original production tracking and court
selection. This experiment measures association on those fixed observations;
it is not a replay of raw YOLO detections through the entire pipeline.
Original detection confidence is absent from annotations, so every method
receives confidence 1.0. Confidence-dependent behavior is therefore not tested
under production conditions. Camera-motion compensation is explicitly disabled
for this stationary-camera experiment. Other tracker settings use the pinned
implementation defaults and are saved before scores are produced.

Backend detection indices associate IDs with original poses. Kalman boxes and
unmatched predictions never replace observed boxes or create new observations.
This keeps crop-cache keys, keypoints and the human frame/pose-index join intact.
Empty and skipped source frames advance tracker age. A backend can withhold an
unconfirmed ID; the pose remains present with an unknown ID.

Human movement exports must bind to the exact annotation and video SHA256.
Verified player decisions score predictions only; they never choose galleries,
seeds or tracker parameters. Duplicates are excluded according to the existing
movement scoring protocol. Model IDs are named by one global one-to-one mapping
to human IDs. Fragmenting a person into many IDs is penalized by that mapping.

The report includes:

- Conditional observation IDF1, adjacent ID changes, distinct raw IDs and
  ID changes after a reviewed gap of at least one second.
- Current roster-stage correct, wrong and unknown rates, per-person scores,
  adjacent switches and assignments on human-excluded poses.
- Saved production IDs as historical context and the current roster stage
  rerun on the original saved tracks as the comparison baseline.
- Source hashes, configurations, package versions, cache provenance and times.

Conditional IDF1 uses only verified observed player poses:
`2 * IDTP / (reviewed positives + assigned reviewed positives)`.
It is not full MOT IDF1: unreviewed/excluded detections and missed people are
outside its denominator. HOTA and detector recall require suitable independent
box/identity ground truth. Roster IDs are scored separately because tracker
continuity can improve while final roster identity gets worse.

## Run

Install optional packages separately from the production environment. The
following commands assume the activated project Python environment. `--no-deps`
preserves its PyTorch/CUDA, NumPy and OpenCV versions; the tracker imports use
the existing scientific packages plus the three explicitly installed packages.

```powershell
python -m pip install --target results/tracker_benchmark_runtime_boxmot26 --no-deps boxmot==26.0.0 filterpy==1.4.5 lapcpp==1.0.0
python scripts/evaluation/compare_player_trackers.py --help
python scripts/evaluation/compare_player_trackers.py `
  --movement results/player_identity_iKhVYAof2as_4k_r000787_baseline_v1/human_movement.json `
  --annotations results/viewer/job_1794301742143025800/iKhVYAof2as_4k_annotations.json `
  --tracker-runtime-dir results/tracker_benchmark_runtime_boxmot26 `
  --output results/player_tracker_comparison_new
```

The default is all three trackers. `--trackers` selects a subset. Cache misses
fail without inference. Existing output directories are refused. Each run saves
`plan.json`, per-method predictions and metrics, then aggregate `metrics.json`.
An interrupted run retains completed method files; use a new output directory
for another run.

## Promotion boundary

One reviewed development video cannot establish generalization. A candidate
must improve final roster identity without increasing wrong assignments, then
pass raw-detection cache replay through court membership, player caps,
substitutions, contacts and final events. Raw detector confidence must be
retained in that follow-up. Different Kalman boxes require their own exact
crop embeddings; never substitute nearby cached crops or fabricate raw
detections from saved derived poses. Production defaults change only after
that validation supports the change.

## Development result, 2026-11-13

Recording: `iKhVYAof2as_4k`, 17,929 processed frames and 90,146 fixed near-side
poses. Movement review revision 787 supplies 30,256 verified player poses over
frames 0–5786 (about 193 seconds); 324 duplicate observations are excluded.
Every one of the 90,146 crop embeddings was a cache hit; zero cache writes and
zero model outputs were computed. Roster discovery uses the entire recording
for every variant, while human scoring covers only the reviewed interval.

Artifacts: `results/player_tracker_comparison_20261113_v2/plan.json` and
`metrics.json`, plus per-method predictions and metrics. The earlier `v1`
attempt retained its plan after a roster-naming error; it has no completed
comparison and is not a measured result.

| Track source + current roster stage | Conditional observation IDF1 | Correct roster ID | Wrong roster ID | Unknown roster ID | Adjacent roster ID changes |
|---|---:|---:|---:|---:|---:|
| Original saved production ByteTrack IDs | 25.56% | 97.58% | 0.20% | 2.23% | 4 |
| ByteTrack reassociation of fixed poses | 27.27% | 97.04% | 0.27% | 2.69% | 8 |
| BoT-SORT reassociation of fixed poses | 33.80% | 97.22% | 0.36% | 2.42% | 27 |
| Deep OC-SORT reassociation of fixed poses | 23.83% | 97.03% | 0.32% | 2.65% | 7 |

All four roster runs used `osnet_tracklet_roster_v5`, without human seeds.
The historical saved file itself contains the older v4 roster output (67.33%
correct); that output is retained in `saved_production.roster` for provenance,
not used as the current-method baseline.

BoT-SORT is the strongest of the three reassociation runs by conditional IDF1
and correct roster coverage. It still loses 109 correct assignments and adds
50 wrong assignments relative to the original tracks with the current roster
stage (29,414 correct / 109 wrong versus 29,523 correct / 59 wrong). Its 27
adjacent roster changes also exceed the baseline's four. No candidate improves
the current workflow's measured identity outcome, so ByteTrack remains the
production tracker. This screening result does not establish that a tracker
cannot improve after a separate raw-cache integration or parameter experiment.

CPU association times over the full recording were 12.68 s for ByteTrack,
36.57 s for BoT-SORT and 42.34 s for Deep OC-SORT; each roster pass took
5.90–6.47 s. These times exclude cache loading, imports, detector/encoder work
and all other pipeline stages. The process used `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`.

## Implementation sources

- [BoxMOT BoT-SORT](https://mikel-brostrom.github.io/boxmot/trackers/botsort/)
- [BoxMOT Deep OC-SORT](https://mikel-brostrom.github.io/boxmot/trackers/deepocsort/)
- [BoT-SORT paper](https://arxiv.org/abs/2206.14651)
- [Deep OC-SORT paper](https://arxiv.org/abs/2302.11813)
