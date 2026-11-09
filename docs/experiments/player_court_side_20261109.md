# Player side and roster identity near the net

This diagnostic compares the net-floor rule with two-sided track history,
OSNet corroboration for ambiguous candidates, and the existing back-facing
prior. The experiment uses saved poses after the per-team cap. It cannot
measure recovery of candidates the cap discarded or reproduce an entire
analysis at a different calibration.

All variants used the same 10,584 cached OSNet crops and one frozen six-player
gallery from F600 of `e2F6pef-te0_2160p`. The saved automatic seed at F29 had
overlapping boxes on the same foreground figure. The diagnostic seed instead
uses six visually distinct boxes. Its labels and 84 visual checks are Codex
diagnostics from one match. The checks do not enter the algorithm or gallery.
The check set has 80 court-side labels, 52 roster IDs and 30 exclusions from
the near-side roster. Some observations have several labels.

## Results

The table compares the original frame policy with the final variant retaining
near-side candidates and adding history, tentative OSNet matches and the back
prior. Each count of changed sides or lost IDs compares a policy with its own
unshifted prediction on 10,310 observations. Positive offsets move the floor
line down. These stability counts measure sensitivity to calibration error.

| Offset | Side changes, frame / final | Lost IDs, frame / final | Correct side, frame / final (of 80) | Correct ID, frame / final (of 52) |
| ---: | ---: | ---: | ---: | ---: |
| −10 px | 439 / 482 | 28 / 28 | 70 / 72 | 43 / 45 |
| −5 px | 243 / 270 | 11 / 11 | 69 / 71 | 42 / 44 |
| 0 px | 0 / 0 | 0 / 0 | 68 / 69 | 41 / 42 |
| +5 px | 346 / 244 | 237 / 135 | 67 / 68 | 41 / 42 |
| +10 px | 642 / 627 | 472 / 457 | 65 / 65 | 37 / 37 |

At zero offset, adding the back prior to the same history and OSNet policy
changed the reviewed scores from 63 to 69 correct sides and 37 to 42 correct
IDs. The original frame policy already uses this prior. The heuristic fired on
13 of 17 visually back-facing poses and two of 15 front or side views at the
existing threshold. These examples are selected diagnostics. The final policy
still misses both near-side jumpers at F7003. It preserves jersey #9 during the
reviewed sideline return. The initial stricter variant had substantially lower
ID coverage. The final variant remains an experiment outside the production
analyzer.

## Orientation from image crops

The published MEBOW checkpoint link redirected to a Penn State service
portal. The separate image probe used the authors' frozen JointBDOE-S weights
and source revision `362f999e22bd50a4e73aca882b58c13f8a96a13c`.
It applied the model to padded saved boxes and accepted angle estimates only
when the output box matched the crop target at IoU ≥ 0.55. The diagnostic
classifies torso yaw as back within 45° of 0°, front within 45° of 180°, and
side in between. A missing match yields an unknown orientation.

JointBDOE-S matched all 32 reviewed near-side crops, got 30/32 front, back or side
labels correct, and got 31/32 correct for back versus other. The current
visibility heuristic got 26/32 for the binary task. These checks come from
one camera and several repeat observations of the same players. The image
probe measures orientation separately. The side and ID comparison above uses
the existing heuristic and does not measure the effect of the new model on
those outputs.
The separate `orientation_rally_frames.json` probe classified the setter at
F6212 as side-facing and both near-side jumpers at F7003 as back-facing.

- [JointBDOE source](https://github.com/hnuzhy/JointBDOE/tree/362f999e22bd50a4e73aca882b58c13f8a96a13c)
- [Author checkpoint](https://huggingface.co/HoyerChou/JointBDOE/blob/6fce990/coco_s_1024_e500_t010_w005_best.pt)

## Replay

Local files under `results/experiments/player_side_e2f_20261109_v1/` and
`results/experiments/player_side_e2f_20261109_v3/` contain the frozen seed,
checks and reports. The benchmark opens SQLite read-only and rejects missing
crops. It executes no neural inference. Choose a fresh output path for replay.

```powershell
python scripts/evaluation/compare_player_court_side.py `
  --annotations results/viewer/job_1794148755226038500_net234/e2F6pef-te0_2160p_annotations.json `
  --seed results/experiments/player_side_e2f_20261109_v1/seed.json `
  --checks results/experiments/player_side_e2f_20261109_v3/checks.json `
  --window 550:830 --window 1940:2270 --window 4360:4450 `
  --window 6150:6270 --window 6940:7050 --window 7260:7550 `
  --preserve-frame-near --output results/experiments/player_side_e2f_replay/report.json
```

Run `scripts/evaluation/probe_body_orientation.py` with the same annotations
and checks to reproduce `orientation_crops.json`. Use `--frame` in place of
`--checks` to predict orientation for all saved boxes at selected frames. The
probe takes the author source and weights as explicit local paths.
