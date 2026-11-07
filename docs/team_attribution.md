# Which side touched the ball: geometry does not answer it

The contact spotter says what happened and when, never who. The plan was to
derive the side from geometry and skip annotating it. **That failed, and the
406 human labels collected to test it are now the only reliable source.**

Labels: `data/contact_team_gold_v1/labels.json`, 406 contacts over the three
gold videos, one keystroke each, collected with
`action_recognition/annotator/team_attribution_annotator.py`. 397 carry a
definite side; 9 were marked unsure and are scored as neither.

The geometric prediction was withheld by the server until a label existed for
that contact, so the labels are not anchored to it.

## What was measured

| derivation | coverage | accuracy where it predicted |
|---|---:|---:|
| nearest player's team | 0.945 | **0.685** |
| nearest player's own foot side | 0.945 | **0.715** |
| ball pixel vs the floor divider | 0.945 | 0.515 |
| net-crossing chain from the previous true side | 0.853 | 0.603 |

Nothing here is usable. A 30% error rate on the side destroys any per-team
statistic, and the chain is worse than the per-contact guess even when handed
the previous contact's true side for free.

`ball_side` is not merely weak, it is degenerate: it answered `team_a` for 374
of 375 contacts. The floor divider sits at image y≈1214 while a contact happens
in the air, well above it, so almost every ball reads as the far side.

## Why it fails, ruled out one at a time

**It is not the calibration.** Sweeping a vertical offset on the floor divider
from -300 to +300 px peaks at **0 px** and 0.715, on all three videos
independently. The divider is where it should be.

**It is not a side-switch or a jersey/half mix-up.** Accuracy is flat at ~70%
across every eighth of every video, never dipping below 50%, so the labels do
not invert anywhere.

**It is the nearest-player assumption.** Accuracy by contact type: `serve`
0.803, `pass` 0.711, `reception` 0.707, `attack` **0.558**. The attack is where
an attacker and one or two blockers are all within centimetres of the ball, and
the closest keypoint belongs to the wrong one about as often as the right one.
Restricting to contacts where a player is within 60 px of the ball makes
accuracy *worse*, not better — congestion, not distance, is the problem.

**The net-crossing idea fails for the same reason as `ball_side`.** Counting
sign changes of the ball against the net tape line treats any high ball as a
crossing, because a ball lofted on its own side also passes above the tape in
image space.

There is no useful operating point. Demanding a margin large enough for 95%
accuracy leaves 8.5% of contacts:

| margin threshold, px | coverage | accuracy |
|---:|---:|---:|
| 0 | 1.000 | 0.685 |
| 200 | 0.301 | 0.708 |
| 400 | 0.104 | 0.897 |
| 500 | 0.085 | 0.969 |

## The real obstacle

One fixed camera, 2D reasoning, and a ball that spends the interesting moments
in the air. Every derivation tried here needs to know which side of the net an
airborne ball is on, and the court homography only answers that for points on
the floor. This is a property of the setup, not a bug in any one heuristic.

## What this leaves

* The 406 labels are ground truth for the side and can train or validate
  anything that comes next. Split by complete source video, as always.
* A **learned** attribution has not been tried: side is a binary target, and
  the features available at a contact (ball position and velocity before and
  after, every player's position and side, the contact type the spotter
  already predicts, the position in the rally) are richer than any single
  heuristic above. 406 examples is small but the task is binary.
* A **depth cue** is unused: a ball on the far side is smaller in pixels than
  one on the near side, and the detector knows the box even though the
  annotations currently store only its centre. Recording the ball's pixel size
  would make this testable.
* Rally structure is still sound in principle — possession alternates, touches
  come in runs of up to three — but it needs a crossing detector that works,
  which is exactly the thing that failed.

Until one of those lands, the pipeline writes `team: null` on every spotted
contact rather than a guess, and per-team statistics are not available.

## The labelling tool

```powershell
python -m action_recognition.annotator.team_attribution_annotator --help
# http://localhost:5057
```

Keys: `A` far side, `B` near side, `S` unsure, `Z` back. One keystroke per
contact, chronological order, the next frames prefetched. The prediction is
revealed only after the answer is recorded, and the running agreement is shown
in the corner.

It never writes to the contact review it reads, and it refuses to start if that
review changed since the labels were begun.

Tests: `tests/test_team_attribution.py`.
