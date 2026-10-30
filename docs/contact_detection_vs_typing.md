# Detection or typing: where the spotting error actually is

Step 1 of the plan that followed the pose-enabled rerun
([`docs/dense_contact_spotting.md`](dense_contact_spotting.md)). The typed
spotting score mixes two jobs — finding the frame of a touch, and naming it —
and the plan to feed a dedicated detector into the existing contact-centred clip
classifier only pays off if typing is the weaker half.

It is not. **The spotter already types located contacts as well as the clip
classifier does, so the two-stage split is not worth building.** The error is
almost entirely in detection.

Rescoring only. Nothing was trained, and no artifact of the run was modified.

```powershell
python scripts\evaluation\score_contact_detection.py `
  --run results\dense_contact_spotting_20261028_v2 `
  --output results\dense_contact_spotting_20261028_v2\detection_typing.json
```

## The split

Three measurements over the finished pose-enabled run, all at +/-5 native
frames:

| | mean | Khe | j0 | j3 |
|---|---:|---:|---:|---:|
| **detection** F1, types collapsed to one `contact` class | 0.8605 | 0.9191 | 0.9012 | 0.7612 |
| detection F1, frames the human marked `non_game` excluded | **0.8978** | 0.9191 | 0.9048 | 0.8696 |
| **typing** accuracy on located contacts, four classes | **0.9208** | 0.9225 | 0.9035 | 0.9364 |
| typing macro F1 on located contacts, four classes | 0.9157 | 0.9214 | 0.8935 | 0.9321 |
| typed end-to-end, four-class macro F1 | 0.7897 | 0.8378 | 0.8099 | 0.7215 |

The halves multiply back: 0.8605 x 0.9208 = 0.792 against the measured 0.7897.
So the decomposition accounts for the end-to-end score, and the 21 points it
loses are 14 points of detection and 8 of typing.

Detection recall is already 0.92-0.96 on every fold (Khe 0.959, j0 0.942,
j3 0.924). **The model finds nearly every contact; it invents extra ones.**

## Why the two-stage plan is dead

The production clip checkpoint scores about 0.92 four-class macro F1 when handed
a human contact frame. The spotter, on contacts it located itself, scores 0.9157
macro F1 / 0.9208 accuracy on the same four classes. There is nothing left for a
second stage to win, and it would arrive with a worse input than the number it
was quoted on — spotter frames are a few frames off, human frames are not.

A dedicated detector head is not worth building either. Peak picking over the
summed contact channels — what a binary head would see, with no training —
scores 0.8636 against the decoded 0.8605. Having to type costs the model
essentially nothing in localisation. (At +/-1 and +/-2 the summed mass is
noticeably sharper, 0.6289 and 0.8121 against 0.5004 and 0.6977, but tolerances
that tight do not matter for match statistics.)

## j3 was not a weak fold, it was a warm-up

j3 looked like the problem fold at 0.7612 detection. 36 of its 60 false
positives sit before its first human contact at frame 5012, in a region the
human labelled `non_game`. Median distance from a j3 false positive to the
nearest real contact is 2728 frames — about 91 seconds. On Khe the same median
is 21 frames.

j3 carries 4958 `non_game` frames; Khe has 904 and j0 613. So the two training
videos of that fold contain almost no warm-up, and the model has never seen one.
It detects real touches there — players do hit balls in warm-up — and scoring
counts every one as a false positive because nobody annotated them.

Excluding `non_game` frames, j3 detection goes 0.7612 -> 0.8696 and its
precision 0.647 -> 0.821, while Khe does not move at all. It is a coverage gap
in the `non_game`/`between` phase class, not a weakness in contact detection.

This is not a claim that `non_game` detection does not matter. For match
statistics the warm-up has to be gated out somehow, and the model's own
`between` phase output is the natural gate once it has seen enough warm-up to
recognise one.

## What typing still gets wrong

Confusions on located contacts, summed over the three folds, six raw classes:

| | count |
|---|---:|
| `reception` -> `dig` | 8 |
| `dig` -> `set` | 7 |
| `set` -> `dig` | 7 |
| `dig` -> `reception` | 6 |
| `serve` -> `attack` | 4 |
| `block` -> `attack` | 3 |
| `set` -> `attack` | 3 |
| everything else | 1-2 each |

`set` <-> `dig` disappears in the four-class view, where both are `pass`.
`reception` <-> `dig` survives as the dominant four-class error. `block` is still
the weakest class: 7 right out of about 13.

## The FSM fails at both halves

Same decomposition on the FSM events from the pose-enabled analyses:

| | mean | Khe | j0 | j3 |
|---|---:|---:|---:|---:|
| detection F1 | 0.4008 | 0.4100 | 0.4310 | 0.3616 |
| detection recall | | 0.500 | 0.529 | 0.538 |
| detection precision | | 0.347 | 0.364 | 0.272 |
| typing accuracy on located contacts | 0.4889 | 0.5135 | 0.4375 | 0.5156 |
| typing macro F1 on located contacts | 0.2958 | 0.3690 | 0.2312 | 0.2872 |

It locates about half the contacts, and of those it names fewer than half
correctly. Its typing per class shows what it actually does: `pass` reaches
0.69-0.73 while `serve` is 0.00, 0.00, 0.12. It labels nearly everything a rally
touch. Both halves fail independently, which is why no threshold tuning will
rescue it.

## What this changes in the plan

* **Drop the two-stage detector plus clip classifier.** Measured, not assumed:
  typing is already at clip-classifier level.
* **Spend the effort on detection precision.** Recall is 0.92-0.96; the cost is
  the roughly 15-19 false positives per video that land inside live rallies,
  plus the warm-up region.
* **Give the model warm-up to learn from.** The cheapest version of this is not
  new labelling: `non_game` segments already exist in all three videos, they are
  just unbalanced. Weighting or sampling them differently is a training change,
  not a data change.
* The `--ball-ensemble` trajectory and a seed ensemble remain untested levers,
  and trajectory quality is still the largest measured effect on this task.

## Limits

* The run being rescored chose its best epoch and threshold on the held-out
  video, and the detector proxy's threshold is chosen the same way. These are
  model-selection estimates.
* Typing accuracy is conditioned on located contacts, so it is not directly
  comparable with a clip classifier's score on human contact frames.
* Some false positives inside live rallies may be real touches the review does
  not cover (`free_ball`, `other`), which are masked in training but still
  count against precision in scoring. The masked counts are small — 8 frames in
  Khe, 1 in j0 — so this does not explain the 15-19 per video.
* Three videos, one camera, one seed.

Tests: `tests/test_contact_detection_score.py`.
