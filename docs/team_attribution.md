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
* A **depth cue**: a ball on the far side is smaller in pixels than one on the
  near side. Tested on 2026-11-10 and shipped as an additive proposal; see
  "The depth cue, measured" below.
* Rally structure is still sound in principle — possession alternates, touches
  come in runs of up to three — but it needs a crossing detector that works,
  which is exactly the thing that failed.

Until one of those lands, the pipeline writes `team: null` on every spotted
contact rather than a guess, and per-team statistics are not available.

## The depth cue, measured (2026-11-10)

The unused cue above was tested and now runs in the pipeline as a separate,
additive proposal. `team` is still `null`.

**What it is.** `ball_size` (longer side of the selected detector box, px) is
saved on every observed ball. The size a ball has exactly at the net follows
from the calibration: the tape is `net_height_m` above the floor line under
it, which gives pixels per metre in the net plane (`setup_1`: 364.9 px over
2.34 m, so a 0.21 m ball is 32.75 px). A contact whose ball is larger is on
the near side, smaller on the far side. Nothing is fitted on labels.

One box cannot split an attacker from a blocker a metre apart, so contacts are
grouped into possessions (a new one starts at a serve, after a serve, after an
attack, or after 4 s) and a possession takes the summed log size ratio of its
non-attack contacts. A possession whose evidence is weaker than 0.10 takes the
side opposite the previous one inside the same rally.

Code: `src/tracking/ball_size.py`, `src/tracking/contact_side.py`; pipeline
hook `EventsMixin._assign_contact_sides`; flag `--no-contact-side`.

**What was measured.** Three development videos, one camera placement
(`setup_1`), ball detector `ball_fullframe_ft1920`, 397 definite human side
labels, spotter `spotter_20261028_v1` (final fit: its contacts on these videos
are in-sample for detection). Reproduce with
`scripts/evaluation/evaluate_contact_side.py`.

| what is scored | n | accuracy |
|---|---:|---:|
| size + possession, human contact times and types | 397 | **0.945** |
| size + possession, the pipeline's own spotter contacts | 388 | **0.925** |
| ... where `side_confident` (size evidence), coverage 0.876 | 340 | 0.956 |
| ... where the side came from alternation only | 48 | 0.708 |
| contact author's side on the same spotter contacts | 388 | **0.977** |

Per video the rule on human contacts scores 0.947 / 0.937 / 0.950. By type on
spotter contacts: serve 1.000, reception 0.984, pass 0.891, attack 0.882.

Earlier steps on the same labels, with the older 1280 detector on two of the
three videos: box size alone 0.851 (serve and reception 1.000, pass 0.927,
attack 0.538, i.e. chance); nearest player on the same contacts 0.684.
Possession order is what lifts the attack.

**How to read it.**

* Against the geometric derivations above (0.515 to 0.715) the cue is real:
  the airborne ball's depth is exactly what the floor homography could not
  give.
* It does not beat the contact author. The author's all-side model has its own
  team HMM and was right on 0.977 of the same contacts. Letting a size side
  overrule the author lowered accuracy at every margin tried (0.925 to 0.977,
  never above 0.977). So the pipeline does not overrule: it lists the
  disagreements as review hints in `contact_side.author_check`. The two
  estimates disagreed on 30 of the 388 contacts, and those 30 held 5 of the
  author's 9 side errors (the 15 with `side_confident` held 2).
* That comparison flatters the author: its final fit was trained on these
  three videos, while the size rule has no trained parameters. On a source the
  author never saw the order may differ. Nobody has measured that; it needs
  ball boxes for the seven other contact-author sources, which the model cache
  does not hold yet.
* The threshold tolerates about +/-8%: with the 1920 detector the rule holds
  0.94 to 0.95 between 31 and 34 px and falls off outside 30 to 36 px. The
  second geometric estimate (court width at the net) reads 34.7 px for
  `setup_1`, at the edge of that band, which is why the pass abstains when the
  two estimates differ by more than 25%.

**When it says nothing.** No net floor line or tape; a camera that is not
end-on (near baseline less than 1.3 times the far one in pixels); the two
geometric estimates disagreeing; or, with 30 or more sized contacts, a median
contact size outside 0.80 to 1.25 of the net size. Events then keep
`side: null` and the report names the reason.

**Limits.** One camera placement. Development videos only, no untouched test.
Detector boxes are coarse (the older 1280 model reported sizes in 3 to 6 px
steps), so this is a side cue, not a depth measurement in metres. A missed or
mistyped contact moves possession boundaries. Sides are derived data: never a
label, never a replacement for the human file above.

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
