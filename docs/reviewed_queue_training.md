# Training the contact spotter on reviewed queues

The production spotter was trained on three fully annotated gold videos from
one camera placement. The first reviewed queue
(`data/contact_review_20261029_v1/review.json`, four videos from a new
placement, every one of 595 candidates verified by a human) adds data from a
second placement. This records what that data can teach, how it was tested,
and what came out.

## What the review contains

| video | kept | unique frames | rejected | frame corrected | type corrected |
|---|---|---|---|---|---|
| `LdOghv6rt_4_4k` | 137 | 130 | 48 | 84 | 19 |
| `QhqMwaCf8iw_4k` | 107 | 98 | 30 | 68 | 26 |
| `TcUru_ou4CQ_4k` | 118 | 112 | 56 | 82 | 25 |
| `qDzHtxhNkBA_4k` | 84 | 74 | 15 | 58 | 16 |
| **total** | **446** | **414** | **149** | **292** | **86** |

Human types over the kept candidates: set 143, attack 104, dig 88,
reception 61, serve 39, block 11. Sides: `team_b` 262, `team_a` 183,
unknown 1.

How the shipped spotter (`spotter_20261028_v1`) did on them, before any
retraining: raw six-class type agreement **0.807** (360/446), four-class
**0.861**. On gold the same model types located contacts at 0.92, so the new
placement costs typing too, not only detection. The largest confusions:
`set -> attack` 17, `dig -> set` 14, `dig -> reception` 9, `attack -> set` 8,
`reception -> set` 7, `attack -> block` 7.

Frame corrections are mostly small: of 292, 263 are within three frames. 15
moved more than eight frames, which reads as the reviewer stepping to a
different touch.

## Three facts that decide what may be trained on

**Rejections are duplicates, not negatives.** 140 of 149 rejected candidates
sit within ten frames of a kept contact; `X` there meant "this is the same
touch", not "no touch here". Only 9 are isolated. The spotter's spare
candidates on this camera are double-fires around real touches, not phantom
contacts. Nine frames do not justify a negative-only loss, so rejections are
counted and otherwise ignored.

**Unreviewed frames are unknown.** The reviewer saw only queued frames and
inserted nothing. 39 serves were found over four ~12-minute videos holding 414
contacts; at the usual four to six touches per rally that is well under one
serve per rally. Either serves are missed by the queue or many happen out of
frame on this placement; this has not been checked by eye. Either way a frame
the queue never offered cannot be used as background.

**No phase track.** The review labels contacts, not `between` /
`serve_flight` / `rally_live`.

So a reviewed video contributes one five-frame segment per kept contact at the
human's frame, with the same three-frame guard ring as gold, and every other
frame is masked. Five frames where merged candidates disagreed on the type
keep the contact for detection and mask the type. Decode constraints are
counted from gold phase tracks only. Implementation:
`action_recognition/rally_spotting/reviewed_contacts.py`
(`reviewed_sparse_contact_targets_v1`); trainer flag
`scripts/models/train_contact_spotter.py --reviewed <review.json>`.

## How it was tested

`scripts/evaluation/run_reviewed_expansion.py`, output
`results/reviewed_expansion_20261029_v1`. Every arm uses the shipped recipe
unchanged (8 epochs, seeds 42/43/44 averaged, deterministic) and is scored
through `ContactSpotter.run`, the path `analyze_video.py --contact-spotter`
takes. No epoch is chosen on held-out data, unlike the earlier
leave-one-video-out runs, so these numbers are lower than those and are not
comparable with them.

* **Gold folds.** Hold out one gold video; train on the other two with
  (`plus`) and without (`base`) all four reviewed videos. Gold labels are
  dense, so precision is real here. This is the guard against contact-only
  targets teaching the model to fire everywhere.
* **New-camera folds.** Hold out one reviewed video; train on all gold with
  (`plus`) and without (`base`) the other three. Precision cannot be measured
  on a reviewed video. Recall over its human contacts can, and so can the
  type the model gives each one at the human's frame. Recall favours `base`:
  the queue was built from its own peaks.

Promotion rule, fixed in code (`PROMOTION_RULE`) before any arm ran — both
must hold:

1. new-camera folds: mean frame-level raw typing accuracy **and** mean recall
   at ±5 are higher for `plus`;
2. gold folds: mean raw six-class macro F1 at ±5 and mean detection precision
   at ±5 each drop by no more than 0.02, the recipe's measured seed spread.

## Results

Measured 2026-10-29, all eleven arms exited cleanly; report
`results/reviewed_expansion_20261029_v1/report.json`. `new_base` rebuilt the
shipped `spotter_20261028_v1.pt` bit for bit (max absolute weight difference
0.0), so the refactored trainer is the production trainer.

**Decision: not promoted.** The new-camera condition holds, the gold
condition fails on precision. `spotter_20261028_v1` stays in the workflow.

Gold folds (mean of three, dense labels, real precision):

| metric | base | plus | change |
|---|---|---|---|
| raw macro F1 ±5 | 0.6846 | 0.7041 | +0.0195 |
| core macro F1 ±5 | 0.7965 | 0.8055 | +0.0089 |
| detection precision ±5 | 0.8281 | 0.8062 | **−0.0219** (limit −0.02) |
| detection recall ±5 | 0.9097 | 0.9163 | +0.0066 |
| typing accuracy on located | 0.8846 | 0.9041 | +0.0195 |
| type at human frame, raw | 0.8269 | 0.8577 | +0.0308 |

Precision drops on every fold (Khe 0.886 → 0.879, j0 0.879 → 0.854,
j3 0.719 → 0.686), and the event count rises on two of three (j0 124 → 130,
j3 146 → 156, against 121 and 119 human contacts).

New-camera folds (mean of four; recall and type only):

| metric | base | plus |
|---|---|---|
| recall ±5 | 0.851 | 0.946 |
| serve recall | 0.708 | 0.938 |
| typed recall, core macro | 0.754 | 0.862 |
| type at human frame, raw | 0.829 | 0.839 |
| type at human frame, core | 0.879 | 0.899 |
| events emitted / human contacts | 366 / 414 | 917 / 414 |
| events more than 10 frames from any queued frame | 0 | 321 |
| events within 10 frames of a queued frame but unmatched | 11 | 202 |

The recall gain is real, but it comes with about 2.5 times as many events.
202 of the extra events sit next to frames the reviewer saw and judged; most
are double-fires beside a kept contact or at a rejected duplicate. Another 321
fall where nobody looked. This review cannot measure precision on the new
camera, but the gold folds move the same way on a smaller scale. The likely
cause is the target design itself: a reviewed video teaches only "contact
here" and never "no contact here", so the model learns that the new placement
is dense with contacts.

What would decide it: review the extra events that the `plus` models produce
on the held-out videos, each video scored by the fold model that never saw
it. A rejection there is a true negative at one of the model's own false
positives, which is exactly what this review lacked, and together they give a
held-out precision on the new camera. Only then is retraining with background
around reviewed contacts justified.
