# Reviewing spotted contacts on a new video

The spotter runs on a video nobody has annotated and produces contacts. They
are predictions. This is how they get checked, and how the check stays
separable from the prediction it corrects.

## Nothing is ground truth until it is

Every record in the review file starts:

```json
{"origin": "model", "label_source": "model", "verified": false, "review": null}
```

A keystroke makes it:

```json
{"origin": "model", "label_source": "human", "verified": true,
 "predicted_type": "set",
 "review": {"status": "kept", "type": "dig", "core_type": "pass",
            "team": "team_b", "reviewed_at": "..."}}
```

The model's own answer stays in `predicted_type` next to the human's in
`review.type`, so agreement can be measured afterwards and the two never blur.
`unsure` leaves `verified: false` and `label_source: "model"` — an unanswered
contact is not an answer. A contact the human adds has `origin: "human"` and
`predicted_type: null`: it can never be mistaken for something the model found.

The review file lives in its own directory and is never written into
`data/full_video_gold` or any other authoritative set. Promoting verified
records into a dataset is a separate, deliberate step that does not exist yet.

## Producing the queue

```powershell
python analyze_video.py --video "data\...\clip.webm" --output "results\batch" `
  --calibration configs\court_calibration_clip_norm.json `
  --contact-spotter models\contact_spotter\spotter_20261028_v1.pt `
  --ball-ensemble --no-net-detection --no-video --no-model-registry `
  --ball-reconstruction conservative
```

Then point the reviewer at the whole output directory:

```powershell
python -m action_recognition.annotator.contact_review_annotator `
  --annotations-dir results\batch `
  --output data\contact_review_v1\review.json
# http://localhost:5059
```

It refuses to start if the analyses changed since a review was begun: both the
annotations and the stats file are fingerprinted, because the contacts come
from the stats and a rerun of either would silently mix two different runs into
one review.

## Build the queue for recall, not for the most likely sequence

The constrained decoder answers "what is the most likely sequence of play".
A reviewer needs "what might be a contact". On four videos from a camera
placement the model had not seen, the decoder emitted 366 contacts where peak
picking over the same probabilities finds 595 — and a 12-minute match holds
closer to the latter. Rejecting a spare candidate costs one keystroke; finding
a contact the queue never offered costs scrubbing the video.

```powershell
python scripts\evaluation\spotter_candidates.py --annotations-dir results\batch --threshold 0.2
```

This writes `<stem>_candidates.json` beside each analysis, and the reviewer
prefers it when present. Each candidate carries `decoded`: whether the
constrained decoder also chose it. The page shows that, because a candidate the
decoder rejected is likelier to be spare. On the four night videos 428 of 595
were also decoded.

Measured on the gold videos, peak picking at 0.2 misses nothing the human
labelled — but those are the model's training videos, so read that as a ceiling
rather than as a recall estimate.

## Reviewing

One contact at a time, chronological, the next frames prefetched. The frame is
drawn with the ball marked and the team divider labelled A (far) and B (near).

| key | what it does |
|---|---|
| `1`–`6` | set the type: serve, reception, set, dig, attack, block |
| `Enter` | keep the predicted type |
| `A` / `B` / `U` | side: far, near, cannot tell |
| `X` | not a contact — reject it |
| `S` | unsure, leaves it unverified |
| `Z`, `←` `→` | move without answering |
| `+` | insert a contact the model missed, in the gap ahead |

A contact needs both a type and a side before it counts as reviewed; whichever
comes second completes it, so the order never matters. Two keystrokes for a
typical contact, one for a rejection.

`+` opens a frame stepper bounded by the current contact and the next one, so a
missed touch is found without scrubbing the whole video. Two contacts may not
share a frame.

## Calibrate first, and normalize it

`net_y` and the team split come from the calibration and feed the spotter's
features. A calibration from another camera placement produces plausible
numbers and silently wrong geometry.

```powershell
python scripts\interactive_court_calibration.py -i frame.png -o configs\court_calibration_clip.json
python scripts\normalize_calibration.py --input configs\court_calibration_clip.json --output configs\court_calibration_clip_norm.json
python scripts\evaluation\overlay_calibration_check.py --frame frame.png --calibration configs\court_calibration_clip_norm.json --output overlay.png
```

The normalize step is not optional. The interactive tool stores the court
starting from the line nearest the camera and calls that line
`back_line_points`; `GameAnalyzer._load_calibration` reads `back_line_points`
as the **bottom** of the court and expects the far line first. The homography
is valid under either reading, so an overlay looks correct while
`court_bottom` lands on the far line and the world y axis points the wrong way.
Normalizing rewrites the order and recomputes the matrix so world `y = -9` is
the far line, `y = 0` the net and `y = +9` the near line, matching every
calibration the gold videos were measured with.

The calibration tool accepts a still image as its input, which is easier than a
video when the opening frame has players standing on the lines.

## What was tried on the first batch, and what it cost

Four videos from a new camera placement in the same gym gave 64-118 contacts
each where a comparable video from the trained camera gave 195, with serves at
3-13 against about 30 rallies. Ball detection was not the problem (46-57%
against the reference's 55%).

Two hypotheses were tested and both were wrong:

* **`--ball-ensemble`.** The spotter was trained on non-ensemble trajectories,
  so a train/inference mismatch was plausible. Rerunning one video without it
  gave 117 contacts and 8 serves against the ensemble's 118 and 8. The ensemble
  is innocent.
* **A grammar rule.** The counted constraints permit `between -> serve_flight`,
  which lets the decoder enter serve flight without ever placing the serve
  contact — and the phase track showed 25-27 serve flights against 3-13 serves,
  which fits exactly. Forbidding every entry into `serve_flight` except from
  `serve` doubled the serves (8 to 16) but collapsed the totals (118 to 83, 64
  to 25) and left only 2-8 serve flights. On the gold videos it cost F1
  (0.987 to 0.977) and serve F1 (1.000 to 0.955). Rejected;
  `require_contact_before_phase` stays in `contact_spotter.py` documented as
  measured and not applied, so it is not tried again.

What is left is domain shift: a camera placement the model has not seen. That
is what the review is for.

## What the reviewer cannot tell you

The spotter was trained on three videos from one camera placement. On new
geometry its quality is unknown until reviewed, which is the point of the
exercise — the tool is built for correcting, not for confirming.

Tests: `tests/test_contact_review_annotator.py`,
`tests/test_normalize_calibration.py`.
