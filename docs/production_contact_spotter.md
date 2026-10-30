# Running the contact spotter in the pipeline

The dense spotter stopped being an experiment and became a pipeline stage. This
is how to train one, how to run it, and exactly what it does and does not
replace.

Why it exists: measured leave-one-video-out on the three gold videos, it puts a
typed contact on the right frame far more often than the rally state machine --
0.79 against 0.156 four-class macro F1 at +/-5 native frames. See
[dense contact spotting](dense_contact_spotting.md) for the measurement and
[detection versus typing](contact_detection_vs_typing.md) for where its
remaining error actually sits.

## Train a checkpoint

```powershell
python scripts\models\train_contact_spotter.py `
  --output models\contact_spotter\spotter_20261028_v1.pt `
  --annotations-dir results\gold_pose_fsm_20261028_v1 `
  --seeds 42 43 44 --epochs 8
```

It trains on **all three** gold videos at once, which is the difference from
the experiment: there is no held-out video, so the checkpoint comes with no
score of its own. Quote the experiment's numbers for quality, never a number
measured on data this checkpoint trained on.

Because nothing is held out, nothing can select an epoch. Across the
experiment's folds and seeds the best epoch landed anywhere between 3 and 15,
so no single value can be read off them. The trainer uses a fixed budget and
averages several seeds; the ensemble is what makes that choice less decisive.
The budget is an engineering decision and says so in the checkpoint provenance.

The checkpoint carries everything inference needs:

* the trained members, one per seed;
* the normalisation mean and deviation fitted on the training sources;
* the **decode constraints**, counted from the unguarded targets of those
  sources. These cannot be recomputed at inference, where there is no labelled
  fold to count them from, so a checkpoint without them is undecodable;
* provenance: source hashes, review hash and audit, seeds, epochs, determinism
  settings, and a list of limits.

Training is deterministic by default. Without that, two runs of one seed were
measured about 0.01 macro F1 apart, and a production checkpoint that cannot be
rebuilt is not auditable.

A `<checkpoint>.json` sidecar holds the same provenance plus the loss history,
so the checkpoint can be audited without loading it.

## Run it

```powershell
python analyze_video.py --video "C:\path\game.webm" --output "results\game" `
  --calibration configs\court_calibration_youtube_4k_setup_1.json `
  --contact-spotter models\contact_spotter\spotter_20261028_v1.pt
```

The spotter runs after the clean FSM replay and after `pose_f` is rebuilt, over
the same final ball coordinates and poses the exported annotations carry. It
needs no labels, so it works on a video nobody has annotated.

## What it replaces, and what it does not

| | |
|---|---|
| `all_events` | the spotter's typed contacts |
| `all_events_fsm` | the FSM's contacts, kept so nothing is lost |
| per-frame `ev` | the spotter's types |
| per-frame `ev_fsm` | the FSM's types |
| `rallies`, `service_attempts`, `rally_lifecycle` | serve-anchored rallies built from the spotter's contacts |
| `rallies_fsm`, `service_attempts_fsm`, `rally_lifecycle_fsm` | the FSM's, kept |
| `contact_spotter` | diagnostics, provenance, decoded phase segments, rally segmentation |

Still FSM-derived in the same run: `match_analytics` (score, points) and
`attack_heatmap`. Score needs team sides the spotter does not produce.
`contact_spotter.not_replaced` names them.

### Rallies (`spotter_rallies_v1`)

`action_recognition/rally_spotting/rally_segmentation.py`. A rally is a serve
that someone then played:

- contacts split into groups at pauses over 5 s and before every serve;
- a group opens on its serve. If the decode missed it, the spotter's own
  per-frame `serve` probability is searched 0.3-3.5 s before the first
  contact; a peak of at least 0.15 becomes a `serve` event with
  `inferred: true`, `source: spotter_serve_probability_peak` and the peak as
  confidence. It joins `all_events`; the review queue still reads only
  decoded contacts;
- a group with no serve evidence joins no rally
  (`contact_spotter.rally_segmentation.groups_without_serve`);
- a serve no contact follows is a service attempt with status
  `serve_not_played`, not a rally. Out, net and an untouched ace look the
  same, so the counters say `serves_not_played` and leave
  `aces_no_reception` / `serve_faults` absent rather than zero;
- a rally ends where the decoded phase track returns to `between`, at most
  3 s after its last contact. Per-team counts stay `null`.

Measured on `JIZ3exHe3hA_4k` (unseen camera, no labels): the decode emitted
15 serves for 23 points. 8 of the 9 serve-less groups had a serve peak of
0.18-0.79 at 1.0-3.0 s before their first contact; 4K crops at those frames
show a near-side serve in 2 cases and the ball leaving the far end line in
the other 6. The ninth, a lone reception with peak 0.04, joins no rally.
Result: 22 rallies all opening on a serve, 1 serve not played, against 14 FSM
rallies whose "serve" sat on a set or dig frame. This is a visual check on
one video, not a labelled score.

Forcing the decoder to place a serve before every `serve_flight` was measured
earlier and rejected (`require_contact_before_phase`); this works after
decoding and does not change any decoded contact.

Rebuild a saved result without rerunning detection (fresh directory only):

```powershell
python scripts/apply_spotter_rallies.py `
  --annotations results\game\game_annotations.json `
  --output results\game_spotter_rallies_v1
```

## What it cannot do

**It cannot tell warm-up from play.** On the one gold video with a real warm-up,
36 of 60 false positives sit before the first human contact, and the decoder
labels those frames `rally_live` or `serve_flight` -- so no phase gate inside
the model removes them. Measured: a gate on the decoder's own surrounding
phases catches 1 of 36, and the phase probability margin carries no signal
either, because at a contact frame the contact class takes the probability
mass. Exclude non-game time with `--match-config`, which the pipeline already
supports. Training on more `non_game` footage is the real fix; the three gold
videos carry 4958, 904 and 613 such frames, so two of them have almost none.

**It does not attribute a contact to a team or a player.** Events carry
`team: null` rather than a guess. For match statistics this is the gap that
matters most, and it may not need new labelling: the side is derivable from the
ball position at the contact frame through the court homography, and the 376
reviewed ball centres in `data/contact_spatial_gold_v1` can validate that
without annotating anything new.

**It is trained on one camera placement.** Three videos, one gym.

**A failed spotter does not fail the analysis.** If the checkpoint will not load
or inference raises, the FSM events already in `all_events` stay as they are
and `contact_spotter.error` records why.

## Refusals

The loader refuses rather than guessing when:

* the checkpoint schema is not `contact_spotter_v1` -- an experiment checkpoint
  from `run_dense_contact_spotting.py` is not production-loadable;
* the checkpoint's feature channels differ from the current `FEATURE_NAMES`,
  which would otherwise feed it silently wrong inputs;
* the label space differs from the current one.

## Checking a borrowed calibration first

`net_y` and the team split both come from the calibration, and both feed the
spotter's features, so running a new video with a calibration measured
elsewhere corrupts the input silently. Look at it first:

```powershell
python scripts\evaluation\overlay_calibration_check.py `
  --frame frame.png --calibration configs\court_calibration_youtube_4k_setup_1.json `
  --output overlay.png
```

Tests: `tests/test_contact_spotter.py`.
