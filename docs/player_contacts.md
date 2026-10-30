# Per-player contacts from saved analysis

Open the original-video viewer and use **Player contacts** at the top of the
sidebar. Select P1–P6 to highlight that person's skeleton in yellow while every
other skeleton stays gray. **All** restores the team overlay. Rosters include
confirmed substitutes with distinct IDs, so this JIZ3 example also offers P7.
Player IDs belong to the saved roster, not jersey numbers or raw ByteTrack IDs.

The selected player's list shows timestamp and original contact type. Click an
entry to seek and play that contact. Counts show reception, dig/pass/set,
attack, block and serve separately. `raw_counts` keeps dig, pass and set
separate as well. Selecting a player is independent of the old team/event
filters: spotter events with `team: null` remain visible here.

## Create a fresh result without inference

```powershell
python scripts/apply_player_contacts.py --help
python scripts/apply_player_contacts.py `
  --annotations results/JIZ3exHe3hA_identity_roster_20261030_v4_regret010/JIZ3exHe3hA_4k_annotations.json `
  --output results/JIZ3exHe3hA_player_contacts_20261030_v1
python viewer.py --port 5052
# http://localhost:5052/load?video=JIZ3exHe3hA_4k&results_dir=results/JIZ3exHe3hA_player_contacts_20261030_v1
```

The output directory must not exist. The command copies the paired stats and
annotations into a fresh directory, adding `player_contacts` to each. Source
events, poses, ball, identities, rallies, analytics and human labels compare
equal. Provenance records both source paths and SHA256 hashes. This command
does not even open the video or load a neural model; no model cache can miss.
It consumes the already cached/final model outputs in the saved JSON.
Normal pipeline export also builds the report after final identity assignment.
Legacy viewer results derive it in memory without rewriting analysis files.

## What an automatic attribution means

`player_contacts_v1` is a conservative **unverified proposal**, not a measured
solution to the known 2D team-attribution problem. A stable person ID tells us
who a skeleton represents; it does not establish who touched the ball.

For each saved contact, inspect the contact frame and up to +/-0.10 seconds.
Ball and skeleton always come from the same saved frame. Search stops at
missing frames, missing/off-camera ball or a new ball track. Missing ball at
the contact frame leaves the contact unresolved. No coordinates are generated.
Observed and reconstructed ball provenance remain in candidate evidence.

Use confident COCO points and body/arm/leg segments, normalize distance by
observed bbox height, and penalize temporal displacement. Both court sides
and unidentified poses compete. A candidate farther than 0.35 body heights,
or separated by less than 0.10 score from another candidate, stays unresolved.
A clear opponent is `other_side`; a near-side pose without a stable roster ID
stays unresolved. Distances/margins are engineering heuristics, not calibrated
confidence. There is no ball-vs-net-floor or nearest-foot side inference.

The report keeps `event_index`, frame, raw type, core type, original event
source/confidence, reason, candidate evidence, identity provenance and ball
source. Events are not deduplicated or retyped. Per-player totals exclude
unresolved and other-side events; existing per-team analytics remain unchanged.

JIZ3 example uses exactly the frozen roster-v4 result with appearance regret
0.10; it does not silently switch to the production default's 0.04. Its 103
saved spotter events yield 19 near-player proposals, 29 other-side proposals
and 55 unresolved. P7 has zero accepted proposals, which does **not** mean he
had zero touches. No attribution accuracy has been measured. Missed contacts
in the source event detector cannot appear in this list.

## Review and persistence

Choose **All** to inspect unresolved events and opponents too. Each dropdown
can select a roster ID, **Other side**, **Unresolved**, or **Reset to proposal**.
Counts/list update immediately after saving. A correction marks attribution
`label_source: human`; an unresolved correction remains unverified. Human
attribution does not verify event detection, event type or roster tracking.

Corrections append to `<stem>_player_contact_reviews.jsonl` beside the result.
Reloading the same result reapplies them. Reset appends a new record, preserving
history. Both analysis SHA256 values bind every review; changing either file
rejects old reviews with HTTP 409. Keep the original pair and log together.
No review is promoted to training data or written into the source JSON.
The API `GET /api/player_contacts/<job_id>` returns current corrected counts
and contacts, while exported analysis JSON retains the original proposals.

Validation covers competing opponents/unidentified players, ambiguity, missing
ball, gap/restart boundaries, full-body contacts, scaling, substitutions,
source preservation, append-only corrections and stale-source rejection.
Frontend checks cover selection/highlight, filtered rows, seeking and review
updates. The JIZ3 viewer is also exercised against the original video.
