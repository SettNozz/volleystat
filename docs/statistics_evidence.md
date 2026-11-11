# Reconciled statistics and video evidence

Open **Player statistics**, then click a count or percentage. The evidence page
lists exactly the contacts behind that cell. **Watch rally** plays the matched
score-rally window. **Open contact in match viewer** opens the original overlay
at that contact's frame. Empty metrics have an empty list, including zeroes.

For percentages the list is the classified denominator; each row marks whether
it also belongs to the numerator. Unknown outcomes and unassigned authors do
not enter a player's percentage. Unassigned contacts are available separately;
unknown-side contacts may belong to either team and never acquire a team from
this view. Digs, passes, sets and blocks have touch evidence without outcome
scoring.

## One derived view

`src/tracking/statistics_reconciliation.py` reconciles final `all_events`, current
author assignments, contact-rally windows, score-rally windows and the saved
`match_analytics.points`. `build_player_match_statistics` derives totals,
coverage and evidence from these same records. HTML, the JSON API and Excel
consume that view. This requires no inference, video analysis or result rewrite.

- Source-local IDs are namespaced. Spotter rally 1 never inherits FSM rally 1's
  point by ordinal. When `service_attempts_fsm` exists, even an empty list
  remains the score-attempt namespace.
- Non-serve contacts require a unique containing score-rally window. Known
  contact/score serve anchors must agree exactly; conflicting anchors, overlap,
  duplicated terminal frames, duplicate point/rally IDs, contradictory sides,
  point/set mismatches and outcome/point contradictions remain explicit.
  Anchor disagreement is a review signal, not proof either detector is wrong.
- Serves use a unique exact serve-frame match. The score-source lifecycle has
  precedence over the contact-source lifecycle. An explicit fault stays a fault
  unless contradictory point/side evidence makes the outcome unresolved.
- Point-winning attacks remain inferences from the terminal recorded contact
  and point winner. A recorded attack fault conflicting with that winner stays
  unresolved, including after winner review. Missing contacts can still affect
  this inference. No contact or outcome accuracy is implied by coverage.
- Reception quality requires the saved reception frame and compatible side.
  Unclassified receptions remain unknown.

## Point review

Expand **Point reconciliation** from the evidence page, or follow its link in
**Match Stats** / **Player statistics**. Conflicting player contacts link to their
unresolved evidence lists. Each score-rally row lets you choose **A · far**,
**B · near**, **Leave unresolved**, or **Reset to proposal**, then save.
Sides refer to that rally's camera geometry, not stable teams across sets.

Reviews append to `<stem>_point_reviews.jsonl`, with schema `point_review_v1`,
timestamp, source-scoped attempt reference and SHA256 of both original analysis
files. Invalid targets/winners are rejected before writing. Changed source
files reject stale requests or saved reviews with HTTP 409. Reset appends a
record; it does not erase history. Saves from viewer tabs are serialized.

A winner review changes the reviewed statistics projection and Excel export.
Original analysis JSON, human annotations, model outputs, FSM score timeline
and attack heatmap retain their provenance. A point review does not resolve an
ambiguous contact/rally join or confirm an attack kill, attack fault, author,
event detection or event type. Human author corrections remain in their own
existing review log.

## Coverage and export

The player table separates human-confirmed authors, automatic authors, derived
outcomes and unresolved outcomes. Author confirmation does not imply a human
verified the outcome. Evidence pages also report unassigned authors.

Excel exports contain Attacks, Serves, Serve reception, Contact evidence and
Point evidence. They use the same current author and point reviews as HTML.
Evidence retains event indexes, source namespaces, inference reasons and point
review provenance. It is not a training dataset or a promotion of predictions
to human labels.

`GET /api/statistics/<job_id>` returns totals, coverage, reconciled contacts,
points, conflicts and the source fingerprint. `POST` accepts the fingerprint,
`attempt_ref` and `winner_side` (`a`, `b`, or null), or `reset: true`.
`/statistics-evidence/<job_id>?player=P1&kind=attack&metric=point` opens a metric;
`player=all` selects assigned roster contacts and `player=unassigned` selects
contacts eligible for the unassigned counter.
