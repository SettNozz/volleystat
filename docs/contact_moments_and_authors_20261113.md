# Contact moments and author eligibility — 2026-11-13

## Reported failures

Source: `results/viewer/job_1794524983478148700_subs_v3`, video
`7YnJlvChSzw_4k`, 29.97002997002997 fps. Source files remain unchanged.

P3's two attack cards at 5:04 were events 83 and 84, at F9141 and F9144.
Both author proposals used exactly the same observed player pose at F9138,
pose index 4, raw track 1705. The spotter emitted separate segments; highlights
deduplicated only event indices, so both counted as independent touches.
The same failure also affected serve events 100/101: F11918/F11925, observed
support F11918/F11916 on raw track 2191.

The setter shown around F13282 contacts the ball at F13289, event 112.
The source pose is index 7 on raw track 2471, consistently P5. The separate
soft author scorer let every roster photo compete for that same nearby pose.
Its appearance distance to P6 was 0.22356 versus 0.23933 to P5, and the HGB
scores were 0.95029 for P6 and 0.28110 for P5. It selected P6 even though the
continuous source track retained P5 across all 31 local observations, including
25 strong identity matches. P6 borrowed P5's geometric contact evidence.

The yellow P1 skeleton was a second, independent UI error: Set analysis retained
the player selection from an earlier highlight/roster view. Selecting an arc
did not update the focused player or contact evidence.

## General rules

`src/tracking/contact_moments.py` supplies the same moment groups to highlights
and statistics/evidence/Excel. It merges only adjacent model proposals with:

- the same assigned/proposed player, status and exact contact type;
- source event frames within 0.30 seconds;
- measured support on the same pose, or the same raw track within 0.10 seconds
  with continuous boxes;
- no intervening contact or human-reviewed event/author decision.

Every member must match the first proposal, preventing transitive chains.
The highest event-confidence representative retains all `source_event_indices`.
Original event arrays, indices, labels and review logs remain intact. Separate
players, attack/block pairs and unsupported or off-camera observations survive.
This is evidence-based consolidation, not a global spotter refractory period.

Author inference retains the frozen feature/checkpoint contract and team HMM.
Before selecting a roster candidate, runtime rejects missing/off-camera contact
evidence, distance above 0.35 body heights, off-court players and disagreement
with strong source roster support. Source support requires at least three
distinct strong frames and 90% agreement in the +/-15-frame local window on
one track segment. The evidence pose must itself have a strong identity match.
Automatic support requires `reason=associated` and appearance distance <=0.25;
verified human identity can also qualify. Gaps, teleports and unknown IDs do
not carry identity across segments. A replacement for a rejected top choice
needs its own repeated source identity support, otherwise attribution abstains.
Original scores, the model's eligible top choice and rejection reasons survive
in the derived report. No source pose is relabelled.

Set analysis clears previous player focus and selects only its own saved setter
and contact evidence. Unknown setters remain unknown.

## Validation and local output

Cache-only replay writes fresh paired JSON to
`results/viewer/job_1794524983478148700_contact_fixes_v2`.
It uses the existing video-bound OSNet cache, with zero missing crops, zero
new model outputs and no rendered video. The intermediate `contact_fixes_v1`
remains available separately.

P3 has four attack moments instead of five, and two serve moments instead of
three. The F13289 set is assigned to P5. Source events and frame observations
are preserved, and raw model scores match the source replay exactly.

Regression tests cover shared highlight/statistic counts, retained event
indices, terminal outcomes, frame rates, human decisions, distinct players and
types, intervening contacts, raw-ID teleports, missing support, chain merging,
roster conflicts, bench intervals, uncertain replacements and stale UI focus.
These tests and this video review do not establish full-match or held-out
contact-author accuracy. Predictions remain unverified; human author reviews
still take precedence and remain source-bound.
