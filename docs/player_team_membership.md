# Stable player team membership

The player jumping at f527 in `XNCS5BLBwig_00-00-00_00-02-30` was exported
as side `a` with no roster ID, although the same continuous raw track 3 was
P6 on side `b` before take-off. Some jump observations also disappeared when
the per-side player cap ran before identity. Airborne ankles are not points
on the ground plane, so instantaneous foot projection cannot determine a
known player's team.

## Processing order

1. Decode or replay cached raw YOLO/ByteTrack observations.
2. Apply court membership and continuous-track real-team memory. Three clear
   ankle observations establish membership. Ambiguous back-facing net poses
   cannot lock the far team. Source-frame gaps over one second and impossible
   box motion reset raw-ID membership; a new video resets the gate.
3. Validate sideline tracklets, then resolve the roster from frozen OSNet
   features on the selected real team. Matched roster photo anchors override
   uncertain feet on their continuous tracklets. Opposite-side detections
   cannot train or enlarge the roster merely by looking similar.
4. Recover observed roster members across a continuous tracklet with at least
   three consistent identity votes. A short appearance change is bridged by
   consistent identity evidence within one second; distinct evidence for
   another person still blocks recovery. A new raw ID additionally requires
   distinct gallery appearance and plausible motion within one second.
   For a wrong-team track after occlusion, three distinct observations can
   confirm a handoff from recent accepted appearance and motion, with frozen
   gallery support. Membership starts at that transition, never at the raw
   track's birth; its previous opponent observations remain unchanged.
   Simultaneous observations cannot receive the same roster ID. Missing
   observations remain missing; no skeleton is synthesized.
5. Apply per-side player limits, prioritizing confirmed handoffs and recognized roster members.
   Regroup `p` with the accepted poses and remap seed/gallery photo keys to
   their exported pose indices.
6. Compute final trajectory, FSM, pose/event features, contacts and statistics
   from the accepted memberships. Render and export those same observations.

## Meaning and provenance

- `poses[].team`: court side (`a`/`b`), for geometry and existing consumers.
- `poses[].team_id`: stable real team, using `team_identity_map` and explicit
  segment overrides. A set-side switch changes `team`, not `team_id`.
- `team_geometry`: instantaneous foot-side evidence retained when available.
- `team_source`: geometry, continuous track, roster anchor or roster evidence.
- `player_identity.team_membership`: method and recovery/correction counts.

`recent_roster_track_handoff` records a confirmed team transition. A temporary
conflict over the exact player ID does not undo that team evidence; the ID can
remain unknown instead of duplicating another simultaneous assignment.

Roster evidence remains model-derived and unverified unless the supplied
roster carries human provenance. Neither a team's six-player capacity nor a
previous integer track ID is proof of identity. Ambiguous appearance,
conflicting tracklet identities, stale/distant reappearance or an already
occupied roster identity cause abstention.

Raw detector/ByteTrack caches remain compatible. Newly retained observations
may need missing OSNet crops; changed final features may need new event
logits/contact probabilities. `--require-model-cache` still fails on those
missing outputs. Existing analyses and review logs are not rewritten; use a
fresh versioned result directory.

Standalone identity-only replay keeps its existing side-only semantics; full
pipeline analysis applies membership before rebuilding derived statistics.
The default six-person roster starts on side `b`; explicit roster seeds keep
existing player names when comparing analyses. This fixes a development-video
regression, not a held-out claim of full-match identity accuracy.

## Regression validation

The full 4,496-frame source was replayed into a new result with the source
model roster frozen at f512 (automatic, unverified; no human labels created).
P6 remains P6 on side `b`, real team `observed_team`, throughout f520–560,
including every exported observation of the reported jump. Re-extracting
pose features sequentially from the final JSON matches every saved `pose_f`.
All exported frames respect the six-per-side cap and one ID per observed
person; seed/gallery keys still address valid exported poses.

Existing YOLO/ByteTrack outputs were reused. Only missing accepted OSNet
crops and changed downstream contact probabilities needed capture. The
original result remains unchanged. Python and UI regressions also cover
occlusion, initial misclassification, recycled IDs, impossible motion,
appearance ambiguity, side switches, pose/centre alignment and photo-key
remapping.

The same clip also exposed P2 at f3001: raw track 518 previously described an
opponent, then acquired the lost P2 after track 307 disappeared. Its frozen
seed-view similarity alone was too weak for the previous recovery cutoff,
although similarity to recently accepted P2 observations was distinct.
Recent handoff recovery corrects 133 observations (f2984–3116) without
recolouring the earlier opponent or changing the P6 jump regression.

Handoff evidence uses a one-second recent window, appearance similarity at
least .84 and a .10 margin to other recent people, plus frozen-gallery
support at least .60. It requires three observations within .33 seconds;
equally plausible boxes abstain. These are engineering thresholds, not
calibrated confidence or a held-out accuracy estimate. Ordinary observations
already belonging to the target team are not new handoff candidates.

Three independent `appearance_motion_roster_recovery` matches can also bind
the new owner of a recycled raw track. Each must independently pass the
existing frozen-gallery similarity (.85), rival margin (.10) and motion
gates; confirmations need distinct frames within .33 seconds. This keeps
membership after the original one-second motion anchor expires. One or two
matches, ambiguous galleries, discontinuous motion and rejected side priors
cannot establish this binding. Earlier observations of the previous owner
remain unchanged. `confirmed_gallery_handoffs` reports these transitions.
Net-facing prior provenance stays in JSON but is not drawn over players.

## Missing servers and prior-only opponents

Six is an upper bound on players per side, not a required visible count. A
server may stand outside the court or camera between rallies. Missing poses
and missing roster IDs remain absent until observation evidence supports them.
The same rule applies during play; it does not depend on an FSM phase guess.

At f140 of `XNCS5BLBwig_00-00-00_00-02-30`, five near-side players are visible
while the sixth prepares to serve off camera. Opponent raw track 49 briefly
entered the near-side candidate pool through the back-facing net prior. Weak
P3 matches were then propagated along that opponent's continuous track.
The frozen f512 P3 anchor itself depicts the correct near-side player.

`team_geometry` now retains the existing foot-side decision before the soft
prior changes it. `side_prior_only` marks unconfirmed prior-based candidates
for automatic seed selection and whole-video roster chunks; they cannot
bootstrap a roster. Explicit roster-photo anchors and confirmed continuous
members retain their precedence through jumps and occlusion.

Before membership propagation, prior-only identity votes on each continuous
tracklet need independent team/anchor evidence or at least three distinct
strong frozen-gallery observations. Strength uses the existing recovery
distance limit (default 0.35), and the confirmation count uses the existing
recovery setting (default 3). Ordinary association, its motion history and
the frozen galleries are preserved. Weak matches cannot become their own
confirmation through subsequent tracklet filling or handoff recovery.
Rejected observations export `identity.reason: uncorroborated_side_prior`;
unassigned soft priors fall back to their pre-prior geometry. No minimum
player count enters these decisions.

Regression coverage includes five visible roster members plus a look-alike
opponent, the real sixth player's later return, genuine net players with
three strong observations, incomplete automatic rosters, confirmed anchors,
jumps, recycled raw IDs and preservation of pose/centre alignment. This is
model-derived validation, not a held-out attribution score.

The fresh result `results/viewer/job_missing_server_20261112_v1` completes a
strict 4,496-frame cache replay with zero new model outputs or cache writes.
F140 has five near-side poses; track 49 remains an opponent without P3 across
its exported observations. P3 still names the correct f512 anchor. F45,
F3366/F3376, the P6 jump, the P2 f3001 handoff and the established net-player
IDs remain covered. Final pose features match fresh sequential extraction;
exported ball coordinates match the previous result on every frame. Only one
changed contact-probability output needed capture; detector and OSNet outputs
were reused. Targeted tests: 177 passing. Result and annotation endpoints
return HTTP 200. Human labels, roster source files and prior results remain
preserved.

## Recycled setter track (2026-11-13)

Source result: `results/viewer/job_1794524983478148700`, video
`7YnJlvChSzw_4k`. At f9089 the near-side setter was still exported as an
opponent on raw track 1661. The track had acquired P5 after occlusion while
retaining its previous opponent membership. Independent appearance/motion
recovery correctly recognized P5 at f8960–8962, but those observations did
not bind continuity. At f8963 the previous team's membership returned.

The fresh paired result is `results/viewer/job_p5_team_20261113_v1`.
The three strong gallery/motion observations now confirm the handoff.
All 330 exported setter observations at f8960–9289 retain P5 and
`observed_team`; f9089 renders blue and the set at f9096 is attributed to
P5. Earlier observations of track 1661 remain unchanged. Prior provenance
stays in JSON; `BACK/NET prior` is removed from browser and video rendering.

Validation covered all 21,300 frames: unchanged ball fields and source file
hashes, the same roster seed/gallery, no duplicate simultaneous roster IDs,
the six-per-side cap, observed input boxes only, matching paired reports,
and freshly reconstructed sequential pose features. The full replay first
used strict model-cache mode. Only changed `contact_probabilities` required
one new cached output; YOLO, OSNet and event logits were reused. The resulting
spotter list has 187 events versus the source's 186. This is a derived-input
change, not new human labels or a held-out identity accuracy measurement.

Evidence and frozen replay snapshots live in `.runs/p5_team_20261113/`;
`validation.json` records the checks and the f9096 contact. Targeted checks:
150 Python tests and 10 UI tests passed. The viewer was restarted without
active analysis jobs and the corrected f9089 was checked in its browser UI.
