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
