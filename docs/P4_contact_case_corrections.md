# P4 contact corrections from user-reviewed examples

The five user comments and three screenshot hashes are stored separately in
`configs/contact_case_reviews/JIZ3exHe3hA_P4_20261030_v1.json`, bound to the exact
JIZ3 stats/annotations SHA256 pair. This does not edit a human gold dataset.
All implementation/artifacts remain in `codex/contact-author-learning`'s
independent worktree. Original analysis JSON is preserved byte for byte.

## Diagnosed causes

1. **Team flips during a jump.** Raw tracks 29 and 229 have near-team P1 identity
   before/after the jump, but their airborne observations are exported as side
   `a`. The previous selector chose a team first, then a nearby person on that
   team. The actually inactive P4 could therefore win over an active skeleton.
2. **Body proximity substituted for an action.** The head or forearm of a
   standing receiver can be near the airborne ball in a 2D projection. Previous
   proposals did not require the selected arm to perform an attack/block.
3. **P4 appearance confusion.** At f6376, raw track 1262 is a receiving player
   incorrectly labelled P4. Against the user-confirmed P4 at f4915, its cached
   single-reference cosine distance is about .54. Standing observations of the
   front player on track 1267 can be about .21 from the same reference. During
   the jump that front player is misclassified as side `a`, with no current
   near-side ReID crop; cached views before and after provide identity evidence.
4. **Being at the net is not touching.** The first replay moved the f384 block
   proposal to P1, but visual inspection showed that nearby player is not the
   central blocker either. The saved hand/ball separation is too large. Final
   logic abstains instead of naming the next nearby person.

## General changes

`src/tracking/contact_actor_context.py` owns the new observed-evidence path:

- Score a candidate's hand/forearm and matching shoulder together. An overhead
  gesture from the other arm cannot validate a defending arm near the ball.
  Blocks need a raised arm near tape height and hand contact support within
  .25 observed body heights; other touches retain the .45 limit. Low/underhand
  attacks remain possible with very close observed hand/ball support.
- Add bounded shoulder-relative arm motion. Whole-body translation does not
  count as an arm swing. At the valid f4915 reception, the chosen P4 arm moves
  about 1.10 body heights/s, versus about .36 for the nearby rival's chosen arm.
  This resolves the previous ambiguous proximity margin while preserving low
  forearm receptions. Missing joints/raw-track changes do not generate motion.
- Search at most ±0.20 s for action evidence, bounded by neighbouring contacts.
  Team priors are weak costs across all candidates, never a hard preselection
  of a body. Off-camera periods and track restarts remain barriers.
- Use at most ±1.2 s of the same continuous raw track for two-sided identity
  support. Long gaps, large box jumps and conflicting identities break support.
  Ground position uses confident observed ankles in ±1 s and standing/landing
  anchors rather than projecting airborne feet as if they were on the floor.
- Court-relative depth is a **position**, not a rotational row label. Ground
  anchors put the defending P4 roughly 4–5 m from the net in the rejected attack
  examples, and the f6376 blocking candidate about 1.7 m from it. Backcourt
  attacks are retained; net proximity is a soft block prior. See
  [FIVB rules 13.2 and 14.1](https://www.fivb.com/wp-content/uploads/2025/01/FIVB-Volleyball_Rules2025_2028-EN-v05.pdf).
- Reuse only existing OSNet embeddings, with every observation key/raw ID/bbox
  checked against the current saved pose and cache hashes verified. The one
  positive human P4 reference at f4915 plus adjacent derived views guides a
  separate identity overlay. Competing roster galleries and one-to-one checks
  prevent simultaneous duplicate P4 labels. A conflicting source P4 stays
  unknown rather than contaminating the gallery.
- A two-raised-arm pose supporting a nearby subsequent block is not automatically
  counted again as the preceding opponent attack. This handles the neighbouring
  f6371/f6376 ambiguity. Block-then-follow-up by the same player remains allowed.

No source keypoint, box or ball coordinate is changed or fabricated. Identity
overlays are marked derived/unverified and applied only to their exact source
pose/raw ID. `P4*` means a derived correction; `ID?` marks a rejected source P4.
The UI suppresses rejected contact boxes instead of displaying them as authors.

## Current examples

| User frame | Result in the selected replay |
|---|---|
| 384, block | P4 excluded. Nearby P1 also lacks enough hand-contact evidence; author unresolved. |
| 1051, attack | P4 excluded. Active track 229 is proposed as P1 using observations before/after the jump. |
| 4915, reception | P4 retained as the contact author; low-arm reception remains valid. |
| 6376, block | P4 removed from receiving track 1262. Blocking track 1267 is proposed as P4 from the confirmed reference and temporal support. This new identity is **not** human-verified. |
| 11223, attack | P4 excluded; observed opponent action proposed, with nearby evidence at f11229. Exact timing/actor remains a model proposal. |

The f6376 human label is an **identity exclusion** for one skeleton, not a claim
that P4 cannot be the blocker on another skeleton. No human P1 label or positive
identity label for track 1267 has been invented. At f4915 the source ball is
interpolated; author confirmation does not turn its coordinates into detector
ground truth. That provenance remains explicit.

Three ablations are retained: `action_only`, `action_temporal`, and selected
`action_P4_reference`. Exclusion cases are used only for evaluation; only the
positive identity reference enters inference. There are no frame-specific
negative vetoes in the scorer. All five development conditions pass for the
selected replay, but this is **not held-out accuracy**. The previous 93.4% team
figure does not validate personal attribution or these corrections.

## Review and reproduce

Completed result in the independent worktree:
`results/P4_action_context_20261030_v4/viewer_result`.

```text
http://localhost:5054/load?video=JIZ3exHe3hA_4k&results_dir=results/P4_action_context_20261030_v4/viewer_result&contact_lab=1
```

Choose **Action + jump context + confirmed P4 reference**. P4 is selected by
default. Five buttons under the video open the user's exact review frames;
source contact and best evidence frames are shown separately. Yellow highlights
P4, cyan highlights a different supported actor, and rejected evidence does not
become an author marker. Switching to **Existing spotter + sequence author** at
the same frame restores the old overlay for comparison.

```powershell
$sharedRoot = 'C:/Users/illya/Documents/volleyball_analitics/volleystat'
python scripts/evaluation/refine_contact_actor_cases.py --help
python scripts/evaluation/refine_contact_actor_cases.py `
  --annotations "$sharedRoot/results/JIZ3exHe3hA_pose_completion_20261030_v1/JIZ3exHe3hA_4k_annotations.json" `
  --base-lab results/volleyball_contact_lab_20261030_v3/viewer_result/JIZ3exHe3hA_4k_contact_lab.json `
  --reviews configs/contact_case_reviews/JIZ3exHe3hA_P4_20261030_v1.json `
  --reid-cache "$sharedRoot/results/player_identity_kalman_20261030_v1/target_features_v4" `
  --output results/P4_action_context_NEW
```

The output must be fresh. The runner prints source, cache/review paths and hashes,
configuration and human-label scope before replay. No training or YOLO/OSNet/SAM
inference occurs, and no full MP4 is rendered. Old runs remain preserved; v1
failed the positive reception check, v2/v3 preceded final visual rejection of
the f384 replacement candidate. Use v4 for review.

Validation: 135 Python regressions and 17 JavaScript checks passed, including
same-arm action support, genuine backcourt/underhand attacks, jump context,
identity conflicts/gaps, ground-role scaling, cached-crop binding, negative-label
non-leakage, the attack/block ambiguity, source-specific overlays and rejection
of unsupported contact markers. Live browser navigation checked all five user
frames and the corrected f6376 highlight. SHA256 checks preserved source JSON,
the embedding/observation cache and all 103 source event times/types. Five case
conditions passed, with f384 explicitly unresolved rather than reassigned.
