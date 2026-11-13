# Promoted contact-author model — 2026-11-07

The default author of a saved contact is now predicted by the expanded
**all-side temporal OSNet + soft HistGradientBoosting** model. Event detection
and event-type checkpoints are unchanged. The logical model `contact_author`
in `configs/model_registry.json` selects the versioned local artifact (or its
registry copy); the loader verifies the complete pinned SHA256 before loading.

The method's development whole-source-held-out result is 470/565 (83.19%) exact
near-person recall, 307/381 (80.58%) on seven new sources, after the photo-reference
guard. Selection used these folds; there is no untouched final-test score. The
all-ten-source final fit has no score on its own training data. Photo rosters
were human-confirmed in evaluation; automatic roster quality is unmeasured.
See the [full experiment history](experiments/contact_expanded_training_20261106_v1.md).

## Installation and default analysis

```powershell
python scripts/models/promote_contact_author.py --help
python scripts/models/promote_contact_author.py install `
  --source results/contact_expanded_training_20261106_v1/final_author/model.joblib
python analyze_video.py --help
python analyze_video.py --video <video.webm> --output results/game_NEW
```

The install copies the unchanged 7 MB artifact into
`models/contact_author/expanded_20261106_v1/model.joblib`, checks its SHA256, and
keeps a separate deployment record. Existing versions are not overwritten.
Weights remain local/gitignored, consistent with other model assets. The original
experiment snapshot/checkpoints retain their historical unpromoted metadata.

The pipeline runs authors after final contact spotting and roster identity,
before exporting the paired JSON files. Both files receive the same
`player_contacts_v1` report; stats model provenance includes `contact_author`.
The viewer reads this saved report and validates its event indices, frames,
types and paired equality. It does not silently replace it with geometry.
Historical results without a stored report retain the legacy viewer fallback.
Historical experimental sidecars keep their existing review workflow.

Controls: `--no-contact-author` restores geometry, `--contact-author-model`
overrides the artifact location (same validated checkpoint contract), and
`--contact-author-roster` supplies a source-bound identity-photo file. Cache-only
analysis (`--require-model-cache`) also forbids missing OSNet crop inference.
Missing model/setup assets have an explicit fallback status in the JSON;
corrupt/incompatible weights and strict cache misses fail rather than claiming
the selected model ran. The model is trained for near side `b` only.

## Photo setup and saved-result replay

No contact-author training decisions are read during inference. Default setup
uses the current identity pass's initial observed boxes and introduction anchors,
preserving its P IDs and automatic/human provenance. A confirmed annotation-tool
roster can instead be exported without any contact labels:

```powershell
python scripts/models/promote_contact_author.py export-roster `
  --project results/contact_author_expansion_20261101_v2 `
  --video-id TcUru_ou4CQ_4k --output results/TcUru_photos_NEW.json
python scripts/apply_player_contacts.py --help
python scripts/apply_player_contacts.py `
  --annotations <shared>/results/spotter_night_20261029_v1/TcUru_ou4CQ_4k_annotations.json `
  --output results/TcUru_authors_NEW --roster results/TcUru_photos_NEW.json `
  --model-cache-dir <shared>/data/inference_cache `
  --assets <shared>/models/player_reid/osnet_msmt17_v1 `
  --require-model-cache --no-model-registry
```

The five actual TcUru people remain five; IDs are not renumbered and P6 is not
invented. A substitute adds a reference person; roster length can exceed six.
The standard court cap remains six. Photo files bind the original video SHA256.

Replay defaults to cache-only. A missing-crop error identifies the exact OSNet
stage. Only a separate `--capture-missing-reid` call allows those missing crops.
No YOLO, SAM, ball or spotter inference is repeated, and no MP4 is rendered.
An existing output directory is refused. Events, poses, ball, roster identities,
rallies and human annotations remain intact; only a new author report and its
model provenance are added to a fresh result.

## Runtime and evidence

Runtime eligibility now requires a measured actor within 0.35 body heights of
the saved ball. Missing/off-camera evidence and distant candidates abstain.
A candidate cannot borrow the touch of a different roster player when that
source pose has at least three strong identity observations, at least 90%
identity agreement, and its own strong match on a continuous local track.
Gaps, teleports, unknown IDs and occluded-only appearance cannot establish this
constraint. When rejection changes the model's top choice, the replacement
also needs that repeated identity support; otherwise attribution is unresolved.
The team HMM, numeric feature matrices, checkpoint and raw scores are unchanged.
Reports retain `model_player_id`, `eligibility_rejections`, `source_player_id`
and `roster_support` so this decision remains reviewable. These are runtime
guards, not a new measured attribution-accuracy result. See the
[reported-case investigation](contact_moments_and_authors_20261113.md).

Both sides compete against identity photos; weak saved pose-side assignments
cannot eliminate a confirmed near-side person. The runtime reproduces the winning
experiment's +/-15-frame, stride-three observed-pose features, clean local-track
OSNet pooling, conservative gallery expansion and training-derived team HMM.
There is no retraining or access to human contact authors. Batch128 reuses the
`9b0b581` inference optimisation. The legacy LF/CRLF-only cache read-through
requires identical normalized encoder source, runtime, weights and settings.

Source skeletons/IDs/team fields and ball coordinates are never rewritten.
Missing ball stays missing; neighboring observed frames can inform the author
without generating a hidden hand or ball location. The method uses future
frames offline. Candidate highlights point to actual source pose indices, with
old raw IDs retained for review. At an evidence frame that pose takes priority
over a conflicting older ID; other skeletons are gray. Predictions remain
unverified and scores uncalibrated. Human review still overrides only author
attribution through the existing source-bound append-only review log.

## Validation

`scripts/evaluation/check_promoted_contact_author.py` compares production feature
matrices against the immutable winning experiment. All 1,137 contacts from ten
videos match exactly, with zero video decoding and zero new model outputs.
This checks implementation equivalence, not final-fit accuracy. Runtime, cache,
five-person setup, pipeline export and viewer tests live in
`tests/test_contact_author_runtime.py`, `test_player_contacts.py` and
`test_player_contacts_ui.cjs`.

The [promotion record](experiments/contact_author_promotion_20261107_v1.md)
preserves test counts, source fingerprints and the five-person cache-only replay.
