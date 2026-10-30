# Human contact-author annotation

This tool collects a human answer to “who touched the ball?” independently of
the algorithms being evaluated. Each contact opens in its original video.
The right panel contains reference photos of the near-side players. Click a
person, **Інша команда**, **Не впевнений**, or **Це не контакт**. Labels start
empty; algorithm actor suggestions are never shown or accepted automatically.

## Prepare and run

Verify both commands with `--help`. Run from the checkout containing the tool.
Paths may point to another checkout's frozen data; output must be a fresh directory.

```powershell
python scripts/evaluation/prepare_contact_author_annotation.py `
  --manifest data/event_clips_three_full_gold_causal_120f_v1/manifest.json `
  --analyses-dir results/gold_pose_fsm_20261028_v1 `
  --timing-review data/contact_spatial_gold_v1/reviews.json `
  --output results/contact_author_annotation_20261031_v1

python -m action_recognition.annotator.contact_author_annotator `
  --project results/contact_author_annotation_20261031_v1 --port 5066
# http://127.0.0.1:5066/
```

Preparation reads the human manifest and saved pose/ball analyses. OpenCV only
reads original video metadata and decodes photo reference frames. No detector,
ReID encoder, event model or training job runs. Original annotations, model
caches and training features remain untouched. Existing output directories are
refused. No annotated MP4 is rendered.

The active manifest contains 414 clips from `Khe4P8QcJkg_5_4k`,
`j0gRhgFCgIs_4_4k`, and `j3BO9GgUFBI_3_4k`. Verified spatial-review timing
overrides original clip contact frames. Audited exact-time duplicates share one
review item while preserving all clip IDs, annotation IDs, review IDs, original
frames and raw action types. Four human-confirmed wrong events are excluded
with an explicit audit reason. Queue: **403 items**, 157 / 126 / 120 per video.
Six items have unverified timing, shown explicitly in the UI. Ambiguous inherited
raw types remain ambiguous; author assignment does not retype a contact.

## Establish the roster once per video

The saved analyses have local tracker IDs, not verified person identities.
Preparation proposes six reference photos from one clear, complete near-side
frame around the first contact. IDs P1–P6 are arbitrary reference-card numbers,
not jersey numbers, court positions, or proven match-long model identities.

1. Click **Показати кадр фотокарток**. Reference boxes and IDs appear on that
   exact frame. Check six different near-team people. Each card's **Кадр фото**
   highlights just that reference; it never claims an identity on other frames.
2. If a crop is wrong, choose **Замінити фото**, seek a clear frame, draw a
   rectangle around that person, and save. Repeat as needed.
3. Click **Підтверджую: це … різних гравців**. Contact labelling becomes enabled.
4. For a substitute, click **Новий гравець / заміна**, draw a clear reference,
   then explicitly confirm a new person. The next ID is P7, then P8, etc.
   Returning people retain their original card.
5. After confirmation, **+ Ракурс** adds a photo only after explicitly confirming
   the same person. The original reference cannot be replaced behind existing labels.

This is a manually confirmed photo roster. It does not claim automatic
whole-video ReID or discover every substitution. The annotator identifies the
person visually. Player IDs are local to each video, with exported identity keys
such as `Khe4P8QcJkg_5_4k:P2`. The current preparation scope is side `b`, the
near team in these three fixed-camera videos; **Інша команда** means side `a`.
Set-side changes are outside this project's scope.

## Review and save

- Choose a video/contact, play ±2 seconds at 0.25×/0.5×/1×, or step single frames.
  The timeline and original video controls allow wider context.
- Click a photo or press 1–9 to assign a person. Cards above P9 remain clickable.
  O = other team, U = unsure, N = not a contact. A/D = frames; arrows = contacts;
  space = context playback. Shortcuts are inactive while editing form controls.
- Auto-next is optional. Each choice saves before moving to the next contact.
  An existing choice is highlighted; **Скинути мітку** removes it with history.
- Refresh/restart resumes at the first item without a decision. Unsure items
  count as reviewed but remain unverified and can be revisited in the contact list.
- Two tabs cannot silently overwrite each other: a stale revision gets HTTP 409.
  Reload and inspect the current choice before repeating a write.

The project directory contains immutable `project.json`, `annotations.sqlite3`
and derived `portraits/`. SQLite stores the current state plus an append-only
revision log for labels, resets and roster changes. Back up the project directory;
stop the annotator before copying the database. Original files must remain
available. Startup checks SHA-256 fingerprints; requests also check size/mtime.
Changed inputs block writes without deleting existing decisions. A changed
queue needs a new project, not an implicit migration of human labels.

## Dataset export and interpretation

**Експорт JSON** downloads `contact_authors_rNNNNNN.json` with schema
`human_contact_author_dataset_v1`. It includes:

- source fingerprints, project SHA/revision, source-video scope and photo rosters;
- full review queue, excluded-event audit and all saved decisions;
- `contacts`: verified `player` / `other_team` decisions with original source
  links, nested human author label and per-video `author_key`;
- unlabelled / unsure / not-contact counts. These rows are excluded from the
  eligible `contacts` array; their decisions and queue rows remain available.

Verification scope is **contact author only** (`not_contact` verifies contact
presence). It does not certify the inherited contact type or timing. Model
photo boxes remain model-derived identity references, never human detector
ground truth. No labels are promoted into any training dataset automatically.

These are the event-type model's training videos. New person labels can support
development of actor attribution, but cannot establish independent event-model
quality or unseen-video generalization. Preserve complete-source-video splits
for future author model selection and keep a separate untouched test source.

## Focused checks

```powershell
python -m pytest tests/test_contact_author_annotator.py -q
node --test tests/test_contact_author_annotation_ui.cjs
```

Checks cover audited time/duplicate provenance, roster confirmation, immutable
confirmed person references, substitution IDs, invalid crops, SQLite resume,
stale-tab protection, revision history, source drift, selective export, blind
review, original-video range serving, portraits and fractional-FPS frame seeks.
