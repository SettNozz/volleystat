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

## New sources outside the existing datasets

Use `scripts/evaluation/prepare_contact_author_expansion.py --help` to prepare
the same blind annotator from existing paired spotter stats/annotations. Pass
each `*_stats.json` explicitly, plus the active event-training manifest and
the completed author snapshot to exclude. Source names and known video hashes
prevent including an old source under a different file name. Duplicate video
content inside the new batch is refused too. This command never runs models.

```powershell
python scripts/evaluation/prepare_contact_author_expansion.py `
  --stats <saved-result-1>_stats.json <saved-result-2>_stats.json `
  --exclude-event-manifest <shared>/data/event_clips_three_full_gold_causal_120f_v1/manifest.json `
  --exclude-author-snapshot results/contact_author_benchmark_20261031_v1/human_labels.json `
  --output results/contact_author_expansion_NEW

python -m action_recognition.annotator.contact_author_annotator `
  --project results/contact_author_expansion_NEW --port 5070
```

Current batch: `results/contact_author_expansion_20261101_v2`, seven videos and
**759** model contact proposals, initially zero author labels:

| Video | Contacts | Proposed people |
|---|---:|---:|
| X6JdbC8SM_U_4k | 87 | 6 |
| JIZ3exHe3hA_4k | 111 | 7, including the known substitute |
| xZbD8fx3D4k_1_4k | 195 | 6 |
| LdOghv6rt_4_4k | 118 | 6 |
| qDzHtxhNkBA_4k | 64 | 6 |
| QhqMwaCf8iw_4k | 82 | 6 |
| TcUru_ou4CQ_4k | 102 | 6 |

These sources are outside the active three-video event training source set and
the completed three-video author gold set. Some night videos have prior
event-type review/research history; this is not a claim they are absent from
every historical experiment or an untouched final test. No new split is assigned
and no training dataset is promoted or rewritten.

All contacts/types/times here are **model proposals**, not human gold. Exact
frame duplicates are grouped with all original indices/types retained. False
proposals can be marked **Це не контакт**. Author assignment verifies only the
person/team; it does not verify time/type. These flags remain explicit in the
export, so timing-sensitive evaluation needs a separate audited-time review.
Algorithm author predictions are never copied into labels or displayed as hints.

Saved identity galleries, when present, retain their local P1/P2 numbering and
substitute reference cards; otherwise six observed near-side boxes propose the
photos. Every roster starts **unconfirmed**, regardless of previous identity
provenance. Confirm or repair the cards before labelling. Additional substitutions
still require explicit new-person confirmation.

`project.json` fingerprints both analysis files and original videos; SQLite
autosaves separately from the old 403-decision project. Existing directories
are refused. Preparation only reads existing analyses and video metadata. It
does not replay YOLO, OSNet, SAM, ball tracking or the spotter, or render MP4.

Tests: `tests/test_contact_author_expansion.py` checks dataset/name/hash exclusions,
model provenance after human author assignment, immutable source data, resume,
saved substitute references and queue duplicate handling.

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

### Remove or restore a card

Each card has **Видалити Pn**. The inline confirmation states how many contacts
are assigned to that person. For a labelled player, **Видалити й зняти N міток**
explicitly returns those contacts to unlabelled; other players and videos retain
their labels. Nothing is removed until this second click. Cancel or navigate to
another contact to dismiss the pending removal.

Removal archives the complete photo references and previous contact decisions
in SQLite. **Відновити Pn** restores the latest removed card and restores its old
labels only where the contact is still unlabelled; newer human decisions win.
Restoration history is retained. Remaining IDs are never renumbered, and removed
IDs cannot be assigned to a new person. Number keys continue to match the visible
P number even when the roster has gaps. All cards may be removed and replaced;
an empty roster must receive new cards before confirmation.

The UI explains beside the contact when roster confirmation is still needed.
Requests time out after 15 seconds instead of leaving **Зберігаю…** forever.
After an ambiguous timeout, it reloads saved state without retrying or advancing
the contact: check the displayed decision before choosing again. Optimistic
revision checks still reject stale writes. Photo elements/URLs stay stable
between ordinary label saves, so all portraits are not fetched again per contact.

## Review and save

The **Тип контакту** selector changes the current event (serve, reception,
set/pass, dig, attack, block, free ball or other). Selecting a different type
autosaves without advancing, even when auto-next for author labels is enabled.
It works independently of roster confirmation and leaves the author decision
and contact time unchanged. **Повернути початковий тип** removes the override.
The contact heading/list update immediately; a rejected/stale save restores the
saved value instead of applying the edit to another contact.

Type reviews are stored under `state.event_reviews` with human provenance,
revision history and `verification_scope: contact_type_only`. Original project
events and source analysis files remain immutable. Export `contacts` and `queue`
apply these reviews to `type`/`raw_types`, mark event type human verified, retain
the original event in `original_event` and include the full `event_reviews` map.
Any stale original `core_type` is retained only inside `original_event`, so a
future dataset builder must derive core taxonomy from the corrected raw type.
Model contact timing stays unverified; editing a type does not audit its frame.

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

## Review movement and fill contact gaps

Both tabs use the same confirmed photo roster. In **Контакти**, step to the
observed touch and click **Підтвердити поточний кадр контакту** to verify or move
the proposed frame. For missed touches, pause on the correct frame, pick a type
in **Тип пропущеного контакту…**, then click **+ Контакт тут** and assign its
author. The proposal queue alone does not measure full-video coverage. Play the
source from frame zero to find contacts outside the queue.

In **Рух та ID**, click **▶ Відрізок** or press A to play or pause, and follow the turquoise box. Each item
contains consecutive observed poses with one raw track ID, up to six seconds.
Click a roster card or press 1–9 once for the whole piece if it follows the same
person. If the box changes person, pause on the last frame of the first person,
check **Позначити лише до поточного кадру включно**, then assign the card. Review
the rest from the next frame. D marks a person outside the roster, U marks an
uncertain span, and F (**Хибна детекція**) marks an empty or duplicate box. False
positives export as verified `false_positive` spans with the source pose indices
and no player ID. Undo the latest span to correct a decision. If a player is
already labelled in an overlapping frame, compare the two boxes and press Enter
if both show that person. The JSON export retains both labels, marking overlapping
frames of later labels in `duplicate_frames` for one-pose-per-player metrics.

Review chronologically for full coverage or use **Сумнівні першими** to prioritize
pieces where the saved model ID changes or goes unknown. The tool hides those
predicted identities. Progress counts reviewed cached poses on the target side
and reports uncertain ones separately. Missing detections and intervals on the
other side are outside that count. Newly encountered substitutes get a new
card, and returning players keep their existing card.

Contact and movement decisions autosave in the same SQLite project. JSON export
includes human-added contacts, reviewed contact frames, and `movement` spans
with source frame/pose indices, roster IDs and explicit verification. Evaluate
wrong IDs, unknowns, identity switches and recoveries after gaps separately.
Keep evaluation ranges out of seed/gallery corrections when comparing replays
on this video. Use another complete source for unseen-video evaluation.

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
