# Contact authors on a saved, unlabelled video

**2026-11-07:** these class-fusion commands are historical experiments. The default
pipeline/replay now uses the [promoted expanded soft HGB](contact_author_production.md).

This experimental workflow applies frozen author models to a new source. It
does not run the video analysis pipeline, retrain a model, render an annotated
MP4, rewrite source results or modify human annotations. Use the original video
plus its existing `*_annotations.json` and `*_stats.json`.

The saved near-side roster supplies photo references and preserves its local
P1–P6 mapping. These automatically initialized references are **unverified**, not
human identity gold. The workflow refuses unsupported roster introductions;
it cannot establish that no substitutions happened.

## Method and scope

`class_fusion` combines five heads: physical actor ExtraTrees, physical actor
HistGradientBoosting, near-only and joint MLP pair rankers, and expanded OSNet
photo/track matching. Logistic regression learns their combination from
whole-source cross-fitted predictions. A separate pooled pose/trajectory model
and soft contact-sequence prior select the team.

The development result on three labelled videos is **146/184 = 79.35%** exact
near-person recall, **78.92%** assignment precision and **79.13%** F1. Nested
algorithm selection gets **75.54%** near-person recall. The >=80% target is not
met. These numbers are conditional on human contact times/types; they are not
new-video metrics or full event detection accuracy. Earlier HSV results in
`contact_author_benchmark.md` describe the preserved baseline experiment.

Deployment reuses the three existing two-source fold members, averaging their
conditional person distributions and sequence-adjusted team probabilities.
No final-fit is performed. The ensemble itself has no held-out quality estimate.
Every inference source must be absent from every member's training sources.
Human training labels are used only to reproduce training-fold sequence priors;
there are no test author targets. Unknown results stay errors when scored later.

## Commands

Verify each command's `--help` first. Every output must be a fresh directory.
The following paths are relative to the isolated contact-author worktree;
`$root` names the shared checkout containing source videos and inference caches.

```powershell
$root = 'C:/Users/illya/Documents/volleyball_analitics/volleystat'
python scripts/evaluation/apply_contact_author_models.py prepare `
  --video "$root/data/videos_for_test/X6JdbC8SM_U_4k.webm" `
  --annotations "$root/results/viewer/job_1793380818/X6JdbC8SM_U_4k_annotations.json" `
  --stats "$root/results/viewer/job_1793380818/X6JdbC8SM_U_4k_stats.json" `
  --output results/X6_author_inputs_NEW

python scripts/evaluation/run_contact_author_reid.py capture `
  --dataset results/X6_author_inputs_NEW --output results/X6_author_osnet_NEW `
  --assets "$root/models/player_reid/osnet_msmt17_v1" `
  --model-cache-dir "$root/data/inference_cache" --device cuda --require-model-cache

python scripts/evaluation/run_contact_author_reid.py prepare `
  --dataset results/X6_author_inputs_NEW --reid results/X6_author_osnet_NEW `
  --output results/X6_author_features_NEW

python scripts/evaluation/apply_contact_author_models.py predict `
  --dataset results/X6_author_features_NEW `
  --training-dataset results/contact_author_reid_temporal_20261031_v1 `
  --evaluation results/contact_author_rich_eval_20261031_v5 `
  --output results/X6_author_predictions_NEW

python -m action_recognition.annotator.contact_author_inference_viewer `
  --predictions results/X6_author_predictions_NEW --port 5068
```

Keep `--require-model-cache` for compatible replay. If it reports missing
outputs, audit the exact crop stage before a separate capture. On the first X6
application, all **1,129 near-side** requested OSNet crops were cached; **877
opposite-side** requested crops were genuinely missing. Only those missing
OSNet outputs were captured. YOLO, SAM, ball inference and contact spotting
were not repeated. HSV features decode only 200 requested source frames and
are saved separately. Frame decoding is not detector inference.

## Outputs and review

The inference snapshot is `inference_inputs.json`, never `human_labels.json`.
Targets are -1; human decisions are empty; timing and roster remain unverified.
Original model identities are excluded before numeric feature extraction, so
an earlier P1/P2 assignment cannot masquerade as new evidence. OSNet reference
roles distinguish automatic references from confirmed human photo references.

`predictions.json` lists every retained saved contact with its event index,
frame/type, person-or-other-team prediction, source-pose actor, uncalibrated
score and three-member agreement. `report.json` records all model hashes,
training/test provenance, counts and `metrics: null`. `source_preservation.json`
compares original video/annotation/stats checksums.

The viewer provides all contacts, player-card filters, disagreement filter,
frame stepping, 0.25/0.5/1x playback and a +/-2 second clip. The selected saved
pose is orange; other poses are grey. During playback, the same local raw track
is highlighted for context, without claiming verified identity on those frames.
No green truth skeleton or editable labels are invented. Accuracy requires a
separate source-bound human author annotation pass.

Tests: `tests/test_contact_author_inference.py` and
`tests/test_contact_author_reid_features.py`.
