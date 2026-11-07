# Contact-author promotion — 2026-11-07

Promoted `expanded_soft_hgb` (`authors_all_sides`) as the default contact-author
model. Event detection and event-type models are unchanged. No retraining.

- Runtime method: `osnet_temporal_all_sides_soft_hgb_v1`.
- Installed artifact: `models/contact_author/expanded_20261106_v1/model.joblib`.
- SHA256: `4fd55ffaea4a7280a1a87bb358c303fcfeb7a364bed92324389fe018dbf96091`.
- Original final-fit artifact and both versioned local installations have the
  same SHA256. Weights remain gitignored; the registry entry, loader, installation
  command and this record are versioned.
- Branch: `codex/expanded-contacts-20261106`.

## Quality evidence

The original [experiment report](contact_expanded_training_20261106_v1.md)
remains unchanged. Development whole-source-held-out near-person recall:
470/565 (83.19%); seven new sources: 307/381 (80.58%). These folds informed
selection. The final model was fitted on all ten sources and has no untouched
held-out score. No training-set accuracy is reported here.

## Promotion checks

- Python: **60 passed**, one MLflow protobuf deprecation warning, with
  `tests/test_contact_author_runtime.py`, `tests/test_player_contacts.py`,
  `tests/test_game_analyzer_rally_integration.py`,
  `tests/test_viewer_upload_workflow.py`, and `tests/test_model_registry.py`.
- JavaScript: **4 passed** in `tests/test_player_contacts_ui.cjs`.
- Production feature parity: **1,137 contacts / 10 sources**; candidate and
  global feature matrices exactly equal the frozen winning experiment, using
  `scripts/evaluation/check_promoted_contact_author.py`. No decoded video or
  new model outputs. Local reports:
  `results/contact_author_promotion_20261107_v1/parity.json` and
  `results/contact_author_promotion_20261107_v1/parity_final.json`.
- Saved-result integration: **102 contacts**, actual five-person photo roster
  `P1, P2, P3, P7, P8`, for `TcUru_ou4CQ_4k`. Paired stats/annotations contain
  identical learned reports. Strict replay computed **zero** model outputs;
  OSNet was not loaded. Output:
  `results/TcUru_contact_author_promoted_20261107_v2_cached`.
- The initial full-queue cache audit found 246 missing OSNet crops, including
  contacts excluded from the training examples. Only that missing embedding
  stage was captured, with batch size 128; no YOLO, SAM, ball or spotter replay,
  MP4 render, dataset rewrite or training occurred. The final replay above used
  `--require-model-cache` successfully.
- Original TcUru source hashes still match `rebuilt_from`. Removing only the
  added/replaced contact report and `models.contact_author` leaves JSON equal
  to the original analysis. Events, frames, ball, poses and identities remain
  unchanged. Source fingerprints:
  - annotations: `9914e01aa82d319e31079019f1abda563b339f53983948de805c1f6cd5dcc316`
  - stats: `21f34e5b68916a3b8d0a36cb474b1da53950eeb0442c6583f0a6f993bd8e5f28`

The integration example is a training source, so it demonstrates runtime and
cache behavior only. Automatic identity-photo setup quality is unmeasured;
evaluation used human-confirmed photo rosters. Predictions remain unverified.

See [deployment and replay instructions](../contact_author_production.md).
