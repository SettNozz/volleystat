# Usage and setup

This guide expands the [README](../README.md). Commands use PowerShell from the repository root. Use a new result directory for each analysis.

## Model assets

The source checkout does not include local videos, production weights, calibration files, inference caches or analysis results. `configs/model_registry.json` defines logical models, local fallback paths and pinned SHA256 values.

| Model | Purpose |
|---|---|
| `ball_detector` | Primary full-frame ball detector, currently configured at input size 1920. |
| `ball_detector_tiled` | Optional second detector for `--ball-ensemble`. |
| `pose_estimator` | YOLO11m-pose; default pose input size is 2560. |
| `player_reid` | Pinned OSNet code/weight bundle for roster identity and cached appearance features. |
| `event_sequence` | Four-class causal event overlay: serve, reception, pass, attack. |
| `contact_spotter` | Optional dense typed contacts and serve-anchored rallies; checkpoint includes its decode constraints and sidecar. |
| `contact_author` | Default promoted contact-author model and deployment record. |
| `net_sam` | SAM net-tape model for the automatic net workflow. Manual calibration is available in the viewer. |

A reachable MLflow `production` alias takes precedence over the local fallback; explicit CLI model paths override registry selection. `--no-model-registry` uses configured local files. Each run records the resolved model source and checksum. The legacy `spotting` registry entry is optional and is not the production contact spotter.

Install the frozen identity bundle once, or reuse an existing verified bundle:

```powershell
python scripts/models/install_player_reid.py
python scripts/models/install_player_reid.py --help
```

Audit model files and available registry backups:

```powershell
python scripts/models/verify_models.py
```

The verifier reports missing files and checksum mismatches. A fresh checkout needs separately provisioned weights; the installer above supplies only the player-identity bundle. Contact-author installation and source-bound photo setup are documented in [contact-author production](contact_author_production.md).

The viewer requires `flask` and `flask-cors`; feature/annotation workflows use `pyarrow`. These are installed separately in the README setup commands. Automatic SAM net segmentation needs the matching SAM package and checkpoint; optional SAM3 post detection has its own isolated environment. See [SAM workflow](sam_people_masks.md) for that experimental environment.

## Viewer workflow

Start `python viewer.py` and open [localhost:5050](http://localhost:5050). A second instance can use `python viewer.py --port 5051`.

1. Choose **Upload** and provide a path or drop a video. Dropped files are copied into `data/uploads/`; the same name and size reuse the existing copy.
2. Open **Preview and edit court & net**. Start from a saved camera setup or the default calibration. Drag court lines/corners and net tape handles. The handle magnifier shows a native-resolution crop; use the mouse wheel to change its zoom.
3. Choose a clearer preview time if needed, then start **Analyze Video**. Edited geometry is saved for that job. Manual mode uses the drawn court centre line as the floor divider and bypasses automatic net detection. **Use automatic settings** restores the automatic workflow.
4. Open the result and inspect overlays and events. The original recording remains the video source; saved annotations supply the overlays.

The preview and result share three net references: upper tape, lower mesh edge and player divider. **Calibration & sync** saves a manual correction for all three together. Named camera setups live in `configs/court_setups.json`; job-local calibration does not change shared setups.

### Review tools

- **Video Analysis:** ball and trail, player skeletons/IDs, net/court, FSM and model overlays, event timeline and rally clips.
- **Set-to-attack trajectories:** adjacent set/attack pairs from the same rally. Filter by side, attack zone and setter; overlay matching paths and click a path or row to play that sequence. Derived points are dashed; gaps and new tracks break paths.
- **Player contacts:** select a roster ID, highlight that player and seek to their saved contacts. Contact-author corrections are stored separately and bound to the analyzed result. Unknown-team contacts show `—` with both teams selected and are excluded by a single-team filter.
- **Visual Stats:** serve/reception maps, attack origin/direction, team contact totals and serve lifecycle summaries.
- **Match Stats:** rallies, service attempts and a contact log, with team/outcome filters.
- **Player statistics:** attack attempts, inferred point-winning attacks and reception outcomes from saved contacts, reviews and point evidence. Missing evidence leaves outcomes unresolved. **Edit players** sets name, jersey and position (outside hitter, middle blocker, setter, opposite hitter) in this browser; Excel export includes them. Pick a player (or click a name) for a personal panel with all categories; **Player highlights** opens a separate page that plays that player's attributed attacks, receptions, serves and other contacts with adjustable lead-in and follow-through.

Near-side attack height is an approximate projection from saved ball, pose and net geometry. Without usable evidence, the badge reports that height is unavailable.

The viewer chooses a ball batch using compatible caches or available GPU memory, then reduces it on CUDA memory exhaustion. CLI batch size is explicit/default 8; batch size is part of the ball cache key.

## CLI calibration

Each camera placement needs its own calibration. The interactive tool and pipeline use different corner-order conventions, so normalize before analysis:

```powershell
python scripts/interactive_court_calibration.py --input "C:/path/game.webm" --output court_calibration_raw.json
python scripts/normalize_calibration.py --input court_calibration_raw.json --output court_calibration_game.json
```

The normalizer preserves the input and refuses an existing output unless explicitly forced. Check its reported image-row ordering: world `y=-9` (far end line), `y=0` (net), `y=+9` (near end line). Use each recording's actual net height; the CLI default profile is `mixed` (2.34 m), with `--net-profile men` (2.43 m) or `--net-height-m` for an explicit value.

Automatic net detection uses SAM tape detection and optional isolated SAM3 post detection, with Hough fallback. Inspect geometry if post spacing is implausible. `--no-net-detection` reuses saved geometry without automatic refresh.

## Analysis options

```powershell
python analyze_video.py --help
```

| Option | Effect |
|---|---|
| `--duration 30` | Process the first 30 seconds for a short review run. |
| `--contact-spotter <checkpoint>` | Replace exported contacts and rallies with the dense spotter's output; retain FSM versions in separate fields. |
| `--match-config <json>` | Set non-game intervals, set boundaries and stable team mappings. Adapt `configs/match_config.example.json` to the actual recording. |
| `--ball-ensemble` | Collect candidates from an additional tiled ball detector. |
| `--ball-smooth light` | Change contact-preserving trajectory smoothing; default is `moderate` (7 frames). |
| `--ball-reconstruction conservative` | Enable derived gap reconstruction; default is `off`. Other modes: `contact-aware`, `compare`. |
| `--no-player-identity` | Keep raw tracking without offline roster identity. |
| `--player-identity-side a` | Move roster identity from default near side `b` to far side `a`; contact-author model scope remains separate. |
| `--no-contact-author` | Use legacy geometric contact-author proposals instead of the promoted model. |
| `--require-model-cache` | Fail on missing compatible model outputs instead of running inference. |
| `--render-video` | Explicitly encode final annotated video/frames. |

### Contact spotting

Viewer uploads use the configured `contact_spotter` checkpoint when present. CLI requires an explicit checkpoint:

```powershell
python analyze_video.py `
  --video "C:/path/game.webm" `
  --calibration court_calibration_game.json `
  --output results/game_spotter_v1 `
  --contact-spotter models/contact_spotter/spotter_20261028_v1.pt
```

The spotter replaces `all_events`, per-frame `ev`, rallies and service attempts. FSM versions remain in `*_fsm` fields. Match score and attack heatmap remain FSM-derived. Spotter contacts have no team assignment from the spotter itself; later side/author stages add their own evidence. See [production contact spotting](production_contact_spotter.md) and [contact-side proposals](team_attribution.md).

### Cached replay and export

Normal analysis saves compatible raw model outputs automatically. Reuse them when changing tracking, smoothing, geometry or roster rules. Keep the same source and compatible inference settings; a fresh result directory does not require fresh inference.

```powershell
python analyze_video.py `
  --video "C:/path/game.webm" `
  --calibration court_calibration_game.json `
  --output results/game_smooth_v2 `
  --ball-smooth light --require-model-cache

python analyze_video.py `
  --video "C:/path/game.webm" `
  --calibration court_calibration_game.json `
  --output results/game_export_v1 `
  --require-model-cache --render-video
```

Keep the original run's other options, including its spotter checkpoint, when reproducing its event path. Changed model inputs can require new predictions even if ball and pose outputs already exist. See [cache contracts and saved-result rendering](model_output_cache.md).

Reconstruction remains separate from observations. `compare` stores conservative/contact-aware variants and selects conservative output. Reconstructed points affect the FSM only with `--reconstruction-affects-fsm`, which admits accepted high-confidence points. Off-camera gaps stay missing.

## Open existing results

The [library](http://localhost:5050/library) indexes results under `results/`; `/api/library` returns its JSON index. Entries missing stats remain visible but cannot be opened.

A CLI result can also be opened directly:

```text
http://localhost:5050/load?video=game&results_dir=results/game_v1
```

`video` is the input stem; `results_dir` is the directory containing its paired JSON files. Persisted upload jobs use `/results/<job_id>`. Keep the original video available for playback and thumbnails.

## Human annotation

Flask tools under `action_recognition/annotator/` support clip/full-video event labels, contact timing/type review, contact authors and player movement. Reviews keep model proposals separate from verified human labels; export is an explicit workflow.

Start with [contact review](contact_review_workflow.md), [contact-author annotation](contact_author_annotation.md) or the [Ukrainian spatial annotation guide](contact_spatial_annotation_uk.md). Training and dataset exports are separate from normal video analysis.
