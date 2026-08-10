# Volleystat — Roadmap

> Compiled from a project discussion (conversation transcript, July 2026). Only content related to volleyball analytics.

## Current State

- **System core:** ball detection, player detection, court markup, net detection. This is the foundation all features build on.
- **Rally detection:** works — rally start/end is detected (serve_preparation → serve_ready → serve), but with errors: sometimes confuses the end of a rally with the start of a new one, gets confused by warm-ups with two balls.
- **Event detection model (pass/serve/reception etc.):** performs poorly — e.g., predicts "serve" where there was a pass. Input: ball coordinates + speed + acceleration + visual features.
- **Web UI (viewer):** upload video → analyze → frame-by-frame event visualization. Visualization is critical for verifying data quality before training models.
- **Labeling tool:** manual event labeling (serve, block, attack, etc.); data was labeled by hand.
- **Tried a video transformer** (fine-tuned on own footage) — recognizes events poorly; retraining from scratch is too time-consuming.
- **Block detection removed:** too few successful blocks in the data to train on.
- **Missing:** score counting, rotation detection, statistics as a product, skeleton usage in the model.

## Key Insight (who it's for and why)

The main value is **per-rotation statistics**. Example from the coach (Uheiev): "on rotation #5 our reception failed 5 times" → in the next game the player placement can be changed for exactly that rotation. The coach named **reception success rate per specific rotation** as the most important piece of information.

**Target audience:** teams and coaches (paid by sponsors). Markets where volleyball is developed — example: Munich, the Otto Brün tournament, 75 mixed teams at once.

## Roadmap

### 1. Skeletons (keypoints) in the system — priority
- Skeleton coordinates are currently not used anywhere.
- Keypoints + ball coordinates → understanding "player is jumping", "player–ball contact" → attack / reception / etc.
- An additional data source that should significantly improve event model accuracy.
- No technical blockers — "just need to sit down and do it".

### 2. Better event detection
- The current model predicts poorly — improve action classification within rallies (leveraging item 1).
- Label acquisition option: labeling via LLM/model, then review (not tried yet).
- Task simplification: track the serve → reception comes for free (serve + next contact being a pass = successful reception; ball out of bounds / on the floor = failed).

### 3. Attack heatmap (low-hanging fruit)
- A sharp change in ball trajectory = hit point.
- Track hit points by court zones → attack heatmap.
- Doesn't require rally detection — only trajectory change. "It's not hard."

### 4. Rotation detection (the flagship product feature)
- 6 rotations; determined by player positions during serve/reception.
- Player identification: jersey numbers / different kits.
- Enables: reception success rate per rotation, moment search like "show all serves received on rotation #5" without manually scrubbing video.

### 5. Score counting
- Determine who won the rally → running score.
- Critical: all statistics rest on the accuracy of this counting.
- Hard problem, currently not solved at all.

### 6. Block detection (postponed)
- Need more data with successful blocks, then bring it back into the model.

## Infrastructure & Product

- **Workflow:** processing runs locally on a GPU machine → results are uploaded to the cloud → the coach gets a link to view statistics.
- **Data collection:** record games yourself on camera (no coach needed for data collection). The video-processing infrastructure can be built in ~a week.
- **Volume:** a full game ≈ 1.5–2 hours of raw video to analyze; GPU required. Speed-up path — cards with more VRAM (bigger batch per inference).
- **Pricing:** ballpark ~$1000/year — roughly covers GPU infrastructure. An amateur coach without a sponsor won't pay ($0) → the product targets sponsored teams.
- **Value validation:** deliver reports to the coach for a month → then ask what changed in the training approach.

## Next Steps (short)

1. Integrate skeletons/keypoints into event model features.
2. Improve the event classification model; try LLM pre-labeling.
3. Build the attack heatmap (via trajectory change).
4. Prototype rotation detection → reception stats per rotation.
5. Score counting.
6. Minimal cloud infra: results upload + web access for the coach.
7. A month of reports to the coach → feedback.
