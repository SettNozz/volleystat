"""
Pre-label clips with a trained model (review afterwards in the annotator).

Rule-derived labels (label_source == "fsm_rule": serve/reception from rally
structure) are kept as-is — the model only fills in the genuinely ambiguous
clips.

Usage:
    python -m action_recognition.inference.predict_event_sequence \
        --manifest data/event_clips_part2/manifest.json \
        --features-dir data/event_clips_part2/features \
        --checkpoint action_recognition/models/checkpoints_lstm_pose_v1/best_event_sequence.pt \
        --output data/event_clips_part2/manifest_predicted.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.data.event_clip_dataset import _fix_length, _load_features
from action_recognition.event_clips.config import EVENT_TYPES
from action_recognition.models.sequence_event_model import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--keep-rule-labels", action="store_true", default=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    model.to(device)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32) \
        if ckpt.get("feature_mean") else None
    std = np.asarray(ckpt["feature_std"], dtype=np.float32) \
        if ckpt.get("feature_std") else None

    with open(args.manifest) as f:
        manifest = json.load(f)

    features_dir = Path(args.features_dir)
    n_pred = n_rule = n_skip = 0
    for clip in manifest["clips"]:
        if args.keep_rule_labels and clip.get("label_source") == "fsm_rule" \
                and clip.get("event_types"):
            n_rule += 1
            continue
        feats = _load_features(features_dir, clip["clip_id"])
        if feats is None:
            n_skip += 1
            continue
        x_np = _fix_length(feats)
        if mean is not None:
            x_np = (x_np - mean) / std
        x = torch.from_numpy(x_np[None]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        label = EVENT_TYPES[int(np.argmax(probs))]
        clip["event_types"] = [label]
        clip["label_source"] = "model"
        clip["model_probs"] = {t: round(float(p), 3)
                               for t, p in zip(EVENT_TYPES, probs)}
        n_pred += 1

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Predicted {n_pred} clips (+{n_rule} kept rule labels, "
          f"{n_skip} skipped no-features) → {args.output}")


if __name__ == "__main__":
    main()
