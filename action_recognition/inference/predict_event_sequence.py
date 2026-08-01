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
from action_recognition.event_clips.feature_schema import (
    LEGACY_PIXEL_SCHEMA,
    apply_checkpoint_feature_contract,
    validate_feature_schema,
)
from action_recognition.models.sequence_event_model import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--keep-rule-labels", action="store_true", default=True)
    ap.add_argument("--min-confidence", type=float, default=0.35,
                    help="Below this probability, write an unknown prediction")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.checkpoint, device)
    checkpoint_event_types = list(ckpt["event_types"])
    sequence_len = int(ckpt.get("sequence_len", 60))
    model.to(device)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32) \
        if ckpt.get("feature_mean") else None
    std = np.asarray(ckpt["feature_std"], dtype=np.float32) \
        if ckpt.get("feature_std") else None
    active_mask = ckpt.get("feature_active_mask")

    with open(args.manifest) as f:
        manifest = json.load(f)

    features_dir = Path(args.features_dir)
    sidecar_schema = None
    sidecar = features_dir / "feature_schema.json"
    if sidecar.exists():
        with open(sidecar) as f:
            sidecar_schema = json.load(f).get("feature_schema")
    data_schema = validate_feature_schema(
        manifest.get("feature_schema")
        or sidecar_schema
        or LEGACY_PIXEL_SCHEMA
    )
    checkpoint_schema = validate_feature_schema(
        ckpt.get("feature_schema") or LEGACY_PIXEL_SCHEMA
    )
    if data_schema != checkpoint_schema:
        raise SystemExit(
            "Feature schema mismatch: "
            f"dataset={data_schema}, checkpoint={checkpoint_schema}. "
            "Train or select a checkpoint produced from the same feature schema."
        )

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
        x_np = _fix_length(feats, sequence_len)
        x_np[:, 7] = np.linspace(0.0, 1.0, sequence_len, dtype=np.float32)
        x_np = apply_checkpoint_feature_contract(
            x_np, mean, std, active_mask,
        )
        x = torch.from_numpy(x_np[None]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        best_index = int(np.argmax(probs))
        label = checkpoint_event_types[best_index]
        confident = float(probs[best_index]) >= args.min_confidence and label != "other"
        clip["event_types"] = [label] if confident else []
        clip["label_source"] = "model" if confident else "model_unknown"
        clip["model_prediction"] = label if confident else "unknown"
        clip["model_probs"] = {t: round(float(p), 3)
                               for t, p in zip(checkpoint_event_types, probs)}
        n_pred += 1

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Predicted {n_pred} clips (+{n_rule} kept rule labels, "
          f"{n_skip} skipped no-features) → {args.output}")


if __name__ == "__main__":
    main()
