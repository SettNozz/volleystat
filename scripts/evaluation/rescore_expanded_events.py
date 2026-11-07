"""Validate saved event fold models with train-only inactive-channel masks.

No model fitting, feature refresh or source edits. Original unmasked runs stay
intact as diagnostics. Both 4-core and 6-raw typing reports are supported.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn

from action_recognition.annotator.contact_author_dataset import sha, write_new
from action_recognition.event_clips.config import EVENT_TYPES
from action_recognition.models.sequence_event_model import load_checkpoint
from scripts.evaluation.train_expanded_contacts import (
    OLD, SCHEMA, emit, event_metrics, normalise_event, raw_event_indices,
    training_channel_mask, training_sources,
)
from scripts.models.train_contact_spotter import enforce_determinism


class ClipTCN(nn.Module):
    """Same saved architecture as the raw-six experiment (strict state loading)."""
    def __init__(self):
        super().__init__(); layers = []; channels = 18
        for dilation in (1, 2, 4, 8):
            layers.extend([nn.Conv1d(channels, 48, 5, padding=2*dilation, dilation=dilation), nn.GELU(), nn.Dropout(.15)])
            channels = 48
        self.body = nn.Sequential(*layers); self.head = nn.Linear(96, 6)

    def forward(self, x):
        z = self.body(x.transpose(1, 2))
        return self.head(torch.cat([z.mean(-1), z[:, :, -20:].mean(-1)], dim=1))


def rescore(run, stages):
    enforce_determinism(); torch.set_num_threads(4)
    core_rows = json.loads((run/"events/dataset_manifest.json").read_bytes())["clips"]
    with np.load(run/"events/dataset.npz") as a: core_x = a["features"]
    for original in stages:
        family = "multiscale" if original == "events" else "tcn"
        target = original+"_validated"; output = run/target; output.mkdir()
        rows = core_rows if original == "events" else json.loads((run/"events_raw/dataset_manifest.json").read_bytes())["clips"]
        chosen = np.arange(len(core_rows)) if original == "events" else raw_event_indices(core_rows)
        X = core_x[chosen]; types = tuple(EVENT_TYPES) if original == "events" else ("serve", "reception", "set", "dig", "attack", "block")
        y = np.array([types.index(c["type"] if original == "events" else c["raw_types"][0]) for c in rows]); groups = np.array([c["video_id"] for c in rows])
        prior = json.loads((run/original/"predictions.json").read_bytes())
        if [c["id"] for c in prior] != [c["id"] for c in rows]: raise ValueError("Changed event dataset")
        predictions = {f"{scope}_{method}": np.array([types.index(c["predictions"][f"{scope}_{method}"]) for c in prior])
            for scope in ("old3", "expanded") for method in ("trees", family)}
        fitted = {}; audit = []
        for path in sorted((run/original/"models").glob(f"*/{family}_seed*.pt")):
            if family == "multiscale":
                model, checkpoint = load_checkpoint(str(path), device="cpu")
                mean = np.array(checkpoint["feature_mean"], np.float32); std = np.array(checkpoint["feature_std"], np.float32)
            else:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model = ClipTCN(); model.load_state_dict(checkpoint["state_dict"])
                mean = np.array(checkpoint["mean"], np.float32); std = np.array(checkpoint["std"], np.float32)
            key = tuple(sorted(checkpoint["training_sources"])); train = np.isin(groups, list(key))
            active = training_channel_mask(X, train)
            if "feature_active_mask" in checkpoint and not np.array_equal(active, checkpoint["feature_active_mask"]):
                raise ValueError("Stored training-channel mask changed")
            model.eval(); tensor = torch.from_numpy(normalise_event(X, mean, std, active))
            with torch.no_grad(): probability = torch.cat([model(tensor[i:i+64]).softmax(-1) for i in range(0, len(y), 64)]).numpy()
            fitted.setdefault(key, []).append(probability)
            audit.append({"checkpoint": str(path.resolve()), "sha256": sha(path), "seed": checkpoint["seed"],
                "training_sources": list(key), "active_channels": active.tolist(), "training_only_mask": True,
                "weights_retrained": False})
        for heldout in sorted(set(groups)):
            for expanded in (False, True):
                key = tuple(sorted(training_sources(groups, heldout, expanded)))
                if key not in fitted or len(fitted[key]) != 3: raise ValueError("Incomplete fold checkpoint ensemble")
                scope = "expanded" if expanded else "old3"; mask = groups == heldout
                predictions[f"{scope}_{family}"][mask] = np.mean(fitted[key], axis=0).argmax(1)[mask]
        results = {n: {"all": event_metrics(y, p, types),
            "old3": event_metrics(y[np.isin(groups, list(OLD))], p[np.isin(groups, list(OLD))], types),
            "new_explicit_corrections_only": event_metrics(y[~np.isin(groups, list(OLD))], p[~np.isin(groups, list(OLD))], types),
            "by_video": {v: event_metrics(y[groups == v], p[groups == v], types) for v in sorted(set(groups))}}
            for n, p in predictions.items()}
        new_y = y[~np.isin(groups, list(OLD))]; majority, count = Counter(new_y).most_common(1)[0]
        write_new(output/"report.json", {"schema": SCHEMA, "types": types, "results": results, "fit_audit": audit,
            "source_report_sha256": sha(run/original/"report.json"), "source_dataset_sha256": sha(run/"events/dataset.npz"),
            "training": "same fixed fold models; rescored only with train-only inactive-channel mask",
            "new_subset_majority_context": {"class": types[majority], "correct": count, "count": len(new_y),
                "accuracy": count/len(new_y), "scope": "descriptive test-set class prior, not a fitted benchmark"},
            "production_promoted": False})
        write_new(output/"predictions.json", [{**c, "predictions": {n: types[p[i]] for n, p in predictions.items()}} for i, c in enumerate(rows)])
        emit({"validated_stage": target, "new_corrections": {n: m["new_explicit_corrections_only"]["accuracy"] for n, m in results.items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=("events", "events_raw"), default=["events", "events_raw"])
    args = parser.parse_args(); rescore(args.run, args.stages)
