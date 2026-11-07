"""Check production features against frozen experiment features; never fit models.

Reads cached embeddings only. No video decode or neural inference, no accuracy
claim from the final model's own training sources.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.models.inference_cache import ModelOutputCache, osnet_feature_store
from src.tracking.contact_author_runtime import (ContactAuthorModel, NewlineCompatibleFeatures,
    build_features, crop_requests, sha)


def check(run, output, cache_dir, assets, model):
    if output.exists(): raise FileExistsError("Choose a fresh parity report")
    manifest = json.loads((run/"manifest.json").read_bytes())
    if sha(run/"human_labels.json") != manifest["snapshot_sha256"]: raise ValueError("Frozen snapshot changed")
    setup = json.loads((run/"human_labels.json").read_bytes()); loaded = ContactAuthorModel(model); reports = {}
    for video, metadata in sorted(setup["videos"].items()):
        folder = run/"features_all_sides"; frozen = json.loads((folder/f"{video}.json").read_bytes())
        source = Path(metadata["annotations_source"]["path"])
        if sha(source) != metadata["annotations_source"]["sha256"]: raise ValueError("Source changed")
        if sha(folder/f"{video}.npz") != frozen["features_sha256"]: raise ValueError("Frozen features changed")
        annotations = json.loads(source.read_bytes())
        contacts = [{k: c["contact"][k] for k in ("frame", "type", "video_id")} for c in frozen["cases"]]
        context = [{"frame": c["frame"], "type": c["type"]} for c in setup["queue"] if c["video_id"] == video]
        roster = {"players": setup["rosters"][video]["players"]}
        _, requests, _ = crop_requests(annotations, contacts, roster)
        cache = ModelOutputCache(metadata["video_source"]["path"], cache_dir, True,
            source_identity={"kind": "video", "sha256": metadata["video_source"]["sha256"]})
        try:
            store = NewlineCompatibleFeatures(osnet_feature_store(cache, assets, assets/"osnet_msmt17.pth", metadata["width"], metadata["height"]))
            vectors = {p["key"]: store.get(f, p["bb"]) for f, players in requests.items() for p in players}
            if any(v is None for v in vectors.values()): raise ValueError("Missing cached parity crop")
            features = build_features(annotations, contacts, roster, vectors, context)
            with np.load(folder/f"{video}.npz") as expected:
                np.testing.assert_array_equal(np.array(features["candidate_features"]), expected["soft"])
                np.testing.assert_array_equal(features["global_features"], expected["global_features"])
            predictions = loaded.predict_features(features, contacts)
            reports[video] = {"contacts": len(contacts), "features_equal": True, "predictions": len(predictions),
                "computed_outputs": dict(cache.computed), "video_decoded": False}
            print(f"Parity {video}: {len(contacts)} contacts, exact feature equality", flush=True)
        finally: cache.db.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"purpose": "implementation parity, not accuracy estimation", "model_sha256": loaded.sha256,
        "source_manifest_sha256": sha(run/"manifest.json"), "videos": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--model-cache-dir", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True); p.add_argument("--model", type=Path, required=True)
    args = p.parse_args(); check(args.run, args.output, args.model_cache_dir, args.assets, args.model)
