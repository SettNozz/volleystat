"""Add per-player contact proposals to a fresh copy of saved analysis JSON.

Uses the promoted author model and cached OSNet crops by default. Missing crops
require --capture-missing-reid. No detector/spotter replay, source edits or MP4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.player_contacts import build_player_contacts, source_fingerprint


def rebuild(annotations_path: Path, output: Path, *, legacy=False, model=None, roster=None, cache_dir=None,
            assets=None, device="auto", require_cache=True, use_registry=True):
    path = annotations_path.resolve()
    if not path.name.endswith("_annotations.json"):
        raise ValueError("Expected <stem>_annotations.json")
    if output.exists():
        raise FileExistsError("Output directory exists; choose a fresh versioned result")
    stem = path.name.removesuffix("_annotations.json")
    stats_path = path.with_name(f"{stem}_stats.json")
    annotations_bytes, stats_bytes = path.read_bytes(), stats_path.read_bytes()
    annotations, stats = json.loads(annotations_bytes), json.loads(stats_bytes)
    if stats.get("video") != annotations.get("video") or float(stats["fps"]) != float(annotations["fps"]):
        raise ValueError("Stats and annotations must belong to the same analysis")
    from src.tracking.contact_author_runtime import attribute_contacts
    report = attribute_contacts(annotations, stats.get("all_events") or [], enabled=not legacy,
        model_path=model, roster=roster, cache_dir=cache_dir, assets=assets, device=device,
        require_cache=require_cache, use_registry=use_registry)
    report["rebuilt_from"] = {"annotations": str(path), "stats": str(stats_path),
                              **source_fingerprint(stats_bytes, annotations_bytes)}
    annotations["player_contacts"] = stats["player_contacts"] = report
    resolution = report.get("attribution", {}).get("model_resolution")
    if resolution: stats.setdefault("models", {})["contact_author"] = resolution
    output.mkdir(parents=True, exist_ok=False)
    for suffix, payload in (("stats", stats), ("annotations", annotations)):
        with (output / f"{stem}_{suffix}.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"output": str(output.resolve()), "totals": report["totals"], "players": report["players"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true", help="Old geometry-only report, no models or video reads")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--roster", type=Path, help="Identity-only photo roster")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--assets", type=Path, help="Pinned OSNet bundle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--require-model-cache", action="store_true", help="Explicit cache-only replay (the default)")
    parser.add_argument("--capture-missing-reid", action="store_true", help="Allow only genuinely missing OSNet crops")
    parser.add_argument("--no-model-registry", action="store_true")
    args = parser.parse_args()
    if args.capture_missing_reid and args.require_model_cache: parser.error("Choose cache-only replay or missing-crop capture")
    print(json.dumps(rebuild(args.annotations, args.output, legacy=args.legacy, model=args.model, roster=args.roster,
        cache_dir=args.model_cache_dir, assets=args.assets, device=args.device,
        require_cache=not args.capture_missing_reid, use_registry=not args.no_model_registry), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
