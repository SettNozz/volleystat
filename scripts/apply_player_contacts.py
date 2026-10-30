"""Add per-player contact proposals to a fresh copy of saved analysis JSON.

Uses only saved events, ball positions, skeletons and roster IDs. Does not open
video, load models, run inference, alter human labels, or render an MP4.
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


def rebuild(annotations_path: Path, output: Path):
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
    report = build_player_contacts(stats.get("all_events") or [], annotations["frames"],
                                   annotations["fps"], annotations.get("player_identity"))
    report["rebuilt_from"] = {"annotations": str(path), "stats": str(stats_path),
                              **source_fingerprint(stats_bytes, annotations_bytes)}
    annotations["player_contacts"] = stats["player_contacts"] = report
    output.mkdir(parents=True, exist_ok=False)
    for suffix, payload in (("stats", stats), ("annotations", annotations)):
        with (output / f"{stem}_{suffix}.json").open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {"output": str(output.resolve()), "totals": report["totals"], "players": report["players"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rebuild(args.annotations, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
