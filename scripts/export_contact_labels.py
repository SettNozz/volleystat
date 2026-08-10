#!/usr/bin/env python3
"""
Export rule-derived contact labels from analysis results.

Serve and reception come from rally structure (very reliable); the remaining
contacts carry their FSM type as a weak pre-label. Output is a plain manifest
that dataset prep / the annotator can consume as pre-labels, so a human only
reviews instead of labeling from scratch.

Usage:
    python scripts/export_contact_labels.py --results-dir results/part_2
    # → results/part_2/contact_labels.json

Output format:
    {
      "video": "...", "fps": 30.0,
      "labels": [
        {"frame": 123, "label": "serve", "team": "team_b",
         "source": "fsm_rule", "confidence": 0.95},
        {"frame": 168, "label": "reception", "outcome": "success", ...},
        {"frame": 199, "label": "pass", "source": "fsm", "confidence": 0.5},
        ...
      ]
    }
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.tracking.serve_reception import annotate_rallies, derive_contact_labels

# FSM contact types passed through as weak labels (model/human refines them)
_WEAK_TYPES = {"dig", "pass", "attack", "block"}
_WEAK_CONFIDENCE = 0.5


def main():
    ap = argparse.ArgumentParser(description="Export rule-derived contact labels")
    ap.add_argument("--results-dir", required=True,
                    help="Directory with <stem>_stats.json from analyze_video")
    ap.add_argument("--output", default=None,
                    help="Output JSON (default: <results-dir>/contact_labels.json)")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    stats_files = sorted(rd.glob("*_stats.json"))
    if not stats_files:
        raise SystemExit(f"No *_stats.json found in {rd}")
    stats_path = stats_files[0]
    with open(stats_path) as f:
        stats = json.load(f)

    fps = float(stats.get("fps", 30.0))
    rallies = stats.get("rallies", [])

    # Ensure serve_reception is present (older results won't have it)
    if rallies and "serve_reception" not in rallies[0]:
        annotate_rallies(rallies, fps)

    labels = derive_contact_labels(rallies)
    strong_frames = {l["frame"] for l in labels}

    # Weak labels for remaining FSM contacts
    for rally in rallies:
        for ev in rally.get("events", []):
            if ev["type"] in _WEAK_TYPES and ev["frame"] not in strong_frames:
                labels.append({
                    "frame": ev["frame"],
                    "label": ev["type"],
                    "team": ev["team"],
                    "source": "fsm",
                    "confidence": _WEAK_CONFIDENCE,
                })

    labels.sort(key=lambda l: l["frame"])

    out_path = Path(args.output) if args.output else rd / "contact_labels.json"
    with open(out_path, "w") as f:
        json.dump({
            "video": stats.get("video", ""),
            "fps": fps,
            "source_stats": str(stats_path),
            "labels": labels,
        }, f, indent=2)

    strong = sum(1 for l in labels if l["source"] == "fsm_rule")
    print(f"Exported {len(labels)} contact labels "
          f"({strong} rule-derived serve/reception, {len(labels) - strong} weak FSM) "
          f"→ {out_path}")


if __name__ == "__main__":
    main()
