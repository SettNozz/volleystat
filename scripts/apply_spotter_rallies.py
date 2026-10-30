"""Rebuild serve-anchored rallies on a saved result; writes a fresh result.

Runs the same ``GameAnalyzer._spot_contacts`` post-pass ``analyze_video.py
--contact-spotter`` runs, over the saved final ball coordinates and poses. No
detector, tracker or FSM is rerun. The FSM's events and rallies are restored
from the ``*_fsm`` copies first, so a result that already went through the
spotter is rebuilt from the same inputs it originally had.

The annotated ``*_analyzed.mp4`` is not copied; the viewer plays the original
video with canvas overlays when it is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.game_analyzer import GameAnalyzer, GameStats


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(annotations_path: Path, output: Path, checkpoint: str | None) -> dict:
    if output.exists():
        raise FileExistsError("Output directory exists; choose a fresh versioned result")
    path = annotations_path.resolve()
    if not path.name.endswith("_annotations.json"):
        raise ValueError("Expected <stem>_annotations.json")
    stem = path.name[: -len("_annotations.json")]
    stats_path = path.with_name(f"{stem}_stats.json")
    annotations = json.loads(path.read_text(encoding="utf-8"))
    original = json.loads(stats_path.read_text(encoding="utf-8"))
    checkpoint = checkpoint or (original.get("contact_spotter") or {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("No --contact-spotter given and the result did not record one")

    frames = annotations["frames"]
    for ann in frames:
        # A result that already ran the spotter keeps the FSM's events here.
        if "ev_fsm" in ann:
            ann["ev"] = list(ann["ev_fsm"])
    already_spotted = bool(original.get("all_events_fsm"))
    fps = float(annotations["fps"])
    stats = GameStats(
        video_name=annotations["video"],
        fps=fps,
        frame_annotations=frames,
        all_events=list(
            original["all_events_fsm"] if already_spotted else original.get("all_events") or []
        ),
        rallies=list(original.get("rallies_fsm") or original.get("rallies") or []),
        service_attempts=list(
            original.get("service_attempts_fsm") or original.get("service_attempts") or []
        ),
        rally_lifecycle=dict(
            original.get("rally_lifecycle_fsm") or original.get("rally_lifecycle") or {}
        ),
    )

    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.contact_spotter_path = str(checkpoint)
    analyzer._contact_spotter = None
    analyzer.match_segments = []
    # The spotter's features need only the frame size and net row the
    # original run used, which the annotations record.
    analyzer.calibration = SimpleNamespace(
        frame_width=annotations["frame_width"],
        frame_height=annotations["frame_height"],
        net_y=annotations["net_y"],
    )
    analyzer._spot_contacts(stats, fps)
    if not stats.contact_spotter.get("diagnostics"):
        raise RuntimeError(f"Contact spotter did not run: {stats.contact_spotter}")

    stats.contact_spotter["rebuilt_from"] = {
        "script": "scripts/apply_spotter_rallies.py",
        "annotations": str(path),
        "annotations_sha256": file_sha256(path),
        "stats": str(stats_path),
    }
    rebuilt = stats.to_dict()
    result = dict(original)
    for key in (
        "total_rallies", "rallies", "service_attempts", "rally_lifecycle",
        "all_events", "all_events_fsm", "rallies_fsm", "service_attempts_fsm",
        "rally_lifecycle_fsm", "contact_spotter",
    ):
        result[key] = rebuilt[key]
    annotations["contact_spotter"] = stats.contact_spotter

    output.mkdir(parents=True)
    for name, payload in (
        (f"{stem}_stats.json", result),
        (f"{stem}_annotations.json", annotations),
    ):
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {
        "output": str(output.resolve()),
        "events": len(result["all_events"]),
        "rally_lifecycle": result["rally_lifecycle"],
        "fsm_rallies": len(result["rallies_fsm"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contact-spotter",
        help="Checkpoint; defaults to the one the saved result recorded",
    )
    args = parser.parse_args()
    report = rebuild(args.annotations, args.output, args.contact_spotter)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
