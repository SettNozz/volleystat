#!/usr/bin/env python3
"""Materialise mp4s for labeled clips whose video was never cut.

Model comparisons need clips the models did not train on. ``event_clips_merged_v5``
holds 130 labeled clips from source stem "3" — the validation stem — but only
their feature Parquets were kept, so no image-based model can be scored on them.
The manifest records ``source_video``, ``start_frame`` and ``end_frame``, which
is everything needed to cut them back out.

Usage:
    python scripts/evaluation/cut_holdout_clips.py --stem 3 \
        --classes pass attack other --per-class 8 \
        --output-dir data/event_clips_holdout3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from action_recognition.event_clips.config import EVENT_TYPES, collapse_labels  # noqa: E402
from action_recognition.event_clips.prepare import cut_clip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/event_clips_merged_v5/manifest.json")
    ap.add_argument("--stem", default="3")
    ap.add_argument("--classes", nargs="+", default=list(EVENT_TYPES))
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--output-dir", default="data/event_clips_holdout3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pool = defaultdict(list)
    for clip in manifest["clips"]:
        if clip.get("source_stem") != args.stem or not clip.get("event_types"):
            continue
        label = collapse_labels(clip["event_types"], clip.get("fsm_event_type"))
        if label in args.classes:
            pool[label].append(clip)

    rng = random.Random(args.seed)
    chosen = []
    for label in args.classes:
        clips = pool.get(label, [])
        rng.shuffle(clips)
        chosen.extend(clips[: args.per_class])
    print(f"selected {len(chosen)} clips: {dict(Counter(collapse_labels(c['event_types'], c.get('fsm_event_type')) for c in chosen))}")

    out_dir = Path(args.output_dir)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for clip in chosen:
        source = clip.get("source_video")
        if not source or not Path(source).exists():
            print(f"  {clip['clip_id']}: source video missing ({source})")
            continue
        target = clips_dir / f"{clip['clip_id']}.mp4"
        if not target.exists():
            if not cut_clip(source, clip["start_frame"], clip["end_frame"],
                            target, clip.get("fps", 30.0)):
                print(f"  {clip['clip_id']}: cut failed")
                continue
        entry = dict(clip)
        entry["path"] = str(target)
        written.append(entry)

    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(json.dumps({"clips": written}, indent=1), encoding="utf-8")
    print(f"{len(written)} clips written -> {out_manifest}")
    print(f"features stay in the source dataset; pass --features-dir "
          f"{Path(args.manifest).parent / 'features'}")


if __name__ == "__main__":
    main()
