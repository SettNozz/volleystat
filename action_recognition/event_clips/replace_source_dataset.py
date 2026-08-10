"""Replace one source video's clips in an existing event dataset.

This is intended for promoting a completed full-video annotation: legacy
clips from that source are removed, the richer gold clips are inserted, and
feature files from both datasets are copied into one training directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _feature_path(directory: Path, clip_id: str) -> Path:
    matches = [directory / f"{clip_id}{suffix}" for suffix in (".parquet", ".npz")]
    existing = [path for path in matches if path.exists()]
    if len(existing) != 1:
        raise ValueError(f"{clip_id}: expected one feature file in {directory}, got {existing}")
    return existing[0]


def replace_source_dataset(
    base_manifest: Path,
    base_features: Path,
    replacement_manifest: Path,
    replacement_features: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base = _read(base_manifest.resolve())
    replacement = _read(replacement_manifest.resolve())
    replacement_clips = [dict(clip) for clip in replacement.get("clips", [])]
    replacement_stems = {str(clip.get("source_stem")) for clip in replacement_clips}
    if not replacement_clips or len(replacement_stems) != 1:
        raise ValueError("replacement manifest must contain exactly one non-empty source")
    replacement_stem = next(iter(replacement_stems))
    kept = [
        dict(clip) for clip in base.get("clips", [])
        if str(clip.get("source_stem")) != replacement_stem
    ]
    removed = len(base.get("clips", [])) - len(kept)
    combined = kept + replacement_clips
    ids = [str(clip["clip_id"]) for clip in combined]
    if len(ids) != len(set(ids)):
        duplicates = [clip_id for clip_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate clip ids after replacement: {duplicates[:5]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    features_out = output_dir / "features"
    features_out.mkdir(exist_ok=True)
    for clip in kept:
        source = _feature_path(base_features.resolve(), str(clip["clip_id"]))
        shutil.copy2(source, features_out / source.name)
    for clip in replacement_clips:
        source = _feature_path(replacement_features.resolve(), str(clip["clip_id"]))
        shutil.copy2(source, features_out / source.name)

    result = {
        "schema_version": max(int(base.get("schema_version", 1)), int(replacement.get("schema_version", 1))),
        "feature_schema": replacement.get("feature_export", {}).get(
            "feature_schema", base.get("feature_schema", "resolution_normalized_v2")
        ),
        "label_source": "human",
        "replacement": {
            "source_stem": replacement_stem, "removed_legacy_clips": removed,
            "inserted_gold_clips": len(replacement_clips),
            "base_manifest": str(base_manifest.resolve()),
            "replacement_manifest": str(replacement_manifest.resolve()),
        },
        "clips": combined,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    report = {
        "clips": len(combined), "removed": removed, "inserted": len(replacement_clips),
        "by_source": dict(Counter(str(c.get("source_stem")) for c in combined)),
        "feature_files": len(list(features_out.glob("*.parquet"))) + len(list(features_out.glob("*.npz"))),
    }
    with (output_dir / "replacement_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    split_policy = {
        "unit": "source_video",
        "train_source_stems": sorted({
            str(clip.get("source_stem")) for clip in kept
        }),
        "validation_source_stems": [replacement_stem],
        "reason": (
            "Keep every clip from the completed full-video annotation together; "
            "random clip splitting would leak adjacent frames, camera, and players."
        ),
    }
    with (output_dir / "split_policy.json").open("w", encoding="utf-8") as handle:
        json.dump(split_policy, handle, indent=2, ensure_ascii=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--replacement-manifest", type=Path, required=True)
    parser.add_argument("--replacement-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replace_source_dataset(
        args.base_manifest, args.base_features, args.replacement_manifest,
        args.replacement_features, args.output_dir,
    ), indent=2))


if __name__ == "__main__":
    main()
