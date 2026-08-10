"""Build the leakage-safe event dataset after 4K human labelling.

The three new 4K videos are higher-resolution copies of legacy sources 3, 4,
and 5.  Keeping both copies would leak the same match across a video-level
train/validation split.  This builder therefore:

1. combines one or more fully human-labelled 4K manifests;
2. removes legacy clips belonging to the replaced source stems;
3. maps each 4K source stem to its canonical legacy stem;
4. hard-links feature Parquets (copy fallback) into self-contained datasets.

The build fails before publishing a manifest if a label, mapping, or feature
file is missing.  That makes accidental partial training datasets visible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from action_recognition.event_clips.config import collapse_labels
from action_recognition.event_clips.feature_schema import RESOLUTION_NORMALIZED_SCHEMA


def _read_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = data.get("clips")
    if not isinstance(clips, list):
        raise ValueError(f"Manifest {path} has no clips list")
    return clips


def _require_human_labels(clips: Iterable[dict], context: str) -> None:
    invalid = [
        str(clip.get("clip_id", "<missing-id>"))
        for clip in clips
        if clip.get("label_source") != "human" or not clip.get("event_types")
    ]
    if invalid:
        preview = ", ".join(invalid[:5])
        raise ValueError(
            f"{context} contains {len(invalid)} clips without completed human labels: "
            f"{preview}"
        )


def _feature_sources(
    clips: Sequence[dict],
    features_dir: Path,
) -> dict[str, Path]:
    sources = {
        clip["clip_id"]: features_dir / f"{clip['clip_id']}.parquet"
        for clip in clips
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} feature Parquet files; first missing: {preview}"
        )
    return sources


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _publish_dataset(
    clips: Sequence[dict],
    feature_sources: Mapping[str, Path],
    output_dir: Path,
    metadata: Mapping[str, object],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_features = output_dir / "features"
    out_features.mkdir(parents=True, exist_ok=True)

    expected_ids = {clip["clip_id"] for clip in clips}
    stale = [p for p in out_features.glob("*.parquet") if p.stem not in expected_ids]
    if stale:
        raise ValueError(
            f"Output feature directory contains {len(stale)} stale files; "
            "choose a new output directory"
        )

    for clip in clips:
        clip_id = clip["clip_id"]
        _link_or_copy(feature_sources[clip_id], out_features / f"{clip_id}.parquet")

    manifest = {
        "feature_schema": RESOLUTION_NORMALIZED_SCHEMA,
        **metadata,
        "clips": list(clips),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (output_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "feature_schema": RESOLUTION_NORMALIZED_SCHEMA,
                "feature_files": len(clips),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def combine_human_manifests(
    manifest_paths: Sequence[Path],
    output_dir: Path,
) -> Path:
    """Combine completed human batches and their adjacent ``features`` dirs."""
    merged: dict[str, dict] = {}
    sources: dict[str, Path] = {}
    for manifest_path in manifest_paths:
        clips = _read_manifest(manifest_path)
        _require_human_labels(clips, str(manifest_path))
        batch_sources = _feature_sources(clips, manifest_path.parent / "features")
        for clip in clips:
            clip_id = clip["clip_id"]
            if clip_id in merged:
                raise ValueError(f"Duplicate human clip_id across batches: {clip_id}")
            normalized = dict(clip)
            normalized["feature_schema"] = RESOLUTION_NORMALIZED_SCHEMA
            merged[clip_id] = normalized
            sources[clip_id] = batch_sources[clip_id]

    clips = list(merged.values())
    return _publish_dataset(
        clips,
        sources,
        output_dir,
        {
            "label_source": "human",
            "source_manifests": [str(path) for path in manifest_paths],
        },
    )


def build_leakage_safe_dataset(
    old_manifest: Path,
    old_features_dir: Path,
    human_manifest: Path,
    output_dir: Path,
    canonical_map: Mapping[str, str],
    excluded_old_stems: Iterable[str],
    val_stems: Iterable[str] = ("3",),
) -> tuple[Path, dict]:
    """Replace legacy video copies with canonicalised human-labelled 4K clips."""
    excluded = {str(stem) for stem in excluded_old_stems}
    val = {str(stem) for stem in val_stems}

    old_all = _read_manifest(old_manifest)
    old_kept = [
        dict(clip)
        for clip in old_all
        if clip.get("event_types") and str(clip.get("source_stem")) not in excluded
    ]
    old_sources = _feature_sources(old_kept, old_features_dir)

    human_clips = _read_manifest(human_manifest)
    _require_human_labels(human_clips, str(human_manifest))
    human_sources = _feature_sources(human_clips, human_manifest.parent / "features")

    unknown = sorted(
        {str(clip.get("source_stem")) for clip in human_clips} - set(canonical_map)
    )
    if unknown:
        raise ValueError(f"No canonical source_stem mapping for: {', '.join(unknown)}")

    canonical_human: list[dict] = []
    for clip in human_clips:
        updated = dict(clip)
        original_stem = str(updated["source_stem"])
        updated["source_stem_original"] = original_stem
        updated["source_stem"] = str(canonical_map[original_stem])
        updated["feature_schema"] = RESOLUTION_NORMALIZED_SCHEMA
        canonical_human.append(updated)

    clips = old_kept + canonical_human
    ids = [clip["clip_id"] for clip in clips]
    duplicates = sorted(cid for cid, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate clip_id in final dataset: {', '.join(duplicates[:5])}")

    sources = {**old_sources, **human_sources}
    collapsed = [
        collapse_labels(clip.get("event_types"), clip.get("fsm_event_type"))
        for clip in clips
    ]
    class_counts = Counter(label for label in collapsed if label is not None)
    stem_counts = Counter(str(clip.get("source_stem")) for clip in clips)
    split_counts = {
        "train": sum(str(clip.get("source_stem")) not in val for clip in clips),
        "validation": sum(str(clip.get("source_stem")) in val for clip in clips),
    }
    report = {
        "old_total": len(old_all),
        "old_labeled": sum(bool(clip.get("event_types")) for clip in old_all),
        "old_kept": len(old_kept),
        "old_replaced": sum(
            bool(clip.get("event_types"))
            and str(clip.get("source_stem")) in excluded
            for clip in old_all
        ),
        "human_added": len(canonical_human),
        "final_clips": len(clips),
        "class_counts": dict(sorted(class_counts.items())),
        "excluded_non_core": sum(label is None for label in collapsed),
        "source_counts": dict(sorted(stem_counts.items())),
        "val_stems": sorted(val),
        "split_counts": split_counts,
    }

    manifest_path = _publish_dataset(
        clips,
        sources,
        output_dir,
        {
            "build": "leakage_safe_4k_human_v1",
            "canonical_source_map": dict(canonical_map),
            "excluded_legacy_source_stems": sorted(excluded),
            "val_stems": sorted(val),
            "summary": report,
        },
    )
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return manifest_path, report


def _parse_mapping(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid mapping {value!r}; expected SOURCE=CANONICAL")
        source, canonical = value.split("=", 1)
        if not source or not canonical:
            raise ValueError(f"Invalid mapping {value!r}; expected SOURCE=CANONICAL")
        result[source] = canonical
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-manifests", nargs="+", required=True)
    parser.add_argument("--human-output-dir", required=True)
    parser.add_argument("--old-manifest", required=True)
    parser.add_argument("--old-features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--canonical-map", nargs="+", required=True)
    parser.add_argument("--exclude-old-stems", nargs="+", required=True)
    parser.add_argument("--val-stems", nargs="+", default=["3"])
    args = parser.parse_args()

    human_manifest = combine_human_manifests(
        [Path(path) for path in args.human_manifests],
        Path(args.human_output_dir),
    )
    manifest_path, report = build_leakage_safe_dataset(
        old_manifest=Path(args.old_manifest),
        old_features_dir=Path(args.old_features_dir),
        human_manifest=human_manifest,
        output_dir=Path(args.output_dir),
        canonical_map=_parse_mapping(args.canonical_map),
        excluded_old_stems=args.exclude_old_stems,
        val_stems=args.val_stems,
    )
    print(json.dumps(report, indent=2))
    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
