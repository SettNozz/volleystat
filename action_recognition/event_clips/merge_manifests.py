"""Merge event-clip manifests into one dataset directory.

Rebuilt after the v2 loss. Feature Parquets are hard-linked where the filesystem
allows it and copied otherwise, so merging a 1200-clip dataset does not duplicate
gigabytes for no reason.

Clip ids embed the source stem and contact frame, so a clip appearing in two
manifests is the same clip; the copy with the stronger label wins
(human > fsm_rule > model > none) rather than whichever was merged last.

Usage:
    python -m action_recognition.event_clips.merge_manifests \
        --manifests data/event_clips_merged_v5/manifest.json \
                    data/event_clips_pilot/manifest_labeled.json \
        --output-dir data/event_clips_merged_v6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from action_recognition.event_clips.manifest_schema import upgrade_manifest

_LABEL_RANK = {"human": 3, "fsm_rule": 2, "model": 1, "predicted": 1}
_CONTEXT_FIELDS = {
    "match_phase",
    "physical_action",
    "context_role",
    "is_service_reception",
    "reception_quality",
    "context_label_source",
    "context_label_confidence",
}


def label_rank(clip: dict) -> int:
    if not clip.get("event_types"):
        return 0
    return _LABEL_RANK.get(str(clip.get("label_source")), 0)


def context_rank(clip: dict) -> int:
    source = str(clip.get("context_label_source") or "")
    return {"human": 3, "weak_fsm": 2, "fsm_rule": 2, "model": 1}.get(
        source, 0
    )


def merge_clip_records(current: dict, candidate: dict) -> tuple[dict, bool]:
    """Merge orthogonal action/context supervision for the same contact."""
    event_better = label_rank(candidate) > label_rank(current)
    context_better = context_rank(candidate) > context_rank(current)
    if not event_better and not context_better:
        return current, False

    merged = dict(current)
    if event_better:
        for key in (
            "event_types", "primary_event_type", "label_source", "model_prediction"
        ):
            if key in candidate:
                merged[key] = candidate[key]
    if context_better:
        for key in _CONTEXT_FIELDS:
            if key in candidate:
                merged[key] = candidate[key]
    # Preserve any new non-label metadata without replacing stronger labels.
    for key, value in candidate.items():
        merged.setdefault(key, value)
    return merged, True


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--features-dirname", default="features")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_features = out_dir / args.features_dirname
    out_features.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    sources: dict[str, Path] = {}
    replaced = 0
    for manifest_path in args.manifests:
        mp = Path(manifest_path)
        data = upgrade_manifest(json.loads(mp.read_text(encoding="utf-8")))
        features_dir = mp.parent / args.features_dirname
        kept = 0
        for clip in data["clips"]:
            cid = clip["clip_id"]
            if cid in merged:
                combined, changed = merge_clip_records(merged[cid], clip)
                if not changed:
                    continue
                merged[cid] = combined
                replaced += 1
            else:
                merged[cid] = clip
            sources[cid] = features_dir / f"{cid}.parquet"
            kept += 1
        print(f"{mp}: {len(data['clips'])} clips, {kept} taken")

    missing = 0
    for cid, src in sources.items():
        if not src.exists():
            missing += 1
            continue
        link_or_copy(src, out_features / f"{cid}.parquet")

    out_manifest = out_dir / "manifest.json"
    out_manifest.write_text(
        json.dumps(upgrade_manifest({"clips": list(merged.values())}), indent=1),
        encoding="utf-8",
    )

    by_label: dict[str, int] = {}
    for clip in merged.values():
        for label in clip.get("event_types") or ["<unlabeled>"]:
            by_label[label] = by_label.get(label, 0) + 1
    by_source: dict[str, int] = {}
    for clip in merged.values():
        by_source[str(clip.get("label_source"))] = by_source.get(str(clip.get("label_source")), 0) + 1

    print(f"\n{len(merged)} clips -> {out_manifest}")
    if replaced:
        print(f"  {replaced} clips replaced by a stronger label")
    if missing:
        print(f"  WARNING: {missing} feature files not found; those clips will fail to load")
    print(f"  labels: {dict(sorted(by_label.items(), key=lambda kv: -kv[1]))}")
    print(f"  label_source: {dict(sorted(by_source.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
