"""Compare fixed contact-author hypotheses on frozen analyses; export a review result.

Gold labels supervise comparison only, not a trained model. Whole source
videos remain separate in reporting. JIZ3 person/contact authors are unlabelled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracking.contact_attribution import METHODS, build_contact_attribution
from src.tracking.player_contacts import source_fingerprint


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def scores(rows):
    known = [(truth, c) for truth, c in rows if truth in ("a", "b")]
    predicted = [(truth, c["side_proposal"]) for truth, c in known if c.get("side_proposal") in ("a", "b")]
    selective = [(truth, c["side_proposal"]) for truth, c in known if c.get("side_accepted") and c.get("side_proposal") in ("a", "b")]
    f1s = []
    for side in ("a", "b"):
        tp = sum(truth == side and c.get("side_proposal") == side for truth, c in known)
        fp = sum(truth != side and c.get("side_proposal") == side for truth, c in known)
        fn = sum(truth == side and c.get("side_proposal") != side for truth, c in known)
        f1s.append(2*tp/max(2*tp+fp+fn, 1))
    return {"labels": len(known), "predicted": len(predicted),
            "top1_coverage": len(predicted)/max(len(known), 1),
            "top1_accuracy": sum(a == b for a, b in predicted)/max(len(predicted), 1),
            "macro_f1_unknown_as_miss": sum(f1s)/2,
            "selective_coverage": len(selective)/max(len(known), 1),
            "selective_accuracy": sum(a == b for a, b in selective)/max(len(selective), 1)}


def evaluate_gold(labels_path):
    raw = labels_path.read_bytes()
    project = json.loads(raw)
    reviews_path = Path(project["reviews_path"])
    if sha(reviews_path.read_bytes()) != project["review_sha256"]:
        raise ValueError("Authoritative spatial reviews changed; do not silently reuse team labels")
    if project.get("schema") != "contact_team_labels_v1":
        raise ValueError("Unknown human team-label schema")
    pooled = {method: [] for method in METHODS}
    by_video = {}
    sources = {}
    for stem, video in sorted(project["videos"].items()):
        path = Path(project["annotations_dir"]) / f"{stem}_annotations.json"
        content = path.read_bytes()
        sources[stem] = {"path": str(path), "sha256": sha(content)}
        payload = json.loads(content)
        items = [item for item in project["items"] if item["video_id"] == stem
                 and project["labels"].get(item["id"], {}).get("team") in ("team_a", "team_b")]
        events = [{"frame": item["frame"], "type": item["contact_type"], "source": "human_reviewed_contact_time"} for item in items]
        needed = {f for e in events for f in range(e["frame"]-14, e["frame"]+15)}
        frames = [a for a in payload["frames"] if a["f"] in needed]
        by_video[stem] = {}
        for method in METHODS:
            report = build_contact_attribution(events, frames, video["fps"], method=method)
            rows = [(project["labels"][item["id"]]["team"][-1], c) for item, c in zip(items, report["contacts"])]
            pooled[method].extend(rows)
            by_video[stem][method] = scores(rows)
        del payload, frames, content
    return {"schema": "contact_author_method_comparison_v1", "labels_path": str(labels_path.resolve()),
            "labels_sha256": sha(raw), "label_provenance": "human team labels; nine unsure contacts excluded",
            "sources": sources, "split_unit": "complete source video; no random clip split or training",
            "ball_input": "saved pipeline ball only; human contact xy is never supplied to methods",
            "timing_context": "audited human contact time; JIZ3 uses predicted source event time",
            "type_context": "human contact type; anatomical method is an oracle-type diagnostic and ineligible for automatic selection",
            "quality_scope": "development team-side comparison, not held-out or per-person/JIZ3 contact accuracy",
            "pooled": {method: scores(rows) for method, rows in pooled.items()}, "by_video": by_video}


def perturbation_stability(events, frames, fps, identity, reports):
    result = {}
    for method, base in reports.items():
        checks = []
        for offset in (-2, 2):
            shifted = [{**e, "frame": int(e["frame"])+offset} for e in events]
            current = build_contact_attribution(shifted, frames, fps, identity, method)
            comparisons = [(a, b) for a, b in zip(base["contacts"], current["contacts"]) if a.get("player_id")]
            checks.append({"frame_shift": offset, "assigned_baseline": len(comparisons),
                           "same_player_fraction": sum(a["player_id"] == b.get("player_id") for a, b in comparisons)/max(len(comparisons), 1)})
        result[method] = checks
    return result


def run(annotations_path, labels_path, output):
    if output.exists():
        raise FileExistsError("Choose a fresh versioned output directory")
    if not annotations_path.name.endswith("_annotations.json"):
        raise ValueError("Expected <stem>_annotations.json")
    stem = annotations_path.name.removesuffix("_annotations.json")
    stats_path = annotations_path.with_name(f"{stem}_stats.json")
    ann_raw, stats_raw = annotations_path.read_bytes(), stats_path.read_bytes()
    payload, stats = json.loads(ann_raw), json.loads(stats_raw)
    if payload["video"] != stats["video"] or payload["fps"] != stats["fps"]:
        raise ValueError("Mismatched source analysis pair")
    gold = evaluate_gold(labels_path)
    print(json.dumps({"development_team_comparison": gold["pooled"]}), flush=True)
    events, frames, fps = stats.get("all_events") or [], payload["frames"], payload["fps"]
    reports = {method: build_contact_attribution(events, frames, fps, payload.get("player_identity"), method)
               for method in METHODS}
    stability = perturbation_stability(events, frames, fps, payload.get("player_identity"), reports)
    # No person labels exist for JIZ3. Team metrics rank independent geometry/
    # motion hypotheses; stable two-sided identity is the tie-break only.
    eligible = ("body_baseline", "hands", "trajectory", "temporal")
    selected = max(eligible, key=lambda method: (
        round(gold["pooled"][method]["macro_f1_unknown_as_miss"], 6),
        round(min(gold["by_video"][v][method]["macro_f1_unknown_as_miss"] for v in gold["by_video"]), 6),
        sum(c["same_player_fraction"] for c in stability[method]), method == "temporal"))
    bundle = {"schema": "player_contact_experiment_v1", "source": source_fingerprint(stats_raw, ann_raw),
              "selected_method": selected, "methods": METHODS, "reports": reports,
              "selection": {"basis": "team-side development macro F1, worst video, then JIZ3 +/-2-frame author stability",
                            "eligible": list(eligible), "person_accuracy": None,
                            "caveat": "Best review candidate under these proxies; human validation of JIZ3 authors is still required"},
              "gold_comparison": gold, "jiz3_stability": stability,
              "provenance": {"annotations": str(annotations_path.resolve()), "stats": str(stats_path.resolve()),
                             "inference": False, "training": False, "source_events_changed": False}}
    output.mkdir(parents=True, exist_ok=False)
    # Byte-identical copies keep observed data and source-bound reviews honest.
    (output / annotations_path.name).write_bytes(ann_raw)
    (output / stats_path.name).write_bytes(stats_raw)
    (output / f"{stem}_player_contact_experiment.json").write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return {"output": str(output.resolve()), "selected_method": selected,
            "totals": {m: r["totals"] for m, r in reports.items()},
            "development_team_comparison": gold["pooled"], "jiz3_stability": stability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--team-labels", type=Path, default=ROOT / "data/contact_team_gold_v1/labels.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.annotations, args.team_labels, args.output), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
