"""Score saved contact-side proposals against the human side labels.

Read-only. Reads finished analyses (``<stem>_stats.json`` and
``<stem>_annotations.json``) and ``data/contact_team_gold_v1/labels.json``;
runs no model and writes nothing unless ``--output`` is given.

Two views, because they answer different questions:

* ``pipeline``: the contacts the analysis itself produced (the spotter's, when
  it ran), matched to human contacts within ``--tolerance-frames``. This is
  what a user gets. The contact author's own side decision on the same
  contacts is scored next to it.
* ``human_contacts``: the same rule applied to the human contact times and
  types, with the analysis's saved ball sizes and geometry. This isolates the
  side rule from contact detection and typing errors.

    python scripts/evaluation/evaluate_contact_side.py --help
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tracking.contact_side import CORE_TYPE, assign_contact_sides  # noqa: E402

SIDE_OF_TEAM = {"team_a": "a", "team_b": "b"}


def find(directory: Path, stem: str, suffix: str) -> Path:
    for candidate in (directory / stem / f"{stem}{suffix}", directory / f"{stem}{suffix}"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{stem}{suffix} not found under {directory}")


def rate(hits) -> dict:
    hits = list(hits)
    return {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4) if hits else None}


def match(events, items, tolerance):
    """Greedy one-to-one nearest-frame matching of events to human contacts."""
    pairs = sorted(
        (abs(e["frame"] - it["frame"]), ei, ii)
        for ei, e in enumerate(events) for ii, it in enumerate(items)
        if abs(e["frame"] - it["frame"]) <= tolerance
    )
    used_e, used_i, out = set(), set(), []
    for _, ei, ii in pairs:
        if ei in used_e or ii in used_i:
            continue
        used_e.add(ei); used_i.add(ii); out.append((ei, ii))
    return out


def calibration_from(annotations) -> SimpleNamespace:
    """The saved geometry the side pass read. No homography is exported."""
    keys = ("net_floor_left", "net_floor_right", "net_tape_top", "net_tape_endpoint_y",
            "net_y", "net_height_m")
    return SimpleNamespace(homography=None, **{key: annotations.get(key) for key in keys})


def evaluate(analyses: Path, labels_path: Path, tolerance: int) -> dict:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    report = {"labels": str(labels_path), "analyses": str(analyses), "tolerance_frames": tolerance,
              "videos": {}}
    pooled = defaultdict(list)
    for stem in sorted(labels["videos"]):
        try:
            stats = json.loads(find(analyses, stem, "_stats.json").read_text(encoding="utf-8"))
            annotations = json.loads(find(analyses, stem, "_annotations.json").read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            report["videos"][stem] = {"status": "missing", "error": str(error)}
            continue
        items = [dict(item, side=SIDE_OF_TEAM.get(labels["labels"].get(item["id"], {}).get("team")))
                 for item in labels["items"] if item["video_id"] == stem]
        events = [dict(event, index=index) for index, event in enumerate(stats["all_events"])
                  if CORE_TYPE.get(event.get("type")) or event.get("core_type")]
        author = {row["event_index"]: row
                  for row in (stats.get("player_contacts") or {}).get("contacts", [])}
        rows = []
        for ei, ii in match(events, items, tolerance):
            event, item = events[ei], items[ii]
            if item["side"] is None:
                continue
            decision = author.get(event["index"], {})
            author_side = {"assigned": "b", "other_side": "a"}.get(decision.get("status"))
            rows.append({
                "video": stem, "type": item["contact_type"], "truth": item["side"],
                "side": event.get("side"), "score": event.get("side_score"),
                "source": event.get("side_source"), "confident": bool(event.get("side_confident")),
                "author": author_side,
                "near_score": decision.get("near_side_score"),
            })
        video = {
            "status": (stats.get("contact_side") or {}).get("status"),
            "net_ball_px": (stats.get("contact_side") or {}).get("net_ball_px"),
            "human_contacts": len(items), "pipeline_contacts": len(events), "matched_definite": len(rows),
            "side": rate(r["side"] == r["truth"] for r in rows if r["side"]),
            "side_coverage": round(sum(1 for r in rows if r["side"]) / len(rows), 4) if rows else None,
            "author": rate(r["author"] == r["truth"] for r in rows if r["author"]),
        }
        # Same rule on the human contact list: isolates it from detection errors.
        human_events = [{"frame": item["frame"], "type": item["contact_type"]} for item in items]
        human_report = assign_contact_sides(
            human_events, annotations["frames"], annotations["fps"], calibration_from(annotations))
        human_rows = [(event, item) for event, item in zip(human_events, items) if item["side"]]
        video["human_contacts_rule"] = {
            "status": human_report["status"],
            **rate(e["side"] == it["side"] for e, it in human_rows if e.get("side")),
            "coverage": round(sum(1 for e, _ in human_rows if e.get("side")) / len(human_rows), 4)
            if human_rows else None,
        }
        report["videos"][stem] = video
        pooled["rows"] += rows
        pooled["human"] += [(e, it) for e, it in human_rows]

    rows = pooled["rows"]
    with_side = [r for r in rows if r["side"]]
    both = [r for r in rows if r["side"] and r["author"]]
    summary = {
        "matched_definite": len(rows),
        "side": rate(r["side"] == r["truth"] for r in with_side),
        "side_coverage": round(len(with_side) / len(rows), 4) if rows else None,
        "side_by_type": {t: rate(r["side"] == r["truth"] for r in with_side if r["type"] == t)
                         for t in sorted({r["type"] for r in with_side})},
        "side_by_source": {s: rate(r["side"] == r["truth"] for r in with_side if r["source"] == s)
                           for s in sorted({r["source"] for r in with_side})},
        "side_by_margin": {
            f">={low}": {**rate(r["side"] == r["truth"] for r in with_side if abs(r["score"]) >= low),
                         "coverage": round(sum(1 for r in with_side if abs(r["score"]) >= low) / len(rows), 4)}
            for low in (0.0, 0.1, 0.2, 0.3, 0.5)
        } if rows else {},
        "author": rate(r["author"] == r["truth"] for r in rows if r["author"]),
        "author_by_type": {t: rate(r["author"] == r["truth"] for r in rows if r["author"] and r["type"] == t)
                           for t in sorted({r["type"] for r in rows if r["author"]})},
        "same_contacts": {
            "n": len(both),
            "side": rate(r["side"] == r["truth"] for r in both)["accuracy"],
            "author": rate(r["author"] == r["truth"] for r in both)["accuracy"],
            "agree": rate(r["side"] == r["author"] for r in both)["accuracy"],
            "both_right": sum(r["side"] == r["truth"] == r["author"] for r in both),
            "only_side_right": sum(r["side"] == r["truth"] != r["author"] for r in both),
            "only_author_right": sum(r["author"] == r["truth"] != r["side"] for r in both),
            "both_wrong": sum(r["side"] != r["truth"] != r["author"] for r in both),
        },
        "side_confident": {
            **rate(r["side"] == r["truth"] for r in with_side if r["confident"]),
            "coverage": round(sum(1 for r in with_side if r["confident"]) / len(rows), 4) if rows else None,
        },
        # The review hint the pipeline records: a confident size side against the author.
        "author_check": {
            "author_errors": sum(r["author"] != r["truth"] for r in both),
            "flagged": sum(r["side"] != r["author"] for r in both),
            "author_errors_flagged": sum(
                r["side"] != r["author"] and r["author"] != r["truth"] for r in both),
            "flagged_confident": sum(r["confident"] and r["side"] != r["author"] for r in both),
            "author_errors_flagged_confident": sum(
                r["confident"] and r["side"] != r["author"] and r["author"] != r["truth"] for r in both),
        },
        # What a consumer would get by letting a confident size side overrule the author.
        "author_overruled_when_margin": {
            f">={low}": rate(
                (r["side"] if abs(r["score"]) >= low else r["author"]) == r["truth"] for r in both)
            for low in (0.0, 0.1, 0.2, 0.3, 0.5)
        },
        "human_contacts_rule": {
            **rate(e["side"] == it["side"] for e, it in pooled["human"] if e.get("side")),
            "coverage": round(sum(1 for e, _ in pooled["human"] if e.get("side")) / len(pooled["human"]), 4)
            if pooled["human"] else None,
            "by_type": {t: rate(e["side"] == it["side"] for e, it in pooled["human"]
                                if e.get("side") and it["contact_type"] == t)
                        for t in sorted({it["contact_type"] for _, it in pooled["human"]})},
            "sources": dict(Counter(e.get("side_source") for e, _ in pooled["human"])),
        },
    }
    report["summary"] = summary
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyses-dir", required=True, type=Path,
                        help="Directory holding <stem>/<stem>_stats.json (or the files directly)")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/contact_team_gold_v1/labels.json")
    parser.add_argument("--tolerance-frames", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON report; refuses to overwrite")
    args = parser.parse_args()
    report = evaluate(args.analyses_dir, args.labels, args.tolerance_frames)
    text = json.dumps(report, indent=2)
    if args.output:
        if args.output.exists():
            parser.error(f"{args.output} exists; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
