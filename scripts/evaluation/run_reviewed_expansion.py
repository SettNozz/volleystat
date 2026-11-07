"""Does adding reviewed queues improve the contact spotter? Held-out arms.

The production spotter trained on three fully annotated gold videos from one
camera placement. A reviewed queue (``contact_review_v1``) adds videos from a
new placement with human-verified contacts but no phase track. This measures
whether training on them helps, with every arm trained by the exact recipe that
ships (``scripts/models/train_contact_spotter.py``: fixed epoch budget, seeds
averaged, deterministic) and scored through ``ContactSpotter.run``, the path
``analyze_video.py --contact-spotter`` uses. No epoch is selected on held-out
data.

Two questions, two sets of arms:

* **gold folds** - hold out one gold video, train on the other two with and
  without every reviewed video. Gold labels are dense, so precision is
  measured honestly here: this is the guard against contact-only targets
  teaching the model to fire everywhere.
* **new-camera folds** - hold out one reviewed video, train on all gold with
  and without the other reviewed videos. A reviewed video is labelled only
  where the queue offered a candidate, so precision cannot be measured there.
  What can: recall over the human contacts, and the type at the human frame.

Recall on a held-out reviewed video favours the baseline: the queue was built
from the baseline's own peaks, so every human contact is a frame the baseline
already scored above 0.2. A gain there is conservative.

    python scripts/evaluation/run_reviewed_expansion.py --list-arms
    python scripts/evaluation/run_reviewed_expansion.py --arm new_base --output-root results/x
    python scripts/evaluation/run_reviewed_expansion.py --report --output-root results/x

Nothing here writes into ``models/`` or a dataset. Arm checkpoints are
held-out-fold models, stored beside their metrics for audit only.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting import reviewed_contacts as rc
from action_recognition.rally_spotting import spotter_dataset as sd
from action_recognition.rally_spotting.contact_spotter import (
    SCHEMA as CHECKPOINT_SCHEMA, ContactSpotter, save_checkpoint,
)
from action_recognition.rally_spotting.metrics import _prf, match_events
from scripts.evaluation.run_dense_contact_spotting import score
from scripts.evaluation.score_contact_detection import (
    detection_scores, typing_scores,
)
from scripts.models import train_contact_spotter as trainer

SCHEMA = "reviewed_expansion_experiment_v1"
GOLD = trainer.STEMS
DEFAULT_REVIEW = ROOT / "data/contact_review_20261029_v1/review.json"
TOLERANCE = 5
# Half-width around a human contact frame over which the type is read off the
# per-frame probabilities, matching the training segment radius.
TYPE_RADIUS = 2
# An event this far from every queued candidate fell where nobody looked.
UNREVIEWED_RADIUS = 10

# Decided before any arm ran. Both conditions must hold.
PROMOTION_RULE = {
    "new_camera": (
        "mean over the new-camera folds of both frame-level raw typing "
        "accuracy and type-agnostic recall at +/-5 is higher with the "
        "reviewed videos than without"
    ),
    "gold": (
        "mean over the gold folds of raw six-class macro F1 at +/-5 drops by "
        "no more than 0.02, and detection precision at +/-5 by no more than "
        "0.02 (0.02 is the measured seed spread of this recipe)"
    ),
    "gold_tolerance": 0.02,
}


def log(message: str) -> None:
    print(f"[reviewed-expansion] {message}", flush=True)


def arms(reviewed: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Every arm by name: what it trains on and what it is scored on."""
    plan: dict[str, dict[str, Any]] = {}
    for held in GOLD:
        rest = [stem for stem in GOLD if stem != held]
        plan[f"gold_{held}_base"] = {
            "question": "gold", "condition": "base", "fold": held,
            "train_gold": rest, "train_reviewed": [], "evaluate": [held],
        }
        plan[f"gold_{held}_plus"] = {
            "question": "gold", "condition": "plus", "fold": held,
            "train_gold": rest, "train_reviewed": list(reviewed),
            "evaluate": [held],
        }
    # One baseline serves every new-camera fold: it never sees a reviewed
    # video, so holding one out changes nothing for it.
    plan["new_base"] = {
        "question": "new_camera", "condition": "base", "fold": None,
        "train_gold": list(GOLD), "train_reviewed": [],
        "evaluate": list(reviewed),
    }
    for held in reviewed:
        plan[f"new_{held}_plus"] = {
            "question": "new_camera", "condition": "plus", "fold": held,
            "train_gold": list(GOLD),
            "train_reviewed": [stem for stem in reviewed if stem != held],
            "evaluate": [held],
        }
    return plan


def type_at_frames(
    probabilities: np.ndarray, truth: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The type the model gives each human contact, at the human's frame.

    Independent of detection and decoding: it asks only "if this is a
    contact, which one", which is what the review corrected 86 times.
    """
    contact = probabilities[:, dc.CONTACT_OFFSET:]
    raw_hits = raw_total = core_hits = core_total = 0
    confusion: Counter[tuple[str, str]] = Counter()
    mass = []
    for event in truth:
        frame = int(event["frame"])
        left = max(0, frame - TYPE_RADIUS)
        right = min(contact.shape[0], frame + TYPE_RADIUS + 1)
        window = contact[left:right]
        mass.append(float(window.sum(axis=1).max()))
        predicted = dc.CONTACT_TYPES[int(window.sum(axis=0).argmax())]
        if event.get("type") is not None:
            raw_total += 1
            raw_hits += int(predicted == event["type"])
            confusion[(event["type"], predicted)] += 1
        truth_core = event.get("core_type")
        if truth_core is not None:
            core_total += 1
            core_hits += int(dc.to_core(predicted) == truth_core)
    return {
        "raw_accuracy": raw_hits / raw_total if raw_total else 0.0,
        "raw_contacts": raw_total,
        "core_accuracy": core_hits / core_total if core_total else 0.0,
        "core_contacts": core_total,
        "median_peak_contact_mass": float(np.median(mass)) if mass else 0.0,
        "confusion_truth_to_predicted": {
            f"{t}->{p}": n for (t, p), n in confusion.most_common() if t != p
        },
    }


def typed_recall(
    events: Sequence[Mapping[str, Any]], truth: Sequence[Mapping[str, Any]],
    project: bool,
) -> dict[str, Any]:
    """Per-class recall: a human contact found with the right type."""
    names = dc.CORE_TYPES if project else dc.CONTACT_TYPES
    rows = {}
    for name in names:
        def typed(rows_: Sequence[Mapping[str, Any]], is_truth: bool) -> list[int]:
            out = []
            for row in rows_:
                value = (
                    (row.get("core_type") if is_truth else dc.to_core(row["type"]))
                    if project else row.get("type")
                )
                if value == name:
                    out.append(int(row["frame"]))
            return out
        target = typed(truth, True)
        if not target:
            continue
        tp, _, fn = match_events(typed(events, False), target, TOLERANCE)
        rows[name] = {"recall": tp / (tp + fn), "found": tp, "contacts": tp + fn}
    return {
        "per_class": rows,
        "macro_recall": float(np.mean([row["recall"] for row in rows.values()]))
        if rows else 0.0,
    }


def evaluate_gold(
    spotter: ContactSpotter, source: Mapping[str, Any],
    annotations: Mapping[str, Any],
) -> dict[str, Any]:
    spotted = spotter.run(annotations)
    events = spotted.events
    truth = source["truth"]
    frames = sorted(int(event["frame"]) for event in events)
    truth_frames = sorted(int(event["frame"]) for event in truth)
    raw = score(events, truth, dc.CONTACT_TYPES, project=False)
    core = score(events, truth, dc.CORE_TYPES, project=True)
    return {
        "decode_status": spotted.decode_status,
        "events": len(events),
        "truth_events": len(truth),
        "raw_macro_f1": raw[f"at_{TOLERANCE}"]["macro_f1"],
        "core_macro_f1": core[f"at_{TOLERANCE}"]["macro_f1"],
        "raw_per_class": raw[f"at_{TOLERANCE}"]["per_class"],
        "detection": detection_scores(frames, truth_frames)[f"at_{TOLERANCE}"],
        "typing": typing_scores(events, truth, TOLERANCE),
        "type_at_frames": type_at_frames(
            spotter.probabilities(source["features"]), truth,
        ),
        "per_type_events": dict(Counter(event["type"] for event in events)),
    }, events


def evaluate_reviewed(
    spotter: ContactSpotter, source: Mapping[str, Any],
    annotations: Mapping[str, Any], candidate_frames: Sequence[int],
) -> dict[str, Any]:
    spotted = spotter.run(annotations)
    events = spotted.events
    truth = source["truth"]
    frames = sorted(int(event["frame"]) for event in events)
    truth_frames = sorted(int(event["frame"]) for event in truth)
    tp, unmatched, fn = match_events(frames, truth_frames, TOLERANCE)
    queued = np.array(sorted(set(candidate_frames) | set(truth_frames)), np.int64)
    unreviewed = [
        frame for frame in frames
        if not queued.size or int(np.abs(queued - frame).min()) > UNREVIEWED_RADIUS
    ]
    return {
        "decode_status": spotted.decode_status,
        "events": len(events),
        "human_contacts": len(truth),
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "found": tp,
        "missed": fn,
        "events_not_matched_to_a_human_contact": unmatched,
        "events_where_nobody_looked": len(unreviewed),
        "precision_note": (
            "not measurable: the reviewer saw only queued candidates, so an "
            "unmatched event may be a contact the queue never offered"
        ),
        "typed_recall_raw": typed_recall(events, truth, project=False),
        "typed_recall_core": typed_recall(events, truth, project=True),
        # A frame where merged candidates disagreed on the core type is a
        # contact for detection but has no type to score.
        "typing": typing_scores(
            events, [row for row in truth if row.get("core_type")], TOLERANCE,
        ),
        "type_at_frames": type_at_frames(
            spotter.probabilities(source["features"]), truth,
        ),
        "per_type_events": dict(Counter(event["type"] for event in events)),
    }, events


def run_arm(name: str, spec: Mapping[str, Any], args: Any, output: Path) -> None:
    arm_dir = output / name
    metrics_path = arm_dir / "metrics.json"
    if metrics_path.exists():
        raise SystemExit(f"refusing to overwrite finished arm {metrics_path}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    determinism = (
        trainer.enforce_determinism() if args.deterministic else {"requested": False}
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    args.reviewed = [args.review] if spec["train_reviewed"] else None
    sources, gold_review, queues = trainer.load_sources(
        args, spec["train_gold"], spec["train_reviewed"] or None,
    )
    members, mean, deviation, weights = trainer.train_members(sources, args, device)
    constraints = trainer.count_constraints(sources, args)
    provenance = {
        "schema": CHECKPOINT_SCHEMA,
        "experiment": SCHEMA,
        "arm": name,
        "spec": dict(spec),
        "note": (
            "held-out-fold model from an experiment arm; for audit, not for "
            "production use"
        ),
        "trained_on": [source["stem"] for source in sources],
        "reviewed_queues": queues,
        "gold_review_sha256": gold_review["review_sha256"],
        "seeds": list(args.seeds),
        "epochs": int(args.epochs),
        "determinism": determinism,
    }
    checkpoint = arm_dir / "checkpoint.pt"
    save_checkpoint(
        checkpoint, members=[m["state_dict"] for m in members],
        config=members[0]["config"], mean=mean, deviation=deviation,
        constraints=constraints, provenance=provenance, window=args.window,
        stride=args.inference_stride, transition_weight=args.transition_weight,
    )
    spotter = ContactSpotter.load(checkpoint, device=device)

    review = rc.load_review(args.review)
    raw_review = json.loads(Path(args.review).read_text(encoding="utf-8"))
    results = {}
    for stem in spec["evaluate"]:
        if stem in GOLD:
            source = sd.load_source(
                stem, args.annotations_dir, args.gold_dir,
                gold_review["contacts"].get(stem, []),
                gold_review["masked"].get(stem, []),
                args.human_source, args.contact_radius, args.guard,
            )
            annotations = json.loads(
                Path(source["annotation_path"]).read_text(encoding="utf-8")
            )
            result, events = evaluate_gold(spotter, source, annotations)
        else:
            source = rc.load_reviewed_source(stem, review, args.contact_radius, args.guard)
            annotations = json.loads(
                Path(source["annotation_path"]).read_text(encoding="utf-8")
            )
            candidates = [
                int(row["frame"]) for row in raw_review["contacts"]
                if row["video_id"] == stem
            ]
            result, events = evaluate_reviewed(spotter, source, annotations, candidates)
        result["annotation_sha256"] = source["annotation_sha256"]
        results[stem] = result
        (arm_dir / f"events_{stem}.json").write_text(
            json.dumps({"events": events, "truth": source["truth"]}, indent=1),
            encoding="utf-8",
        )
        log(f"{name} on {stem}: " + json.dumps({
            key: value for key, value in result.items()
            if isinstance(value, (int, float, str))
        }))

    metrics = {
        "schema": SCHEMA,
        "arm": name,
        "spec": dict(spec),
        "results": results,
        "class_weights": {
            label: float(weights[index]) for index, label in enumerate(dc.LABEL_TYPES)
        },
        "training_history": [
            {"seed": m["seed"], "history": m["history"]} for m in members
        ],
        "train_sources": [
            {
                "stem": source["stem"], "sparse": bool(source.get("sparse")),
                "verified_contacts": source["verified_contacts"],
                "annotation_sha256": source["annotation_sha256"],
            }
            for source in sources
        ],
        "provenance": provenance,
        "seconds": round(time.time() - started, 1),
    }
    metrics_path.write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    log(f"{name} done in {metrics['seconds']:.0f}s")


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def report(output: Path, reviewed: Sequence[str]) -> dict[str, Any]:
    plan = arms(reviewed)
    finished = {}
    for name in plan:
        path = output / name / "metrics.json"
        if path.exists():
            finished[name] = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(set(plan) - set(finished))
    if missing:
        raise SystemExit(f"arms not finished: {missing}")

    gold_rows = []
    for held in GOLD:
        base = finished[f"gold_{held}_base"]["results"][held]
        plus = finished[f"gold_{held}_plus"]["results"][held]
        gold_rows.append({
            "fold": held,
            **{
                f"{key}_{condition}": value
                for condition, result in (("base", base), ("plus", plus))
                for key, value in (
                    ("raw_macro_f1", result["raw_macro_f1"]),
                    ("core_macro_f1", result["core_macro_f1"]),
                    ("detection_precision", result["detection"]["precision"]),
                    ("detection_recall", result["detection"]["recall"]),
                    ("detection_f1", result["detection"]["f1"]),
                    ("typing_raw_accuracy", result["typing"]["raw_accuracy"]),
                    ("type_at_frames_raw", result["type_at_frames"]["raw_accuracy"]),
                    ("events", result["events"]),
                )
            },
            "truth_events": base["truth_events"],
        })
    new_rows = []
    base_new = finished["new_base"]["results"]
    for held in reviewed:
        base = base_new[held]
        plus = finished[f"new_{held}_plus"]["results"][held]
        new_rows.append({
            "fold": held,
            **{
                f"{key}_{condition}": value
                for condition, result in (("base", base), ("plus", plus))
                for key, value in (
                    ("recall", result["recall"]),
                    ("typed_recall_raw_macro", result["typed_recall_raw"]["macro_recall"]),
                    ("typed_recall_core_macro", result["typed_recall_core"]["macro_recall"]),
                    ("serve_recall", result["typed_recall_raw"]["per_class"]
                     .get("serve", {}).get("recall", float("nan"))),
                    ("typing_raw_accuracy", result["typing"]["raw_accuracy"]),
                    ("type_at_frames_raw", result["type_at_frames"]["raw_accuracy"]),
                    ("type_at_frames_core", result["type_at_frames"]["core_accuracy"]),
                    ("events", result["events"]),
                    ("events_where_nobody_looked", result["events_where_nobody_looked"]),
                )
            },
            "human_contacts": base["human_contacts"],
        })

    def mean_of(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return _mean([row[key] for row in rows])

    summary = {
        "gold": {
            key: {
                "base": mean_of(gold_rows, f"{key}_base"),
                "plus": mean_of(gold_rows, f"{key}_plus"),
            }
            for key in ("raw_macro_f1", "core_macro_f1", "detection_precision",
                        "detection_recall", "detection_f1", "typing_raw_accuracy",
                        "type_at_frames_raw")
        },
        "new_camera": {
            key: {
                "base": mean_of(new_rows, f"{key}_base"),
                "plus": mean_of(new_rows, f"{key}_plus"),
            }
            for key in ("recall", "typed_recall_raw_macro", "typed_recall_core_macro",
                        "serve_recall", "typing_raw_accuracy", "type_at_frames_raw",
                        "type_at_frames_core")
        },
    }
    tolerance = PROMOTION_RULE["gold_tolerance"]
    new_ok = (
        summary["new_camera"]["type_at_frames_raw"]["plus"]
        > summary["new_camera"]["type_at_frames_raw"]["base"]
        and summary["new_camera"]["recall"]["plus"]
        > summary["new_camera"]["recall"]["base"]
    )
    gold_ok = (
        summary["gold"]["raw_macro_f1"]["plus"]
        >= summary["gold"]["raw_macro_f1"]["base"] - tolerance
        and summary["gold"]["detection_precision"]["plus"]
        >= summary["gold"]["detection_precision"]["base"] - tolerance
    )

    reproduction = None
    production = ROOT / "models/contact_spotter/spotter_20261028_v1.pt"
    if production.exists():
        shipped = torch.load(str(production), map_location="cpu", weights_only=False)
        rebuilt = torch.load(
            str(output / "new_base" / "checkpoint.pt"), map_location="cpu",
            weights_only=False,
        )
        largest = max(
            float((a[key].double() - b[key].double()).abs().max())
            for a, b in zip(shipped["members"], rebuilt["members"])
            for key in a
        )
        reproduction = {
            "checkpoint": str(production),
            "max_abs_weight_difference": largest,
            "note": (
                "new_base trains on the same three gold videos with the same "
                "recipe and seeds as the shipped checkpoint; zero means the "
                "refactored trainer rebuilt it exactly"
            ),
        }

    result = {
        "schema": SCHEMA,
        "promotion_rule": PROMOTION_RULE,
        "decision": {
            "new_camera_condition": bool(new_ok),
            "gold_condition": bool(gold_ok),
            "promote": bool(new_ok and gold_ok),
        },
        "summary": summary,
        "gold_folds": gold_rows,
        "new_camera_folds": new_rows,
        "reproduction_of_shipped_checkpoint": reproduction,
        "arm_seconds": {name: finished[name]["seconds"] for name in plan},
    }
    (output / "report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = trainer.build_parser(__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-root", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-arms", action="store_true")
    group.add_argument("--arm")
    group.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    reviewed = sorted(rc.load_review(args.review)["contacts"])
    plan = arms(reviewed)
    if args.list_arms:
        for name, spec in plan.items():
            print(name, json.dumps(spec))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required")
    if args.report:
        result = report(args.output_root, reviewed)
        print(json.dumps({"decision": result["decision"], "summary": result["summary"],
                          "reproduction": result["reproduction_of_shipped_checkpoint"]},
                         indent=1))
        return 0
    if args.arm not in plan:
        parser.error(f"unknown arm {args.arm}; see --list-arms")
    run_arm(args.arm, plan[args.arm], args, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
