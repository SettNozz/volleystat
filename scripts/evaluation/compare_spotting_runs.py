"""Put several finished dense-spotting runs side by side.

Every lever tried on this task -- poses, trajectory quality, seed ensembling --
produces another run directory, and the only honest way to read them is
together: the end-to-end score, the detection/typing split underneath it, and
the per-fold spread that says whether a difference is worth believing from one
seed.

Reads finished runs only. Trains nothing and modifies nothing.

    python scripts/evaluation/compare_spotting_runs.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIEWS = ("raw_six_class", "core_four_class")
CONDITIONS = ("peaks", "constrained", "fsm")


def log(message: str) -> None:
    print(message, flush=True)


def load_run(directory: Path) -> dict[str, Any]:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    split_path = directory / "detection_typing.json"
    split = (
        json.loads(split_path.read_text(encoding="utf-8"))
        if split_path.exists() else None
    )
    return {"name": directory.name, "report": report, "split": split}


def end_to_end(run: dict[str, Any], view: str, condition: str,
               tolerance: int) -> float | None:
    node = run["report"]["mean_macro_f1"].get(view, {}).get(condition)
    if not node:
        return None
    return node.get(f"at_{tolerance}")


def per_fold(run: dict[str, Any], view: str, condition: str,
             tolerance: int) -> dict[str, float]:
    rows = {}
    for fold in run["report"]["folds"]:
        node = fold.get(view, {}).get(condition)
        if not isinstance(node, dict):
            continue
        scores = node.get(f"at_{tolerance}")
        if isinstance(scores, dict) and "macro_f1" in scores:
            rows[fold["holdout"]] = float(scores["macro_f1"])
    return rows


def seeds_of(run: dict[str, Any]) -> list[int]:
    arguments = run["report"]["preflight"].get("arguments", {})
    seeds = arguments.get("seeds")
    if seeds:
        return [int(value) for value in seeds]
    seed = arguments.get("seed")
    return [int(seed)] if seed is not None else []


def argument_of(run: dict[str, Any], name: str) -> Any:
    return run["report"]["preflight"].get("arguments", {}).get(name)


def grouped_comparison(
    runs: Sequence[dict[str, Any]], argument: str, view: str, condition: str,
    tolerance: int,
) -> dict[str, Any]:
    """Split runs by one runner argument and pair them on seed.

    A single-seed difference means little on this task, so the paired
    differences are printed next to the spread between seeds inside each
    group: an effect smaller than that spread has not been separated from it.
    """
    groups: dict[str, dict[str, float]] = {}
    ensembles: dict[str, float] = {}
    for run in runs:
        raw = argument_of(run, argument)
        # A run made before the argument existed records nothing, which is not
        # the same claim as having been run with it off.
        value = "absent" if raw is None else str(raw)
        score = end_to_end(run, view, condition, tolerance)
        if score is None:
            continue
        seeds = seeds_of(run)
        key = ",".join(str(seed) for seed in seeds)
        if len(seeds) == 1:
            groups.setdefault(value, {})[key] = score
        else:
            ensembles[f"{value} / seeds {key}"] = score

    log(f"\n# Split by {argument}, {view} {condition} at +/-{tolerance}\n")
    seeds_seen = sorted(
        {seed for scores in groups.values() for seed in scores},
        key=lambda text: int(text) if text.isdigit() else text,
    )
    rows = []
    for value, scores in sorted(groups.items()):
        cells = [value]
        present = [scores[seed] for seed in seeds_seen if seed in scores]
        for seed in seeds_seen:
            cells.append(f"{scores[seed]:.4f}" if seed in scores else "-")
        cells.append(f"{np.mean(present):.4f}" if present else "-")
        cells.append(
            f"{max(present) - min(present):.4f}" if len(present) > 1 else "-"
        )
        rows.append(cells)
    log(table(rows, [argument, *[f"seed {s}" for s in seeds_seen],
                     "mean", "seed spread"]))

    paired: dict[str, float] = {}
    values = sorted(groups)
    if len(values) == 2:
        left, right = values
        differences = []
        rows = []
        for seed in seeds_seen:
            if seed in groups[left] and seed in groups[right]:
                difference = groups[right][seed] - groups[left][seed]
                differences.append(difference)
                paired[seed] = difference
                rows.append([seed, f"{groups[left][seed]:.4f}",
                             f"{groups[right][seed]:.4f}",
                             f"{difference:+.4f}"])
        if rows:
            log(f"\n# Paired per seed: {right} minus {left}\n")
            log(table(rows, ["seed", left, right, "difference"]))
            spreads = [
                max(scores.values()) - min(scores.values())
                for scores in groups.values() if len(scores) > 1
            ]
            largest_spread = max(spreads) if spreads else 0.0
            mean_difference = float(np.mean(differences))
            verdict = (
                "smaller than the spread between seeds, so it is not separated "
                "from seed noise"
                if abs(mean_difference) <= largest_spread else
                "larger than the spread between seeds"
            )
            log(
                f"\nmean difference {mean_difference:+.4f}, largest seed "
                f"spread within a group {largest_spread:.4f} -- {verdict}."
            )
            log(
                "Signs: " + ", ".join(
                    f"seed {seed} {'+' if value > 0 else '-'}"
                    for seed, value in paired.items()
                ) + ". Three paired seeds cannot establish significance; this "
                "is a direction and a magnitude, not a test."
            )
    if ensembles:
        log("\n# Seed ensembles, for the same split\n")
        log(table(
            [[name, f"{score:.4f}"] for name, score in sorted(ensembles.items())],
            ["group / seeds", view],
        ))
    return {
        "argument": argument, "view": view, "condition": condition,
        "per_seed": groups, "ensembles": ensembles, "paired_differences": paired,
    }


def table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    widths = [
        max(len(str(headers[column])), *(len(str(row[column])) for row in rows))
        if rows else len(str(headers[column]))
        for column in range(len(headers))
    ]
    lines = [
        "  ".join(str(headers[c]).ljust(widths[c]) for c in range(len(headers))),
        "  ".join("-" * widths[c] for c in range(len(headers))),
    ]
    lines.extend(
        "  ".join(str(row[c]).ljust(widths[c]) for c in range(len(headers)))
        for row in rows
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, nargs="+", required=True,
        help="finished run directories, in the order to show them",
    )
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument(
        "--group-by", default=None, metavar="ARGUMENT",
        help="a runner argument recorded in preflight (for example "
             "zero_pose_channels) to split the runs by, pairing them on seed "
             "so an effect can be read against the spread between seeds",
    )
    parser.add_argument(
        "--view", default="core_four_class", choices=VIEWS,
        help="which label view the grouped comparison uses",
    )
    parser.add_argument(
        "--condition", default="constrained", choices=CONDITIONS,
        help="which decoding condition the grouped comparison uses",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional JSON file for the assembled comparison",
    )
    args = parser.parse_args(argv)

    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing {args.output}")
    missing = [path for path in args.runs if not (path / "report.json").exists()]
    if missing:
        parser.error(f"no report.json in {missing}")

    runs = [load_run(path) for path in args.runs]
    tolerance = args.tolerance

    log(f"# End-to-end macro F1 at +/-{tolerance} frames\n")
    headers = ["run", "seeds", "det", "raw peaks", "raw constr", "core peaks",
               "core constr", "core fsm"]
    rows = []
    for run in runs:
        def cell(view: str, condition: str) -> str:
            value = end_to_end(run, view, condition, tolerance)
            return f"{value:.4f}" if value is not None else "-"
        determinism = run["report"]["preflight"].get("determinism")
        rows.append([
            run["name"],
            ",".join(str(seed) for seed in seeds_of(run)) or "-",
            "yes" if (determinism or {}).get("requested") else "no",
            cell("raw_six_class", "peaks"),
            cell("raw_six_class", "constrained"),
            cell("core_four_class", "peaks"),
            cell("core_four_class", "constrained"),
            cell("core_four_class", "fsm"),
        ])
    log(table(rows, headers))

    log(f"\n# Core four-class per fold, peaks -> constrained\n")
    folds = sorted({
        fold["holdout"] for run in runs for fold in run["report"]["folds"]
    })
    rows = []
    for run in runs:
        peaks = per_fold(run, "core_four_class", "peaks", tolerance)
        constrained = per_fold(run, "core_four_class", "constrained", tolerance)
        cells = [run["name"]]
        for holdout in folds:
            if holdout in peaks and holdout in constrained:
                cells.append(f"{peaks[holdout]:.4f} -> {constrained[holdout]:.4f}")
            else:
                cells.append("-")
        spread = list(constrained.values())
        cells.append(
            f"{max(spread) - min(spread):.4f}" if len(spread) > 1 else "-"
        )
        rows.append(cells)
    log(table(rows, ["run", *folds, "spread"]))

    split_runs = [run for run in runs if run["split"]]
    if split_runs:
        log(f"\n# Detection and typing, where a split was scored\n")
        rows = []
        for run in split_runs:
            summary = run["split"]["summary"]
            detection = summary["detection_f1"]["constrained"][f"at_{tolerance}"]
            in_play = (
                summary.get("detection_f1_in_play", {})
                .get("constrained", {}).get(f"at_{tolerance}")
            )
            typing = summary["typing"]["constrained"]
            rows.append([
                run["name"],
                f"{detection:.4f}",
                f"{in_play:.4f}" if in_play is not None else "-",
                f"{typing['core_accuracy']:.4f}",
                f"{typing['core_macro_f1']:.4f}",
                f"{summary['typed_core_macro_f1_at_5']:.4f}",
            ])
        log(table(rows, [
            "run", "detection", "detection in-play", "typing acc",
            "typing macroF1", "end-to-end",
        ]))
    else:
        log(
            "\nNo run carries detection_typing.json. Run "
            "scripts/evaluation/score_contact_detection.py on a run to add it."
        )

    seed_runs = [run for run in runs if len(seeds_of(run)) > 1]
    if seed_runs:
        log(f"\n# Per-seed selection scores inside each ensemble\n")
        rows = []
        for run in seed_runs:
            for fold in run["report"]["folds"]:
                for entry in fold.get("per_seed", []):
                    rows.append([
                        run["name"], fold["holdout"], entry["seed"],
                        entry["best_epoch"],
                        f"{entry['selection_macro_f1']:.4f}",
                    ])
        log(table(rows, ["run", "holdout", "seed", "best epoch", "raw macroF1"]))
        log(
            "\nThese are each seed's own raw six-class peak score on the "
            "held-out video, which is what the best epoch was chosen by. They "
            "are not the ensemble's score: the ensemble decodes the averaged "
            "probabilities."
        )

    grouped: dict[str, Any] | None = None
    if args.group_by:
        grouped = grouped_comparison(
            runs, args.group_by, args.view, args.condition, tolerance,
        )

    payload = {
        "schema": "spotting_run_comparison_v1",
        "tolerance": tolerance,
        "runs": [
            {
                "name": run["name"],
                "path": str(path.resolve()),
                "seeds": seeds_of(run),
                "end_to_end": {
                    view: {
                        condition: end_to_end(run, view, condition, tolerance)
                        for condition in CONDITIONS
                    }
                    for view in VIEWS
                },
                "per_fold_core_constrained": per_fold(
                    run, "core_four_class", "constrained", tolerance,
                ),
                "split": run["split"]["summary"] if run["split"] else None,
            }
            for run, path in zip(runs, args.runs)
        ],
        "grouped": grouped,
        "caveats": [
            "Every run chose its best epoch and the control's threshold on the "
            "held-out video, so all of these are model-selection estimates.",
            "Runs built on different analyses differ in ball trajectory as "
            "well as in whatever lever was being tested.",
            "A run made without determinism does not reproduce its own score "
            "on a rerun; measured spread at a fixed seed was about 0.01 macro "
            "F1, so check preflight.determinism before comparing.",
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        log(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
