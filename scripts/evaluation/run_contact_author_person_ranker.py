"""Roster identity seeds and source-held-out evaluation of the person-level author ranker."""
import argparse
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seeds = commands.add_parser("seeds", help="Write player_identity_roster_v2 files from the human photo rosters")
    seeds.add_argument("--benchmark", type=Path, required=True)
    seeds.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate", help="Leave-one-source-out person ranking; fresh output")
    run.add_argument("--benchmark", type=Path, required=True)
    run.add_argument("--identity-dir", type=Path, required=True,
                     help="Directory with <video>/<video>_annotations.json from apply_player_identity.py")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seeds", type=int, nargs="+", default=[42])
    infer = commands.add_parser("predict", help="Fit on all labelled sources; predict an unlabelled source's events")
    infer.add_argument("--benchmark", type=Path, required=True)
    infer.add_argument("--identity-dir", type=Path, required=True)
    infer.add_argument("--inputs", type=Path, required=True,
                       help="Unlabelled author inputs from apply_contact_author_models.py prepare")
    infer.add_argument("--comparison", type=Path, help="Optional earlier predictions.json shown side by side")
    infer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from src.tracking.contact_author_person_evaluation import evaluate, predict_unlabelled, write_seeds
    if args.command == "seeds":
        write_seeds(args.benchmark, args.output)
    elif args.command == "predict":
        predict_unlabelled(args.benchmark, args.identity_dir, args.inputs, args.output, args.comparison)
    else:
        evaluate(args.benchmark, args.identity_dir, args.output, tuple(args.seeds))


if __name__ == "__main__":
    main()
