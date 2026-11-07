"""Prepare frozen human author features; compare algorithms on whole-video folds."""
import argparse
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable,"2")
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    prep = commands.add_parser("prepare",help="Read human labels, cached poses/ball and original colour pixels; no inference")
    prep.add_argument("--project",type=Path,required=True)
    prep.add_argument("--output",type=Path,required=True)
    central = commands.add_parser("central",help="Closest-observed actor timing from frozen poses; no feature/model refresh")
    central.add_argument("--dataset",type=Path,required=True)
    central.add_argument("--output",type=Path,required=True,help="Fresh actor-context JSON path")
    evaluate = commands.add_parser("evaluate",help="Whole-video held-out and nested algorithm comparisons")
    evaluate.add_argument("--dataset",type=Path,required=True)
    evaluate.add_argument("--output",type=Path,required=True)
    evaluate.add_argument("--reuse-models",type=Path,help="Compatible prior fold_models directory; no training repeated")
    evaluate.add_argument("--actor-context",type=Path,help="Optional source-bound central-time actor ablation")
    args = parser.parse_args()
    if args.command=="prepare":
        from src.tracking.contact_author_benchmark import prepare
        prepare(args.project,args.output)
    elif args.command=="central":
        from src.tracking.contact_author_benchmark import prepare_central
        prepare_central(args.dataset,args.output)
    else:
        from src.tracking.contact_author_evaluation import evaluate
        evaluate(args.dataset,args.output,args.reuse_models,args.actor_context)


if __name__=="__main__": main()
