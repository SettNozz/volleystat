"""Capture only missing appearance features and test richer contact-author methods."""
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
    cap = commands.add_parser("capture")
    cap.add_argument("--dataset",type=Path,required=True)
    cap.add_argument("--output",type=Path,required=True)
    cap.add_argument("--assets",type=Path,required=True)
    cap.add_argument("--model-cache-dir",type=Path,required=True)
    cap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    cap.add_argument("--require-model-cache",action="store_true")
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset",type=Path,required=True)
    prep.add_argument("--reid",type=Path,required=True)
    prep.add_argument("--output",type=Path,required=True)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--dataset",type=Path,required=True)
    ev.add_argument("--side-models",type=Path,required=True)
    ev.add_argument("--output",type=Path,required=True)
    ev.add_argument("--single-seed",action="store_true",help="Fast initial ablation; seed 42 only")
    ev.add_argument("--reuse-trained",type=Path,help="Compatible rich-model directory; never repeat fitting")
    args = parser.parse_args()
    if args.command=="capture":
        from src.tracking.contact_author_reid_benchmark import capture
        capture(args.dataset,args.output,args.assets,args.model_cache_dir,args.device,args.require_model_cache)
    elif args.command=="prepare":
        from src.tracking.contact_author_reid_benchmark import prepare
        prepare(args.dataset,args.reid,args.output)
    else:
        from src.tracking.contact_author_reid_evaluation import evaluate
        evaluate(args.dataset,args.side_models,args.output,(42,) if args.single_seed else (42,43,44),args.reuse_trained)


if __name__=="__main__": main()
