"""Prepare or predict contact authors from a saved analysis; never rerun detectors."""
import argparse
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name,"2")
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prep=commands.add_parser("prepare",help="Unlabelled features from saved ball/poses and model contact events")
    prep.add_argument("--video",type=Path,required=True)
    prep.add_argument("--annotations",type=Path,required=True)
    prep.add_argument("--stats",type=Path,required=True)
    prep.add_argument("--output",type=Path,required=True)
    predict=commands.add_parser("predict",help="Reuse three frozen class_fusion fold members; no training")
    predict.add_argument("--dataset",type=Path,required=True)
    predict.add_argument("--training-dataset",type=Path,required=True)
    predict.add_argument("--evaluation",type=Path,required=True)
    predict.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="prepare":
        from src.tracking.contact_author_inference import prepare_input
        prepare_input(args.video,args.annotations,args.stats,args.output)
    else:
        from src.tracking.contact_author_inference import predict as run
        run(args.dataset,args.training_dataset,args.evaluation,args.output)


if __name__=="__main__":main()
