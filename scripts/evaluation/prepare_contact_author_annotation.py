"""Prepare a new, empty human contact-author project from frozen gold inputs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from action_recognition.annotator.contact_author_dataset import prepare_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True,help="Human event-training manifest")
    parser.add_argument("--analyses-dir",type=Path,required=True,help="Existing pose-enabled *_annotations.json directory")
    parser.add_argument("--timing-review",type=Path,required=True,help="contact_spatial_review_v1 reviews.json")
    parser.add_argument("--output",type=Path,required=True,help="Fresh project directory; existing outputs refused")
    args = parser.parse_args()
    prepare_project(args.manifest,args.analyses_dir,args.timing_review,args.output)
    print(f"Open: python -m action_recognition.annotator.contact_author_annotator --project \"{args.output.resolve()}\"")


if __name__=="__main__": main()
