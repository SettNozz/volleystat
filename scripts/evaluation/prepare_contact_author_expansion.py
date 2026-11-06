"""Prepare new-source contact-author annotation from existing spotter analyses."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from action_recognition.annotator.contact_author_expansion import prepare_expansion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats",type=Path,nargs="+",required=True,help="Saved *_stats.json files; paired annotations must exist")
    parser.add_argument("--exclude-event-manifest",type=Path,required=True,help="Active event-training manifest whose sources are excluded")
    parser.add_argument("--exclude-author-snapshot",type=Path,required=True,help="Completed human author export whose sources/hashes are excluded")
    parser.add_argument("--output",type=Path,required=True,help="Fresh annotation project; old labels/queues cannot be overwritten")
    args=parser.parse_args();prepare_expansion(args.stats,args.exclude_event_manifest,args.exclude_author_snapshot,args.output)
    print(f'Open: python -m action_recognition.annotator.contact_author_annotator --project "{args.output.resolve()}" --port 5070')


if __name__=="__main__":main()
