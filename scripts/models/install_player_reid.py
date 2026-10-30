"""Install the pinned, offline OSNet bundle without replacing existing models."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from src.models.player_reid_encoder import DEFAULT_ASSETS, install_assets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir',type=Path,help='Reuse a verified existing bundle; otherwise download official pinned assets')
    parser.add_argument('--output',type=Path,default=DEFAULT_ASSETS,help='New destination directory')
    args=parser.parse_args()
    install_assets(args.output,args.from_dir)
    print(f'Installed pinned OSNet in {args.output.resolve()}')


if __name__=='__main__':
    main()
