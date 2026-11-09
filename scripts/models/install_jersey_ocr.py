"""Install the pinned offline TrOCR jersey-number reader (one explicit download)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.jersey_ocr import DEFAULT_ASSETS, REPOSITORY, REVISION, install_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_ASSETS, help='New destination directory')
    parser.add_argument('--from-dir', type=Path,
                        help='Copy already downloaded files of the pinned revision; the weights checksum is verified')
    args = parser.parse_args()
    install_assets(args.output, args.from_dir)
    print(f'Installed {REPOSITORY}@{REVISION[:7]} in {args.output.resolve()}')


if __name__ == '__main__':
    main()
