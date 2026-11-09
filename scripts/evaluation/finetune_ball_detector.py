"""Fine-tune the production ball detector at another input size (experiment).

Starts from the registry's ``ball_detector`` weights and repeats their own
training arguments (data, optimizer, learning rate, augmentation, seed) with
only ``imgsz``, ``epochs`` and ``batch`` changed, so a run at 1920 and a
control at 1280 differ in the input size alone. Output goes to a fresh
directory under ``results/``; nothing in ``models/`` or the registry changes.

    python scripts/evaluation/finetune_ball_detector.py \
        --output results/ball_finetune_20261108_v1 --name ft1920 --imgsz 1920 --batch 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_registry import resolve_model, sha256_of  # noqa: E402

# Settings that describe where/how a run is stored, not what it learns.
_NOT_COPIED = {'model', 'resume', 'project', 'name', 'exist_ok', 'epochs', 'imgsz', 'batch',
               'device', 'mode', 'task', 'save_dir', 'cfg', 'tracker', 'time', 'source'}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True, help='Fresh experiment directory under results/')
    ap.add_argument('--name', required=True)
    ap.add_argument('--imgsz', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--base', default=None, help='Starting weights (default: registry ball_detector)')
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    base = Path(args.base) if args.base else resolve_model('ball_detector', use_registry=False).path
    run_dir = args.output / args.name
    if run_dir.exists():
        raise SystemExit(f'{run_dir} exists; use a new --name or --output')
    checkpoint = torch.load(base, map_location='cpu', weights_only=False)
    inherited = {k: v for k, v in checkpoint['train_args'].items() if k not in _NOT_COPIED}
    data = Path(inherited['data'])
    if not data.is_absolute():
        # Data is gitignored: a worktree has none, so prefer the working directory.
        data = next((root / data for root in (Path.cwd(), ROOT) if (root / data).exists()), ROOT / data)
        inherited['data'] = str(data)
    settings = dict(inherited, model=str(base), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    workers=args.workers, project=str(args.output.resolve()), name=args.name, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'base_weights': str(base), 'base_sha256': sha256_of(base),
                  'base_train_args': checkpoint['train_args'], 'changed': {'imgsz': args.imgsz,
                  'epochs': args.epochs, 'batch': args.batch, 'workers': args.workers},
                  'settings': settings, 'label_source': 'human YOLO boxes (yolo_fullframe_merged)'}
    (args.output / f'{args.name}_provenance.json').write_text(json.dumps(provenance, indent=2, default=str),
                                                               encoding='utf-8')
    print(json.dumps({k: settings[k] for k in ('model', 'data', 'imgsz', 'epochs', 'batch', 'optimizer',
                                               'lr0', 'scale', 'mosaic', 'seed')}, indent=1))
    YOLO(str(base)).train(**{k: v for k, v in settings.items() if k != 'model'})


if __name__ == '__main__':
    main()
