"""Read jersey numbers on saved near-side poses and write a versioned read cache.

Uses saved poses only (no detection or pose inference). Frames are decoded
sequentially from the original video. Output rows are keyed like identity
observations (``frame:pose_index``) and carry the torso box, so identity
experiments replay them without decoding again.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.jersey_ocr import DEFAULT_ASSETS, JerseyNumberReader, file_sha256, torso_box


def frame_span(value):
    start, end = map(int, value.split(':'))
    if not 0 <= start <= end:
        raise argparse.ArgumentTypeError('Window must be START:END')
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file')
    parser.add_argument('--side', default='b', help="Court side to read (a/b); 'all' reads both")
    parser.add_argument('--stride', type=int, default=3, help='Read every Nth saved frame')
    parser.add_argument('--window', action='append', type=frame_span, help='Inclusive START:END; default all frames')
    parser.add_argument('--assets', type=Path, default=DEFAULT_ASSETS)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--samples', type=Path, help='Optional new directory for a few crop images per reading')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh versioned file')
    source = args.annotations.resolve()
    payload = json.loads(source.read_bytes())
    video = Path(payload['original_video_path'])
    sides = None if args.side == 'all' else {args.side, f'team_{args.side}'}
    wanted = {}
    for ann in payload['frames']:
        f = int(ann['f'])
        if f % args.stride or (args.window and not any(lo <= f <= hi for lo, hi in args.window)):
            continue
        boxes = {}
        for index, pose in enumerate(ann.get('poses') or []):
            if sides is not None and pose.get('team') not in sides:
                continue
            box = torso_box(pose)
            if box is not None:
                boxes[index] = box
        if boxes:
            wanted[f] = boxes
    reader = JerseyNumberReader(args.assets, batch_size=args.batch_size)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    reads, pending, started = {}, [], time.monotonic()
    if args.samples:
        args.samples.mkdir(parents=True)
    saved = Counter()

    def flush():
        results = reader.read([crop for _, _, crop in pending])
        for (key, box, crop), result in zip(pending, results):
            reads[key] = {'box': [round(v, 1) for v in box], **result}
            label = result['number'] or 'none'
            if args.samples and saved[label] < 6:
                saved[label] += 1
                cv2.imwrite(str(args.samples/f'{label}_{key.replace(":", "_")}_{result["prob"]:.2f}.jpg'), crop)
        pending.clear()

    last = max(wanted, default=-1)
    frame_index = 0
    while frame_index <= last:
        ok = cap.grab()
        if not ok:
            break
        if frame_index in wanted:
            ok, image = cap.retrieve()
            if not ok:
                break
            for index, box in wanted[frame_index].items():
                crop = reader.crop(image, box)
                if crop is not None:
                    pending.append((f'{frame_index}:{index}', box, crop))
            if len(pending) >= args.batch_size:
                flush()
        frame_index += 1
    cap.release()
    if pending:
        flush()
    by_player = defaultdict(Counter)
    for key, row in reads.items():
        f, index = map(int, key.split(':'))
        pose = next(a for a in payload['frames'] if int(a['f']) == f)['poses'][index]
        if row['number'] and row['prob'] >= .5:
            by_player[pose.get('player_id') or 'P?'][row['number']] += 1
    report = {'schema': 'jersey_number_reads_v1', 'annotations': str(source),
              'annotations_sha256': file_sha256(source), 'video': str(video),
              'stride': args.stride, 'windows': args.window, 'side': args.side,
              'reader': reader.provenance, 'elapsed_seconds': time.monotonic() - started,
              'crops': len(reads), 'numbers': sum(1 for r in reads.values() if r['number']),
              'label_source': 'model', 'verified': False, 'reads': reads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, separators=(',', ':'))
    print(f"{report['crops']} crops, {report['numbers']} number readings, {report['elapsed_seconds']:.0f} s")
    for pid, counts in sorted(by_player.items()):
        print(pid, counts.most_common(5))


if __name__ == '__main__':
    main()
