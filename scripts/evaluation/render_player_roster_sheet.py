"""Contact sheet of an automatic roster: crops of every roster ID spread over the video.

One row per roster player, plus one row for people who were seen on the same
side but left out of the roster. Reads saved boxes and decodes only the frames
it shows. A review aid: nothing here is a human label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TILE = (150, 300)  # width, height


def tile(image, box, label):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    height, width = image.shape[:2]
    crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    canvas = np.full((TILE[1], TILE[0], 3), 32, np.uint8)
    if crop.size:
        scale = min(TILE[0]/crop.shape[1], (TILE[1]-22)/crop.shape[0])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1]*scale)), max(1, int(crop.shape[0]*scale))))
        left = (TILE[0]-resized.shape[1])//2
        canvas[22:22+resized.shape[0], left:left+resized.shape[1]] = resized
    cv2.putText(canvas, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh .jpg path')
    parser.add_argument('--views', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh file')
    data = json.loads(args.annotations.read_text(encoding='utf-8'))
    identity = data.get('player_identity') or {}
    side = identity.get('side', 'b')
    rows = {}
    for row in data['frames']:
        for pose in row.get('poses') or []:
            pid = pose.get('player_id')
            reason = (pose.get('identity') or {}).get('reason')
            if pid and pose.get('team') in (side, 'team_'+side) and reason in ('associated', 'seed', 'roster_anchor'):
                rows.setdefault(pid, []).append((row['f'], pose['bb']))
    wanted = {}
    for pid, items in sorted(rows.items()):
        picks = np.unique(np.linspace(0, len(items)-1, args.views).round().astype(int))
        wanted[pid] = [items[i] for i in picks]
    by_frame = {row['f']: row for row in data['frames']}
    others = []
    for person in (identity.get('roster_discovery') or {}).get('other_people', [])[:args.views]:
        frame, index = map(int, person['example_key'].split(':'))
        others.append((frame, by_frame[frame]['poses'][index]['bb'], person['presence_frames']))
    needed = sorted({f for items in wanted.values() for f, _ in items} | {f for f, _, _ in others})
    from src.models.inference_cache import video_frames
    images = dict(video_frames(data['original_video_path'], needed, (data['frame_width'], data['frame_height'])))
    fps = data['fps']
    lines = []
    for pid, items in wanted.items():
        tiles = [tile(images[f], box, f'{pid} {f/fps:.0f}s') for f, box in items]
        tiles += [np.full((TILE[1], TILE[0], 3), 32, np.uint8)]*(args.views-len(tiles))
        lines.append(np.hstack(tiles))
    if others:
        tiles = [tile(images[f], box, f'other {frames/fps:.0f}s seen') for f, box, frames in others]
        tiles += [np.full((TILE[1], TILE[0], 3), 32, np.uint8)]*(args.views-len(tiles))
        lines.append(np.hstack(tiles))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.vstack(lines)):
        raise OSError('Cannot write sheet')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
