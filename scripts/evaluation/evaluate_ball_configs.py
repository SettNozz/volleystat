"""Compare ball-detector configurations on held-out human ground truth.

Each ``--config name=weights:imgsz[:COLSxROWS]`` runs the production
``BallDetectorVideo`` (same confidence and NMS thresholds) on:

* window A - every frame of ``6_tiPcRYGcBtc`` 12090-12990 (4K active play).
  Candidates are written in ``dump_ball_candidates.py`` format, so
  ``score_ball_candidates.py --dp`` scores them after the production
  trajectory solver against the 270 human clicks of that window;
* contacts B - single frames at audited contacts of the three gold videos
  (``contact_spatial_gold_v1``, status visible): raw candidates only, since
  the solver needs continuous frames.

Neither source is in ``yolo_fullframe_merged`` (the detector's training set).
Speed is measured with batches of 8 frames (tiles: one call per frame).
Nothing is written outside ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

WINDOW = ('data/labeling_batch/6_tiPcRYGcBtc.mp4', 12090, 12990)
GT_A = 'results/ball_eval/6_tiPcRYGcBtc_12090_12987_gt.json'
GT_B = 'data/contact_spatial_gold_v1/reviews.json'
GOLD_VIDEO = 'data/labeling_intake/2026-07-31_youtube_4k/{stem}.webm'


def parse_config(text: str):
    name, spec = text.split('=', 1)
    parts = spec.rsplit(':', 2)
    if len(parts) == 3 and 'x' in parts[2]:
        weights, imgsz, grid = parts
        grid = tuple(int(v) for v in grid.split('x'))
    else:
        weights, imgsz = spec.rsplit(':', 1)
        grid = None
    return name, Path(weights), int(imgsz), grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', action='append', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, default=Path.cwd(),
                    help='Directory holding data/ and results/ (default: working directory)')
    args = ap.parse_args()

    import cv2
    import numpy as np
    import torch
    from detection.detect_ball_on_video import BallDetectorVideo

    args.output.mkdir(parents=True, exist_ok=False)
    data = args.data_root
    configs = [parse_config(c) for c in args.config]
    detectors = {name: BallDetectorVideo(str(w), imgsz=s, tile_grid=g) for name, w, s, g in configs}

    def detect(det, image):
        return [{'box': [float(v) for v in d[:4]], 'conf': float(d[4])} for d in det.detect_ball(image)]

    # ---- window A: every frame
    video, start, end = WINDOW
    cap = cv2.VideoCapture(str(data / video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    runs = {name: [] for name in detectors}
    frames, timing = [], []
    for index in range(start, end + 1):
        ok, image = cap.read()
        if not ok:
            raise SystemExit(f'decode failed at {index}')
        if len(timing) < 16 and index % 50 == 0:
            timing.append(image)
        frames.append(index)
        for name, det in detectors.items():
            runs[name].append(detect(det, image))
    for name in detectors:
        dump = {'video': video, 'model': name, 'resolution': [3840, 2160], 'start': start, 'stride': 1,
                'conf_floor': 0.1, 'iou': 0.4, 'frame_indices': frames,
                'runs': {name: {'detections': runs[name]}}}
        (args.output / f'A_{name}_candidates.json').write_text(json.dumps(dump), encoding='utf-8')
    print('window A done', len(frames), flush=True)

    # ---- contacts B: single frames
    reviews = json.loads((data / GT_B).read_text(encoding='utf-8'))['reviews']
    wanted = defaultdict(dict)
    for key, r in reviews.items():
        if r.get('status') == 'visible' and r.get('spatial_valid') and r.get('xy'):
            wanted[key.split(':')[0]][int(r['frame'])] = r['xy']
    contacts = []
    for stem, points in sorted(wanted.items()):
        cap = cv2.VideoCapture(str(data / GOLD_VIDEO.format(stem=stem)))
        position = 0
        for frame in sorted(points):
            while position < frame:
                cap.grab(); position += 1
            ok, image = cap.read(); position += 1
            if not ok:
                raise SystemExit(f'decode failed {stem} {frame}')
            contacts.append({'video': stem, 'frame': frame, 'gt': points[frame],
                             'dets': {name: detect(det, image) for name, det in detectors.items()}})
        print('contacts', stem, len(points), flush=True)
    (args.output / 'B_contacts.json').write_text(json.dumps(contacts), encoding='utf-8')

    # ---- speed
    speed = {}
    for name, det in detectors.items():
        for image in timing[:2]:
            det.detect_ball(image)
        torch.cuda.synchronize(); t = time.perf_counter()
        if det.tile_grid:
            for image in timing:
                det.detect_ball(image)
        else:
            for i in range(0, len(timing), 8):
                det.detect_ball_batch(timing[i:i + 8])
        torch.cuda.synchronize()
        speed[name] = round((time.perf_counter() - t) / len(timing) * 1000, 1)
    meta = {'configs': {n: {'weights': str(w), 'imgsz': s, 'tile_grid': g} for n, w, s, g in configs},
            'speed_ms_per_frame': speed, 'window': WINDOW, 'gt_a': GT_A, 'gt_b': GT_B,
            'note': 'speed measured while other GPU work may run; compare within one run'}
    (args.output / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print(json.dumps(speed), flush=True)


if __name__ == '__main__':
    main()
