"""Score the output of ``evaluate_ball_configs.py``.

Window A: recall and wrong picks per minute after the production trajectory
solver (``BallTrajectoryDP`` + gap interpolation), against the human clicks,
at 60 px (one ball diameter at 4K, the usual tolerance) and 30 px. Raw
"ball among candidates" recall and extra candidates per frame are shown too.
Contacts B: raw candidates only (no continuous frames for the solver).
Paired gains/losses are against the first --baseline config.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'evaluation'))
from score_ball_candidates import load_gt, run_dp  # noqa: E402

FPS = 29.97


def centre(d):
    b = d['box']
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def as_points(dets):
    return [{'x': centre(d)[0], 'y': centre(d)[1], 'conf': d['conf'],
             'size': max(d['box'][2] - d['box'][0], d['box'][3] - d['box'][1])} for d in dets]


def hit(points, truth, tol):
    return any(math.dist((p['x'], p['y']), truth) <= tol for p in points)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dirs', nargs='+', type=Path)
    ap.add_argument('--data-root', type=Path, default=Path.cwd())
    ap.add_argument('--baseline', default='base1280')
    args = ap.parse_args()

    gt, _ = load_gt(args.data_root / 'results/ball_eval/6_tiPcRYGcBtc_12090_12987_gt.json')
    visible = {f: xy for f, xy in gt.items() if xy is not None}
    speed, per_a, per_b, rows = {}, {}, {}, []
    for directory in args.dirs:
        meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
        speed.update(meta['speed_ms_per_frame'])
        contacts = json.loads((directory / 'B_contacts.json').read_text(encoding='utf-8'))
        for name in meta['configs']:
            dump = json.loads((directory / f'A_{name}_candidates.json').read_text(encoding='utf-8'))
            raw = {f: as_points(d) for f, d in zip(dump['frame_indices'], dump['runs'][name]['detections'])}
            chosen = run_dp(raw, px_scale=3.0)
            a = {tol: [hit(chosen.get(f, []), xy, tol) for f, xy in sorted(visible.items())] for tol in (60, 30)}
            wrong = sum(1 for f, xy in visible.items() if chosen.get(f) and not hit(chosen[f], xy, 60))
            raw_hits = [hit(raw.get(f, []), xy, 30) for f, xy in sorted(visible.items())]
            extra = sum(sum(1 for p in raw.get(f, []) if math.dist((p['x'], p['y']), xy) > 60)
                        for f, xy in visible.items()) / len(visible)
            b_hits = [hit(as_points(c['dets'][name]), c['gt'], 30) for c in contacts]
            b_extra = sum(sum(1 for p in as_points(c['dets'][name]) if math.dist((p['x'], p['y']), c['gt']) > 60)
                          for c in contacts) / len(contacts)
            per_a[name], per_b[name] = a[60], b_hits
            minutes = len(visible) * 3 / FPS / 60
            rows.append((name, sum(a[60]) / len(visible), sum(a[30]) / len(visible), wrong / minutes,
                         sum(raw_hits) / len(visible), extra, sum(b_hits) / len(b_hits), b_extra, len(b_hits)))
    print(f'A: {len(visible)} human clicks (window, after DP); B: {rows[0][-1]} contacts (raw)\n')
    print(f"{'config':>16} {'A dp@60':>8} {'A dp@30':>8} {'A wrong/min':>11} {'A raw@30':>9} {'A extra/fr':>10} "
          f"{'B raw@30':>9} {'B extra/fr':>10} {'ms/frame':>9}")
    for name, a60, a30, wpm, araw, aext, braw, bext, _ in rows:
        print(f'{name:>16} {a60:8.3f} {a30:8.3f} {wpm:11.1f} {araw:9.3f} {aext:10.2f} {braw:9.3f} {bext:10.2f} '
              f'{speed.get(name, float("nan")):9.1f}')
    if args.baseline in per_a:
        print(f'\npaired vs {args.baseline}: A after DP @60 (+gained/-lost), B raw @30 (+/-)')
        for name in per_a:
            if name == args.baseline:
                continue
            ga = sum(b and not a for a, b in zip(per_a[args.baseline], per_a[name]))
            la = sum(a and not b for a, b in zip(per_a[args.baseline], per_a[name]))
            gb = sum(b and not a for a, b in zip(per_b[args.baseline], per_b[name]))
            lb = sum(a and not b for a, b in zip(per_b[args.baseline], per_b[name]))
            print(f'  {name:>16}: A +{ga}/-{la}   B +{gb}/-{lb}')


if __name__ == '__main__':
    main()
