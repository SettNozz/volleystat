"""Audit saved roster invariants and render spread-out occlusion review strips.

Coverage and review candidates are diagnostics, never identity accuracy labels.
No source annotations, labels, raw tracks or model files are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.player_reid_encoder import file_sha256
from src.tracking.player_identity_workflow import observations_from_annotations
from scripts.evaluation.run_player_identity_pilot import write_json


def audit(annotations_path: Path, output: Path, samples: int = 12):
    if output.exists():
        raise ValueError('Output exists; choose a fresh review directory')
    data = json.loads(annotations_path.read_text(encoding='utf-8'))
    metadata = data['player_identity']
    if metadata.get('status') != 'completed':
        raise ValueError('Identity pass is not complete')
    frames = data['frames']
    observations = observations_from_annotations(frames, metadata['side'])
    by_frame = {f['f']:f for f in frames}
    fps = float(data['fps'])
    counts, reasons, roster_counts = Counter(), Counter(), Counter()
    minutes = {}
    for ann, observed in zip(frames, observations):
        ids = []
        minute = minutes.setdefault(int(ann['f']/fps/60.), Counter())
        roster_counts[len(observed['players'])] += 1
        for obs in observed['players']:
            p = ann['poses'][obs['pose_index']]
            pid = p.get('player_id')
            counts['observations'] += 1
            counts['assigned'] += pid is not None
            minute['observations'] += 1
            minute['assigned'] += pid is not None
            reasons[p['identity']['reason']] += 1
            if pid is not None:
                ids.append(pid)
                counts[pid] += 1
                minute[pid] += 1
        if len(ids) != len(set(ids)):
            raise AssertionError(f'Duplicate roster ID in frame {ann["f"]}')
    if counts['observations'] != metadata['observations'] or counts['assigned'] != metadata['assigned']:
        raise AssertionError('Exported identity counts differ from actual saved poses')
    video = Path(data['original_video_path'])
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Source video unavailable')
    output.mkdir(parents=True)
    # One largest-overlap observation per equal time interval. These are review
    # candidates, including unknowns, with no claim about actual identity errors.
    choices = []
    for interval in np.array_split(np.arange(len(observations)), samples):
        if not len(interval):
            continue
        selected = max((observations[i] for i in interval), key=lambda f:
                       max((p['occlusion'] for p in f['players']), default=0.))
        if selected['players']:
            choices.append(selected)

    def render(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            raise ValueError(f'Cannot decode source frame {f}')
        scale_x, scale_y = 768/im.shape[1], 432/im.shape[0]
        im = cv2.resize(im,(768,432))
        for p in by_frame[f].get('poses') or []:
            if 'identity' not in p:
                continue
            x1,y1,x2,y2 = [round(v*s) for v,s in zip(p['bb'],[scale_x,scale_y,scale_x,scale_y])]
            pid = p.get('player_id')
            colour = (80,230,80) if pid is not None else (40,160,255)
            cv2.rectangle(im,(x1,y1),(x2,y2),colour,2)
            label_at = (x1,max(y1-6,48))
            cv2.putText(im,pid or 'P?',label_at,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4)
            cv2.putText(im,pid or 'P?',label_at,cv2.FONT_HERSHEY_SIMPLEX,.6,colour,1)
        cv2.rectangle(im,(0,0),(768,35),(20,20,20),-1)
        cv2.putText(im,f'f{f} / {f/fps:.2f}s - MODEL, UNVERIFIED',(12,25),
                    cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        return im

    source_frames = np.array(list(by_frame))
    candidates = []
    try:
        seed_frame = metadata['seed']['frame']
        cv2.imwrite(str(output/'initial_roster.jpg'),render(seed_frame))
        for n, choice in enumerate(choices):
            requested = [choice['f']-round(fps),choice['f'],choice['f']+round(fps)]
            actual = [int(source_frames[np.abs(source_frames-f).argmin()]) for f in requested]
            strip = np.concatenate([render(f) for f in actual],axis=1)
            name = f'occlusion_{n+1:02d}_f{choice["f"]}.jpg'
            cv2.imwrite(str(output/name),strip)
            candidates.append({'frame':choice['f'],'seconds':choice['f']/fps,
                               'max_bbox_overlap':max(p['occlusion'] for p in choice['players']),
                               'review_frames':actual,'image':name,'human_status':'pending'})
    finally:
        cap.release()
    report = {'schema':'player_identity_audit_v1','annotations_sha256':file_sha256(annotations_path),
              'audit_source_sha256':file_sha256(Path(__file__)),
              'source_video':str(video.resolve()),'source_video_sha256':file_sha256(video),
              'method':metadata['method'],'source_frames':len(frames),
              'first_frame':frames[0]['f'],'last_frame':frames[-1]['f'],
              'counts':dict(counts),'coverage':counts['assigned']/max(counts['observations'],1),
              'reasons':dict(reasons),'roster_detection_counts':dict(roster_counts),
              'coverage_by_minute':{m:{**dict(c),'coverage':c['assigned']/max(c['observations'],1)}
                                    for m,c in minutes.items()},
              'duplicate_identity_frames':0,'accuracy':None,'verified':False,
              'scope':metadata['scope'],'seed':metadata['seed'],
              'review_selection':'largest bbox overlap per equal time interval; no human error labels',
              'candidates':candidates}
    write_json(output/'audit.json',report)
    print(json.dumps({k:report[k] for k in ('source_frames','counts','coverage','duplicate_identity_frames','accuracy')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=12)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    audit(args.annotations,args.output,args.samples)


if __name__ == '__main__':
    main()
