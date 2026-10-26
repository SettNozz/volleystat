"""Diagnostic only: distinguish saved-pose gaps from raw YOLO-pose gaps.

Read a completed SAM experiment, rerun stock YOLO-pose on five exact source
frames at 640 and 1280; export independent review artifacts, never new labels.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.detection.person_masks import assign,box_iou,decode_mask
from scripts.evaluation.sam3_people_masks import sha256,save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--pose-model',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    if not args.pose_model.is_file():
        raise ValueError('Local YOLO-pose checkpoint required; no download')
    manifest = json.loads((args.experiment/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('A completed SAM experiment is required')
    indices = np.linspace(manifest['start_frame'],manifest['end_frame_exclusive']-1,5).round().astype(int).tolist()
    selected = {}
    with (args.experiment/'masks.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['f'] in indices:
                selected[row['f']] = row
    if set(selected) != set(indices):
        raise ValueError('Missing selected mask records')
    annotation_path = Path(manifest['annotations'])
    annotations = json.loads(annotation_path.read_text(encoding='utf-8-sig'))
    saved = {row['f']:row for row in annotations['frames'] if row['f'] in indices}
    if sha256(annotation_path) != manifest['hashes']['annotations']:
        raise ValueError('Saved poses changed since SAM experiment')
    destination = args.experiment/'pose_diagnostic'
    destination.mkdir(exist_ok=False)
    from ultralytics import YOLO
    model = YOLO(str(args.pose_model))
    cap = cv2.VideoCapture(manifest['source_video'])
    rows = []
    try:
        for index in range(max(indices)+1):
            ok,frame = cap.read()
            if not ok:
                raise RuntimeError(f'Video decode failed at {index}')
            if index not in selected:
                continue
            row = selected[index]
            people = row['persons']
            h,w = people[0]['mask']['size'] if people else (720,1280)
            sx,sy = w/annotations['frame_width'],h/annotations['frame_height']
            image = cv2.resize(frame,(w,h))
            panel = image.copy()
            for person in people:
                mask = decode_mask(person['mask'])
                color = np.array([0,165,255]) if person['review_required'] else np.array([60,230,70])
                panel[mask] = (panel[mask]*.55+color*.45).astype(np.uint8)
            views = [panel]
            result = dict(f=index,t=index/annotations['fps'],saved_pose_count=len(saved[index].get('poses',[])),
                          sam_count=len(people),sam_court_candidate_count=sum(p['review_required'] for p in people))
            for size in (640,1280):
                raw = model.predict(frame,imgsz=size,conf=.25,iou=.7,device=args.device,half=False,verbose=False)[0]
                boxes = raw.boxes.xyxy.cpu().numpy()
                confidences = raw.boxes.conf.cpu().numpy()
                scores = np.array([[box_iou(p['bbox_native'],b) for b in boxes] for p in people]).reshape(len(people),len(boxes))
                matching = {i:j for i,j in assign(scores,.15)}
                candidates = [i for i,p in enumerate(people) if p['review_required']]
                result[str(size)] = dict(raw_pose_count=len(boxes),
                    candidate_matched_raw=sum(i in matching for i in candidates),
                    candidate_unmatched_raw=sum(i not in matching for i in candidates),
                    sam_to_raw_match={str(i):j for i,j in matching.items()},
                    boxes=boxes.tolist(),scores=confidences.tolist(),
                    keypoints=raw.keypoints.data.cpu().numpy().tolist() if raw.keypoints is not None else [])
                view = image.copy()
                for bb,score in zip(boxes,confidences):
                    a,b,c,d = [round(float(v)*s) for v,s in zip(bb,(sx,sy,sx,sy))]
                    cv2.rectangle(view,(a,b),(c,d),(60,230,70),1)
                    cv2.putText(view,f'{score:.2f}',(a,max(35,b-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,(60,230,70),1)
                views.append(view)
            for view,title in zip(views,('SAM3: orange = absent saved pose','Raw YOLO-pose 640 (CPU float32)','Raw YOLO-pose 1280 (CPU float32)')):
                cv2.rectangle(view,(0,0),(w,28),(20,20,20),-1)
                cv2.putText(view,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imwrite(str(destination/f'frame_{index}.jpg'),np.concatenate(views,axis=1))
            rows.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ('640','1280')}),flush=True)
    finally:
        cap.release()
    report = dict(samples=rows,device=args.device,precision='float32',confidence=.25,iou=.7,
        pose_model=str(args.pose_model.resolve()),pose_model_sha256=sha256(args.pose_model),
        experiment_manifest_sha256=sha256(args.experiment/'manifest.json'),
        caveat='Per-frame rerun, not exact original half-precision ByteTrack history. Raw matches suggest downstream filtering/tracking gaps; no raw match is a review candidate, not ground truth.')
    save_json(destination/'report.json',report)


if __name__ == '__main__':
    main()
