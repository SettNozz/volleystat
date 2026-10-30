"""Compare SAM3 people masks with saved final poses on an exact video interval.

Run in .sam3_env; production 3.11 dependencies and source results stay unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.detection.person_masks import FEATURE_NAMES, SCHEMA, PersonMaskAnalyzer


def sha256(path):
    digest = hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def plan_frames(fps,start,duration):
    if not all(math.isfinite(v) for v in (fps,start,duration)) or fps <= 0 or start < 0 or duration <= 0:
        raise ValueError('Require positive FPS/duration and nonnegative start')
    first,last = math.ceil(start*fps),math.ceil((start+duration)*fps)
    if last <= first:
        raise ValueError('Interval contains no frames')
    return first,last


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def overlay(frame,annotation,record,analyzer,cv2):
    left,right = frame.copy(),frame.copy()
    h,w = frame.shape[:2]
    sx,sy = w/analyzer.width,h/analyzer.height
    def box(panel,bb,color,label):
        a,b,c,d = [int(v*s) for v,s in zip(bb,(sx,sy,sx,sy))]
        cv2.rectangle(panel,(a,b),(c,d),color,1)
        cv2.putText(panel,label,(a,max(38,b-4)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
    for pose in annotation.get('poses',[]):
        box(left,pose['bb'],(60,230,70),f"YOLO {pose.get('id')}")
    for person,exported in zip(analyzer.previous,record['persons']):
        status = person['court_status']
        color = (0,165,255) if status == 'inside_candidate' else ((60,230,70) if status == 'existing_court_member' else (130,130,130))
        mask = person['mask']
        right[mask] = (right[mask]*.55+np.array(color)*.45).astype(np.uint8)
        box(right,person['bbox_native'],color,f"S{person['instance_id']} Y{person['yolo_track_id']}")
    for panel,title in ((left,'Saved YOLO-pose'),(right,'SAM3 person masks; orange = review candidate')):
        cv2.rectangle(panel,(0,0),(w,28),(20,20,20),-1)
        cv2.putText(panel,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        ball = annotation.get('ball')
        if ball is not None and not annotation.get('ball_out_of_frame',False):
            cv2.circle(panel,(round(ball[0]*sx),round(ball[1]*sy)),4,(40,40,255),-1)
        cv2.putText(panel,f"f={record['f']} t={record['t']:.3f}s",(8,h-12),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    return np.concatenate((left,right),axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--annotations',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--start',type=float,default=15.)
    parser.add_argument('--duration',type=float,default=30.)
    parser.add_argument('--max-width',type=int,default=1280)
    parser.add_argument('--confidence',type=float,default=.4)
    parser.add_argument('--precision',choices=['float32','bfloat16','float16'],default='bfloat16')
    parser.add_argument('--check',action='store_true',help='Validate inputs/GPU without loading weights or creating output')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; choose a fresh versioned directory')
    if args.max_width < 64 or not 0 <= args.confidence <= 1:
        raise ValueError('Invalid max-width or confidence')
    data = json.loads(args.annotations.read_text(encoding='utf-8-sig'))
    fps = float(data['fps'])
    first,last = plan_frames(fps,args.start,args.duration)
    frames = {}
    for row in data['frames']:
        idx = row['f']
        if type(idx) is not int or idx < 0 or idx in frames:
            raise ValueError('Duplicate/invalid source frame index')
        frames[idx] = row
    if Path(data['video']).stem != args.video.stem or any(i not in frames for i in range(first,last)):
        raise ValueError('Video/annotation identity or interval coverage mismatch')
    width,height = int(data['frame_width']),int(data['frame_height'])
    if not data.get('court_corners'):
        raise ValueError('Court polygon required for candidate filtering')
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size < 1000000:
        raise ValueError('Missing/empty SAM3 checkpoint')
    import cv2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this experiment')
    if args.precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support selected bfloat16 precision')
    cap = cv2.VideoCapture(str(args.video))
    writer = None
    owned = False
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open source video')
        actual = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (width,height) or not math.isclose(cap.get(cv2.CAP_PROP_FPS),fps,rel_tol=1e-5):
            raise ValueError('Video geometry/FPS differs from annotations')
        grid_w = min(args.max_width,width)
        grid_w -= grid_w % 2
        grid_h = round(height*grid_w/width)
        grid_h -= grid_h % 2
        hashes = {name:sha256(path) for name,path in (
            ('video',args.video),('annotations',args.annotations),('checkpoint',args.checkpoint),
            ('worker',Path(__file__)),('features',ROOT/'src/detection/person_masks.py'))}
        manifest = dict(status='preflight',schema=SCHEMA,feature_names=FEATURE_NAMES,
                        source_video=str(args.video.resolve()),annotations=str(args.annotations.resolve()),
                        checkpoint=str(args.checkpoint.resolve()),hashes=hashes,
                        start_frame=first,end_frame_exclusive=last,fps=fps,
                        native_size=[width,height],mask_grid=[grid_w,grid_h],
                        confidence=args.confidence,precision=args.precision,
                        gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                        prompt='person',temporal_method='adjacent_frame_mask_iou',
                        match_iou=.15,temporal_iou=.15,
                        labels='weak_model_derived',accuracy_evaluated=False,
                        pose_reinference_performed=False)
        print(json.dumps(manifest,indent=2),flush=True)
        if args.check:
            return
        args.output.mkdir(parents=True,exist_ok=False)
        owned = True
        save_json(args.output/'manifest.json',manifest)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from PIL import Image
        print('Loading local SAM3 image model...',flush=True)
        load_start = time.perf_counter()
        model = build_sam3_image_model(device='cuda',checkpoint_path=str(args.checkpoint),load_from_HF=False)
        processor = Sam3Processor(model,device='cuda',confidence_threshold=args.confidence)
        load_seconds = time.perf_counter()-load_start
        analyzer = PersonMaskAnalyzer(width,height,data['court_corners'])
        writer = cv2.VideoWriter(str(args.output/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(grid_w*2,grid_h))
        if not writer.isOpened():
            raise RuntimeError('MP4 writer unavailable')
        torch.cuda.reset_peak_memory_stats()
        counts = Counter()
        records,latencies = [],[]
        temporal_ious = []
        processing_start = time.perf_counter()
        with (args.output/'masks.jsonl').open('w',encoding='utf-8') as masks_file, (args.output/'review_candidates.jsonl').open('w',encoding='utf-8') as candidates:
            for idx in range(last):
                ok,frame = cap.read()  # Decode from start: preserve exact WebM source indices.
                if not ok:
                    raise RuntimeError(f'Video ended unexpectedly at source frame {idx}')
                if idx < first:
                    continue
                grid = cv2.resize(frame,(grid_w,grid_h),interpolation=cv2.INTER_AREA)
                image = Image.fromarray(cv2.cvtColor(grid,cv2.COLOR_BGR2RGB))
                torch.cuda.synchronize()
                started = time.perf_counter()
                dtype = {'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[args.precision]
                with torch.inference_mode(),torch.autocast('cuda',dtype=dtype,enabled=args.precision != 'float32'):
                    state = processor.set_image(image)
                    output = processor.set_text_prompt(state=state,prompt='person')
                masks = output['masks'].detach().cpu().numpy()
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:,0]
                scores = output['scores'].detach().float().cpu().numpy().reshape(-1)
                if masks.ndim != 3 or masks.shape != (len(scores),grid_h,grid_w) or not np.isfinite(masks).all():
                    raise ValueError(f'Unexpected SAM mask shape: {masks.shape}')
                order = np.argsort(-scores)
                record = analyzer.process(idx,frames[idx],masks[order] > .5,scores[order])
                record['t'] = idx/fps
                record['ball_confidence'] = frames[idx].get('ball_confidence')
                latencies.append(time.perf_counter()-started)
                masks_file.write(json.dumps(record,allow_nan=False)+'\n')
                for person in record['persons']:
                    counts[person['court_status']] += 1
                    if person['previous_mask_iou'] is not None:
                        temporal_ious.append(person['previous_mask_iou'])
                    if person['review_required']:
                        candidates.write(json.dumps(dict(source_video=args.video.stem,f=idx,t=idx/fps,
                            person={k:v for k,v in person.items() if k != 'mask'},
                            reason='Absent from exported court-filtered poses; not proven raw YOLO miss'))+'\n')
                comparison = overlay(grid,frames[idx],record,analyzer,cv2)
                writer.write(comparison)
                if len(records) in (0,(last-first)//2,last-first-1):
                    cv2.imwrite(str(args.output/f'preview_f{idx}.jpg'),comparison)
                records.append({k:v for k,v in record.items() if k != 'persons'})
                del state,output
                if len(records) == 1 or len(records) % 50 == 0:
                    print(f'Processed {len(records)}/{last-first}; SAM mean={np.mean(latencies):.3f}s; candidates={counts["inside_candidate"]}',flush=True)
        writer.release()
        writer = None
        np.savez_compressed(args.output/'mask_features.npz',frames=np.array([r['f'] for r in records],np.int64),
            features=np.array([r['features'] for r in records],np.float32),valid=np.array([r['valid'] for r in records],bool),
            feature_names=np.array(FEATURE_NAMES),schema=np.array(SCHEMA),source_video=np.array(args.video.stem),
            ball_source=np.array([r['ball_source'] for r in records]),ball_out_of_frame=np.array([r['ball_out_of_frame'] for r in records]))
        report = dict(frames=len(records),person_frame_counts=dict(counts),
            total_saved_yolo_pose_frames=sum(len(frames[r['f']].get('poses',[])) for r in records),
            frames_with_candidates=sum(r['features'][6] > 0 for r in records),
            temporal_mask_pairs=len(temporal_ious),mean_adjacent_mask_iou=float(np.mean(temporal_ious)) if temporal_ious else None,
            mean_inference_seconds=float(np.mean(latencies)),p95_inference_seconds=float(np.percentile(latencies,95)),
            model_load_seconds=load_seconds,processing_seconds=time.perf_counter()-processing_start,
            peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
            peak_cuda_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
            inputs_unchanged=all(sha256(p) == hashes[k] for k,p in [('video',args.video),('annotations',args.annotations)]),
            feature_valid_counts=np.array([r['valid'] for r in records]).sum(axis=0).tolist(),
            accuracy_evaluated=False,pose_reinference_performed=False,
            warning='Counts are person-frame observations, not unique people or verified YOLO misses')
        if not report['inputs_unchanged']:
            raise RuntimeError('Source inputs changed during experiment')
        save_json(args.output/'report.json',report)
        manifest['status'] = 'complete'
        save_json(args.output/'manifest.json',manifest)
        print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        if owned:
            save_json(args.output/'failure.json',dict(error=repr(exc)))
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
