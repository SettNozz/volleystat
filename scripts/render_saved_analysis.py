"""Render saved final annotations with the production renderer, without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.player_reid_encoder import file_sha256
from src.pipeline.game_analyzer import GameAnalyzer


def viewer_metadata_changes(before, after):
    changes={key for key in before.keys()|after.keys() if before.get(key)!=after.get(key)}
    allowed={'court_corners','attack_heatmap','manual_court_correction',
             'net_y','net_x_left','net_x_right','court_setup_id','court_setup_name'}
    if (not changes<=allowed or (changes and after.get('manual_court_correction',{}).get('source')!='viewer')):
        raise ValueError('Render inputs changed during export; immutable input snapshot was retained')
    return sorted(changes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output-video', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, help='Optional short renderer smoke test')
    parser.add_argument('--threads', type=int, default=4, help='OpenCV threads in this isolated rendering process')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('Thread count must be positive')
    cv2.setNumThreads(args.threads)
    snapshot=args.output_video.with_suffix('.render_input.json')
    if args.output_video.exists() or args.output_video.with_suffix('.render.json').exists() or snapshot.exists():
        parser.error('Output exists; choose a fresh file')
    raw=args.annotations.read_bytes()
    fingerprint=hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    with snapshot.open('xb') as stream:
        stream.write(raw)
    calibration_fingerprint=file_sha256(args.calibration)
    source = Path(data['original_video_path'])
    total = max(frame['f'] for frame in data['frames']) + 1
    limit = min(total, args.max_frames) if args.max_frames else total
    if limit <= 0:
        parser.error('Frame count must be positive')
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.net_y_override = None
    analyzer.calibration = analyzer._load_calibration(str(args.calibration.resolve()))
    if analyzer.calibration is None:
        raise ValueError('Explicit calibration unavailable')
    analyzer.calibration = analyzer._scale_calibration(data['frame_width'], data['frame_height'])
    analyzer._projector_calib = None
    analyzer._projector = None
    last = [-300]
    def progress(payload):
        frame = payload.get('frame_current', 0)
        if frame - last[0] >= 300:
            print(f'[render] {frame}/{limit}', flush=True)
            last[0] = frame
    analyzer.progress_callback = progress
    renderer_fingerprint = file_sha256(Path(__file__).resolve().parents[1]/'src/pipeline/game_analyzer.py')
    script_fingerprint = file_sha256(Path(__file__))
    started = time.monotonic()
    ok = analyzer._render_final_video(source, args.output_video, data['frames'], data['fps'],
                                      data['frame_width'], data['frame_height'], limit)
    cap = cv2.VideoCapture(str(args.output_video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    if not ok or count != limit or dimensions != (data['frame_width'], data['frame_height']):
        raise RuntimeError(f'Incomplete output: {count}/{limit}, {dimensions}')
    current_fingerprint=file_sha256(args.annotations)
    changes=[]
    if fingerprint != current_fingerprint:
        changes=viewer_metadata_changes(data,json.loads(args.annotations.read_bytes()))
    if calibration_fingerprint!=file_sha256(args.calibration):
        raise ValueError('Explicit render calibration changed during export')
    with args.output_video.with_suffix('.render.json').open('x', encoding='utf-8') as stream:
        json.dump({'annotations_sha256': fingerprint, 'calibration_sha256': file_sha256(args.calibration),
                   'input_snapshot':str(snapshot.resolve()),'viewer_annotations_sha256':current_fingerprint,
                   'viewer_metadata_changes':changes,
                   'video_sha256': file_sha256(source), 'frames': count, 'width': dimensions[0],
                   'height': dimensions[1], 'elapsed_seconds': time.monotonic() - started,
                   'opencv_threads': cv2.getNumThreads(),
                   'opencv_version': cv2.__version__, 'renderer_source_sha256': renderer_fingerprint,
                   'render_script_sha256': script_fingerprint,
                   'renderer': 'GameAnalyzer._render_final_video; no detector/model inference'}, stream, indent=2)
    print(f'[render] complete {count} frames -> {args.output_video}', flush=True)


if __name__ == '__main__':
    main()
