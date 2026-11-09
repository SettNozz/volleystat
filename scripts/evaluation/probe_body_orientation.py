"""Predict body orientation on saved player crops and score optional visual checks."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou


SOURCE_REV = '362f999e22bd50a4e73aca882b58c13f8a96a13c'
WEIGHTS_SHA256 = 'bc6d63ee0f685a888e5ff94a84d8244ce23a817223010e100459137bacae3e27'


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def orientation_bin(angle):
    if min(angle, 360 - angle) <= 45:
        return 'back'
    return 'front' if abs(angle - 180) <= 45 else 'side'


def load_detector(source, weights, device):
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError('Expected the pinned JointBDOE-S checkpoint')
    sys.path.insert(0, str(source.resolve()))
    from models.common import Bottleneck, C3, Concat, Conv, Focus, SPP
    from models.yolo import Detect, Model

    nn = torch.nn
    allowed = [Bottleneck, C3, Concat, Conv, Focus, SPP, Detect, Model, nn.BatchNorm2d,
               nn.Conv2d, nn.MaxPool2d, nn.ModuleList, nn.Sequential, nn.SiLU, nn.Upsample,
               np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType,
               (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct')]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    return (checkpoint.get('ema') or checkpoint['model']).float().eval().to(device)


@torch.inference_mode()
def probe(model, device, video, boxes):
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords

    predictions = {}
    cap = cv2.VideoCapture(str(video))
    try:
        for f, poses in sorted(boxes.items()):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, f):
                raise ValueError(f'Cannot seek to F{f}')
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode F{f}')
            for index, box in poses:
                x1, y1, x2, y2 = map(round, box)
                pad_x, pad_y = round((x2 - x1) * .15), round((y2 - y1) * .15)
                left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
                crop = image[top:min(image.shape[0], y2 + pad_y), left:min(image.shape[1], x2 + pad_x)]
                scaled, ratio, pad = letterbox(crop, 1024, auto=True, stride=int(model.stride.max()))
                rgb = np.ascontiguousarray(scaled[:, :, ::-1].transpose(2, 0, 1))
                tensor = torch.from_numpy(rgb).to(device).float().unsqueeze(0) / 255.
                detections = non_max_suppression(model(tensor)[0], .25, .45, num_angles=1)[0]
                key = f'{f}:{index}'
                predictions[key] = None
                if not len(detections):
                    continue
                crop_boxes = scale_coords(tensor.shape[2:], detections[:, :4], crop.shape[:2], (ratio, pad))
                target = crop_boxes.new_tensor([[x1 - left, y1 - top, x2 - left, y2 - top]])
                ious = box_iou(target, crop_boxes)[0]
                best = int(ious.argmax())
                if float(ious[best]) < .55:
                    continue
                angle = float(detections[best, 6]) * 360 % 360
                predictions[key] = {'angle': round(angle, 2), 'class': orientation_bin(angle),
                                    'box_iou': round(float(ious[best]), 3)}
    finally:
        cap.release()
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--checks', type=Path, help='Evaluate orientation-labelled diagnostic crops')
    selection.add_argument('--frame', type=int, action='append', help='Predict all saved poses on this frame')
    parser.add_argument('--source', type=Path, required=True, help=f'JointBDOE source at {SOURCE_REV}')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    annotations_sha = sha256(args.annotations)
    payload = json.loads(args.annotations.read_bytes())
    frames = {ann['f']: ann['poses'] for ann in payload['frames']}
    boxes, scores = defaultdict(list), {}
    checks = {}
    if args.checks:
        diagnostic = json.loads(args.checks.read_bytes())
        if diagnostic['annotations_sha256'] != annotations_sha:
            raise ValueError('Orientation checks belong to another annotations file')
        checks = {key: row['orientation'] for key, row in diagnostic['checks'].items() if 'orientation' in row}
        for key in checks:
            f, i = map(int, key.split(':'))
            pose = frames[f][i]
            boxes[f].append((i, pose['bb']))
            scores[key] = pose.get('back')
    else:
        boxes = {f: [(i, pose['bb']) for i, pose in enumerate(frames[f])] for f in args.frame}

    device = torch.device(args.device)
    predictions = probe(load_detector(args.source, args.weights, device), device, payload['original_video_path'], boxes)
    counts = Counter(processed=len(predictions), matched=sum(p is not None for p in predictions.values()))
    for key, label in checks.items():
        predicted = (predictions[key] or {}).get('class')
        counts.update({'reviewed': 1, 'correct_three_class': predicted == label,
                       'correct_binary': predicted is not None and (predicted == 'back') == (label == 'back'),
                       'heuristic_correct_binary': ((scores[key] or 0) >= .45) == (label == 'back')})
    report = {'source_revision': SOURCE_REV, 'weights_sha256': WEIGHTS_SHA256,
              'annotations_sha256': annotations_sha, 'device': str(device),
              'counts': dict(counts), 'checks': checks, 'predictions': predictions, 'back_scores': scores}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(report['counts'])


if __name__ == '__main__':
    main()
