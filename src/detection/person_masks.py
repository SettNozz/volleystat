"""SAM-derived people features; no changes to poses, teams or ball trajectory."""
from __future__ import annotations
import numpy as np

SCHEMA = 'person_mask_features_v1'
FEATURE_NAMES = (
    'ball_to_person_mask_distance', 'ball_inside_person_mask',
    'closest_person_mask_area_ratio', 'closest_person_mask_score',
    'closest_person_mask_previous_iou', 'yolo_pose_mask_match_fraction',
    'sam_unmatched_in_court_count', 'court_person_mask_count',
)


def encode_mask(mask):
    """Uncompressed ROW-MAJOR RLE, NOT COCO's column-major RLE."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    edges = np.flatnonzero(np.diff(np.r_[0, flat, 0]))
    return dict(encoding='rle_row_major_v1', size=list(mask.shape),
                counts=np.diff(np.r_[0, edges, len(flat)]).tolist())


def decode_mask(record):
    shape, counts = record['size'], record['counts']
    if (record.get('encoding') != 'rle_row_major_v1' or len(shape) != 2
            or any(type(v) is not int or v <= 0 for v in shape)
            or any(type(v) is not int or v < 0 for v in counts)
            or sum(counts) != shape[0]*shape[1]):
        raise ValueError('Invalid RLE')
    return np.repeat(np.arange(len(counts)) % 2, counts).astype(bool).reshape(shape)


def mask_iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)/union) if union else 0.


def box_iou(a, b):
    inter = max(0., min(a[2],b[2])-max(a[0],b[0])) * max(0., min(a[3],b[3])-max(a[1],b[1]))
    union = max(0.,a[2]-a[0])*max(0.,a[3]-a[1]) + max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-inter
    return inter/union if union else 0.


def assign(scores, threshold):
    """Maximum weight one-to-one Hungarian assignment, explicit unmatched slots.

    NumPy only, so SAM's isolated environment needs no new dependencies.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('Expected finite pairwise scores')
    n, m = scores.shape
    if not n or not m:
        return []
    valid = (scores >= threshold) & (scores > 0)
    cost = np.concatenate([np.where(valid, -scores, 1e6), np.zeros((n,n))], axis=1)
    cols = cost.shape[1]
    u, v = np.zeros(n+1), np.zeros(cols+1)
    p, way = np.zeros(cols+1, int), np.zeros(cols+1, int)
    for row in range(1,n+1):
        p[0], j0 = row, 0
        mins, used = np.full(cols+1, np.inf), np.zeros(cols+1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1,cols+1):
                if not used[j]:
                    cur = cost[i0-1,j-1]-u[i0]-v[j]
                    if cur < mins[j]:
                        mins[j], way[j] = cur, j0
                    if mins[j] < delta:
                        delta, j1 = mins[j], j
            for j in range(cols+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    mins[j] -= delta
            j0 = j1
            if not p[j0]:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    return [(int(p[j]-1),j-1) for j in range(1,m+1) if p[j] and valid[p[j]-1,j-1]]


def inside_polygon(point, polygon):
    x, y = point
    inside = False
    for (ax,ay),(bx,by) in zip(polygon, polygon[1:]+polygon[:1]):
        cross = (x-ax)*(by-ay)-(y-ay)*(bx-ax)
        if abs(cross) < 1e-6 and min(ax,bx) <= x <= max(ax,bx) and min(ay,by) <= y <= max(ay,by):
            return True
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            inside = not inside
    return inside


class PersonMaskAnalyzer:
    def __init__(self, width, height, court_corners, match_iou=.15, temporal_iou=.15):
        if width <= 0 or height <= 0:
            raise ValueError('Positive native dimensions required')
        self.width, self.height = width, height
        self.court = [list(p) for p in court_corners] if court_corners else []
        self.match_iou, self.temporal_iou = match_iou, temporal_iou
        self.previous, self.previous_frame, self.shape = [], None, None
        self.next_id = 1

    def process(self, frame, annotation, masks, scores):
        if self.previous_frame is not None and frame <= self.previous_frame:
            raise ValueError('Source frame indices must increase')
        masks, scores = np.asarray(masks), np.asarray(scores, dtype=float)
        if (masks.ndim != 3 or scores.ndim != 1 or masks.shape[0] != len(scores)
                or not np.isfinite(masks).all() or not np.isin(masks,[0,1]).all()
                or not np.isfinite(scores).all() or np.any((scores < 0)|(scores > 1))):
            raise ValueError('Expected binary (N,H,W) masks and scores in [0,1]')
        if self.shape is not None and self.shape != masks.shape[1:]:
            raise ValueError('Mask grid must remain constant')
        self.shape = masks.shape[1:]
        h, w = self.shape
        if not h or not w:
            raise ValueError('Empty mask grid')
        sx, sy = self.width/w, self.height/h
        persons = []
        for mask, score in zip(masks.astype(bool), scores):
            yy, xx = np.nonzero(mask)
            if not len(xx):
                continue
            box = [float(xx.min()*sx),float(yy.min()*sy),float((xx.max()+1)*sx),float((yy.max()+1)*sy)]
            persons.append(dict(mask=mask, score=float(score), bbox_native=box,
                                yolo_track_id=None, team=None, pose_index=None,
                                yolo_match_iou=None, previous_mask_iou=None))
        poses = annotation.get('poses',[])
        pairs = np.array([[box_iou(p['bbox_native'],q['bb']) for q in poses] for p in persons]).reshape(len(persons),len(poses))
        for i,j in assign(pairs,self.match_iou):
            persons[i].update(yolo_track_id=poses[j].get('id'),team=poses[j].get('team'),
                              pose_index=j,yolo_match_iou=float(pairs[i,j]))
        for person in persons:
            box = person['bbox_native']
            person['court_status'] = (
                'existing_court_member' if person['pose_index'] is not None else
                'unknown' if not self.court else
                'inside_candidate' if inside_polygon(((box[0]+box[2])/2,box[3]),self.court) else 'outside')
        adjacent = self.previous_frame is not None and frame == self.previous_frame+1
        old = self.previous if adjacent else []
        temporal = np.zeros((len(persons),len(old)))
        for i,person in enumerate(persons):
            for j,prev in enumerate(old):
                a,b = person['yolo_track_id'],prev['yolo_track_id']
                temporal[i,j] = mask_iou(person['mask'],prev['mask']) if a is None or b is None or a == b else -1
        for i,j in assign(temporal,self.temporal_iou):
            persons[i].update(instance_id=old[j]['instance_id'],previous_mask_iou=float(temporal[i,j]))
        for person in persons:
            if 'instance_id' not in person:
                person['instance_id'] = self.next_id
                self.next_id += 1
        eligible = [p for p in persons if p['court_status'] in ('existing_court_member','inside_candidate')]
        features, valid = np.zeros(8,np.float32),np.zeros(8,bool)
        if poses:
            features[5] = sum(p['pose_index'] is not None for p in persons)/len(poses)
            valid[5] = True
        if self.court or not any(p['court_status'] == 'unknown' for p in persons):
            features[6] = sum(p['court_status'] == 'inside_candidate' for p in persons)
            features[7] = len(eligible)
            valid[6:] = True
        ball = annotation.get('ball')
        ball_valid = (not annotation.get('ball_out_of_frame',False) and ball is not None
                      and len(ball) == 2 and np.isfinite(ball).all()
                      and 0 <= ball[0] < self.width and 0 <= ball[1] < self.height)
        if ball_valid and eligible:
            bx,by = ball[0]/sx,ball[1]/sy
            distances = []
            for person in eligible:
                yy,xx = np.nonzero(person['mask'])
                dx = np.maximum(np.maximum(xx-bx,bx-(xx+1)),0)/w
                dy = np.maximum(np.maximum(yy-by,by-(yy+1)),0)/h
                distances.append(float(np.sqrt(dx*dx+dy*dy).min()))
            nearest = eligible[int(np.argmin(distances))]
            features[:4] = [min(distances),any(p['mask'][int(by),int(bx)] for p in eligible),nearest['mask'].mean(),nearest['score']]
            valid[:4] = True
            if nearest['previous_mask_iou'] is not None:
                features[4],valid[4] = nearest['previous_mask_iou'],True
        exported = []
        for person in persons:
            exported.append({k:v for k,v in person.items() if k != 'mask'} | dict(
                mask=encode_mask(person['mask']),provenance='sam3_image_causal_iou_v1',
                review_required=person['court_status'] == 'inside_candidate'))
        self.previous,self.previous_frame = persons,frame
        return dict(f=frame,schema=SCHEMA,features=features.tolist(),valid=valid.tolist(),
                    mask_detection_available=True,ball_source=annotation.get('ball_source','unspecified'),
                    ball_out_of_frame=bool(annotation.get('ball_out_of_frame',False)),persons=exported)
