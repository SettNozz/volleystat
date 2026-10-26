"""Recover real YOLO keypoints from independent, saved SAM person proposals.

Primary YOLO/ByteTrack state is never run on crops. Recovered detections use a
separate negative SAM-mask ID namespace and still pass the normal court gate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from src.detection.person_masks import SCHEMA,assign,box_iou
from src.detection.player_pose import extract_player_poses


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


class SamGuidedPoseRecovery:
    def __init__(self,sidecar,pose_checkpoint):
        self.path=Path(sidecar)
        self.manifest=json.loads((self.path.parent/'manifest.json').read_text(encoding='utf-8'))
        m=self.manifest
        if m.get('status')!='complete' or m.get('schema')!=SCHEMA:
            raise ValueError('SAM pose recovery requires a completed person-mask sidecar')
        self.rows={}
        last=-1
        with self.path.open(encoding='utf-8') as handle:
            for line in handle:
                row=json.loads(line);f=row['f']
                if type(f) is not int or f<=last or row.get('schema')!=SCHEMA:
                    raise ValueError('Invalid/unsorted SAM source frames or schema')
                last=f
                if not row.get('mask_detection_available'):
                    continue
                people=[]
                ids=set()
                for person in row['persons']:
                    if person['court_status'] not in ('inside_candidate','existing_court_member'):
                        continue
                    bb=person['bbox_native'];mid=person['instance_id'];score=person['score']
                    if (type(mid) is not int or mid<=0 or mid in ids or len(bb)!=4
                        or not np.isfinite(bb).all() or bb[2]<=bb[0] or bb[3]<=bb[1]
                        or not np.isfinite(score) or not 0<=score<=1):
                        raise ValueError('Invalid SAM proposal')
                    ids.add(mid)
                    if score>=.4:
                        people.append(dict(bb=bb,id=mid,score=score))
                self.rows[f]=people
        if set(self.rows)!=set(range(m['start_frame'],m['end_frame_exclusive'])):
            raise ValueError('Incomplete SAM sidecar interval')
        self.checkpoint=Path(pose_checkpoint)
        if not self.checkpoint.is_file():
            raise ValueError('Recovery requires the existing local YOLO-pose checkpoint')
        self.model=None
        self.validated=False
        self.covered_frames=0
        self.crop_inferences=0
        self.proposals_recovered=0
        self.accepted_person_frames=0
        self.sidecar_hash=file_hash(self.path)

    def validate_video(self,video,fps,width,height):
        m=self.manifest
        if ([width,height]!=m['native_size'] or not np.isclose(fps,m['fps'],rtol=1e-6)
            or Path(video).stem!=Path(m['source_video']).stem
            or file_hash(video)!=m['hashes']['video']):
            raise ValueError('SAM sidecar belongs to different video bytes/FPS/geometry')
        self.validated=True

    def _predict(self,crop,device):
        if self.model is None:
            from ultralytics import YOLO
            self.model=YOLO(str(self.checkpoint))  # Separate instance: no ByteTrack callbacks/state.
        return self.model.predict(crop,imgsz=640,conf=.15,device=device,
                                  half=(device=='cuda'),verbose=False)[0]

    def recover(self,frame,frame_index,poses,projector,device):
        if not self.validated:
            raise ValueError('Validate SAM source identity before pose recovery')
        if frame_index not in self.rows or projector is None:
            return poses
        self.covered_frames+=1
        proposals=self.rows[frame_index]
        pairs=np.array([[box_iou(p['bb'],q.bbox) for q in poses] for p in proposals]).reshape(len(proposals),len(poses))
        matched={i for i,j in assign(pairs,.15)}
        result=list(poses)
        height,width=frame.shape[:2]
        for i,proposal in enumerate(proposals):
            if i in matched:continue
            bb=proposal['bb'];dx=(bb[2]-bb[0])*.2;dy=(bb[3]-bb[1])*.15
            x1=max(0,int(np.floor(bb[0]-dx)));y1=max(0,int(np.floor(bb[1]-dy)))
            x2=min(width,int(np.ceil(bb[2]+dx)));y2=min(height,int(np.ceil(bb[3]+dy)))
            if x2-x1<8 or y2-y1<8:continue
            self.crop_inferences+=1
            raw=self._predict(frame[y1:y2,x1:x2],device)
            candidates=extract_player_poses(raw,net_y=0)
            scored=[]
            for candidate in candidates:
                # Every coordinate below is produced by YOLO, never by SAM.
                kp=candidate.keypoints
                visible=kp[:,2]>0
                kp[visible,0]+=x1;kp[visible,1]+=y1
                candidate.center=(candidate.center[0]+x1,candidate.center[1]+y1)
                c=candidate.bbox
                candidate.bbox=(c[0]+x1,c[1]+y1,c[2]+x1,c[3]+y1)
                if not np.isfinite(kp).all() or np.count_nonzero(kp[[5,6,11,12],2]>=.3)<2:
                    continue
                overlap=box_iou(candidate.bbox,bb)
                if overlap>=.15:
                    scored.append((overlap,candidate))
            if not scored:continue
            candidate=max(scored,key=lambda item:item[0])[1]
            if any(box_iou(candidate.bbox,p.bbox)>.5 for p in result):
                continue
            candidate.team=projector.team_of(candidate.foot_point())
            candidate.track_id=-proposal['id']
            candidate.source='sam_guided_yolo_pose'
            candidate.mask_instance_id=proposal['id']
            result.append(candidate)
            self.proposals_recovered+=1
        return result

    def summary(self):
        return dict(enabled=True,sidecar=str(self.path.resolve()),sidecar_sha256=self.sidecar_hash,
            sam_checkpoint_sha256=self.manifest['hashes']['checkpoint'],
            pose_checkpoint=str(self.checkpoint.resolve()),source_schema=SCHEMA,
            covered_frames=self.covered_frames,crop_inferences=self.crop_inferences,
            recovered_proposals=self.proposals_recovered,
            accepted_person_frames=self.accepted_person_frames,
            id_namespace='negative_sam_mask_instance_id',
            labels='weak_model_derived',keypoints='measured_by_yolo_pose_on_crop')
