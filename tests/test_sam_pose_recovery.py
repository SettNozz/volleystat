import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.detection.sam_pose_recovery import SamGuidedPoseRecovery
from src.detection.person_masks import SCHEMA
from src.detection.player_pose import CourtProjector,PlayerPose


class Tensor:
    def __init__(self,value):self.value=np.asarray(value,dtype=np.float32)
    def cpu(self):return self
    def numpy(self):return self.value


class Boxes:
    def __init__(self,boxes):self.xyxy=Tensor(boxes);self.id=None
    def __len__(self):return len(self.xyxy.value)


def result(keypoints=True):
    kp=np.zeros((1,17,3),np.float32)
    kp[:]=[40,120,.9]
    return SimpleNamespace(boxes=Boxes([[12,24,72,184]]),
                           keypoints=SimpleNamespace(data=Tensor(kp)) if keypoints else None)


@pytest.fixture
def recovery(tmp_path):
    video=tmp_path/'clip.webm';video.write_bytes(b'input-video')
    weights=tmp_path/'pose.pt';weights.write_bytes(b'local-checkpoint')
    manifest=dict(status='complete',schema=SCHEMA,native_size=[400,300],fps=30,
        source_video=str(video),start_frame=100,end_frame_exclusive=101,
        hashes=dict(video=hashlib.sha256(video.read_bytes()).hexdigest(),checkpoint='sam-checkpoint-hash'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    row=dict(f=100,schema=SCHEMA,mask_detection_available=True,persons=[dict(
        bbox_native=[100,80,160,240],instance_id=7,score=.9,court_status='inside_candidate')])
    sidecar=tmp_path/'masks.jsonl';sidecar.write_text(json.dumps(row)+'\n')
    recovery=SamGuidedPoseRecovery(sidecar,weights)
    recovery.validate_video(video,30,400,300)
    return recovery


def projector():
    return CourtProjector([[20,0,200],[0,20,160],[0,0,1]])


def test_crop_keypoints_are_measured_and_restored_to_native_frame(recovery):
    calls=[]
    def predict(crop,device):
        calls.append((crop.shape,device));return result()
    recovery._predict=predict
    poses=recovery.recover(np.zeros((300,400,3),np.uint8),100,[],projector(),'cpu')
    assert calls==[((208,84,3),'cpu')]
    assert len(poses)==1
    p=poses[0]
    assert p.bbox==(100,80,160,240)
    np.testing.assert_allclose(p.keypoints[5],[128,176,.9])
    assert p.source=='sam_guided_yolo_pose'
    assert p.mask_instance_id==7 and p.track_id==-7
    assert p.team=='team_b'
    assert recovery.proposals_recovered==1


def test_primary_pose_is_untouched_and_matching_mask_skips_crop(recovery):
    pose=PlayerPose('team_b',(130,160),(100,80,160,240),np.ones((17,3),np.float32),track_id=91)
    old=copy.deepcopy(pose)
    def forbidden(*args):raise AssertionError('No crop is needed')
    recovery._predict=forbidden
    poses=recovery.recover(np.zeros((300,400,3),np.uint8),100,[pose],projector(),'cpu')
    assert poses==[pose]
    assert pose.track_id==old.track_id and pose.source=='yolo_pose'
    np.testing.assert_array_equal(pose.keypoints,old.keypoints)
    assert recovery.crop_inferences==0


def test_mask_alone_never_fabricates_keypoints(recovery):
    recovery._predict=lambda *args:result(keypoints=False)
    poses=recovery.recover(np.zeros((300,400,3),np.uint8),100,[],projector(),'cpu')
    assert poses==[] and recovery.proposals_recovered==0


def test_uncovered_frame_and_missing_court_do_not_recover(recovery):
    recovery._predict=lambda *args:pytest.fail('Must not infer')
    image=np.zeros((300,400,3),np.uint8)
    assert recovery.recover(image,99,[],projector(),'cpu')==[]
    assert recovery.recover(image,100,[],None,'cpu')==[]


def test_different_video_bytes_or_geometry_are_rejected(recovery,tmp_path):
    other=tmp_path/'different'/'clip.webm';other.parent.mkdir();other.write_bytes(b'wrong-video')
    with pytest.raises(ValueError,match='different video'):
        recovery.validate_video(other,30,400,300)
    with pytest.raises(ValueError,match='different video'):
        recovery.validate_video(tmp_path/'clip.webm',30,800,600)


def test_requires_validated_source_before_inference(recovery):
    recovery.validated=False
    with pytest.raises(ValueError,match='Validate SAM source'):
        recovery.recover(np.zeros((300,400,3)),100,[],projector(),'cpu')


def test_incomplete_sidecar_rejected(recovery,tmp_path):
    path=tmp_path/'manifest.json';manifest=json.loads(path.read_text())
    manifest['end_frame_exclusive']=102;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Incomplete'):
        SamGuidedPoseRecovery(tmp_path/'masks.jsonl',tmp_path/'pose.pt')


def test_failed_run_rejected(recovery,tmp_path):
    path=tmp_path/'manifest.json';manifest=json.loads(path.read_text())
    manifest['status']='preflight';path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='completed'):
        SamGuidedPoseRecovery(tmp_path/'masks.jsonl',tmp_path/'pose.pt')


def test_recovery_summary_records_separate_identity_and_sources(recovery):
    summary=recovery.summary()
    assert summary['id_namespace']=='negative_sam_mask_instance_id'
    assert summary['keypoints']=='measured_by_yolo_pose_on_crop'
    assert summary['sidecar_sha256']
