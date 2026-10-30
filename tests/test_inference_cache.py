"""Frozen model outputs, raw-pose tracking parity, and pixel-free production replay."""
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.models.inference_cache import (
    CacheMiss, ModelOutputCache, ObservationFeatures, PoseTracker, array_payload,
    file_digest, model_spec, pack_pose_result, unpack_pose_result,
)
from src.pipeline.game_analyzer import CalibrationData, GameAnalyzer


def tracker_yaml():
    return metadata.distribution('ultralytics').locate_file('ultralytics/cfg/trackers/bytetrack.yaml')


def test_immutable_values_survive_reopen_and_require_never_computes(tmp_path):
    source=tmp_path/'video'
    source.write_bytes(b'video one')
    cache=ModelOutputCache(source,tmp_path/'cache')
    stage=cache.stage('detector',{'weights':'first'})
    assert cache.compute(stage,0,lambda:[])==[]
    cache.db.close()
    cached=ModelOutputCache(source,tmp_path/'cache',True)
    assert cached.compute(stage,0,lambda:pytest.fail('inference on cache hit'))==[]
    with pytest.raises(CacheMiss):
        cached.compute(stage,1,lambda:pytest.fail('inference in replay-only mode'))
    with pytest.raises(ValueError,match='replace'):
        cached.put(stage,0,[1])
    cached.db.close()
    source.write_bytes(b'different video')
    other=ModelOutputCache(source,tmp_path/'cache')
    assert other.get(stage,0) is None
    other.db.close()


def test_neural_logits_cached_before_any_downstream_gate(tmp_path):
    import torch
    from src.models.inference_cache import CachedForward
    source=tmp_path/'video'; source.write_bytes(b'video')
    cache=ModelOutputCache(source,tmp_path/'cache')
    calls=[]
    def model(inputs):
        calls.append(True)
        return inputs*2
    forward=CachedForward(model,cache,{'weights':'one'})
    value=torch.tensor([[.1,.8]],dtype=torch.float32)
    first=forward(value)
    cache.require=True
    assert torch.equal(forward(value),first) and len(calls)==1
    # Different phase gates can consume these same raw logits without inference.
    assert torch.argmax(first,dim=1).item()==1
    with pytest.raises(CacheMiss):
        forward(value+1)
    assert len(calls)==1
    cache.db.close()


def test_reid_warm_cache_consumes_neither_images_nor_encoder(tmp_path):
    from copy import deepcopy
    from src.tracking.player_identity_workflow import run_identity_pass
    source=tmp_path/'video';source.write_bytes(b'video')
    cache=ModelOutputCache(source,tmp_path/'cache')
    store=ObservationFeatures(cache,{'weights':'fake'},600,200)
    image=np.zeros((200,600,3),dtype=np.uint8)
    poses=[]
    for i in range(6):
        image[20:180,i*90:i*90+60]=i+1
        poses.append({'id':i,'team':'b','bb':[i*90,20,i*90+60,180]})
    frames=[{'f':i,'poses':deepcopy(poses)} for i in range(2)]
    class Encoder:
        provenance={'weights':'fake'}
        crop=staticmethod(lambda im,bb:im[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2])])
        def encode(self,crops):
            return np.stack([np.eye(6,dtype=np.float32)[int(c[0,0,0])-1] for c in crops])
    first=run_identity_pass(frames,iter([(0,image),(1,image)]),30.,Encoder(),feature_cache=store)
    original_ids={tuple(p['bb']):p['player_id'] for p in frames[0]['poses']}
    for frame in frames:
        frame['poses'].reverse()  # Cached crops cannot depend on annotation array indices.
    cache.require=True
    def forbidden_images():
        pytest.fail('Cached ReID decoded video')
        yield None
    encoder=Encoder()
    encoder.encode=lambda *args:pytest.fail('Cached ReID reran OSNet')
    second=run_identity_pass(frames,forbidden_images(),30.,encoder,feature_cache=store)
    assert first['assigned']==second['assigned']==12
    assert {tuple(p['bb']):p['player_id'] for p in frames[0]['poses']}==original_ids
    cache.db.close()


def test_interrupted_tracker_sequence_cannot_be_used_as_completed(tmp_path):
    source=tmp_path/'video'; source.write_bytes(b'video')
    cache=ModelOutputCache(source,tmp_path/'cache')
    first,read=cache.sequence({'model':'pose','scope':3})
    assert read is False
    cache.put(first,0,{'some':'output'})
    second,read=cache.sequence({'model':'pose','scope':3})
    assert not read and second!=first
    cache.seal_sequence(second)
    assert cache.sequence({'model':'pose','scope':3})==(second,True)
    cache.db.close()


def test_crop_key_ignores_roster_and_pose_index_but_tracks_pixels_and_model(tmp_path):
    source=tmp_path/'video'; source.write_bytes(b'video')
    cache=ModelOutputCache(source,tmp_path/'cache')
    store=ObservationFeatures(cache,{'weights':'one'},100,200)
    vector=np.arange(512,dtype=np.float32)
    store.put(4,[-2.1,3.1,40.1,300],vector)
    np.testing.assert_array_equal(store.get(4,[0,3,40,200]),vector)
    assert store.get(5,[0,3,40,200]) is None
    assert store.get(4,[0,3,41,200]) is None
    assert ObservationFeatures(cache,{'weights':'two'},100,200).get(4,[0,3,40,200]) is None
    cache.db.close()


def raw_pose(frame):
    boxes=np.asarray([[-5+frame*2,-2,310+frame*2,210,.9,0]],dtype=np.float32)
    if frame in (2,5):
        boxes=np.empty((0,6),dtype=np.float32)
    keypoints=np.full((len(boxes),17,3),.75,dtype=np.float32)
    return {'boxes':array_payload(boxes),'keypoints':array_payload(keypoints)}


def test_cached_raw_pose_tracker_matches_installed_ultralytics_callback():
    import torch
    from ultralytics.engine.results import Results
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.track import on_predict_postprocess_end
    from ultralytics.utils import YAML
    image=np.zeros((200,300,3),dtype=np.uint8)
    tracker=BYTETracker(SimpleNamespace(**YAML.load(str(tracker_yaml()))),frame_rate=30)
    expected=[]
    for frame in range(9):
        raw=raw_pose(frame)
        result=Results(image,path='frame.jpg',names={0:'person'},
                       boxes=torch.tensor(raw['boxes']['values'],dtype=torch.float32).reshape(-1,6),
                       keypoints=torch.tensor(raw['keypoints']['values'],dtype=torch.float32).reshape(-1,17,3))
        predictor=SimpleNamespace(args=SimpleNamespace(task='pose'),dataset=SimpleNamespace(mode='image'),
                                  trackers=[tracker],results=[result],vid_path=[None],save_dir=Path('.'))
        on_predict_postprocess_end(predictor,persist=True)
        expected.append(pack_pose_result(predictor.results[0]))
    replay=PoseTracker(tracker_yaml())
    assert [replay.update(raw_pose(frame),(300,200)) for frame in range(9)]==expected


def test_production_cold_then_cache_only_replay_never_calls_models_or_decodes(tmp_path,monkeypatch):
    source=tmp_path/'game.webm'; source.write_bytes(b'fake video bytes')
    checkpoint=tmp_path/'weights.pt'; checkpoint.write_bytes(b'weights one')
    calls={'ball':0,'pose':0,'reads':0}
    deny_pixels=[False]
    geometry=[100.]
    image=np.zeros((200,600,3),dtype=np.uint8)
    class Capture:
        def __init__(self,*args): self.frame=0
        def isOpened(self): return True
        def get(self,key): return {cv2.CAP_PROP_FPS:30.,cv2.CAP_PROP_FRAME_COUNT:8,
                                  cv2.CAP_PROP_FRAME_WIDTH:600,cv2.CAP_PROP_FRAME_HEIGHT:200}[key]
        def read(self):
            assert not deny_pixels[0], 'Warm replay decoded video'
            calls['reads']+=1
            self.frame+=1
            return (True,image) if self.frame<=8 else (False,None)
        def release(self): pass
    class Ball:
        def detect_ball_batch(self,frames):
            calls['ball']+=1
            return [[(10,15,20,25,.9)] for _ in frames]
    class Pose:
        def predict(self,*args,**kwargs):
            calls['pose']+=1
            boxes=np.asarray([[i*90,60,i*90+60,190,.95,0] for i in range(6)],dtype=np.float32)
            kp=np.zeros((6,17,3),dtype=np.float32)
            return [unpack_pose_result({'boxes':array_payload(boxes),'keypoints':array_payload(kp)})]
    def init_ball(a,path):
        a.ball_detector=Ball(); a.ball_detector_2=None
        a._ball_specs={'primary':model_spec(path or checkpoint,backend='fake-ball',batch_size=8)}
        a._configure_ball_selection_for(600)
    def init_pose(a,path):
        a.pose_model=Pose(); a._pose_tracker_path=tracker_yaml()
        a._pose_spec=model_spec(checkpoint,backend='fake-pose')
        a._pose_spec['tracker']=file_digest(tracker_yaml())
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    monkeypatch.setattr(GameAnalyzer,'_init_ball_detector',init_ball)
    monkeypatch.setattr(GameAnalyzer,'_init_pose_model',init_pose)
    monkeypatch.setattr(GameAnalyzer,'_init_event_predictor',lambda *a:None)
    monkeypatch.setattr(GameAnalyzer,'_warmup',lambda *a:None)
    monkeypatch.setattr(GameAnalyzer,'_load_calibration',lambda *a:CalibrationData(geometry[0],0,600,0,200,0,600,600,200))
    options=dict(enable_player_identity=False,enable_net_detection=False,
                 model_cache_dir=str(tmp_path/'cache'),use_model_registry=False,ball_batch_size=8)
    first=GameAnalyzer(**options).process_video(str(source),str(tmp_path/'first'))
    assert calls['ball'] and calls['pose']==8 and calls['reads']==8
    calls.update(ball=0,pose=0,reads=0)
    deny_pixels[0]=True
    second=GameAnalyzer(**options,require_model_cache=True).process_video(str(source),str(tmp_path/'second'))
    assert calls=={'ball':0,'pose':0,'reads':0}
    assert first.frame_annotations==second.frame_annotations
    geometry[0]=150.
    third=GameAnalyzer(**options,require_model_cache=True).process_video(str(source),str(tmp_path/'third'))
    assert calls=={'ball':0,'pose':0,'reads':0}
    assert all(p['team']=='a' for frame in third.frame_annotations for p in frame['poses'])
    changed=tmp_path/'new.pt'; changed.write_bytes(b'new weights')
    with pytest.raises(CacheMiss):
        GameAnalyzer(**options,require_model_cache=True,ball_model_path=str(changed)).process_video(str(source),str(tmp_path/'missing'))
    assert calls=={'ball':0,'pose':0,'reads':0}
    deny_pixels[0]=False
    GameAnalyzer(**options,ball_model_path=str(changed)).process_video(str(source),str(tmp_path/'new-model'))
    assert calls['ball']>0 and calls['pose']==0


def test_net_cache_binds_geometry_and_reuses_writeback_without_sam(tmp_path):
    from dataclasses import replace
    source=tmp_path/'video'; source.write_bytes(b'video')
    weights=tmp_path/'net.pt'; weights.write_bytes(b'weights')
    cache=ModelOutputCache(source,tmp_path/'cache')
    analyzer=GameAnalyzer.__new__(GameAnalyzer)
    analyzer._model_cache=cache
    analyzer.enable_net_detection=True
    analyzer.net_height_m=2.34
    analyzer.model_provenance={}
    analyzer._resolve=lambda *args:SimpleNamespace(path=weights)
    initial=CalibrationData(100,0,600,0,200,0,600,600,200)
    analyzer.calibration=initial
    calls=[]
    def infer(*args):
        calls.append(True)
        analyzer.calibration=replace(initial,net_tape_top=90.,net_tape_bottom=100.)
    analyzer._run_net_detection_uncached=infer
    analyzer._maybe_run_sam_net_detection(str(source),600,200)
    analyzer.calibration=initial
    cache.require=True
    analyzer._maybe_run_sam_net_detection(str(source),600,200)
    assert len(calls)==1 and analyzer.calibration.net_tape_top==90.
    # The persisted calibration is also a valid lookup key on the next run.
    analyzer._maybe_run_sam_net_detection(str(source),600,200)
    assert len(calls)==1
    analyzer.calibration=replace(initial,net_y=120.)
    with pytest.raises(CacheMiss):
        analyzer._maybe_run_sam_net_detection(str(source),600,200)
    assert len(calls)==1
    cache.db.close()
