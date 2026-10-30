"""Production integration: identity is additive, source-bound and rendered last."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.models.player_reid_encoder import OSNetEncoder, file_sha256, install_assets
from src.pipeline.game_analyzer import CalibrationData, GameAnalyzer, GameStats
from src.tracking.player_identity_workflow import observations_from_annotations, load_seed, run_identity_pass


class ColourEncoder:
    provenance={'method':'test_encoder','weights_sha256':'test'}
    crop=staticmethod(lambda image,bb:image[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2])])
    def encode(self,crops):
        return np.stack([np.eye(6,dtype=np.float32)[int(c[0,0,0])-1] for c in crops])


def fixture():
    image=np.zeros((200,600,3),np.uint8)
    poses=[]
    for i in range(6):
        bb=[i*90+5,20,i*90+65,180]
        image[20:180,bb[0]:bb[2]]=i+1
        kp=[[bb[0]+20.,40.,.9]]+[[0.,0.,0.] for _ in range(16)]
        poses.append({'id':i+1,'team':'b','bb':bb,'kp':kp,'source':'yolo_pose'})
    frames=[{'f':f,'ball':None,'state':'rally_active','pose_f':[0.]*8,'ev':['pass'],
             'poses':deepcopy(poses)} for f in range(2)]
    for p in frames[1]['poses']:
        p['id']+=100
    return frames,image


def analyzer(monkeypatch):
    a=GameAnalyzer.__new__(GameAnalyzer)
    a.enable_player_identity=True
    a.player_identity_side='b'
    a.player_identity_seed=None
    a.player_reid_assets=None
    a.player_reid_device='cpu'
    a.model_provenance={}
    a.max_players_per_team=6
    a.progress_callback=None
    def resolve(name,override):
        assert name=='player_reid'
        a.model_provenance[name]={'source':'test'}
        return SimpleNamespace(path=Path('unused'),as_dict=lambda:{'source':'test'})
    a._resolve=resolve
    monkeypatch.setattr('src.models.player_reid_encoder.OSNetEncoder',lambda *args:ColourEncoder())
    return a


def test_pass_adds_roster_id_without_modifying_raw_poses_ball_or_events():
    frames,image=fixture()
    original=deepcopy(frames)
    result=run_identity_pass(frames,[(0,image),(1,image)],30.,ColourEncoder())
    assert result['status']=='completed'
    assert result['assigned']==12
    assert result['method']=='osnet_frozen_gallery_kalman_roster_v4'
    assert result['parameters']['motion_model']=='kalman'
    assert result['seed']['label_source']=='automatic_initialisation'
    assert result['seed']['verified'] is False
    for before,after in zip(original,frames):
        assert before=={**after,'poses':[{k:v for k,v in p.items() if k not in ('player_id','identity')}
                                       for p in after['poses']]}
        assert [p['player_id'] for p in after['poses']]==[f'P{i+1}' for i in range(6)]
        assert all(p['identity']['verified'] is False for p in after['poses'])


def test_missing_source_frame_fails_before_partial_identity_write():
    frames,image=fixture()
    original=deepcopy(frames)
    with pytest.raises(ValueError,match='cover all saved'):
        run_identity_pass(frames,[(0,image)],30.,ColourEncoder())
    assert frames==original


def test_incomplete_roster_does_not_fabricate_six_ids():
    frames,image=fixture()
    for f in frames:f['poses']=f['poses'][:5]
    report=run_identity_pass(frames,[],30.,ColourEncoder())
    assert report['status']=='no_complete_roster'
    assert all(p['player_id'] is None for f in frames for p in f['poses'])


def test_manual_seed_matches_boxes_not_previous_pose_index_or_raw_id(tmp_path):
    frames,_=fixture()
    video=tmp_path/'video.mp4'
    video.write_bytes(b'exact source video bytes')
    seed={'schema':'player_identity_seed_v1','video_sha256':file_sha256(video),'side':'b','frame':0,
          'label_source':'human','verified':True,
          'players':[{'player_id':f'P{i+1}','bb':p['bb']} for i,p in enumerate(frames[0]['poses'])]}
    path=tmp_path/'seed.json'
    path.write_text(json.dumps(seed))
    frames[0]['poses'].reverse()
    for p in frames[0]['poses']:p['id']+=200
    loaded=load_seed(path,observations_from_annotations(frames),video,'b',6)
    assert loaded['assignments']['0:0']=='P6'
    assert loaded['assignments']['0:5']=='P1'
    video.write_bytes(b'a different clip')
    with pytest.raises(ValueError,match='different source video'):
        load_seed(path,observations_from_annotations(frames),video,'b',6)


def test_off_switch_skips_resolution_and_preserves_annotations(monkeypatch):
    frames,image=fixture()
    original=deepcopy(frames)
    a=analyzer(monkeypatch)
    a.enable_player_identity=False
    a._resolve=lambda *args:pytest.fail('Disabled identity must not resolve/load OSNet')
    stats=GameStats(frame_annotations=frames)
    a._apply_player_identity(stats,[],30.)
    assert stats.player_identity['enabled'] is False
    assert frames==original


def test_missing_default_assets_are_explicit_unknown_not_fake_tracker_identity(monkeypatch):
    frames,_=fixture()
    a=analyzer(monkeypatch)
    def missing(*args):raise FileNotFoundError('missing pinned bundle')
    monkeypatch.setattr('src.models.player_reid_encoder.OSNetEncoder',missing)
    stats=GameStats(frame_annotations=frames)
    a._apply_player_identity(stats,[],30.)
    assert stats.player_identity['status']=='model_unavailable'
    assert all(p['player_id'] is None and p['id'] is not None for f in frames for p in f['poses'])
    a.player_reid_assets='explicit_bundle'
    with pytest.raises(FileNotFoundError):a._apply_player_identity(GameStats(frame_annotations=[]),[],30.)


def test_pipeline_default_and_explicit_disable(monkeypatch):
    monkeypatch.setattr(GameAnalyzer,'_load_calibration',lambda *args:None)
    monkeypatch.setattr(GameAnalyzer,'_init_ball_detector',lambda *args:None)
    monkeypatch.setattr(GameAnalyzer,'_init_pose_model',lambda *args:None)
    monkeypatch.setattr(GameAnalyzer,'_init_event_predictor',lambda *args:None)
    monkeypatch.setattr(GameAnalyzer,'_warmup',lambda *args:None)
    assert GameAnalyzer().enable_player_identity is True
    assert GameAnalyzer(enable_player_identity=False).enable_player_identity is False
    assert GameAnalyzer(enable_pose=False).enable_player_identity is False


def test_video_identity_precedes_final_render_and_both_json_exports(monkeypatch,tmp_path):
    frames,image=fixture()
    a=analyzer(monkeypatch)
    a.output_video=True
    a.skip_frames=0
    a.ball_reconstruction_mode='off'
    a.refresh_net_floor=False
    a.net_height_m=2.34
    a.calibration=CalibrationData(100,0,600,0,200,0,600,600,200)
    a._maybe_run_sam_net_detection=lambda *args,**kwargs:None
    a._scale_calibration=lambda *args:a.calibration
    a._configure_ball_selection_for=lambda *args:None
    class Capture:
        def __init__(self):self.index=0
        def isOpened(self):return True
        def get(self,key):return {cv2.CAP_PROP_FPS:30.,cv2.CAP_PROP_FRAME_COUNT:2,
                                 cv2.CAP_PROP_FRAME_WIDTH:600,cv2.CAP_PROP_FRAME_HEIGHT:200}[key]
        def set(self,*args):self.index=0;return True
        def read(self):
            self.index+=1
            return (True,image.copy()) if self.index<=2 else (False,None)
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args:Capture())
    def process(images,fps,total,stats,**kwargs):
        assert kwargs['out_writer'] is None
        list(images)
        stats.frame_annotations=deepcopy(frames)
        stats.total_frames=stats.processed_frames=2
        return stats
    a._process_frames_iter=process
    rendered=[]
    def render(source,path,annotations,*args):
        assert annotations[1]['poses'][0]['id']==101
        assert annotations[1]['poses'][0]['player_id']=='P1'
        rendered.append(path)
    a._render_final_video=render
    stats=a.process_video(str(tmp_path/'video.mp4'),str(tmp_path/'result'))
    assert len(rendered)==1
    assert stats.player_identity['coverage']==1.
    saved_stats=json.loads((tmp_path/'result/video_stats.json').read_text())
    saved_ann=json.loads((tmp_path/'result/video_annotations.json').read_text())
    assert saved_stats['player_identity']==saved_ann['player_identity']
    assert saved_ann['frames'][1]['poses'][0]['id']==101
    assert saved_ann['frames'][1]['poses'][0]['player_id']=='P1'


def test_pose_renderer_distinguishes_person_unknown_and_legacy_track(monkeypatch):
    from src.detection.player_pose import draw_skeleton
    frames,image=fixture()
    p=frames[0]['poses'][0]
    a=GameAnalyzer._poses_from_annotation({'poses':[{**p,'player_id':'P6'}]})[0]
    text=[]
    monkeypatch.setattr(cv2,'putText',lambda frame,label,*args,**kwargs:text.append(label))
    draw_skeleton(image,a,(255,255,255),draw_id=True)
    assert text[-1]=='P6' and a.track_id==1
    a.player_id=None
    draw_skeleton(image,a,(255,255,255),draw_id=True)
    assert text[-1]=='P?'
    a.player_identity_processed=False
    draw_skeleton(image,a,(255,255,255),draw_id=True)
    assert text[-1]=='track #1'


def test_install_refuses_to_overwrite_existing_model_directory(tmp_path):
    existing=tmp_path/'model'
    existing.mkdir()
    marker=existing/'keep.pth'
    marker.write_bytes(b'keep')
    with pytest.raises(FileExistsError):install_assets(existing)
    assert marker.read_bytes()==b'keep'


def test_off_image_box_cannot_wrap_into_a_full_image_identity_crop():
    with pytest.raises(ValueError,match='Empty player crop'):
        OSNetEncoder.crop(np.zeros((20,20,3),np.uint8),[-10,-10,-1,-1])


def test_video_refuses_existing_artifacts_before_opening_source(monkeypatch,tmp_path):
    a=GameAnalyzer.__new__(GameAnalyzer)
    output=tmp_path/'out'
    output.mkdir()
    existing=output/'game_annotations.json'
    existing.write_text('{"keep":true}')
    monkeypatch.setattr(cv2,'VideoCapture',lambda *args:pytest.fail('Existing result must be checked before opening video'))
    with pytest.raises(FileExistsError,match='fresh output'):
        a.process_video(str(tmp_path/'game.webm'),str(output))
    assert existing.read_text()=='{"keep":true}'
