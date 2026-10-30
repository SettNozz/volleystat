import cv2
import numpy as np
import pytest
import torch
from action_recognition.models.vnl_stes_model import build_author_stes
from scripts.evaluation.run_spatial_stes import SpatialClips,spatial_metrics


@pytest.mark.parametrize("chunk_frames",[None,2])
def test_checkpointing_preserves_outputs_gradients_and_batchnorm_state(chunk_frames):
    torch.set_num_threads(4);torch.manual_seed(17)
    plain=build_author_stes(pretrained=False,clip_len=4,predict_location=True,backbone_chunk_frames=chunk_frames).train()
    checked=build_author_stes(pretrained=False,clip_len=4,predict_location=True,activation_checkpointing=True,backbone_chunk_frames=chunk_frames).train()
    checked.load_state_dict(plain.state_dict());assert plain.state_dict().keys()==checked.state_dict().keys()
    x=torch.randn(1,4,3,32,48)
    torch.manual_seed(19);a=plain(x);loss_a=a["im_feat"].square().mean()+a["loc_feat"].square().mean();loss_a.backward()
    torch.manual_seed(19);b=checked(x);loss_b=b["im_feat"].square().mean()+b["loc_feat"].square().mean();loss_b.backward()
    assert a["im_feat"].shape==(1,4,5)
    torch.testing.assert_close(a["loc_feat"],b["loc_feat"],rtol=1e-5,atol=1e-5)
    for key,value in plain.state_dict().items():
        if "running_" in key or "num_batches_tracked" in key:torch.testing.assert_close(value,checked.state_dict()[key],rtol=0,atol=0)
    grads=dict(checked.named_parameters())
    for name,param in plain.named_parameters():
        if param.grad is not None:torch.testing.assert_close(param.grad,grads[name].grad,rtol=2e-4,atol=1e-4)


def record(tmp_path):
    path=tmp_path/"frames.avi";writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),25,(128,72))
    for i in range(10):writer.write(np.full((72,128,3),(20,60,200),np.uint8))
    writer.release()
    return dict(stem="a",cache=str(path),frames=10,events=[dict(frame=2,type="serve",spatial_valid=True,xy_normalized=[.7,.4])],ignored=[],ignored_exact=[])


def test_rectangular_reader_and_variable_clip_keep_xy(tmp_path,monkeypatch):
    r=record(tmp_path);reader=SpatialClips([r],input_width=128,input_height=72,clip_frames=4)
    monkeypatch.setattr("random.random",lambda:.8);monkeypatch.setattr("random.randrange",lambda *a:0)
    x,y,xy,mask=reader.sample(1);reader.close()
    assert x.shape==(1,4,3,72,128) and y.shape==(1,4)
    assert xy[0,2]==pytest.approx([.7,.4]) and mask.sum()==1
    assert abs(float(x[0,0,0].mean())-200)<5


def test_reader_refuses_upscaling_legacy_cache(tmp_path):
    r=record(tmp_path);reader=SpatialClips([r],input_width=1248,input_height=702,clip_frames=4)
    with pytest.raises(ValueError,match="Refuse to upscale"):reader.read(r,0)
    reader.close()


def test_metrics_report_actual_rectangular_input_units():
    coords=np.zeros((4,2));coords[2]=[.6,.5]
    events=[dict(frame=2,spatial_valid=True,xy_normalized=[.5,.5])]
    values=spatial_metrics(coords,events,3840,2160,1248,702)
    assert values["mean_error_input_px"]==pytest.approx(124.8)
    assert values["mean_error_224"]==pytest.approx(22.4)
