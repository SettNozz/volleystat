import sys
from pathlib import Path
import pytest
import torch


def test_author_gsm_backbone_and_three_layer_gru_work_on_cpu():
    deps=Path(__file__).resolve().parents[1]/'.runtime/vnl_stes_deps'
    sys.path.insert(0,str(deps));pytest.importorskip('timm')
    from action_recognition.models.vnl_stes_model import build_author_stes
    from third_party.vnl_stes.model.shift import GatedShift
    model=build_author_stes(clip_len=4,pretrained=False).eval()
    assert model._pred_fine._gru.num_layers==3
    assert model._pred_fine._gru.bidirectional
    assert sum(isinstance(m,GatedShift) for m in model.modules())>0
    x=torch.randn(1,4,3,32,32)
    with torch.no_grad():result=model(x)
    assert result['im_feat'].shape==(1,4,5) and result['loc_feat'] is None
    assert torch.isfinite(result['im_feat']).all()


def test_gsm_padding_respects_tensor_dtype_and_does_not_mix_clips():
    deps=Path(__file__).resolve().parents[1]/'.runtime/vnl_stes_deps'
    sys.path.insert(0,str(deps));pytest.importorskip('timm')
    from action_recognition.models.vnl_stes_model import build_author_stes
    from third_party.vnl_stes.model.shift import _GSM
    build_author_stes(clip_len=4,pretrained=False)
    gsm=_GSM(8,4)
    x=torch.ones(2,8,4,2,2,dtype=torch.float64)
    left=gsm.lshift_zeroPad(x);right=gsm.rshift_zeroPad(x)
    assert left.dtype==x.dtype and right.dtype==x.dtype
    assert not left[:,:,-1].any() and not right[:,:,0].any()
    assert left[:,:,:-1].all() and right[:,:,1:].all()


def test_fast_peak_suppression_and_metrics_match_reference():
    deps=Path(__file__).resolve().parents[1]/'.runtime/vnl_stes_deps';sys.path.insert(0,str(deps))
    from scripts.evaluation.run_author_stes import fast_peaks,event_metrics
    from action_recognition.rally_spotting.metrics import nms_peaks
    from action_recognition.rally_spotting.rgb_spotter import evaluate
    import numpy as np
    rng=np.random.default_rng(7)
    for radius in (0,1,5):
        values=rng.integers(0,5,size=200).astype(np.float32)/4
        assert fast_peaks(values,.25,radius)==nms_peaks(values,.25,radius)
    p=rng.random((200,5)).astype(np.float32);p/=p.sum(1,keepdims=True)
    truth=[dict(frame=30,type='serve'),dict(frame=80,type='attack')]
    assert event_metrics(p,truth,.2)==evaluate(p,truth,.2)
