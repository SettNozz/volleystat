import json
from pathlib import Path
import numpy as np
import pytest
import torch
from action_recognition.rally_spotting.spatial_labels import canonicalize,file_sha,spatial_targets
from scripts.evaluation.run_spatial_stes import losses,spatial_metrics,SpatialClips


def make_project(tmp_path,rows):
    source=tmp_path/"source.json";source.write_text("{}")
    events=[];reviews={}
    for identity,frame,label,xy,status in rows:
        events.append(dict(id=identity,video_id="Khe4P8QcJkg_5_4k",original_frame=frame,label=label,annotation_ids=[identity]))
        reviews[identity]=dict(frame=frame,label=label,xy=xy,xy_normalized=[xy[0]/3840,xy[1]/2160] if xy else None,status=status,timing="clear",reviewer="human",verified=True,spatial_valid=status=="visible",temporal_valid=status!="wrong_event")
    raw=dict(manifest_path=str(source),manifest_sha256=file_sha(source),sources={},events=events,reviews=reviews,
        videos={"Khe4P8QcJkg_5_4k":dict(width=3840,height=2160,num_frames=5000)})
    path=tmp_path/"review.json";path.write_text(json.dumps(raw));return path


def test_duplicate_median_and_provenance(tmp_path):
    path=make_project(tmp_path,[("a",30,"pass",[100,200],"visible"),("b",30,"pass",[104,202],"visible"),("c",50,"attack",None,"occluded"),("d",80,"serve",None,"wrong_event")])
    original=path.read_bytes();p=canonicalize(path)
    assert p["audit"]==dict(reviewed=4,temporal_events=2,spatial_events=1,duplicate_rows_removed=1)
    first=p["events"]["Khe4P8QcJkg_5_4k"][0]
    assert first["xy"]==[102,201] and first["annotation_ids"]==["a","b"]
    xy,mask=spatial_targets(100,p["events"]["Khe4P8QcJkg_5_4k"])
    assert mask.sum()==1 and mask[30] and not mask[50]
    assert path.read_bytes()==original


def test_unknown_conflict_fails(tmp_path):
    path=make_project(tmp_path,[("a",30,"pass",[100,200],"visible"),("b",30,"attack",[100,200],"visible")])
    with pytest.raises(ValueError,match="Unresolved class conflict"):canonicalize(path)


def test_authorized_reception_conflict(tmp_path):
    path=make_project(tmp_path,[("Khe4P8QcJkg_5_4k:3759:reception",3761,"reception",[100,200],"visible"),("Khe4P8QcJkg_5_4k:3765:pass",3761,"pass",[100,200],"visible")])
    p=canonicalize(path)
    assert p["events"]["Khe4P8QcJkg_5_4k"][0]["type"]=="reception"
    assert p["merges"][0]["resolution"]


def test_only_verified_coordinates_receive_spatial_gradient():
    logits=torch.zeros(1,4,5,requires_grad=True);loc=torch.zeros(1,4,2,requires_grad=True)
    y=torch.tensor([[0,1,-100,2]]);xy=torch.ones(1,4,2)*.2;mask=torch.tensor([[False,True,False,False]])
    cls,spatial=losses(dict(im_feat=logits,loc_feat=loc),y,xy,mask)
    (cls+spatial).backward()
    assert loc.grad[0,1].abs().sum()>0
    assert loc.grad[0,[0,2,3]].abs().sum()==0
    assert logits.grad[0,2].abs().sum()==0
    cls,spatial=losses(dict(im_feat=logits,loc_feat=loc),torch.full_like(y,-100),xy,torch.zeros_like(mask))
    assert torch.isfinite(cls+spatial) and float(cls+spatial)==0


def test_horizontal_flip_transforms_verified_points(monkeypatch):
    rec=dict(stem="a",frames=64,events=[dict(frame=3,type="serve",spatial_valid=True,xy_normalized=[.2,.3])],ignored=[],ignored_exact=[],cache="unused")
    reader=SpatialClips([rec])
    monkeypatch.setattr(reader,"read",lambda *args:np.zeros((64,3,8,8),np.uint8))
    monkeypatch.setattr("random.random",lambda:0.)
    _,_,xy,mask=reader.sample(1)
    assert xy[0,3]==pytest.approx([.8,.3]) and mask.sum()==1


def test_coordinate_error_units():
    coords=np.zeros((5,2));coords[3]=[.5,.5]
    result=spatial_metrics(coords,[dict(frame=3,spatial_valid=True,xy_normalized=[.5,.5])],3840,2160)
    assert result["mean_error_224"]==0 and result["within_2px_224"]==1


def test_author_spatial_head_backward_cpu():
    from action_recognition.models.vnl_stes_model import build_author_stes
    model=build_author_stes(pretrained=False,clip_len=4,predict_location=True).eval()
    result=model(torch.randn(1,4,3,32,32))
    assert result["loc_feat"].shape==(1,4,2)
    result["loc_feat"].sigmoid().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model._pred_loc.parameters())
