import json
import cv2
import numpy as np
import pytest
from action_recognition.annotator import contact_spatial_annotator as m

@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    video = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64,48))
    for i in range(10):
        writer.write(np.full((48,64,3), i*20, dtype=np.uint8))
    writer.release()
    source = tmp_path / "data/full_video_gold/tiny/annotations_v1.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"annotations":[{"annotation_id":"a", "video_id":"tiny", "event_type":"set", "frame":4, "label_source":"human"}]}))
    manifest = tmp_path/"manifest.json"
    clip = dict(source_stem="tiny", source_video=str(video), fps=25, contact_frame=4, primary_event_type="set", event_types=["set"], annotation_ids=["a"])
    manifest.write_text(json.dumps({"clips":[clip,clip]}))
    output = tmp_path/"reviews.json"
    app = m.create_app(manifest, output)
    return app.test_client(), manifest, output, source

def payload(client):
    p=client.get("/api/project").json
    return dict(id=p["events"][0]["id"], revision=0, frame=4, xy=[20,15], status="visible", timing="clear", label="pass", reviewer="Illya")

def test_save_resume_export_and_original_untouched(setup):
    client, manifest, output, source = setup
    original=source.read_bytes()
    p=client.get("/api/project").json
    assert len(p["events"])==1
    data=payload(client)
    r=client.post("/api/review", json=data)
    assert r.status_code==200 and r.json["spatial_valid"]
    assert r.json["xy_normalized"]==[20/64,15/48]
    assert source.read_bytes()==original
    assert client.post("/api/review",json=data).status_code==409
    resumed=m.create_app(manifest,output).test_client()
    assert len(resumed.get("/api/project").json["reviews"])==1
    data.update(revision=1,label="attack",status="occluded",xy=None)
    r=resumed.post("/api/review",json=data)
    assert r.json["temporal_valid"] and not r.json["spatial_valid"]
    assert r.json["xy"] is None
    exported=json.loads(resumed.get("/api/export").data)
    assert exported["events"][0]["label"]=="attack"
    assert len(json.loads(output.read_text())["history"])==2

@pytest.mark.parametrize("patch",[{"xy":[64,15]},{"xy":[-1,0]},{"xy":[float('nan'),0]},{"frame":-1},{"frame":10},{"frame":True},{"reviewer":""},{"status":""},{"xy":None}])
def test_invalid_data_not_saved(setup,patch):
    client,_,output,_=setup
    data=payload(client);data.update(patch)
    assert client.post("/api/review",json=data).status_code==400
    assert not output.exists()

def test_uncertain_timing_masks_spatial_loss(setup):
    client,*_=setup
    data=payload(client);data["timing"]="uncertain"
    r=client.post("/api/review",json=data).json
    assert not r["spatial_valid"] and not r["temporal_valid"]

def test_exact_native_frame_and_bounds(setup):
    client,*_=setup
    for frame in [8,2,3,0]:
        r=client.get(f"/api/frame/tiny/{frame}")
        assert r.status_code==200
        image=cv2.imdecode(np.frombuffer(r.data,np.uint8),cv2.IMREAD_COLOR)
        assert image.shape==(48,64,3)
        assert abs(float(image.mean())-frame*20)<4
    assert client.get("/api/frame/tiny/10").status_code==400
    assert client.get("/api/frame/unknown/0").status_code==400

def test_changed_source_requires_new_project(setup):
    client,manifest,output,source=setup
    client.post("/api/review",json=payload(client))
    source.write_text(source.read_text()+" ")
    with pytest.raises(ValueError,match="Source labels changed"):
        m.create_app(manifest,output)

@pytest.mark.parametrize("status", ["out_of_frame", "uncertain", "wrong_event"])
def test_unusable_coordinates_are_null(setup,status):
    client,*_=setup
    data=payload(client);data["status"]=status
    r=client.post("/api/review",json=data).json
    assert r["xy"] is None and not r["spatial_valid"]
    assert r["ball_out_of_frame"] == (status=="out_of_frame")
    assert r["temporal_valid"] == (status!="wrong_event")
