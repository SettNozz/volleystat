import json

import pytest

from action_recognition.rally_spotting.spatial_labels import canonicalize, file_sha, spatial_targets


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
