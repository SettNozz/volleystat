from copy import deepcopy
import json

import numpy as np
import pytest

from action_recognition.annotator.contact_author_dataset import fingerprint_file,sha,write_new
from action_recognition.annotator.contact_author_inference_viewer import create_app
from src.tracking.contact_author_inference import model_roster,prepare_input,strip_model_identities
from src.tracking.contact_author_features import extract_contact


def pose(raw=1):
    return {"id":raw,"team":"b","bb":[20,10,80,180],"kp":[[40+j,50+j,.9] for j in range(17)],
            "player_id":"P99","identity":{"method":"old_model"}}


def source():
    return {"fps":30.,"frame_width":200,"frame_height":200,"frames":[
        {"f":f,"poses":[pose()],"ball":[55,65]} for f in range(30)],"player_identity":{
            "enabled":True,"side":"b","seed":{"frame":0,"assignments":{"0:0":"P1"},
            "label_source":"automatic_initialisation","verified":False},"gallery_keys":{"P1":["0:0"]}}}


def test_old_model_identity_cannot_change_new_numeric_contact_features():
    annotations=source();before=deepcopy(annotations)
    clean=strip_model_identities(annotations["frames"])
    assert annotations==before and "player_id" not in clean[0]["poses"][0]
    first=extract_contact({"frame":15,"type":"pass"},{r["f"]:r for r in clean},30,(200,200))[0]
    annotations["frames"][15]["poses"][0]["player_id"]="P2"
    second=extract_contact({"frame":15,"type":"pass"},{r["f"]:r for r in strip_model_identities(annotations["frames"])},30,(200,200))[0]
    np.testing.assert_array_equal(first,second)


def test_inference_keeps_automatic_roster_and_has_no_fabricated_human_targets(tmp_path,monkeypatch):
    annotations=source();roster=model_roster(annotations)
    assert roster["players"][0]["player_id"]=="P1" and not roster["verified"]
    annotations["player_identity"]["gallery_keys"]["P2"]=["1:0"]
    with pytest.raises(ValueError,match="introductions"):model_roster(annotations)
    annotations=source();video=tmp_path/"video.webm";video.write_bytes(b"immutable-video")
    annotations["original_video_path"]=str(video);annotations["model_cache"]={"source":{"sha256":sha(video)}}
    ann=tmp_path/"ann.json";stats=tmp_path/"stats.json";write_new(ann,annotations)
    write_new(stats,{"all_events":[{"frame":15,"type":"pass","source":"test_model"}]})
    originals={p:sha(p) for p in (video,ann,stats)}
    def colours(metadata,requests,path):
        np.savez_compressed(path,unused=np.array([1]))
        return {key:(np.ones(384)/np.sqrt(384),np.ones(384)/np.sqrt(384)) for rows in requests.values() for key,_ in rows}
    monkeypatch.setattr("src.tracking.contact_author_inference.capture_colours",colours)
    output=tmp_path/"inputs";prepare_input(video,ann,stats,output)
    assert not (output/"human_labels.json").exists()
    snapshot=json.loads((output/"inference_inputs.json").read_bytes())
    assert not snapshot["human_labels"] and snapshot["decisions"]==[]
    assert all("author" not in c and not c["timing_verified"] for c in snapshot["contacts"])
    arrays=np.load(output/"features.npz");assert arrays["targets"].tolist()==[-1] and not arrays["strict"].any()
    assert originals=={p:sha(p) for p in originals}
    with pytest.raises(ValueError,match="fresh"):prepare_input(video,ann,stats,output)


@pytest.fixture
def review(tmp_path):
    video=tmp_path/"video.webm";video.write_bytes(bytes(range(256)))
    annotation=tmp_path/"annotations.json";write_new(annotation,source())
    dataset=tmp_path/"dataset";dataset.mkdir();output=tmp_path/"predictions";output.mkdir()
    metadata={"video_source":fingerprint_file(video),"annotations_source":fingerprint_file(annotation),
              "fps":30.,"width":200,"height":200,"num_frames":30}
    contact={"id":"one","video_id":"video","frame":15,"type":"pass"}
    snapshot={"videos":{"video":metadata},"rosters":{"video":model_roster(source())},"contacts":[contact]}
    write_new(dataset/"inference_inputs.json",snapshot)
    write_new(dataset/"manifest.json",{"snapshot_file":"inference_inputs.json","snapshot_sha256":sha(dataset/"inference_inputs.json")})
    actor={"frame":15,"pose_index":0,"raw_track_id":1,"bbox":pose()["bb"]}
    records=[{**contact,"truth":None,"predictions":{"class_fusion_ensemble":{
        "label":"P1","label_source":"derived","verified":False,"actor":actor,"score":.8,"ensemble_agreement":1.}}}]
    write_new(output/"predictions.json",records)
    write_new(output/"report.json",{"schema":"contact_author_unlabelled_inference_v1","dataset":str(dataset),
              "dataset_manifest_sha256":sha(dataset/"manifest.json"),"predictions_sha256":sha(output/"predictions.json"),
              "metrics":None,"method":"class_fusion_ensemble"})
    return output,annotation


def test_unlabelled_viewer_exposes_saved_evidence_without_label_write_routes(review):
    directory,_=review;client=create_app(directory).test_client()
    assert client.get("/").status_code==200
    assert client.get("/api/report").get_json()["report"]["metrics"] is None
    frames=client.get("/api/frames/video?start=14&end=16").get_json()["frames"]
    assert [r["f"] for r in frames]==[14,15,16] and frames[1]["poses"][0]["id"]==1
    assert client.get("/api/frames/video?start=20&end=1").status_code==400
    assert client.get("/api/frames/unknown").status_code==400
    assert client.get("/video/video",headers={"Range":"bytes=0-15"}).status_code==206
    assert client.post("/api/label",json={"player_id":"P2"}).status_code in (404,405)


def test_unlabelled_viewer_rejects_source_or_prediction_drift(review):
    directory,_=review
    (directory/"predictions.json").write_text("[]")
    with pytest.raises(ValueError,match="artifacts changed"):create_app(directory)


def test_unlabelled_viewer_rejects_source_changes(review):
    directory,annotation=review;annotation.write_text("{}")
    with pytest.raises(ValueError,match="Source observations changed"):create_app(directory)


def test_unlabelled_viewer_cannot_present_a_prediction_as_human_truth(review):
    directory,_=review;path=directory/"predictions.json";records=json.loads(path.read_bytes())
    records[0]["truth"]="P1";path.write_text(json.dumps(records))
    report_path=directory/"report.json";report=json.loads(report_path.read_bytes());report["predictions_sha256"]=sha(path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="no human truth"):create_app(directory)
