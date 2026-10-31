"""Read-only review is bound to frozen truth, predictions and source poses."""
import json

import cv2
import numpy as np
import pytest

from action_recognition.annotator.contact_author_benchmark_viewer import create_app
from action_recognition.annotator.contact_author_dataset import fingerprint_file, sha, write_new


@pytest.fixture
def review(tmp_path):
    video = tmp_path/"game.avi"
    writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),30.,(100,100))
    for _ in range(5): writer.write(np.full((100,100,3),128,np.uint8))
    writer.release()
    source = tmp_path/"annotations.json"
    write_new(source,{"frames":[{"f":2,"poses":[{"id":1,"bb":[10,10,40,90],"kp":[[20,30,.9]]*17}],"ball":[30,30]}]})
    dataset = tmp_path/"dataset"; directory = tmp_path/"evaluation"
    snapshot = {"contacts":[{"id":"c","frame":2,"video_id":"game","author":{"player_id":"P1"}}],
        "videos":{"game":{"fps":30.,"width":100,"height":100,"num_frames":5,"video_id":"game",
                            "video_source":fingerprint_file(video),"annotations_source":fingerprint_file(source)}},
        "rosters":{"game":{"players":[{"player_id":"P1","references":[{"frame":2,"bbox":[10,10,40,90]}]}]}}}
    write_new(dataset/"human_labels.json",snapshot)
    write_new(dataset/"manifest.json",{"snapshot_sha256":sha(dataset/"human_labels.json")})
    records = [{"id":"c","frame":2,"video_id":"game","truth":"P1","predictions":{"method":{
        "label":"P1","label_source":"derived","verified":False,"actor":{"frame":2,"pose_index":0,"raw_track_id":1}}}}]
    write_new(directory/"predictions.json",records)
    write_new(directory/"comparison.json",{"dataset":str(dataset),"dataset_manifest_sha256":sha(dataset/"manifest.json"),
                                          "predictions_sha256":sha(directory/"predictions.json")})
    return directory,source


def test_review_serves_original_evidence_and_has_no_label_write_endpoint(review):
    directory,_ = review; app = create_app(directory); client = app.test_client()
    assert client.get("/").status_code==200
    assert client.get("/api/report").get_json()["records"][0]["truth"]=="P1"
    observed = client.get("/api/frame/game/2").get_json()
    assert observed["poses"][0]["id"]==1 and len(observed["poses"][0]["kp"])==17
    assert client.get("/api/frame/game/3").get_json()["poses"]==[]
    assert client.get("/photo/game/P1").mimetype=="image/jpeg"
    assert client.get("/video/game",headers={"Range":"bytes=0-99"}).status_code==206
    assert client.post("/api/label",json={"player_id":"P2"}).status_code in (404,405)
    assert client.get("/api/frame/unknown/2").status_code==400


def test_review_rejects_prediction_drift_and_source_pose_changes(review):
    directory,source = review
    original = (directory/"predictions.json").read_bytes()
    (directory/"predictions.json").write_bytes(original+b" ")
    with pytest.raises(ValueError,match="predictions changed"): create_app(directory)
    (directory/"predictions.json").write_bytes(original)
    source.write_text("{}")
    with pytest.raises(ValueError,match="annotations changed"): create_app(directory)
