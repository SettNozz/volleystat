"""Contact-author review preserves source provenance and person identity meaning."""
from copy import deepcopy
import json
import sqlite3

import cv2
import numpy as np
import pytest

from action_recognition.annotator.contact_author_dataset import (
    AnnotationStore, Conflict, SCHEMA, contact_queue, fingerprint_file, initial_roster, write_new,
)
from action_recognition.annotator.contact_author_annotator import create_app


@pytest.fixture
def project_dir(tmp_path):
    source = tmp_path/"manifest.json"; source.write_text('{"label_source":"human"}')
    video = tmp_path/"game.avi"
    writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),30.,(200,100))
    assert writer.isOpened()
    for f in range(20): writer.write(np.full((100,200,3),f*10,dtype=np.uint8))
    writer.release()
    roster = {"confirmed":False,"players":[{"player_id":f"P{n}","identity_confirmation":None,
        "references":[{"frame":2,"bbox":[n*20,10,n*20+15,90],"label_source":"model"}]} for n in range(1,7)]}
    project = {"schema":SCHEMA,"sources":[fingerprint_file(source)],"videos":{"game":{
        "video_id":"game","name":"game","fps":30.,"width":200,"height":100,"num_frames":20,
        "target_side":"b","video_source":fingerprint_file(video),"initial_roster":roster,"contact_count":4}},
        "contacts":[{"id":f"c{n}","video_id":"game","frame":n+2,"type":"pass","timing_verified":True,
                     "annotation_ids":[f"ann{n}"],"clip_ids":[f"clip{n}"],"author_label_source":None} for n in range(4)],
        "excluded":[],"source_scope":{"clips":4},"evaluation_scope":"development_on_event_model_training_sources"}
    directory = tmp_path/"project"; write_new(directory/"project.json",project)
    AnnotationStore(directory).initialize()
    return directory


def confirm(store):
    return store.roster_action({"video_id":"game","action":"confirm","revision":store.load()["revision"]})


def test_queue_uses_audited_time_merges_duplicates_and_retains_links():
    clips = [{"clip_id":f"clip{n}","source_stem":"game","annotation_ids":[f"ann{n}"],
              "contact_frame":10+n,"primary_event_type":"pass","full_event_types":["pass"]} for n in range(4)]
    events = [{"id":f"e{n}","video_id":"game","annotation_ids":[f"ann{n}"]} for n in range(4)]
    review = {"events":events,"reviews":{
        "e0":{"verified":True,"temporal_valid":True,"frame":20},
        "e1":{"verified":True,"temporal_valid":True,"frame":20},
        "e2":{"verified":True,"status":"wrong_event"},
        "e3":{"verified":False,"frame":40}}}
    rows,excluded = contact_queue({"clips":clips},review)
    assert len(rows)==2 and len(excluded)==1
    assert rows[0]["frame"]==13 and not rows[0]["timing_verified"]
    assert rows[1]["frame"]==20 and rows[1]["original_frames"]==[10,11]
    assert rows[1]["annotation_ids"]==["ann0","ann1"]
    assert rows[1]["author_label_source"] is None
    review["events"][1]["annotation_ids"] = ["ann0"]
    review["reviews"]["e1"]["frame"] = 21
    with pytest.raises(ValueError,match="different audited"): contact_queue({"clips":clips},review)


def test_initial_roster_six_clear_people_no_automatic_confirmation():
    def person(n,team="team_b"):
        return {"id":n,"team":team,"bb":[n*40,0,n*40+25,80],"kp":[[n*40,30,.9]]*17}
    frame = {"f":100,"poses":[person(i) for i in range(6)]+[person(7,"team_a")]}
    result = initial_roster({"frames":[frame]},100)
    assert not result["confirmed"] and len(result["players"])==6
    assert [p["player_id"] for p in result["players"]]==[f"P{i}" for i in range(1,7)]
    assert all(p["identity_confirmation"] is None for p in result["players"])
    assert all(p["references"][0]["label_source"]=="model" for p in result["players"])


def test_author_labels_require_roster_and_valid_person(project_dir):
    store = AnnotationStore(project_dir)
    assert store.load()["labels"]=={}
    with pytest.raises(ValueError,match="Confirm"): store.label({"revision":0,"contact_id":"c0","decision":"player","player_id":"P1"})
    confirm(store)
    for decision,pid in (("player","P99"),("other_team","P1"),("bad",None)):
        with pytest.raises(ValueError): store.label({"revision":1,"contact_id":"c0","decision":decision,"player_id":pid})
    assert store.load()["revision"]==1
    assert store.load()["labels"]=={}


def test_confirmed_ids_cannot_change_person_and_substitutes_get_new_ids(project_dir):
    store = AnnotationStore(project_dir)
    payload = {"video_id":"game","player_id":"P1","frame":3,"bbox":[2,2,40,90],"action":"replace_reference","revision":0}
    store.roster_action(payload)
    confirm(store)
    with pytest.raises(Conflict,match="original reference"): store.roster_action({**payload,"revision":2})
    with pytest.raises(ValueError,match="same existing person"):
        store.roster_action({**payload,"action":"add_view","revision":2})
    store.roster_action({**payload,"action":"add_view","same_person_confirmed":True,"revision":2})
    references = store.load()["rosters"]["game"]["players"][0]["references"]
    assert len(references)==2 and references[0]["frame"]==3
    with pytest.raises(ValueError,match="new person"):
        store.roster_action({**payload,"action":"add_player","revision":3})
    for n in (7,8):
        result = store.roster_action({**payload,"action":"add_player","new_person_confirmed":True,"revision":n-4})
        assert result["rosters"]["game"]["players"][-1]["player_id"]==f"P{n}"
    assert len(store.load()["rosters"]["game"]["players"])==8


@pytest.mark.parametrize("box",[[-1,0,20,30],[0,0,300,90],[0,0,2,3],[0,0,float('nan'),30],None])
def test_bad_reference_boxes_rejected(project_dir,box):
    store = AnnotationStore(project_dir)
    with pytest.raises(ValueError): store.roster_action({"video_id":"game","action":"replace_reference","player_id":"P1","frame":2,"bbox":box,"revision":0})
    assert store.load()["revision"]==0


def test_restart_stale_tab_history_and_export_selection(project_dir):
    store = AnnotationStore(project_dir); confirm(store)
    for n,(decision,pid) in enumerate((("player","P2"),("other_team",None),("unsure",None),("not_contact",None))):
        store.label({"revision":n+1,"contact_id":f"c{n}","decision":decision,"player_id":pid})
    resumed = AnnotationStore(project_dir); resumed.initialize()
    assert resumed.load()["revision"]==5
    with pytest.raises(Conflict,match="another tab"): resumed.label({"revision":1,"contact_id":"c0","decision":"reset"})
    export = resumed.export()
    assert len(export["contacts"])==2 and export["contacts"][0]["author_key"]=="game:P2"
    assert export["contacts"][1]["team"]=="a"
    assert not export["decisions"]["c2"]["verified"]
    assert export["decisions"]["c3"]["verification_scope"]=="contact_presence"
    assert len(export["queue"])==4 and not export["training_promotion"]
    assert export["contacts"][0]["annotation_ids"]==["ann0"]
    with sqlite3.connect(resumed.path) as con: assert con.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]==5
    resumed.label({"revision":5,"contact_id":"c0","decision":"reset"})
    assert len(resumed.export()["contacts"])==1


def test_changed_source_blocks_write_without_losing_labels(project_dir):
    store = AnnotationStore(project_dir); confirm(store)
    source = store.project["sources"][0]["path"]
    with open(source,"a") as stream: stream.write("changed")
    with pytest.raises(Conflict,match="Source changed"): store.label({"revision":1,"contact_id":"c0","decision":"other_team"})
    assert store.load()["revision"]==1


def test_project_fingerprint_binds_existing_labels(project_dir):
    store = AnnotationStore(project_dir); confirm(store)
    project = deepcopy(store.project); project["contacts"][0]["frame"] = 8
    store.project_path.write_text(json.dumps(project))
    with pytest.raises(Conflict): store.check_sources()
    with pytest.raises(Conflict,match="another contact queue"): AnnotationStore(project_dir).initialize()


def test_http_video_portrait_blind_queue_save_and_export(project_dir):
    app = create_app(project_dir); client = app.test_client()
    assert client.get("/").status_code==200
    assert client.get("/contact_author.js").status_code==200
    project = client.get("/api/project").get_json()
    assert project["state"]["labels"]=={}
    assert "predicted_author" not in json.dumps(project)
    assert client.get("/video/game",headers={"Range":"bytes=0-99"}).status_code==206
    image = client.get("/portrait/game/P1/0")
    assert image.status_code==200 and image.mimetype=="image/jpeg"
    assert client.get("/portrait/game/P99/0").status_code==400
    assert client.post("/api/roster",json={"video_id":"game","action":"confirm","revision":0}).status_code==200
    payload = {"revision":1,"contact_id":"c0","decision":"player","player_id":"P3"}
    assert client.post("/api/label",json=payload).status_code==200
    assert client.post("/api/label",json=payload).status_code==409
    assert client.get("/api/project").get_json()["progress"]["game"]["reviewed"]==1
    export = client.get("/api/export")
    assert export.status_code==200 and "attachment" in export.headers["Content-Disposition"]
    assert export.get_json()["contacts"][0]["author_key"]=="game:P3"
