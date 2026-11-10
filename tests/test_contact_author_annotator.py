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


def test_contact_frame_review_adds_human_contact_and_preserves_original_proposal(project_dir):
    client = create_app(project_dir).test_client()
    frame_url = "/api/contact-frame"
    added = {"video_id": "game", "action": "add", "type": "dig", "frame": 7, "revision": 0}
    assert client.post(frame_url, json=added).get_json()["progress"]["game"]["total"] == 5
    assert client.post(frame_url, json={**added, "revision": 1}).status_code == 400
    timing = {"video_id": "game", "action": "timing", "contact_id": "c0", "frame": 8, "revision": 1}
    assert client.post(frame_url, json=timing).status_code == 200
    store = AnnotationStore(project_dir)
    confirm(store)
    store.label({"contact_id": "game:human:r1", "decision": "player", "player_id": "P1", "revision": 3})
    exported = store.export()
    contact = exported["contacts"][0]
    assert (contact["frame"], contact["type"], contact["author_key"]) == (7, "dig", "game:P1")
    assert contact["event_verified"] and contact["timing_verified"]
    corrected = next(row for row in exported["queue"] if row["id"] == "c0")
    assert corrected["frame"] == 8 and corrected["timing_source"] == "human"
    assert store.contacts["c0"]["frame"] == 2


def test_movement_review_splits_track_and_rejects_overlapping_player_id(project_dir, tmp_path):
    project = json.loads((project_dir / "project.json").read_bytes())
    annotation = tmp_path / "poses.json"
    a = {"id": 10, "team": "b", "bb": [2, 2, 20, 80], "player_id": "P4"}
    b = {**a, "id": 20, "bb": [25, 2, 45, 80]}
    write_new(annotation, {"frames": [{"f": 0, "poses": [a, b]}, {"f": 1, "poses": [b, a]},
                                      {"f": 2, "poses": []}, {"f": 3, "poses": [a]}]})
    source = fingerprint_file(annotation)
    project["sources"].append(source)
    project["videos"]["game"]["annotations_source"] = source
    directory = tmp_path / "movement"
    write_new(directory / "project.json", project)
    client = create_app(directory).test_client()
    queue = client.get("/api/movement/game").get_json()["segments"]
    assert [(row["id"], row["end"]) for row in queue] == [("0:0", 1), ("0:1", 1), ("3:0", 3)]
    assert client.get("/api/movement/game/0:0").get_json() == {"boxes": [a["bb"], a["bb"]]}
    url = "/api/movement"
    label = {"video_id": "game", "segment_id": "0:0", "decision": "player", "player_id": "P1", "end": 0}
    assert client.post(url, json={**label, "revision": 0}).status_code == 400
    store = AnnotationStore(directory)
    confirm(store)
    assert client.post(url, json={**label, "revision": 1}).status_code == 200
    conflict = client.post(url, json={**label, "revision": 2, "segment_id": "0:1"})
    assert conflict.status_code == 400
    assert conflict.get_json()["overlap"] == {"frame": 0, "bbox": a["bb"]}
    assert client.post(url, json={**label, "revision": 2, "end": 1, "player_id": "P2"}).status_code == 200
    assert client.post(url, json={**label, "revision": 2, "decision": "undo"}).status_code == 409
    rows = store.export()["movement"]["videos"]["game"]["spans"]
    assert [(row["player_id"], row["start"], row["pose_indices"]) for row in rows] == [("P1", 0, [0]), ("P2", 1, [1])]
    assert client.post(url, json={**label, "revision": 3, "decision": "undo"}).status_code == 200
    assert len(store.export()["movement"]["videos"]["game"]["spans"]) == 1
    false_positive = {**label, "decision": "false_positive", "revision": 4, "end": 1}
    assert client.post(url, json=false_positive).status_code == 400  # A false detection has no player.
    assert client.post(url, json={**false_positive, "player_id": None}).status_code == 200
    resumed = AnnotationStore(directory).export()["movement"]["videos"]["game"]["spans"]
    assert [(row["decision"], row["player_id"]) for row in resumed] == [("player", "P1"), ("false_positive", None)]
    assert resumed[-1]["verified"] and resumed[-1]["pose_indices"] == [1]
    duplicate = {**label, "segment_id": "0:1", "revision": 5, "end": 1, "confirm_overlap": True}
    assert client.post(url, json=duplicate).status_code == 200
    rows = store.export()["movement"]["videos"]["game"]["spans"]
    assert "duplicate_frames" not in rows[0]
    assert rows[-1]["player_id"] == "P1" and rows[-1]["duplicate_frames"] == [0]
    for revision in (6, 7):
        assert client.post(url, json={**label, "revision": revision, "decision": "undo"}).status_code == 200
    remaining = store.export()["movement"]["videos"]["game"]["spans"]
    assert len(remaining) == 1 and "duplicate_frames" not in remaining[0]


def test_contact_type_correction_preserves_author_source_and_timing_and_can_be_reset(project_dir):
    store=AnnotationStore(project_dir);confirm(store)
    store.label({"contact_id":"c0","decision":"player","player_id":"P2","revision":1})
    original_author=deepcopy(store.load()["labels"]["c0"]);original_project=(project_dir/"project.json").read_bytes()
    store.event_action({"contact_id":"c0","action":"set","type":"attack","revision":2})
    store.event_action({"contact_id":"c0","action":"set","type":"set","revision":3})
    resumed=AnnotationStore(project_dir);row=resumed.export()["contacts"][0]
    assert row["type"]=="set" and row["raw_types"]==["set"]
    assert row["author"]==original_author and row["event_label_source"]=="human" and row["event_verified"]
    assert row["original_event"]["type"]=="pass" and row["timing_verified"]
    assert row["event_review"]["verification_scope"]=="contact_type_only"
    assert (project_dir/"project.json").read_bytes()==original_project
    with pytest.raises(Conflict):resumed.event_action({"contact_id":"c0","type":"block","revision":2})
    resumed.event_action({"contact_id":"c0","action":"reset","revision":4})
    reset=resumed.export()["contacts"][0]
    assert reset["type"]=="pass" and "event_review" not in reset and reset["author"]==original_author
    with resumed.connect() as con:assert con.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]==5


def test_type_review_is_independent_of_roster_confirmation_and_author_decision(project_dir):
    store=AnnotationStore(project_dir);store.event_action({"contact_id":"c0","type":"set","revision":0})
    exported=store.export()
    assert exported["contacts"]==[] and exported["queue"][0]["type"]=="set"
    assert exported["event_reviews"]["c0"]["label_source"]=="human"
    assert store.load()["labels"]=={} and not store.load()["rosters"]["game"]["confirmed"]


@pytest.mark.parametrize("event_type",["banana",None,[],{}])
def test_invalid_event_type_does_not_change_saved_state(project_dir,event_type):
    store=AnnotationStore(project_dir);before=store.load()
    with pytest.raises(ValueError,match="Unknown contact type"):store.event_action({"contact_id":"c0","type":event_type,"revision":0})
    assert store.load()==before


def test_event_review_api_export_keeps_model_timing_unverified(project_dir,tmp_path):
    project=json.loads((project_dir/"project.json").read_bytes());contact=project["contacts"][0]
    contact.update(type="attack",raw_types=["attack"],event_label_source="model",event_verified=False,
                   timing_verified=False,timing_source="model",core_type="attack")
    directory=tmp_path/"model_events";write_new(directory/"project.json",project)
    client=create_app(directory).test_client()
    response=client.post("/api/event",json={"contact_id":"c0","type":"set","revision":0})
    assert response.status_code==200 and response.get_json()["state"]["labels"]=={}
    row=client.get("/api/export").get_json()["queue"][0]
    assert row["type"]=="set" and row["event_verified"] and not row["timing_verified"]
    assert row["original_event"]["type"]=="attack" and row["original_event"]["event_label_source"]=="model"
    assert "core_type" not in row and row["original_event"]["core_type"]=="attack"
    assert client.post("/api/event",json={"contact_id":"c0","type":"block","revision":0}).status_code==409
    assert client.post("/api/event",json={"contact_id":"bad","type":"set","revision":1}).status_code==400


def test_remove_player_requires_explicit_confirmation_and_reserves_its_id(project_dir):
    store=AnnotationStore(project_dir);before=store.load()
    with pytest.raises(ValueError,match="Confirm player removal"):
        store.roster_action({"video_id":"game","action":"remove_player","player_id":"P6","revision":0})
    assert store.load()==before
    result=store.roster_action({"video_id":"game","action":"remove_player","player_id":"P6","removal_confirmed":True,"revision":0})
    assert [p["player_id"] for p in result["rosters"]["game"]["players"]]==[f"P{n}" for n in range(1,6)]
    assert result["labels"]=={} and result["rosters"]["game"]["removed_players"][0]["player"]["player_id"]=="P6"
    result=store.roster_action({"video_id":"game","action":"add_player","new_person_confirmed":True,
        "frame":3,"bbox":[2,2,40,90],"revision":1})
    assert result["rosters"]["game"]["players"][-1]["player_id"]=="P7"
    with pytest.raises(Conflict):
        store.roster_action({"video_id":"game","action":"remove_player","player_id":"P1","removal_confirmed":True,"revision":0})


def test_labelled_player_removal_is_reversible_and_does_not_overwrite_new_labels(project_dir):
    store=AnnotationStore(project_dir);confirm(store)
    store.label({"contact_id":"c0","decision":"player","player_id":"P6","revision":1})
    original=deepcopy(store.load()["labels"]["c0"])
    payload={"video_id":"game","action":"remove_player","player_id":"P6","removal_confirmed":True,"revision":2}
    with pytest.raises(Conflict,match="1 contact labels"):store.roster_action(payload)
    assert store.load()["labels"]["c0"]==original and store.load()["revision"]==2
    store.roster_action({**payload,"unassign_contacts_confirmed":True})
    assert store.export()["contacts"]==[]
    archive=store.load()["rosters"]["game"]["removed_players"][0]
    assert archive["unassigned_labels"]["c0"]==original
    store.roster_action({"video_id":"game","action":"restore_player","player_id":"P6","revision":3})
    assert store.load()["labels"]["c0"]==original
    store.roster_action({**payload,"unassign_contacts_confirmed":True,"revision":4})
    store.label({"contact_id":"c0","decision":"player","player_id":"P2","revision":5})
    store.roster_action({"video_id":"game","action":"restore_player","player_id":"P6","revision":6})
    assert store.load()["labels"]["c0"]["player_id"]=="P2"
    assert store.load()["rosters"]["game"]["restored_players"][-1]["unassigned_labels"]["c0"]==original


def test_delete_all_cards_can_add_new_person_without_reusing_old_ids(project_dir):
    store=AnnotationStore(project_dir)
    for n in range(1,7):
        store.roster_action({"video_id":"game","action":"remove_player","player_id":f"P{n}","removal_confirmed":True,"revision":n-1})
    assert store.load()["rosters"]["game"]["players"]==[]
    with pytest.raises(ValueError,match="Empty roster"):confirm(store)
    store.roster_action({"video_id":"game","action":"add_player","new_person_confirmed":True,
                        "frame":3,"bbox":[2,2,40,90],"revision":6})
    assert store.load()["rosters"]["game"]["players"][0]["player_id"]=="P7"


def test_removal_is_scoped_to_one_video_and_keeps_other_person_labels(project_dir,tmp_path):
    project=json.loads((project_dir/"project.json").read_bytes())
    project["videos"]["other"]=deepcopy(project["videos"]["game"])
    project["videos"]["other"].update(video_id="other",name="other")
    project["contacts"].append({"id":"other_contact","video_id":"other","frame":3,"type":"pass","timing_verified":True})
    path=tmp_path/"two_sources";write_new(path/"project.json",project);store=AnnotationStore(path);store.initialize()
    for video in ("game","other"):store.roster_action({"video_id":video,"action":"confirm","revision":store.load()["revision"]})
    for key in ("c0","other_contact"):store.label({"contact_id":key,"decision":"player","player_id":"P6","revision":store.load()["revision"]})
    store.roster_action({"video_id":"game","action":"remove_player","player_id":"P6","removal_confirmed":True,
                        "unassign_contacts_confirmed":True,"revision":store.load()["revision"]})
    assert set(store.load()["labels"])=={"other_contact"}
    assert store.export()["contacts"][0]["author_key"]=="other:P6"


def test_remove_and_restore_are_available_through_roster_api(project_dir):
    client=create_app(project_dir).test_client()
    removed=client.post("/api/roster",json={"video_id":"game","action":"remove_player","player_id":"P6",
                                          "removal_confirmed":True,"revision":0})
    assert removed.status_code==200 and len(removed.get_json()["state"]["rosters"]["game"]["players"])==5
    assert client.get("/portrait/game/P6/0").status_code==400
    restored=client.post("/api/roster",json={"video_id":"game","action":"restore_player","player_id":"P6","revision":1})
    assert restored.status_code==200 and len(restored.get_json()["state"]["rosters"]["game"]["players"])==6


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
