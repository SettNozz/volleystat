from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from action_recognition.annotator.contact_author_dataset import AnnotationStore,sha,write_new
from action_recognition.annotator.contact_author_expansion import excluded_sources,model_contacts,prepare_expansion,proposed_roster


def test_model_queue_keeps_provenance_merges_duplicates_and_never_prefills_authors():
    events=[{"frame":10,"type":"pass","confidence":.8,"source":"dense_contact_spotter","player_id":"P4"},
            {"frame":10,"type":"pass","confidence":.7,"source":"dense_contact_spotter"},
            {"frame":20,"type":"attack","confidence":.9}, {"frame":21,"type":"rally_end"}]
    before=deepcopy(events);rows,excluded=model_contacts("new",events,{10,20,21})
    assert len(rows)==2 and rows[0]["event_indices"]==[0,1] and len(excluded)==1
    assert events==before and "player_id" not in rows[0] and "author" not in rows[0]
    assert all(r["event_label_source"]=="model" and not r["timing_verified"] and r["author_label_source"] is None for r in rows)
    with pytest.raises(ValueError,match="outside"):model_contacts("new",[{"frame":11,"type":"pass"}],{10})


def test_exclusions_cover_training_names_author_names_and_renamed_content():
    names,hashes=excluded_sources({"clips":[{"source_stem":"train","source_video":"C:\\data\\train_alias.webm"}]},
                                {"videos":{"old":{"video_source":{"path":"old_alias.webm","sha256":"fingerprint"}}}})
    assert names=={"train","train_alias","old","old_alias"} and hashes=={"fingerprint"}


@pytest.fixture
def saved(tmp_path):
    video=tmp_path/"new.avi";writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),30.,(200,100))
    assert writer.isOpened()
    for n in range(20):writer.write(np.full((100,200,3),n*10,np.uint8))
    writer.release()
    poses=[{"id":n,"team":"b","bb":[n*30,10,n*30+20,90],"kp":[[n*30,30,.9]]*17} for n in range(6)]
    annotations=tmp_path/"new_annotations.json";stats=tmp_path/"new_stats.json"
    write_new(annotations,{"original_video_path":str(video),"fps":30.,"frame_width":200,"frame_height":100,
                           "frames":[{"f":f,"poses":deepcopy(poses)} for f in range(20)]})
    write_new(stats,{"contact_spotter":{"enabled":True},"all_events":[{"frame":8,"type":"pass","source":"dense_contact_spotter"}]})
    manifest=tmp_path/"training.json";authors=tmp_path/"authors.json"
    write_new(manifest,{"clips":[{"source_stem":"train"}]});write_new(authors,{"videos":{"old":{"video_source":{"sha256":"other"}}}})
    return stats,annotations,video,manifest,authors


def test_prepared_project_is_blind_resumable_and_author_verification_does_not_verify_model_timing(saved,tmp_path):
    stats,annotations,video,manifest,authors=saved;originals={p:sha(p) for p in saved}
    output=tmp_path/"expansion";prepare_expansion([stats],manifest,authors,output)
    store=AnnotationStore(output);assert store.load()["labels"]=={} and store.load()["revision"]==0
    assert not store.load()["rosters"]["new"]["confirmed"]
    store.roster_action({"video_id":"new","action":"confirm","revision":0})
    contact=store.project["contacts"][0]
    store.label({"contact_id":contact["id"],"decision":"player","player_id":"P1","revision":1})
    resumed=AnnotationStore(output);export=resumed.export();row=export["contacts"][0]
    assert row["author"]["label_source"]=="human" and row["author"]["verified"]
    assert row["event_label_source"]=="model" and not row["timing_verified"] and not row["event_verified"]
    assert not export["training_promotion"] and len(export["queue"])==1
    assert originals=={p:sha(p) for p in saved}
    with pytest.raises(ValueError,match="fresh"):prepare_expansion([stats],manifest,authors,output)


def test_preparation_refuses_existing_dataset_or_same_content_under_a_different_name(saved,tmp_path):
    stats,_,video,manifest,authors=saved
    manifest.write_text(json.dumps({"clips":[{"source_stem":"new"}]}))
    with pytest.raises(ValueError,match="already in"):prepare_expansion([stats],manifest,authors,tmp_path/"named")
    manifest.write_text(json.dumps({"clips":[{"source_stem":"train"}]}))
    authors.write_text(json.dumps({"videos":{"different_name":{"video_source":{"sha256":sha(video)}}}}))
    with pytest.raises(ValueError,match="another name"):prepare_expansion([stats],manifest,authors,tmp_path/"renamed")


def test_saved_substitute_gallery_is_only_an_unconfirmed_photo_proposal(saved):
    _,annotations,_,_,_=saved;ann=json.loads(annotations.read_bytes())
    ann["player_identity"]={"side":"b","gallery_keys":{f"P{n+1}":[f"0:{n}"] for n in range(6)}}
    ann["player_identity"]["gallery_keys"]["P7"]=["10:0"]
    roster=proposed_roster(ann,8)
    assert len(roster["players"])==7 and not roster["confirmed"]
    assert all(p["identity_confirmation"] is None and p["references"][0]["label_source"]=="model" for p in roster["players"])
