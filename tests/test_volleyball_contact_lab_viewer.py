from copy import deepcopy
import json

import pytest
import viewer
from src.tracking.player_contacts import source_fingerprint
from src.tracking.volleyball_contact_lab import SCHEMA


@pytest.fixture
def lab(tmp_path):
    ann = tmp_path/"clip_annotations.json"; stat = tmp_path/"clip_stats.json"; path = tmp_path/"clip_contact_lab.json"
    ann.write_text(json.dumps({"video":"clip","fps":30,"frames":[]}),encoding="utf-8")
    stat.write_text(json.dumps({"video":"clip","fps":30,"all_events":[{"frame":30,"type":"pass"}]}),encoding="utf-8")
    bundle = {"schema":SCHEMA,"source":source_fingerprint(stat.read_bytes(),ann.read_bytes()),
              "selected_method":"joint","roster_ids":["P1"],"methods":{"joint":{"events":[
                  {"frame":32,"original_frame":30,"type":"pass","source_event_index":0,
                   "player_id":"P1","label_source":"derived","verified":False},
                  {"frame":90,"original_frame":90,"type":"contact","source_event_index":None,
                   "player_id":None,"label_source":"derived","verified":False}]}}}
    path.write_text(json.dumps(bundle),encoding="utf-8")
    viewer.jobs["contact-lab-test"]={"status":"done","stats_path":str(stat),"annotations_path":str(ann)}
    with viewer.app.test_client() as client:yield client,ann,stat,path,bundle
    viewer.jobs.pop("contact-lab-test",None)


def test_lab_is_read_only_and_supports_new_untyped_proposals(lab):
    client,ann,stat,path,bundle=lab
    original=(ann.read_bytes(),stat.read_bytes(),path.read_bytes())
    assert client.get('/contact-lab/contact-lab-test').status_code==200
    assert client.get('/api/contact_lab/contact-lab-test').get_json()==bundle
    assert client.post('/api/contact_lab/contact-lab-test',json={"verified":True}).status_code==405
    assert original==(ann.read_bytes(),stat.read_bytes(),path.read_bytes())


def test_missing_and_changed_sources_are_not_silently_reused(lab):
    client,ann,stat,path,bundle=lab
    assert client.get('/api/contact_lab/missing-job').status_code==404
    ann.write_bytes(ann.read_bytes()+b' ')
    assert client.get('/api/contact_lab/contact-lab-test').status_code==409
    path.unlink()
    assert client.get('/api/contact_lab/contact-lab-test').status_code==404


def test_original_video_is_used_when_no_annotated_render_exists(lab):
    client,*_=lab
    viewer.jobs['contact-lab-test']['original_video_path_url']='C%3A%2Fvideo.webm'
    page=client.get('/contact-lab/contact-lab-test').get_data(as_text=True)
    assert '/api/original_video?path=C%3A%2Fvideo.webm' in page
    assert 'src="/api/video/contact-lab-test"' not in page


def test_load_link_restores_lab_after_server_restart(lab):
    client,ann,stat,*_=lab
    response=client.get('/load',query_string={'video':'clip','results_dir':str(stat.parent),'contact_lab':'1'})
    assert response.status_code==302 and response.headers['Location'].startswith('/contact-lab/preloaded_clip_')
    assert client.get(response.headers['Location']).status_code==200
    viewer.jobs.pop(response.headers['Location'].rsplit('/',1)[-1],None)


@pytest.mark.parametrize('field',["verified","original_frame","source_event_index","player_id"])
def test_lab_rejects_fake_verification_or_changed_source_provenance(lab,field):
    client,ann,stat,path,bundle=lab
    event=bundle['methods']['joint']['events'][0]
    event[field]={'verified':True,'original_frame':99,'source_event_index':99,'player_id':'P999'}[field]
    path.write_text(json.dumps(bundle),encoding='utf-8')
    assert client.get('/api/contact_lab/contact-lab-test').status_code==409
