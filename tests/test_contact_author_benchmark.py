"""Human person metrics, source boundaries and roster-reference invariants."""
from copy import deepcopy
import json

import numpy as np
import torch

from action_recognition.annotator.contact_author_dataset import fingerprint_file, sha, write_new
from src.tracking.contact_author_benchmark import encode, pose_colour, track_segments, prepare_central
from src.tracking.contact_author_evaluation import PairRanker, fit_rankers, fixed_predict, metrics, prediction, ranker_scores, roster_assignment, sequence_scores


def person(raw=1,x=20):
    kp = [[x,30,.9] for _ in range(17)]
    for j,xy in {5:(x,10),6:(x+20,10),11:(x+2,45),12:(x+18,45),13:(x+2,75),14:(x+18,75)}.items():
        kp[j] = [*xy,.9]
    return {"id":raw,"bb":[x-5,0,x+25,95],"kp":kp,"team":"team_b"}


def case(target="P1",video="a"):
    return {"options":["other_team","P1","P2"],"target":["other_team","P1","P2"].index(target),
        "contact":{"video_id":video},"pairs":[
            {"class_index":j,"player_id":pid,"score":.1,"frame":30,"pose_index":0,"anchor_track":False,
             "colour_box":d,"colour_pose":d,"track_colour_box":d,"track_colour_pose":d,
             "action_supported":True,"action_distance":.1} for j,pid,d in ((1,"P1",.1),(2,"P2",.4))]}


def test_colour_uses_torso_regions_not_player_number_and_rejects_missing_joints():
    image = np.zeros((100,100,3),np.uint8); image[:50] = (255,0,0); image[50:] = (0,0,255)
    p = person(); vector = pose_colour(image,p)
    assert vector.shape==(384,) and abs(np.linalg.norm(vector)-1)<1e-5
    renamed = {**p,"id":999,"player_id":"P100"}
    np.testing.assert_array_equal(vector,pose_colour(image,renamed))
    broken = deepcopy(p); broken["kp"][5][2] = 0.
    assert pose_colour(image,broken) is None


def test_track_segments_split_gaps_teleports_and_ignore_roster_names():
    frames = [{"f":f,"poses":[person(x=x)]} for f,x in ((0,20),(1,22),(30,23),(31,400))]
    observations,tracks = track_segments(frames,30)
    assert len(tracks)==3
    assert observations[0,0]["segment"]==observations[1,0]["segment"]
    assert observations[30,0]["segment"]!=observations[1,0]["segment"]
    assert observations[31,0]["segment"]!=observations[30,0]["segment"]


def test_raw_tracker_baseline_abstains_without_photo_anchor():
    c = case()
    raw = fixed_predict(c,raw=True)
    assert raw["label"]=="unknown" and raw["team"]=="b"
    assert fixed_predict(c,anchor=False)["label"]=="P1"
    c["pairs"][1]["anchor_track"] = True
    assert fixed_predict(c,raw=True)["label"]=="P2"


def test_metrics_do_not_hide_near_errors_behind_other_team_majority():
    cases = [case("other_team") for _ in range(6)]+[case("P1") for _ in range(2)]
    result = metrics(cases,[prediction(c,"other_team") for c in cases],np.ones(8,bool))
    assert result["author_accuracy"]==.75
    assert result["near_author_recall"]==0 and result["balanced_author_accuracy"]==.5
    assert result["near_assignments"]==0 and result["near_assignment_precision"] is None


def test_person_ids_scoped_by_source_and_unknowns_remain_errors():
    cases = [case("P1","a"),case("P1","b"),case("other_team","b")]
    predictions = [prediction(cases[0],"P1",.9),prediction(cases[1],"unknown",0.,team="b"),prediction(cases[2],"P1",.9)]
    result = metrics(cases,predictions,np.ones(3,bool))
    assert result["near_author_recall"]==.5 and result["team_accuracy"]==2/3
    assert result["near_assignment_precision"]==.5
    assert result["near_identity_accuracy_given_near_team"]==.5
    assert result["coverage"]==2/3


def test_ranker_normalization_fits_training_sources_only():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(9,3,4)).astype(np.float32); x[6:] = 1000
    classes = np.tile([0,1,2],(9,1)); y = np.array([0,1,2]*3)
    train = np.arange(9)<6
    fitted = fit_rankers(x,classes,y,train,hidden=0,epochs=2,seeds=(42,))
    np.testing.assert_allclose(fitted["mean"],x[:6].reshape(-1,4).mean(axis=0))
    assert fitted["training_count"]==6 and fitted["training_supported"]==6
    scores,_ = ranker_scores(fitted,x,classes,3)
    assert scores.shape==(9,3) and np.isfinite(scores).all()
    # Even invalid held-out targets must not enter training support/output shape.
    changed = y.copy(); changed[~train] = 999
    other = fit_rankers(x,classes,changed,train,hidden=0,epochs=2,seeds=(42,))
    for key,value in fitted["members"][0]["network"].state_dict().items():
        torch.testing.assert_close(value,other["members"][0]["network"].state_dict()[key])


def test_pair_network_does_not_learn_numeric_roster_ids():
    torch.manual_seed(42); network = PairRanker(4,0).eval()
    x = torch.randn(2,3,4); classes = torch.tensor([[0,1,2],[0,1,2]])
    before,_ = network(x,classes,3)
    changed = torch.tensor([[0,2,1],[0,2,1]])
    after,_ = network(x,changed,3)
    torch.testing.assert_close(before[:,[0,2,1]],after)
    assert encode([None,0.,float("nan"),2.])==[0.,0.,0.,1.,0.,0.,2.,1.]


def test_hungarian_prevents_duplicate_ids_and_keeps_unknown_slots():
    c = case()
    for p in c["pairs"]: p["segment"] = "A"
    for p in deepcopy(c["pairs"]):
        p["segment"] = "B"; p["track_colour_pose"] = .12 if p["player_id"]=="P1" else .15
        c["pairs"].append(p)
    before = deepcopy(c)
    assert roster_assignment(c)=={"A":"P1","B":"P2"}
    assert c==before
    for p in c["pairs"]:
        for field in ("track_colour_pose","track_colour_box","colour_pose","colour_box"): p[field] = None
    assert set(roster_assignment(c).values())=={"unknown"}


def test_sequence_priors_and_predictions_ignore_held_out_authors():
    groups = np.array(["a","a","b","b","c","c"])
    cases = [{"contact":{"id":str(i)}} for i in range(6)]
    snapshot = {"videos":{v:{"fps":30.} for v in ("a","b","c")},"queue":[
        {"id":str(i),"video_id":groups[i],"frame":30 if i%2==0 else 60,"type":"serve" if i%2==0 else "reception","timing_verified":True} for i in range(6)]}
    train = groups!="c"; y = np.array([0,1,1,0,0,1]); scores = np.array([.1,.8,.8,.1,.3,.55])
    first,priors = sequence_scores(snapshot,cases,groups,train,y,scores)
    changed = y.copy(); changed[~train] = 1-changed[~train]
    second,other = sequence_scores(snapshot,cases,groups,train,changed,scores)
    assert priors==other and priors["serve>reception:normal"]["count"]==2
    np.testing.assert_array_equal(first,second)


def test_central_actor_uses_known_frame_without_author_targets_or_new_coordinates(tmp_path):
    def make_dataset(directory,target,missing=False,off_camera=False):
        frames = [{"f":f,"poses":[person()],"ball":None if missing and f==4 else [24,35],"ball_out_of_frame":off_camera and f==4} for f in range(9)]
        source = directory/"source.json"; write_new(source,{"frames":frames})
        snapshot = {"videos":{"game":{"fps":30.,"annotations_source":fingerprint_file(source)}},
                    "queue":[{"video_id":"game","frame":4}]}
        write_new(directory/"human_labels.json",snapshot)
        write_new(directory/"cases.json",[{"contact":{"id":"c","video_id":"game","frame":4,"type":"pass","author":{"player_id":target}},
            "pairs":[{"segment":"1:0","class_index":1,"player_id":"P1","frame":2,"pose_index":0,"score":.01}]}])
        (directory/"features.npz").write_bytes(b"unchanged feature sentinel")
        write_new(directory/"manifest.json",{"cases_sha256":sha(directory/"cases.json"),
            "snapshot_sha256":sha(directory/"human_labels.json"),"features_sha256":sha(directory/"features.npz")})
        return directory
    first = make_dataset(tmp_path/"a","P1"); second = make_dataset(tmp_path/"b","P99")
    before = sha(first/"features.npz")
    outputs = []
    for n,path in enumerate((first,second)):
        output = tmp_path/f"context{n}.json"; prepare_central(path,output)
        outputs.append(json.loads(output.read_bytes()))
    assert outputs[0]["cases"]==outputs[1]["cases"]
    chosen = outputs[0]["cases"][0]["pairs"][0]
    assert chosen["frame"]==4 and chosen["ball"]==[24,35] and chosen["bbox"]==person()["bb"]
    assert sha(first/"features.npz")==before
    missing = make_dataset(tmp_path/"missing","P1",missing=True)
    prepare_central(missing,tmp_path/"fallback.json")
    fallback = json.loads((tmp_path/"fallback.json").read_bytes())["cases"][0]["pairs"][0]
    assert fallback["frame"]==3 and fallback["actor_time_method"]=="closest_observed_to_audited_time"
    off = make_dataset(tmp_path/"off_camera","P1",missing=True,off_camera=True)
    prepare_central(off,tmp_path/"off.json")
    assert json.loads((tmp_path/"off.json").read_bytes())["cases"][0]["pairs"]==[]
