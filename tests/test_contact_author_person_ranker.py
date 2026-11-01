import numpy as np
import pytest

from src.tracking.contact_author_person_evaluation import evaluate, metrics
from src.tracking.contact_author_person_ranker import (chain_posterior, count_transitions, decide,
                                                       person_candidates, pose_geometry, roster_seed)


def pose(x, y, team="b", pid=None, raw=1, h=400.):
    kp = [[0., 0., 0.] for _ in range(17)]
    for j, (dx, dy) in {5: (-40, 80), 6: (40, 80), 7: (-50, 160), 8: (50, 160), 9: (-55, 230), 10: (55, 230),
                        11: (-30, 220), 12: (30, 220), 15: (-30, 390), 16: (30, 390)}.items():
        kp[j] = [x+dx, y+dy, .9]
    return {"bb": [x-60, y, x+60, y+h], "kp": kp, "team": team, "player_id": pid, "id": raw}


def test_geometry_uses_hand_distance_in_body_heights():
    g = pose_geometry(pose(1000, 500), (945, 730))
    assert g["w"] == pytest.approx(0.)
    assert pose_geometry(pose(1000, 500), (1000, 330))["b"] > .3


def test_hidden_at_contact_frame_person_still_competes_and_missing_ball_is_not_measured():
    frames = {f: {"ball": [945, 730], "poses": [pose(1000, 500, pid="P1", raw=7), pose(2000, 500, pid="P2", raw=8),
                                                  pose(1500, 100, team="a", raw=9)]} for f in range(90, 111)}
    frames[100] = {"ball": None, "poses": [pose(2000, 500, pid="P2", raw=8)]}
    x = person_candidates(frames, {"frame": 100, "type": "reception"}, ["P1", "P2"], 2160)
    assert x["P1"]["present0"] == 0. and x["P1"]["d1"] == pytest.approx(0.)
    assert x["P2"]["d1"] > 1. and x["P1"]["rank3"] == 0.
    assert x["other_team"]["is_other"] == 1. and x["P1"]["t_reception"] == 1.


def test_evidence_points_at_the_saved_pose_closest_to_the_ball():
    frames = {f: {"ball": [945, 730] if f == 103 else [3000, 100], "poses": [pose(2000, 500, team="a", raw=4),
                                                                          pose(1000, 500, pid="P1", raw=7)]} for f in range(90, 111)}
    evidence = {}
    person_candidates(frames, {"frame": 100, "type": "dig"}, ["P1", "P2"], 2160, evidence=evidence)
    assert evidence["P1"]["frame"] == 103 and evidence["P1"]["pose_index"] == 1 and evidence["P1"]["raw_track_id"] == 7
    assert evidence["P1"]["bbox"] == frames[103]["poses"][1]["bb"] and "P2" not in evidence


def test_unidentified_pose_borrows_dominant_raw_track_vote_only():
    frames = {f: {"ball": [945, 730], "poses": [pose(1000, 500, pid="P3", raw=7)]} for f in range(0, 101)}
    for f in range(95, 106):
        frames[f] = {"ball": [945, 730], "poses": [pose(1000, 500, pid=None, raw=7)]}
    x = person_candidates(frames, {"frame": 100, "type": "set"}, ["P3", "P4"], 2160)
    assert x["P3"]["n3"] == 7 and x["P3"]["src_pid"] < 1.
    x = person_candidates(frames, {"frame": 100, "type": "set"}, ["P3", "P4"], 2160, vote_share=1.01)
    assert x["P3"]["n3"] == 0 and x["P4"]["missing"] == 1.


def test_chain_transitions_use_training_sources_and_switch_team_after_serve():
    contacts = [{"video_id": v, "frame": f, "type": t} for v in ("a", "b") for f, t in ((0, "serve"), (30, "reception"), (60, "set"))]
    near = [1, 0, 0, 0, 1, 1]
    transitions = count_transitions(contacts, near, {"a"})
    assert transitions[("serve", "reception")] < .5 and transitions[("reception", "set")] > .5
    p = np.array([.9, .5, .45, .9, .5, .45])
    posterior = chain_posterior(contacts, p, transitions, {"b"})
    assert posterior[4] < .5 and posterior[5] < .5 and list(posterior[:3]) == list(p[:3])


def test_decide_and_metrics():
    assert decide({"P1": .2, "P2": .7, "other_team": .9}, .8) == "P2"
    assert decide({"P1": .2, "other_team": .1}, .3) == "other_team"
    cases = [{"options": ["other_team", "P1", "P2"], "target": t} for t in (1, 2, 0)]
    m = metrics(cases, {0: "P1", 1: "P1", 2: "other_team"})
    assert (m["near_correct"], m["near_assignment_precision"], m["team_accuracy"]) == (1, .5, 1.)


def test_roster_seed_binds_substitute_to_detector_box():
    snapshot = {"revision": 3, "videos": {"v": {"video_source": {"sha256": "x", "path": "C:\\a\\v.webm"}}},
                "rosters": {"v": {"players": [
                    {"player_id": "P1", "references": [{"frame": 10, "bbox": [0, 0, 10, 10]}]},
                    {"player_id": "P7", "references": [{"frame": 50, "bbox": [102, 0, 160, 400]}]}]}}}
    annotations = {"frames": [{"f": 10, "poses": [{"bb": [0, 0, 10, 10], "team": "b", "id": 1}]},
                              {"f": 40, "poses": [{"bb": [100, 0, 160, 400], "team": "b", "id": 5}]},
                              {"f": 50, "poses": [{"bb": [100, 0, 160, 400], "team": "b", "id": 5}]}]}
    seed = roster_seed(snapshot, "v", annotations)
    intro = seed["introductions"][0]
    assert intro["anchor"]["bb"] == [100, 0, 160, 400] and intro["available_from_frame"] == 40
    assert seed["max_on_court"] == 1 and seed["provenance"]["contact_author_targets_used"] is False
    annotations["frames"][0]["poses"].append({"bb": [20, 0, 30, 10], "team": "b", "id": 2})
    with pytest.raises(ValueError):
        roster_seed(snapshot, "v", annotations)


def test_evaluate_refuses_existing_output(tmp_path):
    with pytest.raises(ValueError):
        evaluate(tmp_path, tmp_path, tmp_path)
