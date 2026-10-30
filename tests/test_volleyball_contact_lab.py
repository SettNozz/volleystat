from copy import deepcopy

import numpy as np
import pytest

from src.tracking.volleyball_contact_lab import (
    consensus, dense_evidence, joint_rank, learn_transitions,
    peaks, sequence_posteriors,
)
from scripts.evaluation.run_volleyball_contact_lab import (
    detection, grouped_sequence, matched_indices, pipeline_team_scores, retime, tune,
)


def flight():
    return [{"f":f,"ball":[200+(f-30)*5,200+(f-30)**2*.3] if f<30
             else [200-(f-30)*5,200-(f-30)*8+(f-30)**2*.3]} for f in range(60)]


def test_piecewise_fit_detects_impulse_but_not_smooth_flight_without_mutation():
    rows = flight(); original = deepcopy(rows)
    evidence, support = dense_evidence(rows)
    assert np.argmax(evidence[:,0]) == 30
    assert evidence[30,0] > .5 and support[30]
    smooth = [{"f":f,"ball":[100+f*5,100+f*f*.3]} for f in range(60)]
    assert dense_evidence(smooth)[0][:,0].max() < 1e-5
    assert rows == original


@pytest.mark.parametrize("mutation", ["missing","offcamera","interpolated","reconstructed","restart"])
def test_no_impulse_evidence_through_missing_derived_or_restarted_ball(mutation):
    rows = flight()
    if mutation == "missing": rows[30]["ball"] = None
    if mutation == "offcamera": rows[30]["ball_out_of_frame"] = True
    if mutation == "interpolated": rows[30]["ball_interp"] = True
    if mutation == "reconstructed": rows[30]["ball_source"] = "contact_aware_reconstructed"
    if mutation == "restart": rows[30]["ball_new"] = True
    evidence, support = dense_evidence(rows)
    assert not support[30] and evidence[30,0] == 0


def test_temporal_interaction_requires_same_track_and_never_fills_hidden_pose():
    rows = flight()
    for f,row in enumerate(rows):
        kp = [[0,0,0] for _ in range(17)]
        kp[9] = [200,200,1]
        row["poses"] = [{"id":7,"team":"b","bb":[180,100,240,300],"kp":kp}]
    original = deepcopy(rows)
    good = dense_evidence(rows)[0][30,1]
    assert good > .5
    rows[27]["poses"][0]["id"] = 8
    assert dense_evidence(rows)[0][30,1] == 0
    rows[30]["poses"] = []
    assert dense_evidence(rows)[0][30,1] == 0
    assert original[30]["poses"][0]["id"] == 7


def test_sequence_transition_training_and_inference_are_separate():
    train = [[{"frame":0,"type":"attack","team_label":"a"},
              {"frame":30,"type":"dig","team_label":"b"}] for _ in range(30)]
    prior = learn_transitions(train,30)
    events = [{"frame":0,"type":"attack","team_label":"b"},
              {"frame":30,"type":"dig","team_label":"a"}]
    predicted = sequence_posteriors(events,[.95,.60],prior,30)
    assert predicted[0] > .9 and predicted[1] < .5
    # Held-out human answers cannot change the posterior, only availability.
    flipped = [{**e,"team_label":"a"} for e in events]
    np.testing.assert_array_equal(predicted,sequence_posteriors(flipped,[.95,.60],prior,30))
    np.testing.assert_allclose(sequence_posteriors(events,[.95,.60],prior,30,strength=0),[.95,.60])


def test_unknown_contacts_serves_and_long_gaps_reset_sequence():
    prior = {"attack>pass:normal":{"same":.01}}
    for second in ({"frame":200,"type":"dig","team_label":"a"},
                   {"frame":30,"type":"serve","team_label":"a"},
                   {"frame":30,"type":"dig","team_label":None}):
        events = [{"frame":0,"type":"attack","team_label":"a"},second]
        np.testing.assert_allclose(sequence_posteriors(events,[.95,.60],prior,30),[.95,.60])


def test_nms_consensus_and_close_attack_block_retiming():
    assert peaks([0,.8,.9,0,.7],.5,1) == [2,4]
    assert consensus([10,12],[11,30]) == [10,12,30]
    evidence = np.zeros((40,3)); evidence[11,2] = .9
    events = [{"frame":10,"type":"attack"},{"frame":12,"type":"block"}]
    assert [e["frame"] for e in retime(events,evidence)] == [10,12]


def test_tuning_uses_only_supplied_training_videos_and_masks_are_scoring_only():
    video = {"evidence":np.zeros((40,3)),"truth":[10],"valid":np.ones(40,bool),"non_game":np.zeros(40,bool)}
    video["evidence"][10,0] = .8; video["evidence"][30,0] = .3
    chosen = tune([video],0)
    assert peaks(video["evidence"][:,0],**chosen) == [10]
    video["non_game"][30] = True
    scores = detection([10,30],[10],video["valid"],video["non_game"])
    assert scores["all"]["at_5"]["fp"] == 1
    assert scores["excluding_non_game"]["at_5"]["fp"] == 0


def test_joint_ranking_keeps_observed_candidate_provenance():
    actors = [{"frame":1,"score":.1,"player_id":"P2"},{"frame":2,"score":.12,"player_id":"P1"}]
    evidence = np.zeros((3,3)); evidence[2,:2] = 1
    ranked = joint_rank(actors,evidence)
    assert ranked[0]["player_id"] == "P1"
    assert actors[0]["score"] == .1 and "flight_evidence" not in actors[0]


def test_duplicate_source_labels_are_one_context_node_and_unsure_resets():
    items = [{"id":"a","frame":10},{"id":"b","frame":10},{"id":"c","frame":20},{"id":"d","frame":30}]
    labels = {"a":{"team":"team_b"},"b":{"team":"team_b"},"c":{"team":"unsure"},"d":{"team":"team_a"}}
    contacts = [{"frame":10,"contact_type":"reception"},{"frame":20,"contact_type":"set"}]
    rows = grouped_sequence(items,labels,contacts)
    assert len(rows)==3 and rows[0]["case_ids"]==["a","b"]
    assert rows[0]["team_label"]=="team_b"
    assert rows[1]["team_label"] is None and rows[2]["team_label"] is None


def test_pipeline_scoring_matches_times_before_using_team_labels():
    events = [{"frame":11},{"frame":20},{"frame":35}]
    video = {"valid":np.ones(50,bool),"non_game":np.zeros(50,bool),"truth":[10,20,30],
             "sequence":[{"frame":10,"team_label":"team_b"},{"frame":20,"team_label":None},
                         {"frame":30,"team_label":"team_a"}]}
    score = pipeline_team_scores(events,[.9,.1,.9],video)["all"]
    assert score["matched_known_team"]==2 and score["correct_team"]==1
    assert score["matched_unknown_team_excluded"]==1
    assert score["joint_detection_team"]["tp"]==1 and score["joint_detection_team"]["fp"]==1
    assert matched_indices([{"frame":13},{"frame":15}],[10,14],3)==[(0,10),(1,14)]
