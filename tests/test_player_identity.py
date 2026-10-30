"""Identity failures: crossings, ambiguity, exits and human-test leakage."""
import copy
import json

import numpy as np
import pytest

from src.tracking.player_identity import appearance, compare, default_seed, run_method, score
from scripts.evaluation.run_player_identity_pilot import digest, write_json


def player(frame, index, x, feature, raw_id=None, occlusion=0.):
    return {"key": f"{frame}:{index}", "pose_index": index, "raw_id": raw_id,
            "bb": [x, 0., x+20., 100.], "appearance": feature, "occlusion": occlusion}


def fixture():
    frames = [
        {"f": 0, "players": [player(0, 0, 0, [1., 0.], 10), player(0, 1, 60, [0., 1.], 20)]},
        {"f": 3, "players": [player(3, 0, 50, [1., 0.], 101), player(3, 1, 10, [0., 1.], 102)]},
    ]
    seed = {"frame": 0, "assignments": {"0:0": "P1", "0:1": "P2"},
            "label_source": "human", "verified": True}
    return frames, seed


def test_appearance_resolves_crossing_that_nearest_motion_swaps():
    frames, seed = fixture()
    motion = run_method(frames, seed, 30., "motion")
    visual = run_method(frames, seed, 30., "motion_appearance")
    assert motion["3:0"]["player_id"] == "P2"
    assert visual["3:0"]["player_id"] == "P1"
    assert visual["3:1"]["player_id"] == "P2"


def test_ambiguous_overlap_abstains_and_does_not_duplicate_id():
    frames, seed = fixture()
    frames[1]["players"] = [player(3, i, 30, [1., 0.], occlusion=.9) for i in range(2)]
    predictions = run_method(frames, seed, 30., "motion")
    assert all(predictions[p["key"]]["player_id"] is None for p in frames[1]["players"])


def test_long_exit_expires_even_when_appearance_matches():
    frames, seed = fixture()
    frames[1] = {"f": 121, "players": [player(121, 0, 0, [1., 0.])]}
    predictions = run_method(frames, seed, 30., "motion_appearance")
    assert predictions["121:0"]["player_id"] is None


def test_backward_pass_recovers_frames_before_seed():
    frames, _ = fixture()
    frames[1]["players"] = [player(3, 0, 1, [1., 0.]), player(3, 1, 61, [0., 1.])]
    seed = {"frame": 3, "assignments": {"3:0": "P1", "3:1": "P2"}}
    predictions = run_method(frames, seed, 30., "motion_appearance")
    assert predictions["0:0"]["player_id"] == "P1"


def test_seed_ids_unique_and_complete():
    frames, seed = fixture()
    seed["assignments"]["0:1"] = "P1"
    with pytest.raises(ValueError, match="unique"):
        run_method(frames, seed, 30., "motion")


def test_duplicate_raw_ids_cannot_give_one_person_to_two_detections():
    frames, seed = fixture()
    for p in frames[1]["players"]:
        p["raw_id"] = 10
    predictions = run_method(frames, seed, 30., "raw_track")
    assert all(predictions[p["key"]]["player_id"] is None for p in frames[1]["players"])


def test_checks_do_not_change_predictions_and_unknown_counts_as_failure():
    frames, seed = fixture()
    observations = {"frames": frames, "source": {}, "fps": 30.}
    check = {"key": "3:0", "player_id": "P1", "label_source": "human", "verified": True}
    blind = compare(observations, seed)
    checked = compare(observations, seed, [check])
    for method in blind["methods"]:
        assert blind["methods"][method]["predictions"] == checked["methods"][method]["predictions"]
    assert checked["methods"]["raw_track"]["human_checks"]["accuracy_including_unknown"] == 0.
    assert checked["methods"]["raw_track"]["human_checks"]["accuracy_when_assigned"] is None
    assert checked["methods"]["motion_appearance"]["human_checks"]["accuracy_including_unknown"] == 1.
    assert score({}, [check], set())["checked"] == 0


def test_appearance_distinguishes_white_and_black_and_handles_clipping():
    white, black = np.full((100, 40, 3), 255, np.uint8), np.zeros((100, 40, 3), np.uint8)
    assert appearance(white, [0, 0, 40, 100]) != appearance(black, [0, 0, 40, 100])
    assert appearance(white, [-100, -100, -1, -1]) is None


def project(tmp_path):
    frames, _ = fixture()
    annotations = tmp_path/"source_annotations.json"
    annotations.write_text('{"human_labels":"untouched"}')
    video = tmp_path/"original.webm"
    video.write_bytes(b"placeholder; frame route tested on real pilot separately")
    observations = {"frames": frames, "fps": 30., "roster_size": 2, "side": "b", "width": 100, "height": 100,
                    "source": {"annotations": str(annotations), "annotations_sha256": digest(annotations),
                               "video": str(video), "video_size": video.stat().st_size,
                               "video_mtime_ns": video.stat().st_mtime_ns}}
    write_json(tmp_path/"observations.json", observations)
    write_json(tmp_path/"comparison_0000.json", compare(observations, default_seed(frames, 2)))
    return annotations


def test_review_keeps_human_labels_separate_and_blocks_seed_redefinition(tmp_path):
    from action_recognition.annotator.player_identity_annotator import create_app
    annotations = project(tmp_path)
    original = annotations.read_bytes()
    client = create_app(tmp_path).test_client()
    assert client.get("/").status_code == 200
    seed_payload = {"frame": 0, "role": "seed", "assignments": {"0:0": "P1", "0:1": "P2"}}
    assert client.post("/api/labels", json={"frame": 3, "role": "check", "assignments": {"3:0": "P1"}}).status_code == 400
    assert client.post("/api/labels", json=seed_payload).status_code == 200
    assert client.post("/api/labels", json={"frame": 3, "role": "check", "assignments": {"3:0": "P1"}}).status_code == 200
    assert client.post("/api/labels", json=seed_payload).status_code == 200
    bad_seed = copy.deepcopy(seed_payload)
    bad_seed["assignments"] = {"0:0": "P2", "0:1": "P1"}
    assert client.post("/api/labels", json=bad_seed).status_code == 409
    assert client.post("/api/compare", json={}).status_code == 200
    assert (tmp_path/"comparison_0001.json").exists()
    labels = json.loads((tmp_path/"labels.json").read_text())
    assert labels["checks"][0]["label_source"] == "human"
    assert annotations.read_bytes() == original


def test_changed_source_or_cache_cannot_reuse_labels(tmp_path):
    from action_recognition.annotator.player_identity_annotator import create_app
    annotations = project(tmp_path)
    client = create_app(tmp_path).test_client()
    client.post("/api/labels", json={"frame": 0, "role": "seed", "assignments": {"0:0": "P1", "0:1": "P2"}})
    path = tmp_path/"observations.json"
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="Observations changed"):
        create_app(tmp_path)
    annotations.write_text("changed")
    with pytest.raises(ValueError, match="Source annotations changed"):
        create_app(tmp_path)


def test_extra_experiment_rescores_checks_without_changing_predictions(tmp_path):
    from action_recognition.annotator.player_identity_annotator import create_app
    project(tmp_path)
    client = create_app(tmp_path).test_client()
    seed_payload = {"frame": 0, "role": "seed", "assignments": {"0:0": "P1", "0:1": "P2"}}
    client.post('/api/labels', json=seed_payload)
    extra = json.loads((tmp_path/'comparison_0000.json').read_text())
    extra['observations_sha256'] = digest(tmp_path/'observations.json')
    extra['methods'] = {'osnet_fixed_recovery': extra['methods']['motion_appearance']}
    write_json(tmp_path/'extra.json', extra)
    before = digest(tmp_path/'extra.json')
    client = create_app(tmp_path,tmp_path/'extra.json').test_client()
    assert 'osnet_fixed_recovery' in client.get('/api/project').get_json()['comparison']['methods']
    client.post('/api/labels',json={'frame':3,'role':'check','assignments':{'3:0':'P1'}})
    result = client.post('/api/compare',json={}).get_json()
    assert result['methods']['osnet_fixed_recovery']['human_checks']['correct'] == 1
    assert digest(tmp_path/'extra.json') == before


def test_extra_experiment_refuses_other_observation_cache(tmp_path):
    from action_recognition.annotator.player_identity_annotator import create_app
    project(tmp_path)
    write_json(tmp_path/'extra.json',{'observations_sha256':'wrong'})
    with pytest.raises(ValueError,match='different observations'):
        create_app(tmp_path,tmp_path/'extra.json')
