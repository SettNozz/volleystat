from copy import deepcopy
import json

import pytest

from scripts.complete_player_poses import rebuild
from src.tracking.player_contacts import build_player_contacts
from src.tracking.pose_completion import CompletionConfig, complete_player_poses


def pose(f, pid="P1", x=100, raw_id=1):
    x += f * 2
    points = [[x+40, 110, .95] for _ in range(17)]
    for joint, dx, y in ((5, 20, 140), (6, 60, 140), (7, 15, 165), (8, 65, 165),
                         (9, 10, 190), (10, 70, 190), (11, 25, 190), (12, 55, 190),
                         (13, 25, 220), (14, 55, 220), (15, 25, 250), (16, 55, 250)):
        points[joint] = [x+dx, y, .95]
    return {"player_id": pid, "team": "b", "id": raw_id,
            "bb": [x, 100, x+80, 260], "kp": points}


def frames(gap=(4, 5, 6)):
    return [{"f": f, "ball": [50, 50], "poses":
             ([pose(f)] if f not in gap else []) + [pose(f, "P2", x=160, raw_id=2)]}
            for f in range(15)]


def complete(fs, config=None):
    return complete_player_poses(fs, 30, (1000, 1000), {"side": "b"}, config)


def test_short_occlusion_uses_both_endpoints_and_keeps_observations_untouched():
    source = frames()
    before = deepcopy(source)
    estimates, report = complete(source)
    assert source == before
    assert report["counts"]["missing_person_frames"] == 3
    assert report["uses_future_frames"] is True
    assert report["contact_attribution_uses_estimates"] is False
    middle = estimates[5][0]
    assert middle["player_id"] == "P1"
    assert middle["keypoints"][9] == [120., 190.]
    assert middle["joint_evidence"][9]["left_frame"] == 3
    assert middle["joint_evidence"][9]["right_frame"] == 7
    assert middle["verified"] is False and middle["label_source"] == "derived"
    assert "id" not in middle and all(len(p) == 2 for p in middle["keypoints"])


def test_missing_joint_completion_preserves_every_measured_joint():
    source = frames(gap=())
    for f in (4, 5, 6):
        source[f]["poses"][0]["kp"][10][2] = .01
    before = deepcopy(source)
    estimates, report = complete(source)
    assert source == before
    assert report["counts"]["partial_person_frames"] == 3
    for f in (4, 5, 6):
        entry = next(e for e in estimates[f] if e["player_id"] == "P1")
        assert entry["observed_pose_index"] == 0
        assert entry["estimated_joints"] == [10]
        assert entry["keypoints"][10] == pose(f)["kp"][10][:2]


@pytest.mark.parametrize("change", ["long_gap", "no_occluder", "unstable", "edge", "jump", "different_person", "missing_frame", "duplicate_id"])
def test_unsafe_or_unbounded_gaps_stay_empty(change):
    source = frames()
    if change == "long_gap":
        source = frames(gap=range(2, 13))
    elif change == "no_occluder":
        source = [{**a, "poses": [p for p in a["poses"] if p["player_id"] == "P1"]} for a in source]
    elif change == "unstable":
        source[2]["poses"] = [source[2]["poses"][-1]]
    elif change == "edge":
        for a in source:
            for p in a["poses"]:
                if p["player_id"] == "P1": p["bb"][0] = 0
    elif change == "jump":
        for a in source[7:]:
            a["poses"][0] = pose(a["f"], x=500)
    elif change == "different_person":
        for a in source[7:]: a["poses"][0]["player_id"] = "P3"
    elif change == "missing_frame":
        source = [a for a in source if a["f"] != 5]
    elif change == "duplicate_id":
        source[3]["poses"].append(deepcopy(source[3]["poses"][0]))
    result, _ = complete(source)
    assert not any(e["player_id"] == "P1" for entries in result.values() for e in entries)


def test_rejected_identity_is_not_restored_from_raw_track_or_proximity():
    source = frames()
    for f in (4, 5, 6):
        unknown = pose(f, pid=None)
        unknown["identity"] = {"reason": "appearance_contradiction"}
        source[f]["poses"].append(unknown)
    result, report = complete(source)
    assert not result
    assert report["diagnostics"]["existing_or_unidentified_pose_frames"] == 3


def test_arms_have_shorter_limit_than_torso_and_fast_arm_motion_is_rejected():
    source = frames(gap=range(4, 10))
    result, _ = complete(source)
    assert result
    for entries in result.values():
        for entry in entries:
            assert not set(entry["estimated_joints"]).intersection({7, 8, 9, 10})
    source = frames()
    for a in source[7:]: a["poses"][0]["kp"][10][0] += 150
    result, _ = complete(source)
    assert all(10 not in entry["estimated_joints"] for entries in result.values() for entry in entries)


def test_cannot_add_seventh_player_or_extrapolate_before_first_observation():
    source = frames()
    for f in (4, 5, 6):
        source[f]["poses"] += [pose(f, f"P{i}", x=300+i*90, raw_id=i) for i in range(3, 8)]
    assert not complete(source)[0]
    assert not complete(frames(gap=(0, 1, 2)))[0]


def test_completion_is_resolution_invariant_and_has_no_gap_field_at_observed_frames():
    source = frames()
    original, _ = complete(source)
    scaled = deepcopy(source)
    for a in scaled:
        for p in a["poses"]:
            p["bb"] = [x*2 for x in p["bb"]]
            p["kp"] = [[x*2, y*2, c] for x, y, c in p["kp"]]
    result, _ = complete_player_poses(scaled, 30, (2000, 2000), {"side": "b"})
    assert set(result) == set(original) == {4, 5, 6}
    assert result[5][0]["keypoints"][9] == [x*2 for x in original[5][0]["keypoints"][9]]


def test_pose_estimates_cannot_change_contact_report_or_source_exports(tmp_path):
    annotations = {"video": "clip", "fps": 30, "frame_width": 1000, "frame_height": 1000,
                   "frames": frames(), "player_identity": {"side": "b"}}
    stats = {"video": "clip", "fps": 30, "all_events": [{"frame": 5, "type": "pass"}]}
    ann = tmp_path / "clip_annotations.json"
    stat = tmp_path / "clip_stats.json"
    ann.write_text(json.dumps(annotations), encoding="utf-8")
    stat.write_text(json.dumps(stats), encoding="utf-8")
    before = ann.read_bytes(), stat.read_bytes()
    output = tmp_path / "fresh"
    rebuild(ann, output)
    exported = json.loads((output / ann.name).read_bytes())
    assert (ann.read_bytes(), stat.read_bytes()) == before
    for a in exported["frames"]:
        a.pop("pose_estimates", None)
    exported.pop("pose_completion")
    assert exported == annotations
    result = json.loads((output / stat.name).read_bytes())
    result.pop("pose_completion")
    assert result == stats
    new_frames = json.loads((output / ann.name).read_bytes())["frames"]
    assert build_player_contacts(stats["all_events"], new_frames, 30) == build_player_contacts(stats["all_events"], annotations["frames"], 30)
    with pytest.raises(FileExistsError): rebuild(ann, output)


def test_invalid_order_and_configuration_fail():
    with pytest.raises(ValueError, match="ordered"):
        complete(frames()[::-1])
    with pytest.raises(ValueError, match="bounds"):
        complete(frames(), CompletionConfig(max_gap_seconds=1.))


def test_dropout_evaluation_hides_reference_and_scores_matched_joints():
    from scripts.evaluation.evaluate_pose_completion import evaluate
    payload = {"fps": 30, "frame_width": 1000, "frame_height": 1000,
               "frames": frames(gap=()), "player_identity": {"side": "b"}}
    before = deepcopy(payload)
    report = evaluate(payload, samples_per_gap=2)
    assert payload == before
    assert all(group["evaluated_joint_frames"] > 0 for group in report["groups"])
    for group in report["groups"]:
        estimates = group["errors"]["all"]
        assert len({v["joint_frames"] for v in estimates.values()}) == 1
        assert estimates["bidirectional"]["median_error_body_heights"] < 1e-6
        assert estimates["last_pose"]["median_error_body_heights"] > 0
        assert group["available_reference_joint_frames"] >= group["evaluated_joint_frames"]
