"""Same-frame near-camera serve fragments and strict cached crop replay."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.detection.player_pose import PlayerPose
from src.detection.pose_fragments import (
    PoseFragmentRecovery, merge_fragments, recover_from_crop, recovery_crop, same_person,
)
from src.models.inference_cache import (
    CacheMiss, ModelOutputCache, array_payload,
)


def server_pair(scale=1.):
    # Observed coordinates at f3376, XNCS5BLBwig. These are model outputs,
    # not human-verified pose labels. Face confidence is irrelevant to merging.
    lower = np.zeros((17, 3), dtype=np.float32)
    upper = lower.copy()
    lower[5:] = [[2182.5,831,.74],[2332.5,853.5,.87],[2188.5,989.2,.53],
                 [2407.5,994.5,.86],[2247,1045.5,.33],[2445,1032,.59],
                 [2170.5,1194.8,1],[2284.5,1211.2,1],[2128.5,1456.5,1],
                 [2307,1522.5,1],[2064,1732.5,.99],[2281.5,1791,.99]]
    upper[5:] = [[2169,791.2,.99],[2350.5,855,1],[2121,711.4,.91],
                 [2410.5,991.5,.97],[2154,572.6,.58],[2463,1047.8,.81],
                 [2170.5,1167.8,1],[2295,1182,1],[2152.5,1434.8,.92],
                 [2290.5,1429.5,.94],[2146.5,1525.5,.18],[2299.5,1536,.18]]
    def pose(points, box, tid):
        points[:, :2] *= scale
        box = tuple(v*scale for v in box)
        return PlayerPose('team_b', ((box[0]+box[2])/2,(box[1]+box[3])/2), box, points, tid)
    return (pose(lower, (1992.8,762.3,2486.6,1859.8), 530),
            pose(upper, (2063.2,520,2478,1524.3), 593))


@pytest.mark.parametrize('scale', [.25, 1., 2.])
def test_reported_server_is_one_track_with_observed_arm_and_leg_chains(scale):
    lower, upper = server_pair(scale)
    raw = deepcopy([lower, upper])
    merged = merge_fragments([lower, upper])
    assert len(merged) == 1 and merged[0].track_id == 530
    np.testing.assert_array_equal(merged[0].keypoints[[7,9]], upper.keypoints[[7,9]])
    np.testing.assert_array_equal(merged[0].keypoints[[13,15]], lower.keypoints[[13,15]])
    assert [p['id'] for p in merged[0].pose_repair['observations']] == [530,593]
    for before, after in zip(raw, [lower, upper]):
        np.testing.assert_array_equal(before.keypoints, after.keypoints)
        assert before.bbox == after.bbox
    reversed_result = merge_fragments([upper, lower])
    assert reversed_result[0].track_id == 530
    np.testing.assert_array_equal(merged[0].keypoints, reversed_result[0].keypoints)


@pytest.mark.parametrize('scale', [.25, 1., 2.])
def test_f45_duplicate_with_narrow_hip_estimate_uses_matching_torso_scale(scale):
    # Model observations of the foreground number-6 player at f45. The two
    # hip widths differ by 1.88x despite near-identical torso bone lengths.
    first = np.zeros((17, 3), dtype=np.float32)
    second = first.copy()
    first[5:] = [[1114.5,949.5,1],[1300.5,960,1],[1021.5,1039.5,1],
                 [1378.5,1044,.98],[1041.8,1113.8,.94],[1419.8,1038,.84],
                 [1131,1223.2,1],[1233,1232.2,1],[1181.2,1428,1],
                 [1206,1437.8,1],[1210.5,1617,.87],[1171.5,1611,.81]]
    second[5:] = [[1142.2,935.2,.97],[1255.5,955.5,.97],[1053,1055.2,.91],
                  [1318.5,1122,.93],[1032.8,1120.5,.63],[1317,1222.5,.73],
                  [1156.5,1221,1],[1210.5,1227.8,1],[1177.5,1455,1],
                  [1167.8,1467,1],[1195.5,1674,.99],[1151.2,1674,.99]]
    poses = []
    for tid, box, points in ((9, (984.6,751.2,1460.7,1681.4), first),
                             (18, (914.1,841.9,1449,1812), second)):
        points[:, :2] *= scale
        box = tuple(v*scale for v in box)
        poses.append(PlayerPose('team_b', ((box[0]+box[2])/2,(box[1]+box[3])/2), box, points, tid))
    assert same_person(*poses)
    merged = merge_fragments(poses)
    assert len(merged) == 1
    assert {p['id'] for p in merged[0].pose_repair['observations']} == {9,18}
    # The partly overlapping teammate is smaller and must remain distinct.
    teammate = deepcopy(poses[0])
    teammate.track_id = 4
    teammate.bbox = tuple(v*scale for v in (1280.1,845.4,1428.3,1370.5))
    teammate.keypoints[:, :2] = (teammate.keypoints[:, :2] - np.array([1200,1200])*scale) * .45 + np.array([1350,1100])*scale
    assert len(merge_fragments([*poses,teammate])) == 2


def test_overlapping_people_need_joint_and_scale_agreement_not_box_iou():
    lower, upper = server_pair()
    upper.keypoints[:, 0] += 100
    assert merge_fragments([lower,upper]) == [lower,upper]
    lower, upper = server_pair()
    upper.keypoints[11, 0] += 100  # A distant player inside the same large box.
    upper.keypoints[12, 0] -= 30
    assert not same_person(lower,upper)
    lower, upper = server_pair()
    upper.keypoints[5:7, 2] = 0
    upper.keypoints[13:15, 2] = 0
    assert not same_person(lower,upper)  # Two hips alone cannot merge raw people.


def test_distinct_roster_members_never_merge_and_nonfinite_inputs_abstain():
    lower, upper = server_pair()
    lower.player_id, upper.player_id = 'P1','P2'
    assert not same_person(lower,upper)
    upper.player_id = None
    upper.keypoints[5, 0] = np.nan
    assert not same_person(lower,upper)


def test_neighbour_chain_cannot_merge_three_people_transitively():
    first, _ = server_pair()
    middle, last = deepcopy(first), deepcopy(first)
    middle.track_id, last.track_id = 531,532
    middle.keypoints[:,0] += 40
    last.keypoints[:,0] += 80
    assert same_person(first,middle) and same_person(middle,last)
    assert not same_person(first,last)
    assert len(merge_fragments([first,middle,last])) == 2


def test_partials_request_only_missing_body_half_and_complete_poses_skip_crop():
    lower, upper = server_pair()
    lower.keypoints[5:7, 2] = .1
    crop = recovery_crop(lower, (3840,2160))
    assert crop[1] < lower.bbox[1] and crop[3] < lower.bbox[3]
    assert recovery_crop(upper, (3840,2160))[3] > upper.bbox[3]
    assert recovery_crop(merge_fragments(server_pair())[0], (3840,2160)) is None
    assert recovery_crop(replace(lower,bbox=(0,0,20,100)), (3840,2160)) is None
    lower.keypoints[15:17, 2] = 0
    assert recovery_crop(lower,(3840,2160)) is None


def test_crop_requires_unique_precise_hips_and_additional_observed_joints():
    lower, upper = server_pair()
    lower.keypoints[5:7, 2] = .1
    crop = recovery_crop(lower,(3840,2160))
    upper.keypoints[11:13] = lower.keypoints[11:13]
    upper.keypoints[:, :2] -= crop[:2]
    upper.bbox = (upper.bbox[0]-crop[0],upper.bbox[1]-crop[1],upper.bbox[2]-crop[0],upper.bbox[3]-crop[1])
    fixed = recover_from_crop(lower,[upper],crop)
    assert fixed.track_id == lower.track_id and min(fixed.keypoints[5:7,2]) >= .4
    assert fixed.pose_repair['crop'] == list(crop)
    assert recover_from_crop(lower,[upper,deepcopy(upper)],crop) is lower
    upper.keypoints[:, 0] += 100
    assert recover_from_crop(lower,[upper],crop) is lower


def test_crop_cache_replay_uses_no_pixels_and_missing_raw_stage_fails(tmp_path):
    source = tmp_path/'video.mp4'
    source.write_bytes(b'video')
    cache = ModelOutputCache(source,tmp_path/'cache')
    spec = dict(weights_sha256='frozen',runtime={})
    recovery = PoseFragmentRecovery('unused.pt',spec,(3840,2160),cache)
    lower, upper = server_pair()
    lower.keypoints[5:7, 2] = .1
    crop = recovery_crop(lower,(3840,2160))
    upper.keypoints[11:13] = lower.keypoints[11:13]
    upper.keypoints[:, :2] -= crop[:2]
    box = np.array([[upper.bbox[0]-crop[0],upper.bbox[1]-crop[1],
                     upper.bbox[2]-crop[0],upper.bbox[3]-crop[1],.9,0]])
    raw = dict(boxes=array_payload(box),keypoints=array_payload(upper.keypoints[None]))
    frozen = deepcopy(raw)
    calls = []
    recovery._predict = lambda image: calls.append(image.shape) or raw
    first = recovery.repair([lower],3366,lambda:np.zeros((2160,3840,3),dtype=np.uint8))
    assert calls and first[0].pose_repair
    assert raw == frozen
    cache.require = True
    def forbidden():
        raise AssertionError('Warm replay must not decode pixels or run a model')
    recovery._predict = lambda _: forbidden()
    replay = recovery.repair([lower],3366,forbidden)
    np.testing.assert_array_equal(first[0].keypoints,replay[0].keypoints)
    with pytest.raises(CacheMiss,match='pose_fragment_crop_raw'):
        recovery.repair([lower],3367,forbidden)
    cache.db.close()


def test_duplicate_repair_runs_in_production_before_any_court_gate():
    from src.pipeline.analyzer.detection import DetectionMixin
    analyzer = DetectionMixin()
    result = analyzer._repair_pose_fragments(list(server_pair()),np.empty((0,0,3)),3376,None)
    assert len(result) == 1 and result[0].track_id == 530
