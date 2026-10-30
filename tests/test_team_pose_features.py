import numpy as np
from src.detection.player_pose import PlayerPose
from src.tracking.pose_feature_extractor import PoseFeatureExtractor
from action_recognition.event_clips.team_pose_features import (
    team_contact_features,point_segment_distance,observation_stats,normalize_observations,VALUE_NAMES,FEATURE_NAMES,
)
from scripts.evaluation.refresh_team_pose_dataset import intervals


def person(team='team_a'):
    kp=np.zeros((17,3),np.float32)
    for i in (0,5,6,7,8,9,10,11,12,13,14,15,16):kp[i]=[100,100,.9]
    kp[7]=[80,100,.9];kp[9]=[120,100,.9]
    return PlayerPose(team,(100,100),(60,50,140,150),kp,track_id=1)


def test_forearm_contact_uses_segment_not_only_wrists():
    p=person();fx=PoseFeatureExtractor();fx.update(0,[p],(100,100),90)
    values,valid=team_contact_features([p],(100,100),90,fx)
    assert values.shape==(24,) and valid.shape==(24,)
    assert valid[4] and values[4]==0
    assert values[0]==np.float32(1/6) and values[12]==0
    assert valid[12] and not valid[13:].any()


def test_missing_ball_invalidates_contact_distances_without_erasing_visible_team():
    p=person();fx=PoseFeatureExtractor();fx.update(0,[p],None,90)
    values,valid=team_contact_features([p],None,90,fx)
    assert not valid[1:8].any()
    assert valid[0] and valid[8] and valid[9] and valid[11]
    assert not valid[10]  # No ankle baseline yet.


def test_all_joint_confidence_missing_does_not_invent_contact_evidence():
    p=person();p.keypoints[:]=0;fx=PoseFeatureExtractor();fx.update(0,[p],(100,100),90)
    values,valid=team_contact_features([p],(100,100),90,fx)
    assert valid[1] and not valid[2:11].any()
    assert values[2:11].tolist()==[0]*9


def test_team_swap_swaps_blocks_without_reading_labels():
    a=person('team_a');b=person('team_b');fx=PoseFeatureExtractor()
    va,ma=team_contact_features([a],(100,100),90,fx)
    vb,mb=team_contact_features([b],(100,100),90,fx)
    np.testing.assert_array_equal(va[:12],vb[12:]);np.testing.assert_array_equal(ma[:12],mb[12:])


def test_train_statistics_exclude_missing_observations_and_impute_mean():
    values=np.array([[2,0],[4,100000]],np.float32);valid=np.array([[True,False],[True,False]])
    mean,std=observation_stats([(values,valid)])
    np.testing.assert_array_equal(mean,[3,0]);np.testing.assert_array_equal(std,[1,1])
    normalized=normalize_observations(values,valid,mean,std)
    np.testing.assert_array_equal(normalized,[[-1,0,1,0],[1,0,1,0]])


def test_causal_warmup_ranges_merge_without_losing_frames():
    assert intervals([dict(start_frame=100,end_frame=130),dict(start_frame=190,end_frame=220)],90)==[[10,220]]
    assert intervals([dict(start_frame=0,end_frame=3),dict(start_frame=300,end_frame=310)],90)==[[0,3],[210,310]]


def test_schema_is_explicit_and_ordered():
    assert len(VALUE_NAMES)==42 and len(FEATURE_NAMES)==84
    assert len(set(FEATURE_NAMES))==84
    assert FEATURE_NAMES[42:] == tuple('valid_'+n for n in VALUE_NAMES)
    assert np.isclose(point_segment_distance((1,1),(0,0),(2,0)),1)


def test_candidate_channels_cannot_mix_contact_and_raised_arms_of_other_player():
    from action_recognition.event_clips.team_pose_features import candidate_team_features,contact_candidate
    near=person();other=person();other.track_id=2
    other.keypoints[:,:2]+=[500,-100]
    other.keypoints[9]=[600,0,.9]
    fx=PoseFeatureExtractor();fx.update(0,[near,other],(100,100),90)
    assert contact_candidate([other,near],(100,100)) is near
    actual,valid=candidate_team_features([near,other],(100,100),90,fx)
    expected,mask=team_contact_features([near],(100,100),90,fx)
    np.testing.assert_array_equal(actual[1:12],expected[1:12])
    np.testing.assert_array_equal(valid[1:12],mask[1:12])
    assert np.isclose(actual[0],2/6)


def test_candidate_missing_ball_or_joints_keeps_count_without_fake_person():
    from action_recognition.event_clips.team_pose_features import candidate_team_features
    p=person();fx=PoseFeatureExtractor()
    v,m=candidate_team_features([p],None,90,fx)
    assert m[0] and m[12] and not m[1:12].any()
    p.keypoints[:]=0
    v,m=candidate_team_features([p],(100,100),90,fx)
    assert not m[1:12].any() and not v[1:12].any()
