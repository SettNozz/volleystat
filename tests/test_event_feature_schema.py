from collections import deque

import numpy as np
import pytest

from action_recognition.event_clips.compute_features import compute_clip_features
from action_recognition.event_clips.config import NUM_FEATURES, NUM_TRAJ_FEATURES, SEQUENCE_LEN
from action_recognition.event_clips.feature_schema import (
    LEGACY_PIXEL_SCHEMA,
    RESOLUTION_NORMALIZED_SCHEMA,
    apply_checkpoint_feature_contract,
    validate_feature_schema,
)
from action_recognition.event_clips.prepare import (
    build_additional_segments,
    build_segments,
    build_trajectory_segments,
)
from action_recognition.inference.streaming_event_predictor import (
    StreamingEventPredictor,
    gate_event_probabilities,
)
from action_recognition.rally_spotting.prepare import build_continuous_features


def _scaled_frames(scale):
    return [
        {"ball": [200.0 * scale, 300.0 * scale],
         "p": [[180.0 * scale, 330.0 * scale], [900.0 * scale, 280.0 * scale]]},
        {"ball": [224.0 * scale, 282.0 * scale],
         "p": [[190.0 * scale, 324.0 * scale], [880.0 * scale, 290.0 * scale]]},
        {"ball": [260.0 * scale, 270.0 * scale],
         "p": [[205.0 * scale, 318.0 * scale], [860.0 * scale, 300.0 * scale]]},
    ]


def test_normalized_features_are_resolution_independent():
    features_720 = compute_clip_features(
        _scaled_frames(1.0),
        net_y=299.0,
        fps=30.0,
        frame_width=1280,
        frame_height=720,
        feature_schema=RESOLUTION_NORMALIZED_SCHEMA,
    )
    features_4k = compute_clip_features(
        _scaled_frames(3.0),
        net_y=897.0,
        fps=30.0,
        frame_width=3840,
        frame_height=2160,
        feature_schema=RESOLUTION_NORMALIZED_SCHEMA,
    )

    np.testing.assert_allclose(features_720.to_numpy(), features_4k.to_numpy(), atol=1e-6)
    assert features_4k.attrs["feature_schema"] == RESOLUTION_NORMALIZED_SCHEMA


def test_legacy_default_keeps_raw_pixel_semantics():
    features = compute_clip_features(_scaled_frames(1.0), net_y=299.0, fps=30.0)

    assert features.attrs["feature_schema"] == LEGACY_PIXEL_SCHEMA
    assert features.loc[0, "ball_x"] == pytest.approx(200.0)
    assert features.loc[0, "ball_y"] == pytest.approx(300.0)
    assert features.loc[1, "ball_speed"] == pytest.approx(30.0 * np.hypot(24.0, -18.0))
    assert features.loc[1, "ball_net_dist"] == pytest.approx(17.0)


def test_normalized_schema_requires_explicit_frame_size():
    with pytest.raises(ValueError, match="requires frame_size"):
        compute_clip_features(
            _scaled_frames(1.0),
            net_y=299.0,
            fps=30.0,
            feature_schema=RESOLUTION_NORMALIZED_SCHEMA,
        )


def test_streaming_and_offline_feature_builders_have_parity():
    frames = _scaled_frames(3.0)
    frames[0]["f"] = 10
    frames[1] = {"f": 11, "ball": None, "p": frames[1]["p"]}
    frames[2]["f"] = 12
    _, spotting = build_continuous_features({
        "frame_width": 3840,
        "frame_height": 2160,
        "fps": 30.0,
        "net_y": 897.0,
        "frames": frames,
    })
    expected = np.concatenate((spotting[:, :10], spotting[:, 12:20]), axis=1)

    predictor = StreamingEventPredictor.__new__(StreamingEventPredictor)
    predictor.feature_schema = RESOLUTION_NORMALIZED_SCHEMA
    predictor._buf = deque(maxlen=SEQUENCE_LEN)
    predictor._recent = deque(maxlen=predictor.SMOOTH_WINDOW)
    predictor._frame_count = 0
    predictor._prev_ball = None
    predictor._last_fsm_state = ""

    for frame in frames:
        predictor.add_frame(
            tuple(frame["ball"]) if frame["ball"] else None,
            {"team_a": frame["p"], "team_b": []},
            net_y=897.0,
            fps=30.0,
            frame_size=(3840, 2160),
        )

    actual = np.stack(predictor._buf)
    # Streaming assigns window-local frame_norm immediately before inference.
    expected[:, 7] = 0.0
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual[1, :NUM_TRAJ_FEATURES].sum() == pytest.approx(0.0)
    assert actual[2, 2] == pytest.approx(0.0)  # no velocity across an occlusion


def test_checkpoint_contract_masks_channels_absent_during_training():
    features = np.asarray([[12.0, 999.0, -3.0]], dtype=np.float32)
    normalized = apply_checkpoint_feature_contract(
        features,
        feature_mean=[10.0, 0.0, -1.0],
        feature_std=[2.0, 1.0, 4.0],
        feature_active_mask=[True, False, True],
    )
    np.testing.assert_allclose(normalized, [[1.0, 0.0, -0.5]])


def test_normalized_clip_streaming_and_spotting_agree_on_gaps_and_restarts():
    frames = [
        {"f": 0, "ball": [100.0, 200.0]},
        {"f": 1, "ball": None},
        {"f": 2, "ball": [300.0, 200.0]},
        {"f": 3, "ball": [900.0, 400.0], "ball_new": True},
        {"f": 4, "ball": [900.0, 400.0], "ball_out_of_frame": True},
        {"f": 5, "ball": [500.0, 300.0]},
    ]
    for frame in frames:
        frame["p"] = [[150.0, 240.0]]
        frame["pose_f"] = [0.1] * 8
    clip = compute_clip_features(
        frames, 350.0, 30.0, frame_width=1280, frame_height=720,
        feature_schema=RESOLUTION_NORMALIZED_SCHEMA,
    ).to_numpy()
    _, spotting = build_continuous_features({
        "frames": frames, "net_y": 350.0, "fps": 30.0,
        "frame_width": 1280, "frame_height": 720,
    })
    np.testing.assert_allclose(clip, np.concatenate((spotting[:, :10], spotting[:, 12:20]), axis=1))

    predictor = StreamingEventPredictor.__new__(StreamingEventPredictor)
    predictor.feature_schema = RESOLUTION_NORMALIZED_SCHEMA
    predictor._buf = deque(maxlen=SEQUENCE_LEN)
    predictor._recent = deque(maxlen=predictor.SMOOTH_WINDOW)
    predictor.reset()
    for frame in frames:
        predictor.add_frame(
            None if frame.get("ball_out_of_frame") else frame["ball"],
            {"team_a": frame["p"], "team_b": []}, 350.0, 30.0,
            pose_features=frame["pose_f"], frame_size=(1280, 720),
            ball_new=bool(frame.get("ball_new")),
        )
    clip[:, 7] = 0.0  # streaming frame_norm is assigned per inference window
    np.testing.assert_allclose(clip, np.stack(predictor._buf))
    assert np.all(clip[[1, 4], :2] == 0.0)
    assert np.all(clip[[2, 3, 5], 2] == 0.0)


def test_legacy_clip_features_keep_last_known_ball_compatibility():
    frames = [{"ball": [100.0, 200.0]}, {"ball": None}, {"ball": [130.0, 200.0]}]
    clip = compute_clip_features(frames, 350.0, 30.0)
    assert clip.loc[1, "ball_x"] == 100.0
    assert clip.loc[2, "ball_speed"] == 900.0


def test_fsm_gate_keeps_only_phase_possible_classes_and_renormalizes():
    labels = ["serve", "reception", "pass", "attack"]
    raw = np.asarray([0.7, 0.1, 0.15, 0.05], dtype=np.float32)

    serving = gate_event_probabilities(raw, labels, "serving")
    rally = gate_event_probabilities(raw, labels, "rally_active")
    idle = gate_event_probabilities(raw, labels, "serve_ready")

    np.testing.assert_allclose(serving, [0.875, 0.125, 0.0, 0.0])
    np.testing.assert_allclose(rally, [0.0, 0.0, 0.75, 0.25])
    np.testing.assert_allclose(idle, np.zeros(4))


def test_missing_checkpoint_schema_selects_legacy_mode():
    assert validate_feature_schema(None) == LEGACY_PIXEL_SCHEMA
    assert NUM_FEATURES > NUM_TRAJ_FEATURES


def test_first_serve_contact_gets_two_seconds_of_preroll():
    stats = {
        "rallies": [{
            "start_frame": 300,
            "serve_reception": {
                "serve": {"frame": 300},
                "reception": {"frame": 330},
            },
            "events": [
                {"frame": 300, "type": "serve", "team": "team_b"},
                {"frame": 330, "type": "reception", "team": "team_a"},
            ],
        }],
    }

    segments = list(build_segments(stats, fps=30.0))

    assert len(segments) == 2
    assert segments[0]["start_frame"] == 240
    assert segments[0]["contact_frame"] == 300
    assert segments[0]["event_types"] == ["serve"]
    assert segments[0]["label_source"] == "fsm_rule"
    assert segments[1]["start_frame"] == 300


def test_additional_contact_mining_excludes_labeled_frames_with_tolerance():
    stats = {
        "all_events": [
            {"frame": 100, "type": "serve", "team": "team_b"},
            {"frame": 200, "type": "pass", "team": "team_a"},
            {"frame": 300, "type": "attack", "team": "team_a"},
        ],
    }

    segments = list(build_additional_segments(
        stats,
        fps=30.0,
        excluded_frames=[100, 202],
        exclusion_tolerance=5,
    ))

    assert len(segments) == 1
    assert segments[0]["contact_frame"] == 300
    assert segments[0]["start_frame"] == 240
    assert segments[0]["primary_event_type"] == "attack"
    assert segments[0]["event_types"] == []
    assert segments[0]["label_source"] is None
    assert segments[0]["candidate_source"] == "all_events"


def test_trajectory_mining_finds_coherent_turn_and_respects_exclusion():
    points = [
        (100, 500), (105, 500), (110, 500), (115, 500),
        (115, 495), (115, 490), (115, 485),
    ]
    frames = [
        {"f": frame, "ball": list(point), "ball_new": False, "ball_interp": False}
        for frame, point in enumerate(points, start=10)
    ]

    segments = list(build_trajectory_segments(
        frames,
        fps=30.0,
        frame_width=1000,
        frame_height=1000,
        max_candidates=5,
    ))

    assert len(segments) == 1
    assert segments[0]["contact_frame"] == 13
    assert segments[0]["candidate_source"] == "trajectory_change"
    assert segments[0]["direction_change_deg"] == pytest.approx(90.0)

    excluded = list(build_trajectory_segments(
        frames,
        fps=30.0,
        frame_width=1000,
        frame_height=1000,
        excluded_frames=[14],
        exclusion_tolerance=2,
    ))
    assert excluded == []
