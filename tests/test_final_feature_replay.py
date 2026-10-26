from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline.game_analyzer import GameAnalyzer, GameStats, _EventOverlayState
from src.tracking.pose_feature_extractor import POSE_FEATURE_NAMES, PoseFeatureExtractor


def _pose(x, team, track_id):
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index in (5, 6):
        keypoints[index] = [x, 100.0, 1.0]
    for index in (9, 10):
        keypoints[index] = [x, 80.0, 1.0]
    for index in (15, 16):
        keypoints[index] = [x, 190.0, 1.0]
    return {"id": track_id, "team": team, "bb": [x - 20, 50, x + 20, 200], "kp": keypoints}


class _RecordingPredictor:
    def __init__(self):
        self.calls = ["provisional"]
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.calls.clear()

    def add_frame(self, ball, teams, net_y, fps, **kwargs):
        self.calls.append((ball, teams, net_y, fps, kwargs))
        return "attack" if ball is not None else None


def _analyzer(predictor):
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer._event_predictor = predictor
    analyzer.calibration = SimpleNamespace(
        net_y=150.0, net_tape_top=90.0, frame_width=1280, frame_height=720,
    )
    return analyzer


def test_final_replay_replaces_stale_features_and_predictions_without_changing_observations():
    predictor = _RecordingPredictor()
    analyzer = _analyzer(predictor)
    poses = [_pose(100.0, "a", 1), _pose(600.0, "b", 2)]
    frames = [
        {"f": 0, "ball": [600.0, 80.0], "ball_raw": [100.0, 80.0],
         "ball_source": "reconstructed_conservative", "ball_confidence": 0.9,
         "ball_new": True, "state": "rally_active", "poses": poses,
         "pose_f": [-1.0] * 8, "event_model": "serve"},
        {"f": 1, "ball": None, "ball_out_of_frame": True,
         "state": "serve_ready", "poses": poses, "event_model": "serve"},
    ]
    original = deepcopy(frames)
    # Features follow exported ball; FSM reconstruction policy is independent.
    analyzer.reconstruction_affects_fsm = False
    analyzer._recompute_features_and_predictions(frames, 30.0)

    assert predictor.resets == 1
    assert len(predictor.calls) == 2
    assert predictor.calls[0][0] == (600.0, 80.0)
    assert predictor.calls[0][4] == {
        "fsm_state": "rally_active", "pose_features": frames[0]["pose_f"],
        "frame_size": (1280, 720), "ball_new": True,
    }
    expected = PoseFeatureExtractor().update(
        0, analyzer._poses_from_annotation(frames[0]), (600.0, 80.0), 90.0,
    )
    assert frames[0]["pose_f"] == expected
    assert frames[0]["pose_f"][POSE_FEATURE_NAMES.index("np_team_side")] == 1.0
    assert frames[0]["event_model"] == "attack"
    assert predictor.calls[1][0] is None
    assert predictor.calls[1][4]["fsm_state"] == "serve_ready"
    assert "event_model" not in frames[1]
    for before, after in zip(original, frames):
        assert {k: v for k, v in before.items() if k not in {"pose_f", "event_model"}} == {
            k: v for k, v in after.items() if k not in {"pose_f", "event_model"}
        }
    # Running another pass/video must not accumulate pose or predictor history.
    first = deepcopy(frames)
    analyzer._recompute_features_and_predictions(frames, 30.0)
    assert predictor.resets == 2
    assert frames == first


@pytest.mark.parametrize("calibrated", [True, False])
def test_final_pose_features_are_recomputed_with_prediction_disabled(calibrated):
    analyzer = _analyzer(None)
    if not calibrated:
        analyzer.calibration = None
    frames = [{"f": 0, "ball": None, "poses": [], "pose_f": [99.0] * 8,
               "event_model": "attack"}]
    analyzer._recompute_features_and_predictions(frames, 30.0)
    assert frames[0]["pose_f"] == [0.0] * 8
    assert "event_model" not in frames[0]


def test_replay_rebuilds_jump_history_in_frame_order():
    analyzer = _analyzer(None)
    frames = [{"f": f, "ball": [100.0, 80.0], "poses": [_pose(100.0, "b", 1)]}
              for f in range(12)]
    for index in (15, 16):
        frames[-1]["poses"][0]["kp"][index][1] = 160.0
    analyzer._recompute_features_and_predictions(frames, 30.0)
    jump = POSE_FEATURE_NAMES.index("np_jump")
    assert frames[0]["pose_f"][jump] == 0.0
    assert frames[-1]["pose_f"][jump] == pytest.approx(0.2)
    first = deepcopy(frames)
    analyzer._recompute_features_and_predictions(frames, 30.0)
    assert frames == first


def test_replay_predictor_uses_exported_player_centers():
    predictor = _RecordingPredictor()
    analyzer = _analyzer(predictor)
    # Full poses may be in mixed team order; p is grouped by team.
    frame = {"f": 0, "ball": [600.0, 80.0], "state": "rally_active",
             "poses": [_pose(600.0, "b", 2), _pose(100.0, "a", 1)],
             "p": [[100.1, 125.1], [600.1, 125.1]]}
    analyzer._recompute_features_and_predictions([frame], 30.0)
    assert predictor.calls[0][1] == {
        "team_a": [[100.1, 125.1]], "team_b": [[600.1, 125.1]],
    }


def test_shared_overlay_preserves_cooldown_gating_and_sticky_display():
    overlay = _EventOverlayState(30.0)
    assert overlay.update("serve", "serve_ready", 0) is None
    assert overlay.update("serve", "serving", 1) == "serve"
    assert overlay.update("serve", "serving", 2) is None
    assert overlay.update("serve", "serving", 241) == "serve"
    assert overlay.update("attack", "rally_active", 242) == "attack"
    assert overlay.update("pass", "rally_end", 243) == "attack"
    assert overlay.update(None, "rally_end", 244) is None
    assert overlay.update("pass", "rally_end", 302) is None


def test_processing_runs_feature_replay_after_trajectory_and_clean_fsm(monkeypatch):
    analyzer = _analyzer(_RecordingPredictor())
    analyzer._court_gate = None
    analyzer.ball_gate = SimpleNamespace(reset=lambda: None, acquired_this_frame=False)
    analyzer.ball_batch_size = 1
    analyzer.ball_detector_2 = None
    analyzer.skip_frames = 0
    analyzer.output_video = False
    analyzer.model_provenance = {}
    analyzer.ball_trajectory_dp = True
    analyzer.suppress_static_ball_sources = False
    analyzer.match_segments = []
    analyzer.team_identity_map = {}
    analyzer._emit_progress = lambda **kwargs: None
    analyzer._detect_ball = lambda frame, index: ((100.0, 80.0), 0.9)
    analyzer._detect_players = lambda frame, **kwargs: ({"team_a": [], "team_b": []}, [])
    analyzer._make_fsm = lambda fps: SimpleNamespace(
        get_current_status=lambda: {"game_state": "serving"}, update=lambda **kwargs: [],
    )
    def solve(stats):
        stats.frame_annotations[0]["ball"] = [500.0, 80.0]
        return True
    analyzer._apply_trajectory_dp = solve
    def smooth(stats):
        stats.frame_annotations[0]["ball"] = [600.0, 80.0]
    analyzer._smooth_ball_trajectory = smooth
    def clean_fsm(annotations, fps):
        annotations[0]["state"] = "rally_active"
        return [], [], []
    analyzer._recompute_events_on_clean_trajectory = clean_fsm
    analyzer._get_projector = lambda: None
    monkeypatch.setattr("src.pipeline.game_analyzer.AttackHeatmap", lambda fps: SimpleNamespace(
        analyze=lambda *args, **kwargs: {"total_attacks": 0},
    ))
    # Heatmap only needs these calibration fields in this focused integration.
    analyzer.calibration.net_floor_left = analyzer.calibration.net_floor_right = None
    analyzer.calibration.net_x_left = 0
    analyzer.calibration.net_x_right = 1280
    stats = analyzer._process_frames_iter([(0, np.zeros((2, 2, 3), np.uint8))], 30.0, 1, GameStats())
    assert analyzer._event_predictor.resets == 2
    assert len(analyzer._event_predictor.calls) == 1
    call = analyzer._event_predictor.calls[0]
    assert call[0] == (600.0, 80.0)
    assert call[4]["fsm_state"] == "rally_active"
    assert stats.frame_annotations[0]["event_model"] == "attack"
