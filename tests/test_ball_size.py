"""The saved ball box size follows the observation, never the frame."""

from src.pipeline.game_analyzer import GameAnalyzer, GameStats
from src.tracking.ball_size import (
    drop_unobserved_ball_sizes, has_observed_ball, observed_ball_size,
)
from src.tracking.ball_track_gate import BallCandidate


def test_observed_ball_size_rounds_and_rejects_empty_values():
    assert observed_ball_size(36.04) == 36.0
    assert observed_ball_size(None) is None
    assert observed_ball_size(0.0) is None


def test_only_detector_observations_keep_a_size():
    frames = [
        {"f": 0, "ball": [10.0, 10.0], "ball_size": 30.0},
        {"f": 1, "ball": None, "ball_size": 30.0},                       # removed later
        {"f": 2, "ball": [12.0, 10.0], "ball_interp": True, "ball_size": 30.0},
        {"f": 3, "ball": [13.0, 10.0], "ball_source": "reconstructed", "ball_size": 30.0},
        {"f": 4, "ball": [14.0, 10.0], "ball_source": "detected", "ball_size": 31.5},
        {"f": 5, "ball": None, "ball_out_of_frame": True, "ball_size": 30.0},
    ]
    assert drop_unobserved_ball_sizes(frames) == 4
    assert [f["ball_size"] for f in frames] == [30.0, None, None, None, 31.5, None]
    assert has_observed_ball(frames[0]) and not has_observed_ball(frames[2])


def test_trajectory_solver_writes_size_for_detections_not_for_bridged_frames():
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.suppress_static_ball_sources = False
    analyzer.ball_reconstruction_mode = "off"
    analyzer._ball_px_scale = 1.0
    # A steady flight with frame 3 missing: the solver bridges that frame.
    analyzer._ball_candidates = {
        frame: [BallCandidate(x=100.0 + 12.0 * frame, y=200.0, confidence=0.9, size=30.0 + frame)]
        for frame in (0, 1, 2, 4, 5, 6)
    }
    stats = GameStats(video_name="t", fps=30.0)
    stats.frame_annotations = [{"f": frame, "ball": None} for frame in range(7)]

    assert analyzer._apply_trajectory_dp(stats) is True

    by_frame = {ann["f"]: ann for ann in stats.frame_annotations}
    assert by_frame[2]["ball_size"] == 32.0
    assert by_frame[4]["ball_size"] == 34.0
    assert by_frame[3]["ball"] is not None and by_frame[3]["ball_interp"] is True
    assert by_frame[3]["ball_size"] is None


def test_live_gate_remembers_the_selected_box_size():
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.ball_trajectory_dp = False
    chosen = BallCandidate(x=5.0, y=6.0, confidence=0.8, size=41.5)

    class Gate:
        def update(self, _frame, candidates):
            return candidates[0] if candidates else None

    analyzer.ball_gate = Gate()
    assert analyzer._select_ball_candidate([chosen], 0) == ((5.0, 6.0), 0.8)
    assert analyzer._last_ball_size == 41.5
    assert analyzer._select_ball_candidate([], 1) == (None, 0.0)
    assert analyzer._last_ball_size is None
