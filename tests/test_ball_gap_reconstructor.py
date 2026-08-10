from src.tracking.ball_gap_reconstructor import (
    BallGapReconstructor,
    ContactHint,
    ReconstructionConfig,
)


def _ann(frame, ball=None, *, bbox=None, ball_new=None, state=None):
    poses = []
    if bbox is not None:
        poses = [{"bb": bbox, "kp": [[0, 0, 0]] * 17}]
    return {
        "f": frame,
        "ball": list(ball) if ball is not None else None,
        "ball_new": ball_new,
        "poses": poses,
        "state": state,
    }


def test_short_gap_preserves_raw_and_builds_both_variants():
    frames = [_ann(0, (10, 10)), _ann(1), _ann(2), _ann(3, (40, 10))]
    summaries = BallGapReconstructor(1280, 720).apply(frames)

    assert frames[1]["ball_raw"] is None
    assert frames[1]["ball"] == [20.0, 10.0]
    assert frames[1]["ball_source"] == "interpolated_short"
    assert frames[1]["ball_conservative"] == [20.0, 10.0]
    assert frames[1]["ball_contact_aware"] == [20.0, 10.0]
    assert summaries["conservative"].reconstructed == 2


def test_long_gap_without_occlusion_stays_empty():
    frames = [_ann(0, (100, 200))]
    frames += [_ann(frame) for frame in range(1, 12)]
    frames.append(_ann(12, (340, 200)))

    BallGapReconstructor(1280, 720).apply(frames)

    assert all(frame["ball"] is None for frame in frames[1:-1])
    assert all(frame["ball_source"] == "missing" for frame in frames[1:-1])


def test_player_occlusion_allows_conservative_reconstruction():
    frames = [_ann(0, (100, 200))]
    frames += [_ann(frame, bbox=(90, 100, 360, 400)) for frame in range(1, 9)]
    frames.append(_ann(9, (280, 200)))

    BallGapReconstructor(1280, 720).apply(frames)

    assert all(frame["ball"] is not None for frame in frames)
    assert frames[4]["ball_source"] == "reconstructed_conservative"
    assert frames[4]["ball_confidence"] < 0.8


def test_track_restart_outside_live_play_is_not_bridged():
    frames = [_ann(0, (100, 200))]
    frames += [_ann(frame, bbox=(0, 0, 1280, 720)) for frame in range(1, 5)]
    frames.append(_ann(5, (220, 200), ball_new=True))

    BallGapReconstructor(1280, 720).apply(frames)

    assert all(frame["ball"] is None for frame in frames[1:-1])


def test_occluded_track_restart_during_live_play_is_reacquired():
    frames = [
        _ann(0, (60, 200), state="rally_active"),
        _ann(1, (80, 200), state="rally_active"),
        _ann(2, (100, 200), state="rally_active"),
    ]
    frames += [
        _ann(frame, bbox=(0, 0, 1280, 720), state="rally_active")
        for frame in range(3, 8)
    ]
    frames += [
        _ann(8, (220, 200), ball_new=True, state="rally_active"),
        _ann(9, (240, 200), state="rally_active"),
        _ann(10, (260, 200), state="rally_active"),
    ]

    BallGapReconstructor(1280, 720).apply(frames)

    assert all(frame["ball"] is not None for frame in frames)
    assert frames[5]["ball_source"] == "reconstructed_conservative"


def test_edge_exit_is_marked_out_of_frame_without_coordinates():
    frames = [_ann(0, (40, 300)), _ann(1, (15, 300))]
    frames += [_ann(frame) for frame in range(2, 7)]
    frames += [_ann(7, (18, 300)), _ann(8, (45, 300))]

    BallGapReconstructor(1280, 720).apply(frames)

    assert all(frame["ball"] is None for frame in frames[2:7])
    assert all(frame["ball_out_of_frame"] is True for frame in frames[2:7])


def test_contact_hint_creates_contact_aware_direction_break_only():
    config = ReconstructionConfig(conservative_max_gap_frames=20)
    frames = [_ann(0, (100, 300)), _ann(1, (130, 300))]
    frames += [_ann(frame, bbox=(180, 180, 340, 500)) for frame in range(2, 9)]
    frames += [_ann(9, (360, 240)), _ann(10, (390, 210))]

    BallGapReconstructor(1280, 720, config=config).apply(
        frames,
        active_mode="contact_aware",
        contact_hints=[ContactHint(5, "reception")],
    )

    assert frames[5]["ball_contact_reconstructed"] is True
    assert frames[5]["ball_source"] == "reconstructed_contact_aware"
    assert frames[5]["ball_contact_aware"] is not None
    assert frames[5]["ball_conservative"] is not None


def test_leading_and_trailing_gaps_are_not_extrapolated():
    frames = [_ann(0), _ann(1), _ann(2, (100, 100)), _ann(3), _ann(4)]

    BallGapReconstructor(1280, 720).apply(frames)

    assert frames[0]["ball"] is None
    assert frames[1]["ball"] is None
    assert frames[3]["ball"] is None
    assert frames[4]["ball"] is None
