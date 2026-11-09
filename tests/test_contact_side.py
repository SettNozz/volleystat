"""Side-of-net proposals from ball box size and possession order."""

from types import SimpleNamespace

import pytest

from src.pipeline.game_analyzer import GameAnalyzer, GameStats
from src.tracking import contact_side as cs

FPS = 30.0


def calibration(**overrides):
    """End-on camera: net tape 234 px above the floor line => 21 px ball at the net."""
    values = dict(
        net_floor_left=(100.0, 600.0), net_floor_right=(1100.0, 600.0),
        net_tape_top=366.0, net_tape_endpoint_y=None, net_y=366.0, net_height_m=2.34,
        homography=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def frames(sizes):
    """One annotation per frame; ``sizes`` maps frame -> observed box size."""
    return [
        {"f": index, "ball": [10.0, 10.0] if index in sizes else None,
         "ball_size": sizes.get(index)}
        for index in range(max(sizes) + 20)
    ]


def event(frame, kind):
    return {"frame": frame, "type": kind, "team": None}


def test_net_plane_size_comes_from_tape_height_over_the_floor_line():
    geometry = cs.net_plane_ball_px(calibration())
    assert geometry["status"] == "ok"
    assert geometry["ball_px"] == pytest.approx(0.21 * 234 / 2.34)


def test_missing_or_degenerate_net_geometry_abstains():
    assert cs.net_plane_ball_px(None)["status"] == "no_net_geometry"
    assert cs.net_plane_ball_px(calibration(net_floor_left=None))["status"] == "no_net_geometry"
    # Tape below the floor line: not a usable net.
    assert cs.net_plane_ball_px(calibration(net_tape_top=700.0, net_y=700.0))["status"] == "no_net_geometry"


def test_side_on_camera_and_inconsistent_calibration_abstain():
    # world -> image with equal baselines: both halves equally far away.
    side_on = [[100.0, 0.0, 600.0], [0.0, 20.0, 600.0], [0.0, 0.0, 1.0]]
    assert cs.net_plane_ball_px(calibration(homography=side_on))["status"] == "camera_not_end_on"
    # End-on (near baseline wider), but the court width implies a far larger scale.
    end_on = [[400.0, 0.0, 600.0], [0.0, 20.0, 600.0], [0.0, 0.03, 1.0]]
    report = cs.net_plane_ball_px(calibration(homography=end_on))
    assert report["baseline_width_ratio"] >= cs.MIN_END_ON_WIDTH_RATIO
    assert report["status"] == "geometry_inconsistent"


def test_large_ball_is_near_small_ball_is_far_and_team_is_untouched():
    events = [event(10, "serve"), event(60, "reception")]
    report = cs.assign_contact_sides(events, frames({10: 40.0, 60: 14.0}), FPS, calibration())
    assert report["status"] == "completed" and report["net_ball_px"] == pytest.approx(21.0)
    assert [e["side"] for e in events] == ["b", "a"]
    assert all(e["team"] is None for e in events)
    assert events[0]["side_ball_px"] == 40.0 and events[0]["side_source"] == "ball_size_possession"


def test_attack_takes_the_side_of_its_possession_not_of_its_own_box():
    # Reception and set are clearly far; the attack box at the net reads near.
    events = [event(10, "serve"), event(60, "reception"), event(90, "set"), event(120, "attack")]
    sizes = {10: 40.0, 60: 14.0, 90: 15.0, 120: 23.0}
    cs.assign_contact_sides(events, frames(sizes), FPS, calibration())
    assert [e["side"] for e in events] == ["b", "a", "a", "a"]


def test_weak_possession_alternates_against_the_previous_one_inside_a_rally():
    # After a clearly far attack, a lone dig whose box sits at the net size.
    events = [event(10, "serve"), event(60, "reception"), event(90, "set"), event(120, "attack"),
              event(150, "dig")]
    sizes = {10: 40.0, 60: 14.0, 90: 15.0, 120: 20.0, 150: 21.5}
    cs.assign_contact_sides(events, frames(sizes), FPS, calibration())
    assert events[4]["side"] == "b" and events[4]["side_source"] == "possession_alternation"
    # A serve opens a new rally and owes nothing to the previous possession.
    events = [event(10, "attack"), event(200, "serve")]
    cs.assign_contact_sides(events, frames({10: 14.0}), FPS, calibration())
    assert events[1]["side"] is None


def test_contact_without_any_evidence_stays_unassigned():
    # 13 s after the serve: too long a pause to assume the sides alternated.
    events = [event(10, "serve"), event(400, "dig")]
    report = cs.assign_contact_sides(events, frames({10: 40.0}), FPS, calibration())
    assert events[1]["side"] is None and events[1]["side_score"] is None
    assert report["assigned"] == 1 and report["unassigned"] == 1


def test_non_contact_events_are_ignored_and_reconstructed_points_are_not_evidence():
    events = [event(10, "serve"), {"frame": 30, "type": "net_cross", "team": None}]
    annotations = frames({10: 40.0})
    annotations[10]["ball_size"] = None          # e.g. a reconstructed point
    report = cs.assign_contact_sides(events, annotations, FPS, calibration())
    assert report["status"] == "no_ball_sizes"
    assert "side" not in events[1] and events[0]["side"] is None


def test_box_sizes_that_disagree_with_the_calibration_abstain_for_the_whole_video():
    # Every contact twice the net size: the calibration or detector scale is off.
    count = cs.MIN_CONTACTS_FOR_SIZE_CHECK
    events = [event(10 + 30 * i, "set") for i in range(count)]
    sizes = {e["frame"]: 60.0 for e in events}
    report = cs.assign_contact_sides(events, frames(sizes), FPS, calibration())
    assert report["status"] == "size_distribution_mismatch"
    assert all(e["side"] is None for e in events)


def test_pipeline_pass_is_additive_and_can_be_disabled():
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.calibration = calibration()
    stats = GameStats(video_name="t", fps=FPS)
    stats.frame_annotations = frames({10: 40.0, 60: 14.0})
    stats.all_events = [event(10, "serve"), event(60, "reception")]
    before = [dict(e) for e in stats.all_events]

    analyzer._assign_contact_sides(stats, FPS)
    assert stats.contact_side["status"] == "completed"
    assert [e["side"] for e in stats.all_events] == ["b", "a"]
    for old, new in zip(before, stats.all_events):
        assert {k: new[k] for k in old} == old            # nothing existing changed
    assert stats.to_dict()["contact_side"]["method"] == cs.METHOD

    analyzer.enable_contact_side = False
    stats.all_events = [dict(e) for e in before]
    analyzer._assign_contact_sides(stats, FPS)
    assert stats.contact_side == {"status": "disabled"} and stats.all_events == before


def test_pipeline_pass_never_fails_the_analysis(monkeypatch):
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.calibration = calibration()
    stats = GameStats(video_name="t", fps=FPS)
    stats.all_events = [event(10, "serve")]

    def boom(*_args, **_kwargs):
        raise RuntimeError("bad geometry")

    monkeypatch.setattr(cs, "assign_contact_sides", boom)
    analyzer._assign_contact_sides(stats, FPS)
    assert stats.contact_side["status"] == "error"
    assert stats.all_events == [event(10, "serve")]
