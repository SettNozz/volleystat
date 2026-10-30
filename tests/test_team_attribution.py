"""Deriving the side of a contact, and measuring that derivation honestly.

The point of the labelling tool is to find out whether geometry answers "which
team touched the ball" well enough to skip annotating it. That only means
something if the prediction is withheld until the human has answered, and if
the two derivations are scored separately -- they fail differently, and the
cheap one fails in a way that is easy to miss.
"""

import numpy as np
import pytest

from action_recognition.rally_spotting import team_attribution as ta


def _pose(team, points, confidence=0.9):
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, (x, y) in points.items():
        keypoints[index] = [float(x), float(y), confidence]
    return {"team": team, "kp": keypoints, "id": f"{team}-track"}


FLOOR_LEFT = (1200.0, 1200.0)
FLOOR_RIGHT = (2832.0, 1200.0)


def test_team_name_accepts_both_spellings():
    assert ta.team_name("a") == "team_a"
    assert ta.team_name("team_b") == "team_b"
    assert ta.team_name(None) is None
    assert ta.team_name("spectator") is None


def test_divider_y_interpolates_along_a_sloped_line():
    assert ta.divider_y_at(1200.0, (1200.0, 1000.0), (2832.0, 1200.0)) == pytest.approx(1000.0)
    assert ta.divider_y_at(2832.0, (1200.0, 1000.0), (2832.0, 1200.0)) == pytest.approx(1200.0)
    assert ta.divider_y_at(2016.0, (1200.0, 1000.0), (2832.0, 1200.0)) == pytest.approx(1100.0)


def test_divider_needs_a_line():
    assert ta.divider_y_at(100.0, None, FLOOR_RIGHT) is None


def test_ball_side_reads_the_half_the_pixel_falls_in():
    above = ta.ball_side_team((2000.0, 900.0), FLOOR_LEFT, FLOOR_RIGHT)
    below = ta.ball_side_team((2000.0, 1500.0), FLOOR_LEFT, FLOOR_RIGHT)
    assert above["team"] == "team_a"
    assert below["team"] == "team_b"
    assert above["margin_px"] == pytest.approx(-300.0)


def test_ball_side_is_wrong_for_a_high_ball_and_says_so():
    # A near-side player serving: the ball is high, so the pixel sits above the
    # divider and the cheap derivation claims the far side.
    result = ta.ball_side_team((2000.0, 600.0), FLOOR_LEFT, FLOOR_RIGHT)
    assert result["team"] == "team_a"
    assert "airborne" in result["caveat"]


def test_nearest_player_picks_the_closer_side_and_reports_the_margin():
    poses = [
        _pose("a", {9: (1000.0, 1000.0)}),
        _pose("b", {10: (2010.0, 610.0)}),
    ]
    result = ta.nearest_player_team(poses, (2000.0, 600.0))
    assert result["team"] == "team_b"
    assert result["distance_px"] == pytest.approx(np.hypot(10.0, 10.0))
    assert result["margin_px"] > 0
    assert result["wrist_distance_px"] == pytest.approx(result["distance_px"])


def test_nearest_player_ignores_low_confidence_keypoints():
    poses = [_pose("a", {9: (2000.0, 600.0)}, confidence=0.05)]
    assert ta.nearest_player_team(poses, (2000.0, 600.0))["team"] is None


def test_nearest_player_needs_a_ball_and_poses():
    assert ta.nearest_player_team([], (1.0, 2.0))["team"] is None
    assert ta.nearest_player_team([_pose("a", {9: (0.0, 0.0)})], None)["team"] is None


def test_attribute_prefers_the_player_derivation():
    poses = [_pose("b", {10: (2010.0, 610.0)})]
    result = ta.attribute(poses, (2000.0, 600.0), FLOOR_LEFT, FLOOR_RIGHT)
    assert result["nearest_player"]["team"] == "team_b"
    assert result["ball_side"]["team"] == "team_a"
    assert result["methods_agree"] is False
    assert result["prediction"] == "team_b"
    assert result["prediction_source"] == "nearest_player"


def test_attribute_falls_back_to_the_ball_when_no_pose_helps():
    result = ta.attribute([], (2000.0, 1500.0), FLOOR_LEFT, FLOOR_RIGHT)
    assert result["prediction"] == "team_b"
    assert result["prediction_source"] == "ball_side"


def test_attribute_reports_nothing_rather_than_guessing():
    result = ta.attribute([], None, FLOOR_LEFT, FLOOR_RIGHT)
    assert result["prediction"] is None
    assert result["prediction_source"] is None
    assert result["methods_agree"] is None


def _record(human, nearest, side):
    return {
        "human": human,
        "attribution": {
            "nearest_player": {"team": nearest},
            "ball_side": {"team": side},
            "prediction": nearest or side,
        },
    }


def test_scoring_separates_coverage_from_accuracy():
    records = [
        _record("team_a", "team_a", "team_a"),
        _record("team_b", "team_b", "team_a"),
        _record("team_a", None, "team_b"),
    ]
    nearest = ta.score_against_human(records, "nearest_player").to_dict()
    assert nearest["total"] == 3
    assert nearest["predicted"] == 2
    assert nearest["correct"] == 2
    # ``to_dict`` rounds for a readable report, so compare at that precision.
    assert nearest["coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert nearest["accuracy_where_predicted"] == pytest.approx(1.0)

    side = ta.score_against_human(records, "ball_side").to_dict()
    assert side["predicted"] == 3
    assert side["correct"] == 1
    assert side["confusion"]["team_b->team_a"] == 1


def test_scoring_skips_contacts_the_human_left_unlabelled():
    records = [
        _record("team_a", "team_a", "team_a"),
        _record("unsure", "team_b", "team_b"),
        _record(None, "team_a", "team_a"),
    ]
    report = ta.score_against_human(records, "prediction").to_dict()
    # Only the one real label counts; "unsure" is neither right nor wrong.
    assert report["total"] == 1
    assert report["predicted"] == 1
    assert report["correct"] == 1
