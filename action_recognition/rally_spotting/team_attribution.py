"""Which side of the net made a contact, from geometry alone.

The contact spotter says what happened and when, never who. For match
statistics the side matters as much as the type, so the question is whether it
can be derived rather than annotated.

Two independent derivations, because they fail differently:

* **nearest player** -- the team of the player whose nearest keypoint is
  closest to the ball at the contact frame. Players stand on the floor, where
  the calibrated divider is trustworthy, and a contact is by definition made by
  a player. This is the principled one.
* **ball side** -- which side of the projected net-floor line the ball pixel
  sits on. Cheap, and wrong whenever the ball is high: an airborne ball on the
  near side projects past the divider and reads as the far side. Kept because
  it needs no poses, and because measuring how often it is wrong is the point.

``team_a`` is the far side of the court, above the divider in the image;
``team_b`` is the near side. That is the convention the pipeline's own poses
use and it was confirmed against the saved analyses, not assumed.

Nothing here decides anything on its own. It produces a prediction and the
evidence behind it, so a human label can judge both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "team_attribution_v1"
TEAMS = ("team_a", "team_b")
# COCO-17 wrists, reported separately because a contact is often but not
# always made with the hands.
WRIST_KEYPOINTS = (9, 10)
MIN_KEYPOINT_CONFIDENCE = 0.3


def team_name(value: str | None) -> str | None:
    """Normalise ``a``/``team_a`` to ``team_a``."""
    if value is None:
        return None
    text = str(value)
    if text in TEAMS:
        return text
    if text in ("a", "b"):
        return f"team_{text}"
    return None


def divider_y_at(
    x: float, left: Sequence[float] | None, right: Sequence[float] | None,
) -> float | None:
    """Image y of the net-floor line at ``x``, or ``None`` without one."""
    if not left or not right:
        return None
    x0, y0 = float(left[0]), float(left[1])
    x1, y1 = float(right[0]), float(right[1])
    if abs(x1 - x0) < 1e-9:
        return y0
    return y0 + (y1 - y0) * (float(x) - x0) / (x1 - x0)


def ball_side_team(
    ball: Sequence[float] | None,
    net_floor_left: Sequence[float] | None,
    net_floor_right: Sequence[float] | None,
) -> dict[str, Any]:
    """Side of the divider the ball pixel falls on.

    The ball is usually in the air, and this treats it as if it were on the
    floor, so a high ball reads as belonging to the far side. The returned
    ``margin_px`` says how far from the line it was, which is the only honest
    way to read the answer.
    """
    if ball is None:
        return {"team": None, "reason": "no ball position"}
    line_y = divider_y_at(ball[0], net_floor_left, net_floor_right)
    if line_y is None:
        return {"team": None, "reason": "no net floor line"}
    margin = float(ball[1]) - float(line_y)
    return {
        "team": "team_b" if margin > 0 else "team_a",
        "margin_px": margin,
        "divider_y": float(line_y),
        "caveat": "treats an airborne ball as if it were on the floor",
    }


def _keypoint_distances(
    pose: Mapping[str, Any], ball: Sequence[float],
) -> tuple[float | None, float | None]:
    """Closest keypoint distance, and the closest wrist distance."""
    keypoints = pose.get("kp") or []
    best = None
    best_wrist = None
    for index, point in enumerate(keypoints):
        if len(point) < 3 or float(point[2]) < MIN_KEYPOINT_CONFIDENCE:
            continue
        distance = float(np.hypot(float(point[0]) - ball[0], float(point[1]) - ball[1]))
        if best is None or distance < best:
            best = distance
        if index in WRIST_KEYPOINTS and (best_wrist is None or distance < best_wrist):
            best_wrist = distance
    return best, best_wrist


def nearest_player_team(
    poses: Sequence[Mapping[str, Any]], ball: Sequence[float] | None,
) -> dict[str, Any]:
    """Team of the player closest to the ball, with the margin to the other side.

    ``margin_px`` is the gap between the closest player of the chosen team and
    the closest player of the other team. A small margin means the frame does
    not settle it, which is exactly when a human label is worth having.
    """
    if ball is None:
        return {"team": None, "reason": "no ball position"}
    if not poses:
        return {"team": None, "reason": "no poses on this frame"}
    best: dict[str, dict[str, Any]] = {}
    for pose in poses:
        team = team_name(pose.get("team"))
        if team is None:
            continue
        distance, wrist = _keypoint_distances(pose, ball)
        if distance is None:
            continue
        current = best.get(team)
        if current is None or distance < current["distance_px"]:
            best[team] = {
                "distance_px": distance,
                "wrist_distance_px": wrist,
                "track_id": pose.get("id"),
            }
    if not best:
        return {"team": None, "reason": "no pose with usable keypoints"}
    ordered = sorted(best.items(), key=lambda item: item[1]["distance_px"])
    team, winner = ordered[0]
    runner_up = ordered[1][1]["distance_px"] if len(ordered) > 1 else None
    return {
        "team": team,
        "distance_px": winner["distance_px"],
        "wrist_distance_px": winner["wrist_distance_px"],
        "track_id": winner["track_id"],
        "other_distance_px": runner_up,
        "margin_px": (
            None if runner_up is None else runner_up - winner["distance_px"]
        ),
    }


def attribute(
    poses: Sequence[Mapping[str, Any]],
    ball: Sequence[float] | None,
    net_floor_left: Sequence[float] | None,
    net_floor_right: Sequence[float] | None,
) -> dict[str, Any]:
    """Both derivations for one contact, plus whether they agree."""
    nearest = nearest_player_team(poses, ball)
    side = ball_side_team(ball, net_floor_left, net_floor_right)
    agree = (
        None if nearest["team"] is None or side["team"] is None
        else nearest["team"] == side["team"]
    )
    return {
        "schema": SCHEMA,
        "nearest_player": nearest,
        "ball_side": side,
        "methods_agree": agree,
        # The principled one leads; the cheap one is the fallback, and that
        # order is a design decision, not a measurement.
        "prediction": nearest["team"] or side["team"],
        "prediction_source": (
            "nearest_player" if nearest["team"] else
            ("ball_side" if side["team"] else None)
        ),
    }


@dataclass
class AgreementReport:
    """How a derivation did against human labels."""

    method: str
    total: int = 0
    predicted: int = 0
    correct: int = 0
    confusion: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.predicted / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.predicted if self.predicted else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "total": self.total,
            "predicted": self.predicted,
            "correct": self.correct,
            "coverage": round(self.coverage, 4),
            "accuracy_where_predicted": round(self.accuracy, 4),
            "confusion": dict(sorted(self.confusion.items())),
        }


def score_against_human(
    records: Sequence[Mapping[str, Any]], method: str,
) -> AgreementReport:
    """Compare one derivation's predictions with the human labels.

    ``records`` carry ``human`` (``team_a``/``team_b``) and the attribution
    payload. Contacts the human left unlabelled are skipped entirely rather
    than counted as either outcome.
    """
    report = AgreementReport(method=method)
    for record in records:
        human = team_name(record.get("human"))
        if human is None:
            continue
        report.total += 1
        attribution = record.get("attribution") or {}
        if method == "prediction":
            predicted = attribution.get("prediction")
        else:
            predicted = (attribution.get(method) or {}).get("team")
        if predicted is None:
            continue
        report.predicted += 1
        key = f"{human}->{predicted}"
        report.confusion[key] = report.confusion.get(key, 0) + 1
        if predicted == human:
            report.correct += 1
    return report
