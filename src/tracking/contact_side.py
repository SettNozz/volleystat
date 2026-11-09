"""Which side of the net touched the ball, from the ball's apparent size.

A camera behind a baseline sees the far half of the court further away than
the near half, so the ball is smaller there. The size the ball has exactly at
the net follows from the calibration: the net tape is ``net_height_m`` above
the floor line under it, which gives pixels per metre in the net plane. A ball
larger than that is on the near side (``b``), a smaller one on the far side
(``a``).

One box is a coarse measurement, and at the net an attacker and a blocker are
only a metre or two apart in depth. So contacts are grouped into possessions
(one side's consecutive touches) and a possession takes the summed evidence of
its non-attack contacts; a possession with weak evidence takes the side
opposite the previous one inside the same rally.

Derived data, never a label: every event keeps ``team`` as it was and gets
separate ``side*`` fields. Nothing is emitted when the geometry cannot support
the method (no net geometry, a camera that is not end-on, or box sizes that
disagree with the calibration).

See ``docs/team_attribution.md`` for what was measured and on what.
"""

import math
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "contact_side_v1"
METHOD = "ball_size_possession_v1"

BALL_DIAMETER_M = 0.21
COURT_HALF_WIDTH_M = 4.5
COURT_HALF_LENGTH_M = 9.0

# Raw contact vocabulary -> the four core classes the rule is written in.
# ``block`` is an attack-class touch at the net, as in the dense-contact data.
CORE_TYPE = {
    "serve": "serve",
    "reception": "reception", "reception_pass": "reception",
    "set": "pass", "setting_pass": "pass", "dig": "pass", "pass": "pass",
    "free_ball": "pass", "followup_contact": "pass",
    "attack": "attack", "block": "attack",
}

WINDOW_SECONDS = 1.0 / 6.0        # box sizes within +/- this of the contact
POSSESSION_GAP_SECONDS = 4.0      # a longer pause starts a new possession
WEAK_EVIDENCE = 0.10              # |log size ratio| below this is undecided
# Sanity gates. The size threshold was measured usable within about +/-8% of
# its best value, so calibration that is off by much more must abstain.
MIN_END_ON_WIDTH_RATIO = 1.3      # near baseline / far baseline width, px
GEOMETRY_AGREEMENT = (0.80, 1.25)  # horizontal vs vertical net-plane scale
SIZE_AGREEMENT = (0.80, 1.25)     # median contact size vs the net size
MIN_CONTACTS_FOR_SIZE_CHECK = 30


def _project(homography: Sequence[Sequence[float]], x: float, y: float) -> Optional[Tuple[float, float]]:
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3):
        return None
    u, v, w = matrix @ np.array([x, y, 1.0])
    if not np.isfinite([u, v, w]).all() or abs(w) < 1e-9:
        return None
    return float(u / w), float(v / w)


def net_plane_ball_px(calibration: Any, ball_diameter_m: float = BALL_DIAMETER_M) -> Dict[str, Any]:
    """Expected ball size, in pixels, exactly at the net plane.

    Returns ``{"status": "ok", "ball_px": ...}`` or a status naming why the
    calibration cannot support the size cue. Reads only the saved geometry.
    """
    get = lambda name: getattr(calibration, name, None) if calibration is not None else None  # noqa: E731
    left, right = get("net_floor_left"), get("net_floor_right")
    homography = get("homography")
    height_m = float(get("net_height_m") or 0.0)
    if not (left and right) or height_m <= 0.0:
        return {"status": "no_net_geometry"}

    endpoints = get("net_tape_endpoint_y")
    tape_y = (
        (float(endpoints[0]) + float(endpoints[1])) / 2.0 if endpoints
        else get("net_tape_top") if get("net_tape_top") is not None else get("net_y")
    )
    if tape_y is None:
        return {"status": "no_net_geometry"}
    floor_y = (float(left[1]) + float(right[1])) / 2.0
    rise_px = floor_y - float(tape_y)
    if not math.isfinite(rise_px) or rise_px <= 1.0:
        return {"status": "no_net_geometry"}
    vertical = ball_diameter_m * rise_px / height_m
    report: Dict[str, Any] = {
        "status": "ok", "ball_px": round(vertical, 2),
        "source": "net_tape_height_over_floor_line",
        "net_height_m": height_m, "net_rise_px": round(rise_px, 1),
    }

    if homography is not None:
        corners = [
            _project(homography, sx * COURT_HALF_WIDTH_M, sy * COURT_HALF_LENGTH_M)
            for sy in (-1, 1) for sx in (-1, 1)
        ]
        net_ends = [_project(homography, sx * COURT_HALF_WIDTH_M, 0.0) for sx in (-1, 1)]
        if all(corners) and all(net_ends):
            widths = sorted(abs(corners[i + 1][0] - corners[i][0]) for i in (0, 2))
            ratio = widths[1] / max(widths[0], 1e-6)
            horizontal = ball_diameter_m * abs(net_ends[1][0] - net_ends[0][0]) / (2 * COURT_HALF_WIDTH_M)
            report.update(
                baseline_width_ratio=round(ratio, 2),
                ball_px_from_court_width=round(horizontal, 2),
            )
            if ratio < MIN_END_ON_WIDTH_RATIO:
                return {**report, "status": "camera_not_end_on"}
            agreement = horizontal / vertical
            if not GEOMETRY_AGREEMENT[0] <= agreement <= GEOMETRY_AGREEMENT[1]:
                return {**report, "status": "geometry_inconsistent"}
    return report


def contact_ball_size(sizes: Dict[int, float], frame: int, radius: int) -> Tuple[Optional[float], int]:
    """Median observed box size around a contact, and how many boxes it used."""
    found = [sizes[index] for index in range(frame - radius, frame + radius + 1) if index in sizes]
    return (float(median(found)), len(found)) if found else (None, 0)


def _possessions(contacts: List[Dict[str, Any]], gap_frames: float) -> List[List[int]]:
    groups: List[List[int]] = []
    for index, contact in enumerate(contacts):
        previous = contacts[index - 1] if index else None
        starts = (
            previous is None
            or contact["core"] == "serve"
            or previous["core"] in ("serve", "attack")
            or contact["frame"] - previous["frame"] > gap_frames
        )
        if starts:
            groups.append([])
        groups[-1].append(index)
    return groups


def _sign(value: float) -> int:
    return 1 if value > 0 else -1


def assign_contact_sides(
    events: List[Dict[str, Any]],
    annotations: Sequence[Dict[str, Any]],
    fps: float,
    calibration: Any,
) -> Dict[str, Any]:
    """Add ``side`` fields to contact events in place; return the report.

    ``side`` is ``"a"`` (far), ``"b"`` (near) or None. ``team`` is not touched.
    """
    fps = float(fps or 30.0)
    parameters = {
        "window_seconds": WINDOW_SECONDS, "possession_gap_seconds": POSSESSION_GAP_SECONDS,
        "weak_evidence": WEAK_EVIDENCE, "ball_diameter_m": BALL_DIAMETER_M,
    }
    report: Dict[str, Any] = {
        "schema": SCHEMA, "method": METHOD, "label_source": "derived", "verified": False,
        "parameters": parameters,
        "limitations": (
            "Unverified side proposals from detector box size and possession order. "
            "Event detection and typing errors propagate; a missed contact shifts "
            "possession boundaries. Measured on one end-on camera placement only."
        ),
    }
    contacts = [
        {"event": event, "frame": int(event["frame"]),
         "core": event.get("core_type") if event.get("core_type") in set(CORE_TYPE.values())
         else CORE_TYPE.get(event.get("type"))}
        for event in events if isinstance(event.get("frame"), (int, float))
    ]
    contacts = sorted((c for c in contacts if c["core"]), key=lambda c: c["frame"])
    for contact in contacts:
        contact["event"].update(side=None, side_score=None, side_source=None, side_ball_px=None)

    geometry = net_plane_ball_px(calibration)
    report["geometry"] = geometry
    report["contacts"] = len(contacts)
    if not contacts:
        return {**report, "status": "no_contacts"}
    if geometry["status"] != "ok":
        return {**report, "status": geometry["status"]}
    net_px = float(geometry["ball_px"])

    sizes = {
        int(ann["f"]): float(ann["ball_size"])
        for ann in annotations if ann.get("ball_size") and ann.get("ball")
    }
    if not sizes:
        return {**report, "status": "no_ball_sizes"}
    radius = max(1, int(round(WINDOW_SECONDS * fps)))
    for contact in contacts:
        size, used = contact_ball_size(sizes, contact["frame"], radius)
        contact["size"], contact["boxes"] = size, used
        contact["evidence"] = math.log(size / net_px) if size else 0.0
        contact["event"]["side_ball_px"] = round(size, 1) if size else None

    sized = [c["size"] for c in contacts if c["size"]]
    report["contacts_with_ball_size"] = len(sized)
    if sized:
        report["median_contact_ball_px"] = round(float(median(sized)), 1)
    if len(sized) >= MIN_CONTACTS_FOR_SIZE_CHECK:
        agreement = float(median(sized)) / net_px
        if not SIZE_AGREEMENT[0] <= agreement <= SIZE_AGREEMENT[1]:
            # The boxes do not straddle the net size: wrong calibration, another
            # detector scale or a camera the cue does not fit. Say nothing.
            return {**report, "status": "size_distribution_mismatch"}

    groups = _possessions(contacts, POSSESSION_GAP_SECONDS * fps)
    scores: List[float] = []
    sources: List[Optional[str]] = []
    for group in groups:
        evidence = [contacts[i]["evidence"] for i in group
                    if contacts[i]["core"] != "attack" and contacts[i]["evidence"] != 0.0]
        if not evidence:
            evidence = [contacts[i]["evidence"] for i in group if contacts[i]["evidence"] != 0.0]
        scores.append(float(sum(evidence)) if evidence else 0.0)
        sources.append("ball_size_possession" if evidence else None)
    gap_frames = POSSESSION_GAP_SECONDS * fps
    for index in range(1, len(groups)):
        first, last = contacts[groups[index][0]], contacts[groups[index - 1][-1]]
        if first["core"] == "serve":
            continue                        # a new rally owes nothing to the last one
        if first["frame"] - last["frame"] > gap_frames:
            continue                        # a long pause hides a missed restart
        if abs(scores[index]) < WEAK_EVIDENCE and abs(scores[index - 1]) >= WEAK_EVIDENCE:
            scores[index] = -WEAK_EVIDENCE * _sign(scores[index - 1])
            sources[index] = "possession_alternation"

    by_source: Dict[str, int] = {}
    assigned = 0
    for group, score, source in zip(groups, scores, sources):
        if source is None:
            continue
        for i in group:
            contacts[i]["event"].update(
                side="b" if score > 0 else "a",
                side_score=round(score, 3),
                side_source=source,
            )
            assigned += 1
            by_source[source] = by_source.get(source, 0) + 1
    report.update(
        status="completed", net_ball_px=net_px, possessions=len(groups),
        assigned=assigned, unassigned=len(contacts) - assigned, by_source=by_source,
        sides={side: sum(1 for c in contacts if c["event"]["side"] == side) for side in ("a", "b")},
    )
    return report
