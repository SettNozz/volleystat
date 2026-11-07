"""Put a calibration into the corner order the pipeline reads.

``scripts/interactive_court_calibration.py`` stores the court corners starting
from the line nearest the camera, and names that line ``back_line_points``.
``GameAnalyzer._load_calibration`` reads ``back_line_points`` as the bottom of
the court and expects the corners to start at the far end line. Two first-party
tools, two conventions, and the disagreement is silent: the homography is
correct either way, so the overlay looks right while ``court_bottom`` ends up
on the far line and the world y axis points the other way.

This rewrites a calibration into the convention the loader expects:

* ``court_corners`` ordered far-left, far-right, near-right, near-left;
* ``front_line_points`` the far line, ``back_line_points`` the near one;
* ``perspective_matrix`` recomputed so world ``y = -9`` is the far end line,
  ``y = 0`` the net and ``y = +9`` the near end line, matching every
  calibration the gold videos were measured with.

The input is never modified, and an existing output is refused.

    python scripts/normalize_calibration.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# World metres, in the frame ``CourtProjector`` documents: x across the court,
# y along it, y = 0 under the net, negative towards the far end line.
WORLD_COURT = np.array(
    [[-4.5, -9.0], [4.5, -9.0], [4.5, 9.0], [-4.5, 9.0]], dtype=np.float64,
)


def order_corners(corners: Sequence[Sequence[float]]) -> list[list[float]]:
    """far-left, far-right, near-right, near-left, whatever order came in.

    Far means higher in the image, which for a camera behind one baseline is
    always the smaller y. Within each pair, left is the smaller x.
    """
    points = [[float(p[0]), float(p[1])] for p in corners]
    if len(points) != 4:
        raise ValueError("a court quad needs exactly four corners")
    by_depth = sorted(points, key=lambda p: p[1])
    far = sorted(by_depth[:2], key=lambda p: p[0])
    near = sorted(by_depth[2:], key=lambda p: p[0])
    return [far[0], far[1], near[1], near[0]]


def normalize(calibration: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the rewritten calibration and what changed."""
    import cv2

    raw = calibration.get("court_corners")
    if not raw or len(raw) != 4:
        raise ValueError("calibration has no four-corner court quad")
    ordered = order_corners(raw)
    far_left, far_right, near_right, near_left = ordered

    result = dict(calibration)
    result["court_corners"] = ordered
    result["front_line_points"] = [far_left, far_right]
    result["back_line_points"] = [near_left, near_right]
    result["left_sideline_points"] = [far_left, near_left]
    result["right_sideline_points"] = [far_right, near_right]
    homography = cv2.getPerspectiveTransform(
        WORLD_COURT.astype(np.float32),
        np.asarray(ordered, dtype=np.float32),
    )
    result["perspective_matrix"] = homography.tolist()

    def project(x: float, y: float) -> tuple[float, float]:
        values = homography @ np.array([x, y, 1.0])
        return float(values[0] / values[2]), float(values[1] / values[2])

    metadata = dict(result.get("metadata") or {})
    metadata["normalized_from"] = {
        "court_corners": [[float(p[0]), float(p[1])] for p in raw],
        "note": (
            "reordered to the loader's convention by "
            "scripts/normalize_calibration.py; the court quad itself is "
            "unchanged, only its order and the world axis direction"
        ),
    }
    result["metadata"] = metadata

    changed = {
        "corner_order_changed": [
            [float(p[0]), float(p[1])] for p in raw
        ] != ordered,
        "far_line_y": project(0.0, -9.0)[1],
        "net_line_y": project(0.0, 0.0)[1],
        "near_line_y": project(0.0, 9.0)[1],
    }
    changed["ordering_ok"] = (
        changed["far_line_y"] < changed["net_line_y"] < changed["near_line_y"]
    )
    return result, changed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true",
        help="allow writing over an existing output",
    )
    args = parser.parse_args(argv)

    if args.input.resolve() == args.output.resolve():
        parser.error("refusing to rewrite the input in place")
    if args.output.exists() and not args.force:
        parser.error(f"refusing to overwrite existing {args.output}")

    calibration = json.loads(args.input.read_text(encoding="utf-8"))
    result, changed = normalize(calibration)
    if not changed["ordering_ok"]:
        parser.error(
            "after reordering the far line is still not above the near one; "
            "this calibration is not a plain court quad and needs a look"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"  corner order changed: {changed['corner_order_changed']}")
    print(
        f"  world y=-9 -> {changed['far_line_y']:.0f} (far), "
        f"y=0 -> {changed['net_line_y']:.0f} (net), "
        f"y=+9 -> {changed['near_line_y']:.0f} (near)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
