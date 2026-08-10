"""Named court setups (camera placements) and the homography they imply.

One recorded game is one camera placement.  The camera is re-mounted between
games and is never in exactly the same spot, so a single global
``court_calibration.json`` cannot describe them all: a setup is stored under a
name, and a result points at the setup it was measured with.

The world frame matches :class:`src.detection.player_pose.CourtProjector`:
``x`` runs across the court in ``[-4.5, 4.5]``, ``y`` runs along it in
``[-9, 9]``, and ``y = 0`` is the line on the floor under the net.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

COURT_HALF_WIDTH_M = 4.5
COURT_HALF_LENGTH_M = 9.0
ATTACK_LINE_M = 3.0

# Corner order is the one written by scripts/interactive_court_calibration.py,
# so existing calibrations load without translation.
CORNER_NAMES = ("front_left", "front_right", "back_right", "back_left")
WORLD_CORNERS: Tuple[Tuple[float, float], ...] = (
    (-COURT_HALF_WIDTH_M, -COURT_HALF_LENGTH_M),
    (COURT_HALF_WIDTH_M, -COURT_HALF_LENGTH_M),
    (COURT_HALF_WIDTH_M, COURT_HALF_LENGTH_M),
    (-COURT_HALF_WIDTH_M, COURT_HALF_LENGTH_M),
)

Point = Tuple[float, float]


class CourtSetupError(ValueError):
    """Raised when four points cannot describe a court plane."""


def _as_points(corners: Sequence[Sequence[float]]) -> List[Point]:
    if corners is None or len(corners) != 4:
        raise CourtSetupError("a court needs exactly four corners")
    points: List[Point] = []
    for corner in corners:
        if corner is None or len(corner) != 2:
            raise CourtSetupError("each corner must be an [x, y] pair")
        x, y = float(corner[0]), float(corner[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            raise CourtSetupError("corner coordinates must be finite")
        points.append((x, y))
    return points


def _cross_signs(points: Sequence[Point]) -> List[float]:
    signs = []
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        signs.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    return signs


def validate_corners(corners: Sequence[Sequence[float]]) -> List[Point]:
    """Return the corners if they form a usable, convex, non-degenerate quad.

    A self-intersecting or collapsed quad still yields a 3x3 matrix from
    OpenCV, but it maps the court inside out.  Rejecting it here keeps a
    mis-drag from silently producing mirrored zones.
    """
    points = _as_points(corners)
    signs = _cross_signs(points)
    if any(abs(sign) < 1e-6 for sign in signs):
        raise CourtSetupError("three court corners are collinear")
    if not (all(sign > 0 for sign in signs) or all(sign < 0 for sign in signs)):
        raise CourtSetupError(
            "court corners must be convex in front-left, front-right, "
            "back-right, back-left order"
        )
    return points


def homography_from_corners(
    corners: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Build the world -> image homography from the four court corners."""
    points = validate_corners(corners)
    source = np.array(WORLD_CORNERS, dtype=np.float64)
    target = np.array(points, dtype=np.float64)

    # Direct linear transform: 4 point pairs give 8 equations for the 8 free
    # parameters of a homography (h33 fixed at 1).
    rows = []
    for (wx, wy), (ix, iy) in zip(source, target):
        rows.append([wx, wy, 1, 0, 0, 0, -ix * wx, -ix * wy])
        rows.append([0, 0, 0, wx, wy, 1, -iy * wx, -iy * wy])
    matrix = np.array(rows, dtype=np.float64)
    rhs = target.reshape(-1)
    try:
        solution = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as exc:
        raise CourtSetupError(f"degenerate court corners: {exc}") from exc

    homography = np.append(solution, 1.0).reshape(3, 3)
    if not np.all(np.isfinite(homography)):
        raise CourtSetupError("court corners produced a non-finite homography")
    try:
        np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:
        raise CourtSetupError(f"court homography is not invertible: {exc}") from exc
    return homography.tolist()


def corners_from_homography(
    homography: Sequence[Sequence[float]],
) -> List[Point]:
    """Project the four world corners back into image space."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise CourtSetupError(f"expected a 3x3 homography, got {matrix.shape}")
    points: List[Point] = []
    for wx, wy in WORLD_CORNERS:
        vector = matrix @ np.array([wx, wy, 1.0])
        if abs(vector[2]) < 1e-12:
            raise CourtSetupError("homography sends a court corner to infinity")
        points.append((float(vector[0] / vector[2]), float(vector[1] / vector[2])))
    return points


def scale_corners(
    corners: Sequence[Sequence[float]],
    from_size: Tuple[float, float],
    to_size: Tuple[float, float],
) -> List[Point]:
    """Rescale image-space corners between two frame resolutions."""
    from_w, from_h = float(from_size[0]), float(from_size[1])
    to_w, to_h = float(to_size[0]), float(to_size[1])
    if from_w <= 0 or from_h <= 0:
        raise CourtSetupError("source frame size must be positive")
    sx, sy = to_w / from_w, to_h / from_h
    return [(float(x) * sx, float(y) * sy) for x, y in _as_points(corners)]


def court_guide_lines(
    homography: Sequence[Sequence[float]],
) -> Dict[str, List[List[float]]]:
    """Image-space guide lines a user can align the quad against.

    The far baseline is often occluded by the net and players, so fitting the
    quad by its own corners alone is guesswork.  The centre line and the two
    attack lines are almost always visible and pin the depth scale.
    """
    matrix = np.asarray(homography, dtype=np.float64)

    def project(wx: float, wy: float) -> List[float]:
        vector = matrix @ np.array([wx, wy, 1.0])
        return [float(vector[0] / vector[2]), float(vector[1] / vector[2])]

    half = COURT_HALF_WIDTH_M
    length = COURT_HALF_LENGTH_M
    attack = ATTACK_LINE_M
    return {
        "center_line": [project(-half, 0.0), project(half, 0.0)],
        "attack_line_near": [project(-half, -attack), project(half, -attack)],
        "attack_line_far": [project(-half, attack), project(half, attack)],
        "sideline_left": [project(-half, -length), project(-half, length)],
        "sideline_right": [project(half, -length), project(half, length)],
        "baseline_near": [project(-half, -length), project(half, -length)],
        "baseline_far": [project(-half, length), project(half, length)],
    }


def world_x_screen_sign(homography: Sequence[Sequence[float]]) -> float:
    """Return +1 when a larger world ``x`` appears further right on screen.

    Whether ``front`` was clicked on the near or the far baseline decides the
    sign of the world frame, and a court clicked the other way round is
    rotated 180 degrees.  Zone names are meant to read the way a coach sees
    the video, so the left/right split is taken from the screen, not from the
    sign convention of whoever calibrated.
    """
    matrix = np.asarray(homography, dtype=np.float64)

    def image_x(wx: float) -> float:
        vector = matrix @ np.array([wx, 0.0, 1.0])
        return float(vector[0] / vector[2])

    return 1.0 if image_x(COURT_HALF_WIDTH_M) >= image_x(-COURT_HALF_WIDTH_M) else -1.0


@dataclass
class CourtSetup:
    """One camera placement, reusable across the games recorded with it."""

    name: str
    court_corners: List[List[float]]
    frame_width: float
    frame_height: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    net_y: Optional[float] = None
    net_x_left: Optional[float] = None
    net_x_right: Optional[float] = None
    updated_at: float = field(default_factory=time.time)
    note: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise CourtSetupError("a court setup needs a name")
        self.court_corners = [
            [round(float(x), 2), round(float(y), 2)]
            for x, y in validate_corners(self.court_corners)
        ]
        self.frame_width = float(self.frame_width)
        self.frame_height = float(self.frame_height)
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise CourtSetupError("frame size must be positive")

    def corners_for(self, width: float, height: float) -> List[Point]:
        return scale_corners(
            self.court_corners,
            (self.frame_width, self.frame_height),
            (width, height),
        )

    def homography_for(self, width: float, height: float) -> List[List[float]]:
        return homography_from_corners(self.corners_for(width, height))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "court_corners": self.court_corners,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "net_y": self.net_y,
            "net_x_left": self.net_x_left,
            "net_x_right": self.net_x_right,
            "updated_at": round(float(self.updated_at), 3),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CourtSetup":
        return cls(
            name=data.get("name", ""),
            court_corners=data.get("court_corners"),
            frame_width=data.get("frame_width", 0),
            frame_height=data.get("frame_height", 0),
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            net_y=_optional_float(data.get("net_y")),
            net_x_left=_optional_float(data.get("net_x_left")),
            net_x_right=_optional_float(data.get("net_x_right")),
            updated_at=float(data.get("updated_at") or time.time()),
            note=str(data.get("note") or ""),
        )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


class CourtSetupStore:
    """A small JSON-backed collection of named setups."""

    VERSION = 1

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> List[CourtSetup]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        setups = []
        for entry in data.get("setups") or []:
            try:
                setups.append(CourtSetup.from_dict(entry))
            except (CourtSetupError, TypeError, ValueError):
                # A corrupt entry must not hide the healthy ones.
                continue
        setups.sort(key=lambda setup: setup.updated_at, reverse=True)
        return setups

    def get(self, setup_id: str) -> Optional[CourtSetup]:
        for setup in self.load():
            if setup.id == setup_id:
                return setup
        return None

    def save(self, setup: CourtSetup) -> CourtSetup:
        """Insert or replace by id, or by name when the id is new."""
        setups = self.load()
        setup.updated_at = time.time()
        by_id = {existing.id: index for index, existing in enumerate(setups)}
        by_name = {
            existing.name.lower(): index for index, existing in enumerate(setups)
        }
        if setup.id in by_id:
            setups[by_id[setup.id]] = setup
        elif setup.name.lower() in by_name:
            index = by_name[setup.name.lower()]
            setup.id = setups[index].id
            setups[index] = setup
        else:
            setups.append(setup)
        self._write(setups)
        return setup

    def delete(self, setup_id: str) -> bool:
        setups = self.load()
        remaining = [setup for setup in setups if setup.id != setup_id]
        if len(remaining) == len(setups):
            return False
        self._write(remaining)
        return True

    def _write(self, setups: List[CourtSetup]) -> None:
        payload = {
            "version": self.VERSION,
            "setups": [setup.to_dict() for setup in setups],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)
