"""Bounded, offline pose estimates between observed poses of the same person.

Nothing is inserted into detector ``poses``. Estimates have no detector score
or tracker ID, and must not become contact/training ground truth. Future
observations are required: this is not streaming pose detection.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math

import numpy as np

SCHEMA = "temporal_pose_completion_v1"
TORSO = (5, 6, 11, 12)
ARMS = (7, 8, 9, 10)


@dataclass(frozen=True)
class CompletionConfig:
    max_gap_seconds: float = .30
    max_arm_gap_seconds: float = .15
    anchor_confidence: float = .40
    missing_confidence: float = .30
    min_anchor_points: int = 8
    min_torso_points: int = 3
    max_center_motion_heights: float = .60
    max_joint_motion_heights: float = .60
    min_overlap: float = .20
    min_occluded_fraction: float = .50
    max_on_court: int = 6


def _side(pose):
    return {"a": "a", "b": "b", "team_a": "a", "team_b": "b"}.get(pose.get("team"))


def _box(pose):
    value = np.asarray(pose.get("bb") or [], dtype=float)
    return value if value.shape == (4,) and np.isfinite(value).all() and value[2] > value[0] and value[3] > value[1] else None


def _points(pose):
    value = np.asarray(pose.get("kp") or [], dtype=float)
    return value if value.shape == (17, 3) and np.isfinite(value).all() else None


def _height(box):
    return float(box[3] - box[1])


def _center(box):
    return (box[:2] + box[2:]) * .5


def _overlap(box, other):
    """Fraction of the estimated person's image box covered by another box."""
    intersection = np.maximum(0., np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
    return float(np.prod(intersection) / np.prod(box[2:] - box[:2]))


def _quality(pose, config):
    points = _points(pose)
    return (_box(pose) is not None and points is not None
            and np.count_nonzero(points[:, 2] >= config.anchor_confidence) >= config.min_anchor_points
            and np.count_nonzero(points[list(TORSO), 2] >= config.anchor_confidence) >= config.min_torso_points)


def _interpolate(left, right, fraction):
    """Interpolate global body motion and body-relative shape separately."""
    lb, rb = _box(left), _box(right)
    box = lb * (1. - fraction) + rb * fraction
    lp, rp = _points(left), _points(right)
    relative = ((lp[:, :2] - _center(lb)) / _height(lb) * (1. - fraction)
                + (rp[:, :2] - _center(rb)) / _height(rb) * fraction)
    return box, relative * _height(box) + _center(box)


def complete_player_poses(frames, fps, frame_size, identity=None, config=None):
    """Return frame-indexed overlays and diagnostics; never mutate inputs.

    Endpoints need a second adjacent supporting observation. Missing full
    poses also need observed bbox overlap during the gap. Unknown/rejected
    identities are never relabelled; matching unidentified detections block
    a duplicate estimate. No extrapolation or missing-frame bridging.
    """
    config = config or CompletionConfig()
    fps = float(fps)
    width, height = map(float, frame_size)
    if (not math.isfinite(fps) or fps <= 0 or not all(math.isfinite(v) and v > 0 for v in (width, height))
            or not 0 < config.max_arm_gap_seconds <= config.max_gap_seconds <= .5
            or config.max_on_court < 1):
        raise ValueError("Invalid FPS, image size or pose completion bounds")
    by_frame = {a["f"]: a for a in frames}
    if len(by_frame) != len(frames) or any(type(f) is not int for f in by_frame) or list(by_frame) != sorted(by_frame):
        raise ValueError("Pose completion requires unique ordered integer source frames")
    identity = identity or {}
    target = identity.get("side", "b")
    if target not in ("a", "b"):
        raise ValueError("Identity side must be a or b")
    rows = defaultdict(dict)
    duplicates = set()
    for ann in frames:
        for index, pose in enumerate(ann.get("poses") or []):
            pid = pose.get("player_id")
            if not pid or _side(pose) != target:
                continue
            if ann["f"] in rows[pid]:
                duplicates.add((pid, ann["f"]))
            rows[pid][ann["f"]] = (index, pose)
    reasons = Counter()
    full_allowed = {}

    def stable(pid, f, direction, joint=None):
        for current in (f, f + direction):
            if (pid, current) in duplicates or current not in rows[pid]:
                return False
            pose = rows[pid][current][1]
            if not _quality(pose, config):
                return False
            if joint is not None and _points(pose)[joint, 2] < config.anchor_confidence:
                return False
        return True

    def motion_ok(left, right):
        lb, rb = _box(left), _box(right)
        scale = (_height(lb) + _height(rb)) * .5
        return (np.linalg.norm(_center(lb) - _center(rb)) / scale <= config.max_center_motion_heights
                and .65 <= _height(lb) / _height(rb) <= 1.55)

    def inside(box):
        return box[0] > 2 and box[1] > 2 and box[2] < width-2 and box[3] < height-2

    def duplicate_observation(ann, predicted_box, predicted_points, left, right):
        for pose in ann.get("poses") or []:
            box, points = _box(pose), _points(pose)
            if box is None:
                continue
            # Never override a rejected ReID assignment using a track ID.
            if (not pose.get("player_id") and _side(pose) == target
                    and pose.get("id") is not None and pose.get("id") in (left.get("id"), right.get("id"))):
                return True
            if points is None or _overlap(predicted_box, box) < .4:
                continue
            valid = points[:, 2] >= config.anchor_confidence
            if np.count_nonzero(valid) >= 4 and np.median(
                np.linalg.norm(points[valid, :2] - predicted_points[valid], axis=1)
            ) / _height(predicted_box) < .20:
                return True
        return False

    # First establish which truly missing person-frames can be filled.
    for pid, observations in rows.items():
        ordered = sorted(observations)
        for l, r in zip(ordered, ordered[1:]):
            gap = r - l - 1
            if not 1 <= gap <= math.floor(config.max_gap_seconds * fps):
                continue
            reasons["short_identity_gaps"] += 1
            if any(f not in by_frame for f in range(l+1, r)):
                reasons["missing_source_frames"] += 1
                continue
            left, right = observations[l][1], observations[r][1]
            if not stable(pid, l, -1) or not stable(pid, r, 1):
                reasons["unstable_anchors"] += 1
                continue
            if not motion_ok(left, right) or not inside(_box(left)) or not inside(_box(right)):
                reasons["implausible_motion_or_edge"] += 1
                continue
            accepted = []
            occluded = 0
            for f in range(l+1, r):
                ann = by_frame[f]
                fraction = (f-l) / (r-l)
                box, points = _interpolate(left, right, fraction)
                others = [b for p in ann.get("poses") or [] if (b := _box(p)) is not None]
                overlap = max((_overlap(box, b) for b in others), default=0.)
                if overlap < config.min_overlap:
                    continue
                occluded += 1
                if sum(_side(p) == target for p in ann.get("poses") or []) >= config.max_on_court:
                    reasons["court_already_full_frames"] += 1
                    continue
                if duplicate_observation(ann, box, points, left, right):
                    reasons["existing_or_unidentified_pose_frames"] += 1
                    continue
                accepted.append((f, box, overlap))
            if occluded / gap < config.min_occluded_fraction:
                reasons["insufficient_occlusion_gaps"] += 1
                continue
            for f, box, overlap in accepted:
                full_allowed[pid, f] = {"box": box, "overlap": overlap, "left_frame": l, "right_frame": r}

    output = {}
    # Fill only absent/weak joints, with independent endpoints per joint.
    for pid, observations in rows.items():
        for joint in range(17):
            valid = [f for f, (_, p) in sorted(observations.items())
                     if _quality(p, config) and _points(p)[joint, 2] >= config.anchor_confidence]
            limit = config.max_arm_gap_seconds if joint in ARMS else config.max_gap_seconds
            for l, r in zip(valid, valid[1:]):
                gap = r - l - 1
                if not 1 <= gap <= math.floor(limit * fps):
                    continue
                if (not stable(pid, l, -1, joint) or not stable(pid, r, 1, joint)
                        or any(f not in by_frame or (pid, f) in duplicates for f in range(l, r+1))):
                    continue
                left, right = observations[l][1], observations[r][1]
                if not motion_ok(left, right):
                    continue
                lb, rb = _box(left), _box(right)
                lp, rp = _points(left), _points(right)
                displacement = float(np.linalg.norm(
                    (lp[joint, :2]-_center(lb))/_height(lb) - (rp[joint, :2]-_center(rb))/_height(rb)))
                if displacement > config.max_joint_motion_heights:
                    reasons["joint_motion_rejected"] += 1
                    continue
                for f in range(l+1, r):
                    existing = observations.get(f)
                    missing = existing is None
                    if missing and (pid, f) not in full_allowed:
                        continue
                    if existing is not None:
                        current_points = _points(existing[1])
                        if current_points is None or _box(existing[1]) is None or current_points[joint, 2] >= config.missing_confidence:
                            continue
                    fraction = (f-l) / (r-l)
                    box, points = _interpolate(left, right, fraction)
                    if not inside(box):
                        continue
                    if existing is not None:
                        # Align to reliable observed torso, preserving its actual
                        # motion instead of trusting a truncated occluded bbox.
                        torso = [j for j in TORSO if current_points[j, 2] >= config.anchor_confidence
                                 and lp[j, 2] >= config.anchor_confidence and rp[j, 2] >= config.anchor_confidence]
                        if len(torso) < 2:
                            continue
                        shift = np.median(current_points[torso, :2] - points[torso], axis=0)
                        if np.linalg.norm(shift) / _height(box) > .25:
                            continue
                        points += shift
                        # The missing joint itself needs an observed occluder;
                        # a confidence dip in open space is not hidden anatomy.
                        occluders = [b for i, p in enumerate(by_frame[f].get("poses") or [])
                                     if i != existing[0] and (b := _box(p)) is not None]
                        point = points[joint]
                        if not any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3] for b in occluders):
                            continue
                    if not (0 <= points[joint, 0] < width and 0 <= points[joint, 1] < height):
                        continue
                    key = (f, pid)
                    if key not in output:
                        known = _points(existing[1]) if existing is not None else None
                        output[key] = {
                            "player_id": pid, "team": target, "source": SCHEMA,
                            "label_source": "derived", "verified": False,
                            "observed_pose_index": existing[0] if existing is not None else None,
                            "bb": [round(float(v), 3) for v in (existing[1]["bb"] if existing else full_allowed[pid, f]["box"])],
                            "keypoints": [[round(float(v), 3) for v in p[:2]] if p[2] >= config.missing_confidence else None
                                          for p in known] if known is not None else [None] * 17,
                            "estimated_joints": [], "joint_evidence": [],
                        }
                    entry = output[key]
                    entry["keypoints"][joint] = [round(float(v), 3) for v in points[joint]]
                    entry["estimated_joints"].append(joint)
                    entry["joint_evidence"].append({
                        "joint": joint, "left_frame": l, "right_frame": r,
                        "left_pose_index": observations[l][0], "right_pose_index": observations[r][0],
                        "gap_frames": gap, "body_relative_change": round(displacement, 5),
                    })
    by_output_frame = defaultdict(list)
    counts = Counter()
    # Keep the six-person cap even when two identities have simultaneous gaps.
    for (f, pid), entry in sorted(output.items()):
        if entry["observed_pose_index"] is None:
            estimated = set(entry["estimated_joints"])
            if len(estimated) < 6 or len(estimated.intersection(TORSO)) < config.min_torso_points:
                continue
            count = sum(_side(p) == target for p in by_frame[f].get("poses") or [])
            count += sum(e["observed_pose_index"] is None for e in by_output_frame[f])
            if count >= config.max_on_court:
                continue
            counts["missing_person_frames"] += 1
        else:
            counts["partial_person_frames"] += 1
        counts["estimated_joint_frames"] += len(entry["estimated_joints"])
        by_output_frame[f].append(entry)
    report = {"schema": SCHEMA, "side": target, "offline": True, "uses_future_frames": True,
              "label_source": "derived", "verified": False, "parameters": asdict(config),
              "counts": dict(counts), "diagnostics": dict(reasons),
              "contact_attribution_uses_estimates": False,
              "limitations": "Interpolated hidden anatomy, not image evidence. Fast hand motion is not recovered. Detector poses, identities and events are unchanged."}
    return dict(by_output_frame), report
