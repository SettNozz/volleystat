"""Use nearby observations to check sideline poses before final analysis."""

from bisect import bisect_left
from math import hypot

from src.detection.player_pose import (
    COURT_ENTRY_MARGIN_SERVE,
    COURT_ENTRY_MARGIN_X,
    KP_CONF_MIN,
    L_ANKLE,
    R_ANKLE,
)


def reject_unsupported_sideline_poses(annotations, projector, fps, *, sample_step=1):
    """Remove sideline poses without court evidence on a continuous tracklet.

    A raw ID can jump to another person. Split on long gaps or large box jumps
    before consulting either past or future observations. Keep legal serve
    positions and leave ankle occlusion to the original court gate.
    """
    max_gap = max(fps, sample_step)
    support_frames = 5 * fps
    sideline_limit = projector.half_width + COURT_ENTRY_MARGIN_X
    serve_limit = projector.half_length + COURT_ENTRY_MARGIN_SERVE
    active = {}
    tracklets = []
    for frame_index, ann in enumerate(annotations):
        frame = ann['f']
        for pose_index, pose in enumerate(ann.get('poses') or []):
            track_id = pose.get('id')
            box = pose.get('bb')
            if track_id is None or not box:
                continue
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            height = max(y2 - y1, 1)
            previous = active.get(track_id)
            if previous is not None:
                last_frame, last_center, last_height, tracklet = previous
                gap = frame - last_frame
                # Scale motion by the apparent player height and time elapsed.
                motion_limit = (0.5 + 3 * gap / fps) * min(last_height, height)
                displacement = hypot(center[0] - last_center[0], center[1] - last_center[1])
                if gap > max_gap or displacement > motion_limit:
                    previous = None
            if previous is None:
                tracklet = ([], [])
                tracklets.append(tracklet)
            active[track_id] = (frame, center, height, tracklet)

            keypoints = pose.get('kp') or []
            if len(keypoints) <= R_ANKLE:
                continue
            ankles = [keypoints[i] for i in (L_ANKLE, R_ANKLE) if keypoints[i][2] >= KP_CONF_MIN]
            if not ankles:
                continue
            foot = (sum(p[0] for p in ankles) / len(ankles), max(p[1] for p in ankles))
            world_x, world_y = projector.image_to_world(foot)
            anchors, sideline = tracklet
            if abs(world_x) > sideline_limit:
                sideline.append((frame, frame_index, pose_index))
            elif abs(world_x) <= projector.half_width and abs(world_y) <= serve_limit:
                anchors.append(frame)

    rejected = {}
    for anchors, sideline in tracklets:
        for frame, frame_index, pose_index in sideline:
            index = bisect_left(anchors, frame)
            past = index > 0 and frame - anchors[index - 1] <= support_frames
            future = index < len(anchors) and anchors[index] - frame <= support_frames
            if not (past or future):
                rejected.setdefault(frame_index, set()).add(pose_index)

    for frame_index, indices in rejected.items():
        ann = annotations[frame_index]
        poses = ann['poses']
        points = ann.get('p')
        if points is not None and len(points) == len(poses):
            # Export groups player centres by side but keeps pose order within each side.
            grouped = sorted(range(len(poses)), key=lambda i: poses[i].get('team') not in ('a', 'team_a'))
            ann['p'] = [point for i, point in zip(grouped, points) if i not in indices]
        ann['poses'] = [pose for i, pose in enumerate(poses) if i not in indices]
    return sum(map(len, rejected.values()))
