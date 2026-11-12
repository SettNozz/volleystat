"""Reconcile same-frame YOLO body fragments without inventing keypoints.

Boxes alone cannot distinguish an occluded teammate from a duplicate person.
Require agreement of named body joints and body scale before sharing joints.
Raw detector/tracker outputs remain immutable; every donor is recorded.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.detection.player_pose import PlayerPose, estimate_back_facing_score

SCHEMA = "yolo_pose_fragments_v1"
TORSO = (5, 6, 11, 12)
CORE = (5, 6, 11, 12, 13, 14)
PARTS = ((0, 1, 2, 3, 4), (5, 6), (7, 9), (8, 10), (11, 12), (13, 15), (14, 16))
CONF = .40


def same_person(a: PlayerPose, b: PlayerPose) -> bool:
    """Conservative anatomical agreement, independent of predicted court side."""
    if a.player_id and b.player_id and a.player_id != b.player_id:
        return False
    boxes = np.asarray([a.bbox, b.bbox])
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        return False
    intersection = np.maximum(0, np.minimum(boxes[0, 2:], boxes[1, 2:])
                              - np.maximum(boxes[0, :2], boxes[1, :2]))
    if np.prod(intersection) / np.min(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)) < .60:
        return False
    p, q = a.keypoints, b.keypoints
    if p.shape != (17, 3) or q.shape != (17, 3) or not np.isfinite(p).all() or not np.isfinite(q).all():
        return False
    shared = [i for i in TORSO if min(p[i, 2], q[i, 2]) >= CONF]
    if len(shared) < 3:
        shared = [i for i in CORE if min(p[i, 2], q[i, 2]) >= CONF]
    if len(shared) < 3 or not all(i in shared for i in (11, 12)):
        return False
    # Hip width is local body scale, so distant players inside a foreground
    # player's large box cannot borrow its generous pixel-distance threshold.
    widths = [np.linalg.norm(k[11, :2] - k[12, :2]) for k in (p, q)]
    if min(widths) <= 1e-6 or max(widths) / min(widths) > 1.5:
        return False
    distances = np.linalg.norm(p[shared, :2] - q[shared, :2], axis=1)
    return bool(np.median(distances) <= .40 * min(widths)
                and np.max(distances) <= .65 * min(widths))


def combine(anchor: PlayerPose, donors: list[PlayerPose]) -> PlayerPose:
    """Choose coherent observed limb chains; never average incompatible arms."""
    sources = [anchor, *donors]
    points = anchor.keypoints.copy()
    joint_sources = [0] * 17
    for part in PARTS:
        # Prefer complete limbs, then their weakest joint, then mean quality.
        def quality(index):
            conf = sources[index].keypoints[list(part), 2]
            return (int(np.count_nonzero(conf >= CONF)), float(np.min(conf)), float(np.mean(conf)))
        winner = max(range(len(sources)), key=quality)
        points[list(part)] = sources[winner].keypoints[list(part)]
        for joint in part:
            joint_sources[joint] = winner
    box = np.asarray([p.bbox for p in sources])
    bbox = (*np.min(box[:, :2], axis=0), *np.max(box[:, 2:], axis=0))
    result = replace(anchor, keypoints=points, bbox=tuple(float(v) for v in bbox),
                     center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
    result.pose_repair = {
        "schema": SCHEMA, "label_source": "model", "verified": False,
        "joint_sources": joint_sources,
        "observations": [{"id": p.track_id, "source": p.source, "bb": list(p.bbox),
                          "kp": p.keypoints.tolist(), "repair": p.pose_repair} for p in sources],
    }
    result.back_facing_score = estimate_back_facing_score(result)
    return result


def merge_fragments(poses: list[PlayerPose]) -> list[PlayerPose]:
    """Keep one observed track per anatomical group; prevent transitive merges."""
    remaining = list(range(len(poses)))
    result = []
    while remaining:
        first = remaining.pop(0)
        group = [first]
        for index in remaining[:]:
            if all(same_person(poses[index], poses[member]) for member in group):
                group.append(index)
                remaining.remove(index)
        if len(group) == 1:
            result.append(poses[first])
            continue
        # Grounded legs retain the original continuous track through a serve.
        anchor = max(group, key=lambda i: (bool(poses[i].player_id),
                     int(np.count_nonzero(poses[i].keypoints[[15, 16], 2] >= CONF)),
                     float(np.sum(poses[i].keypoints[:, 2]))))
        result.append(combine(poses[anchor], [poses[i] for i in group if i != anchor]))
    return result


def recovery_crop(pose: PlayerPose, size: tuple[int, int]):
    """Expand a large, half-body observation to contain the missing body half."""
    width, height = size
    p = pose.keypoints
    if p.shape != (17, 3) or not np.isfinite(p).all():
        return None
    x1, y1, x2, y2 = pose.bbox
    if not np.isfinite(pose.bbox).all():
        return None
    h, w = y2 - y1, x2 - x1
    if h < .25 * height or w <= 0 or h <= 0 or min(p[11:13, 2]) < CONF:
        return None
    upper_missing = max(p[5:7, 2]) < CONF
    lower_missing = max(p[15:17, 2]) < CONF
    # A full occlusion has no trustworthy body half to extend.
    if upper_missing == lower_missing:
        return None
    if upper_missing and min(p[15:17, 2]) < CONF:
        return None
    if lower_missing and min(p[5:7, 2]) < CONF:
        return None
    hip_y = float(np.mean(p[11:13, 1]))
    # Focus on the missing half and its hip anchors. Repeating the whole
    # body crop can reproduce the same confident lower-half detection.
    crop = (max(0, int(np.floor(x1 - .35*w))),
            max(0, int(np.floor(y1 - .60*h if upper_missing else hip_y - .25*h))),
            min(width, int(np.ceil(x2 + .35*w))),
            min(height, int(np.ceil(hip_y + .20*h if upper_missing else y2 + .60*h))))
    return crop if crop[2] > crop[0] and crop[3] > crop[1] else None


def recover_from_crop(pose: PlayerPose, candidates: list[PlayerPose], crop) -> PlayerPose:
    """Only use a uniquely matched YOLO observation that adds missing joints."""
    matches = []
    for candidate in candidates:
        points = candidate.keypoints.copy()
        points[points[:, 2] > 0, :2] += crop[:2]
        box = candidate.bbox
        shifted = replace(candidate, keypoints=points,
                          bbox=(box[0]+crop[0], box[1]+crop[1], box[2]+crop[0], box[3]+crop[1]),
                          source="yolo_pose_crop")
        # The targeted crop can end above the knees and the source can lack
        # both shoulders. In that case two precise hips are the only common
        # anchors. This exception is only for a uniquely matched crop donor,
        # never for merging independent full-frame people.
        hips = pose.keypoints[11:13]
        other = shifted.keypoints[11:13]
        widths = [np.linalg.norm(k[0, :2] - k[1, :2]) for k in (hips, other)]
        hip_match = (np.isfinite(points).all() and min(other[:, 2]) >= .70
                     and min(widths) > 1e-6 and max(widths) / min(widths) <= 1.35
                     and np.max(np.linalg.norm(hips[:, :2] - other[:, :2], axis=1)) <= .30 * min(widths))
        if same_person(pose, shifted) or hip_match:
            matches.append(shifted)
    if len(matches) != 1:
        return pose
    result = combine(pose, matches)
    if np.count_nonzero(result.keypoints[:, 2] >= CONF) <= np.count_nonzero(pose.keypoints[:, 2] >= CONF):
        return pose
    result.pose_repair["crop"] = list(crop)
    return result


class PoseFragmentRecovery:
    """Cache only new crop inference; cached replay never decodes or loads YOLO."""

    def __init__(self, weights, spec, size, cache=None):
        import hashlib
        import inspect
        self.weights, self.size, self.cache = weights, size, cache
        self.weights_sha256 = spec["weights_sha256"]
        self.model = None
        self.settings = dict(imgsz=1280, conf=.15)
        self.stage = cache.stage("pose_fragment_crop_raw", {
            "weights_sha256": spec["weights_sha256"], "runtime": spec["runtime"],
            "backend": hashlib.sha256(inspect.getsource(type(self)._predict).encode()).hexdigest(),
            "settings": self.settings,
            "precision": "fp16_cuda_fp32_cpu", "preprocess": "ultralytics_predict",
        }) if cache is not None else None

    def _predict(self, image):
        import torch
        from ultralytics import YOLO
        from src.models.inference_cache import file_digest, pack_pose_result
        if self.model is None:
            if file_digest(self.weights) != self.weights_sha256:
                raise ValueError("Pose recovery weights differ from the cached full-frame model")
            # This instance never shares tracker callbacks or worker state.
            self.model = YOLO(str(self.weights))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return pack_pose_result(self.model.predict(image, **self.settings, device=device,
                                half=device == "cuda", verbose=False)[0])

    def repair(self, poses, frame_index, load_frame):
        from src.detection.player_pose import extract_player_poses
        from src.models.inference_cache import digest, unpack_pose_result
        repaired = []
        for pose in poses:
            crop = recovery_crop(pose, self.size)
            if crop is None:
                repaired.append(pose)
                continue
            key = digest({"frame": frame_index, "crop": crop})
            def infer():
                image = load_frame()
                if image is None or (image.shape[1], image.shape[0]) != self.size:
                    raise ValueError(f"Cannot load native pose recovery frame {frame_index}")
                return self._predict(image[crop[1]:crop[3], crop[0]:crop[2]])
            raw = self.cache.compute(self.stage, key, infer) if self.cache is not None else infer()
            candidates = extract_player_poses(unpack_pose_result(raw), net_y=0)
            fixed = recover_from_crop(pose, candidates, crop)
            if fixed is not pose:
                fixed.pose_repair["crop_cache"] = {"stage": self.stage, "item": key,
                                                  "settings": dict(self.settings)}
            repaired.append(fixed)
        return merge_fragments(repaired)
