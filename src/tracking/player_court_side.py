"""Offline player-side evidence from floor geometry and continuous tracks."""

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass

from src.tracking.court_pose_consistency import pose_ankles, pose_tracklets


@dataclass(frozen=True)
class SideEvidence:
    side: str | None
    reason: str
    distance: float
    band: float

    @property
    def tentative(self):
        return self.reason in ('uncertain', 'back_prior') and abs(self.distance) <= 2 * self.band


def measure_poses(annotations, projector, frame_height):
    """Measure pixel residuals once, independently of any tested line offset."""
    measurements = {}
    scale = frame_height / 2160
    for ann in annotations:
        for i, pose in enumerate(ann.get('poses') or []):
            x1, y1, x2, y2 = pose['bb']
            ankles = pose_ankles(pose)
            x = sum(p[0] for p in ankles) / len(ankles) if ankles else (x1 + x2) / 2
            y = max(p[1] for p in ankles) if ankles else y2
            ankle_distances = [p[1] - projector.net_floor_y(p[0])
                               for p in ankles if p[2] >= .5]
            height = y2 - y1
            band = max(24 * scale, min(48 * scale, height * .10))
            measurements[(ann['f'], i)] = (
                y - projector.net_floor_y(x), ankle_distances, height, band,
                (pose.get('back') or 0) >= .45 and projector.bbox_occludes_net(pose['bb']),
            )
    return measurements


def frame_sides(annotations, measurements, offset, *, use_back=True):
    """Replay the current net-floor branch on the fixed saved candidate pool."""
    result = {}
    previous_state = None
    for ann in annotations:
        serve = previous_state in ('serve_prep', 'serve_ready')
        for i, _ in enumerate(ann.get('poses') or []):
            key = (ann['f'], i)
            distance, _, height, band, back = measurements[key]
            distance -= offset
            tolerance = max(5, min(18, .06 * height))
            if use_back and back:
                tolerance = max(tolerance, max(10, min(40 if serve else 30,
                                                       height * (.14 if serve else .10))))
            result[key] = SideEvidence('b' if distance >= -tolerance else 'a',
                                       'frame_floor', distance, band)
        previous_state = ann.get('state')
    return result


def temporal_sides(annotations, measurements, fps, offset, *, use_back=False):
    """Use nearby trusted ankles on both sides of a timestamp to resolve it."""
    result = {}
    anchors = {}
    for key, (distance, ankles, _, band, back) in measurements.items():
        distance -= offset
        side = None
        if len(ankles) == 2:
            if min(ankles) - offset > band:
                side = 'b'
            elif max(ankles) - offset < -band:
                side = 'a'
        if side is not None:
            anchors[key] = side
            reason = 'trusted_ankles'
        elif use_back and back and abs(distance) <= band:
            side, reason = 'b', 'back_prior'
        else:
            reason = 'uncertain'
        result[key] = SideEvidence(side, reason, distance, band)

    for tracklet in pose_tracklets(annotations, fps):
        trusted = [(frame, anchors[(frame, i)]) for frame, _, i in tracklet if (frame, i) in anchors]
        times = [frame for frame, _ in trusted]
        for frame, _, i in tracklet:
            key = (frame, i)
            evidence = result[key]
            if abs(evidence.distance) > 2 * evidence.band and key in anchors:
                continue
            lo = bisect_left(times, frame - fps)
            hi = bisect_right(times, frame + fps)
            votes = Counter(side for t, side in trusted[lo:hi] if t != frame)
            if not votes:
                continue
            side, count = votes.most_common(1)[0]
            if count < 3 or count < .8 * sum(votes.values()):
                if evidence.reason == 'back_prior' and votes['a']:
                    result[key] = SideEvidence(None, 'uncertain', evidence.distance, evidence.band)
                continue
            if key in anchors and side != anchors[key]:
                # Revisit a conflicting observation only when support brackets it.
                past = bisect_left(times, frame) - 1
                future = bisect_right(times, frame)
                if not (lo <= past and future < hi and trusted[past][1] == trusted[future][1] == side):
                    continue
            result[key] = SideEvidence(side, 'track_history', evidence.distance, evidence.band)
    return result


def preserve_frame_sides(temporal, fallback):
    """Keep existing near-side candidates and recover plausible near-side misses."""
    result = {}
    for key, evidence in temporal.items():
        if fallback[key].side == 'b':
            result[key] = fallback[key]
        elif evidence.reason in ('track_history', 'trusted_ankles'):
            result[key] = evidence
        else:
            result[key] = SideEvidence('a', 'uncertain', evidence.distance, evidence.band)
    return result
