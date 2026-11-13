"""Confirm short-term roster handoffs without treating raw IDs as people."""
from collections import Counter, defaultdict, deque

import numpy as np

from src.tracking.court_pose_consistency import pose_tracklets


def _unit(value):
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value)
    return value / norm if np.isfinite(value).all() and norm > 1e-8 else None


def _nearby(box, previous, seconds):
    height = max(box[3] - box[1], 1)
    old_height = max(previous[3] - previous[1], 1)
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    old_center = np.array([(previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2])
    return (np.linalg.norm(center - old_center) <= (.5 + seconds) * min(height, old_height)
            and abs(np.log(height / old_height)) <= .5)


def recover_recent_roster_handoffs(annotations, report, features, galleries, fps, target):
    """Bind a roster handoff after three independent appearance/motion matches.

    Frozen galleries still define the roster. Recent accepted observations
    provide short-term appearance/motion context, not new training/gallery
    labels. A raw ID may have belonged to an opponent before the handoff;
    membership starts at the confirmed transition, never at the track's birth.
    """
    tokens = {(frame, index): token for token, track in enumerate(pose_tracklets(annotations, fps))
              for frame, _, index in track}
    recent = defaultdict(lambda: deque(maxlen=30))
    pending = {}
    gallery_pending = {}
    bindings = {}
    counts = Counter()

    def vector(frame, index):
        return _unit(features.get(f'{frame}:{index}'))

    def gallery_similarity(pid, value):
        sims = galleries[pid] @ value
        return float(np.sort(sims)[-min(3, len(sims)):].mean())

    def mark(pose, pid, occupied):
        # Team continuity is useful even if another ambiguous box currently
        # holds this person ID. Never export two simultaneous copies of it.
        previous = pose.get('player_id')
        pose['team_id'] = target
        pose['team_source'] = 'recent_roster_track_handoff'
        if not previous and pid not in occupied:
            pose['player_id'] = pid
            pose['identity'] = {'method': report.get('method'), 'reason': 'recent_roster_track_handoff',
                                'label_source': 'model', 'verified': False}
            occupied.add(pid)
            counts['recovered_ids'] += 1
        counts['team_observations'] += 1

    for ann in annotations:
        frame = ann['f']
        poses = ann.get('poses') or []
        for history in recent.values():
            while history and frame - history[0][0] > fps:
                history.popleft()
        occupied = {p.get('player_id') for p in poses if p.get('player_id')}
        by_token = {tokens[(frame, i)]: i for i, p in enumerate(poses) if (frame, i) in tokens}
        vectors = {i: vector(frame, i) for i in range(len(poses))}
        continued = set()

        for token, binding in list(bindings.items()):
            pid, last_frame, last_box, reference = binding
            if frame - last_frame > fps:
                bindings.pop(token)
                continue
            i = by_token.get(token)
            if i is None:
                continue
            pose, value = poses[i], vectors[i]
            if not _nearby(pose['bb'], last_box, (frame - last_frame) / fps):
                bindings.pop(token)
                continue
            # A new confidently different person ends the binding, including
            # another raw-ID hijack without a large image-space displacement.
            other = pose.get('player_id')
            similarity = float(value @ reference) if value is not None else None
            if similarity is not None and (similarity < .5 or (other and other != pid and similarity < .7)):
                bindings.pop(token)
                continue
            mark(pose, pid, occupied)
            if value is not None and similarity >= .7:
                reference = value
            bindings[token] = (pid, frame, pose['bb'], reference)
            continued.add(i)

        # The stricter frozen-gallery/motion recovery runs before this pass.
        # Its isolated matches must also establish continuity: otherwise a
        # recycled opponent ID regains its old team as soon as the original
        # owner's one-second motion anchor expires. Require three distinct,
        # unambiguous gallery matches, never a track-wide majority/backfill.
        for i, pose in enumerate(poses):
            pid, value = pose.get('player_id'), vectors[i]
            token = tokens.get((frame, i))
            if (i in continued or pid not in galleries or value is None or token is None
                    or (pose.get('identity') or {}).get('reason') != 'appearance_motion_roster_recovery'):
                continue
            own = gallery_similarity(pid, value)
            rival = max((gallery_similarity(other, value) for other in galleries if other != pid), default=-1)
            if own < .85 or own - rival < .10:
                continue
            key = (token, pid)
            history = gallery_pending.get(key, [])
            if history and (frame - history[-1][0] > .2 * fps
                            or not _nearby(pose['bb'], history[-1][1]['bb'],
                                           (frame - history[-1][0]) / fps)):
                history = []
            history = [row for row in history if 0 < frame - row[0] <= .33 * fps]
            history.append((frame, pose, occupied, value))
            gallery_pending[key] = history
            if len(history) < 3:
                continue
            for _, old_pose, old_occupied, _ in history:
                mark(old_pose, pid, old_occupied)
            reference = _unit(np.mean([row[3] for row in history], axis=0))
            bindings[token] = (pid, frame, pose['bb'], reference)
            continued.add(i)
            counts['confirmed_gallery_handoffs'] += 1
            gallery_pending.pop(key)
        for key in list(gallery_pending):
            if frame - gallery_pending[key][-1][0] > .2 * fps:
                gallery_pending.pop(key)

        # Dormant IDs only. Visible owners and competing candidates block a
        # handoff; appearance alone cannot duplicate a roster identity.
        proposals = []
        for i, pose in enumerate(poses):
            value, token = vectors[i], tokens.get((frame, i))
            if (i in continued or pose.get('player_id') or pose.get('team_id') == target
                    or (pose.get('identity') or {}).get('reason') == 'uncorroborated_side_prior'
                    or value is None or token is None):
                continue
            scores = {}
            for pid, history in recent.items():
                if pid in occupied or len(history) < 3:
                    continue
                last_frame, last_box, _ = history[-1]
                gap = (frame - last_frame) / fps
                if not 0 < gap <= 1 or not _nearby(pose['bb'], last_box, gap):
                    continue
                if gallery_similarity(pid, value) < .60:
                    continue
                scores[pid] = max(float(value @ v) for _, _, v in history)
            if not scores:
                continue
            pid = max(scores, key=scores.get)
            score = scores[pid]
            # Compare with all recent people, including currently visible
            # rivals, so similar uniforms cannot win just through vacancy.
            rival = max((max(float(value @ v) for _, _, v in history)
                         for other, history in recent.items() if other != pid and history), default=-1)
            if score >= .84 and score - rival >= .10:
                proposals.append((score, i, token, pid))

        competing = defaultdict(list)
        for score, _, _, pid in proposals:
            competing[pid].append(score)
        ambiguous = {pid for pid, values in competing.items()
                     if len(values) > 1 and sorted(values, reverse=True)[0] - sorted(values, reverse=True)[1] < .05}
        used = set()
        accepted_pending = set()
        for score, i, token, pid in sorted(proposals, reverse=True):
            if pid in used or pid in occupied or pid in ambiguous:
                continue
            used.add(pid)
            key = (token, pid)
            history = pending.get(key, [])
            if history and frame - history[-1][0] > .2 * fps:
                history = []
            history = [row for row in history if frame - row[0] <= .33 * fps]
            history.append((frame, poses[i], occupied, vectors[i]))
            pending[key] = history
            accepted_pending.add(key)
            if len(history) < 3:
                continue
            for old_frame, pose, old_occupied, _ in history:
                mark(pose, pid, old_occupied)
            bindings[token] = (pid, frame, poses[i]['bb'], vectors[i])
            continued.add(i)
            counts['confirmed_handoffs'] += 1
            pending.pop(key)
        for key in list(pending):
            if key not in accepted_pending and frame - pending[key][-1][0] > .2 * fps:
                pending.pop(key)

        for i, pose in enumerate(poses):
            pid, value = pose.get('player_id'), vectors[i]
            if pid not in galleries or value is None:
                continue
            identity = pose.get('identity') or {}
            trusted = (identity.get('reason') in ('associated', 'roster_anchor', 'occluded_appearance_only')
                       and (identity.get('appearance_distance') or 0) <= .30)
            if trusted or i in continued:
                recent[pid].append((frame, pose['bb'], value))

    return dict(counts)
