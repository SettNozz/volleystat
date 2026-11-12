"""Persistent roster membership; court side is only a frame-local projection."""
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right

import numpy as np

from src.tracking.court_pose_consistency import pose_tracklets
from src.tracking.match_analytics import DEFAULT_TEAM_IDENTITY_MAP


def tentative_roster_side(pose):
    """A side changed only by a net-facing prior has no team confirmation."""
    return (pose.get('team_source') == 'frame_geometry'
            and pose.get('team_prior') in ('serve_back_facing_net_override', 'back_facing_net_override'))


def _reject_uncorroborated_side_priors(annotations, report, features, galleries, fps):
    """Do not let a few weak near-net matches manufacture a missing owner.

    Keep ordinary association and its motion history intact. Before roster
    membership can propagate an ID, prior-only tracklets need three distinct
    strong frozen-gallery observations or independent team/anchor evidence.
    """
    blocked = set()
    maximum = (report.get('parameters') or {}).get('recovery_appearance_max', .35)
    minimum = (report.get('parameters') or {}).get('recovery_confirmations', 3)
    rejected = 0
    for tracklet in pose_tracklets(annotations, fps):
        rows = [(frame, i, annotations[index]['poses'][i]) for frame, index, i in tracklet]
        votes = defaultdict(list)
        for frame, i, pose in rows:
            if pose.get('player_id') in galleries:
                votes[pose['player_id']].append((frame, i, pose))
        unsupported = set()
        for pid, matches in votes.items():
            if any(not tentative_roster_side(pose) for _, _, pose in matches):
                continue
            strong = set()
            for frame, i, pose in matches:
                value = features.get(f'{frame}:{i}')
                if value is None:
                    continue
                value = np.asarray(value, dtype=np.float32)
                norm = np.linalg.norm(value)
                if norm > 1e-8 and np.isfinite(value).all():
                    distance = float(1 - np.max(galleries[pid] @ (value / norm)))
                    if distance <= maximum:
                        strong.add(frame)
            if len(strong) < minimum:
                unsupported.add(pid)
        if not unsupported:
            continue
        for frame, i, pose in rows:
            pid = pose.get('player_id')
            if pid and pid not in unsupported:
                continue
            blocked.add((frame, i))
            if pid:
                rejected += 1
            pose['player_id'] = None
            pose['identity'] = {'method': report.get('method'), 'reason': 'uncorroborated_side_prior',
                                'label_source': 'model', 'verified': False}
    return blocked, rejected


def team_map_at_frame(frame, identity_map=None, segments=()):
    result = {**DEFAULT_TEAM_IDENTITY_MAP, **(identity_map or {})}
    for segment in segments:
        if segment.get('start_frame', -1) <= frame <= segment.get('end_frame', -1):
            result.update(segment.get('team_identity_map') or {})
            break
    if set(result) != {'team_a', 'team_b'} or len(set(result.values())) != 2:
        raise ValueError('Player membership requires two distinct court-side team identities')
    return result


def roster_side_at_frame(frame, side, identity_map=None, segments=()):
    member = team_map_at_frame(0, identity_map)[f'team_{side}']
    return next(s[-1] for s, team in team_map_at_frame(frame, identity_map, segments).items() if team == member)


def _centres(ann):
    poses = ann.get('poses') or []
    order = sorted(range(len(poses)), key=lambda i: poses[i].get('team') not in ('a', 'team_a'))
    points = ann.get('p')
    if points is not None and len(points) == len(poses):
        return {id(poses[i]): point for i, point in zip(order, points)}
    return {id(p): [round((p['bb'][0] + p['bb'][2]) / 2, 1),
                    round((p['bb'][1] + p['bb'][3]) / 2, 1)] for p in poses}


def sync_centres(ann, points):
    poses = sorted(ann.get('poses') or [], key=lambda p: p.get('team') not in ('a', 'team_a'))
    ann['p'] = [points[id(p)] for p in poses]


def apply_seed_membership(annotations, seed, side, fps, *, identity_map=None, segments=()):
    """A matched roster photo outranks uncertain feet on its continuous tracklet."""
    target = team_map_at_frame(0, identity_map)[f'team_{side}']
    anchors = set(seed['assignments']) | {a['key'] for a in seed.get('introductions') or []}
    points = [_centres(ann) for ann in annotations]
    for tracklet in pose_tracklets(annotations, fps):
        if not any(f'{frame}:{i}' in anchors for frame, _, i in tracklet):
            continue
        for frame, index, pose_index in tracklet:
            pose = annotations[index]['poses'][pose_index]
            mapping = team_map_at_frame(frame, identity_map, segments)
            local_side = next(s[-1] for s, team in mapping.items() if team == target)
            if pose['team'][-1] != local_side:
                pose.setdefault('team_geometry', pose['team'])
                pose['team'] = local_side
            pose['team_id'] = target
            pose['team_source'] = 'roster_anchor_tracklet'
    for ann, original in zip(annotations, points):
        sync_centres(ann, original)


def preserve_roster_membership(annotations, report, features, fps, *, identity_map=None, segments=()):
    """Recover observed roster people without promoting opponents into the roster.

    Galleries/roster are frozen by the preceding identity pass on the selected
    team. Continuous tracklets may bridge jumps and brief occlusion. A new raw
    ID additionally needs distinct appearance and nearby motion evidence.
    Neither a predicted box nor an unobserved skeleton is ever created.
    """
    target = team_map_at_frame(0, identity_map)[f"team_{report.get('side', 'b')}"]
    gallery = {}
    for pid, keys in report.get('gallery_keys', {}).items():
        vectors = [features[key] for key in keys if key in features]
        if vectors:
            matrix = np.asarray(vectors, dtype=np.float32)
            gallery[pid] = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    ids = list(gallery)
    blocked, rejected = _reject_uncorroborated_side_priors(annotations, report, features, gallery, fps)
    points = [_centres(ann) for ann in annotations]
    rows = {(ann['f'], i): p for ann in annotations for i, p in enumerate(ann.get('poses') or [])}
    scores = {}
    # Batch each gallery comparison, keeping the same three-view evidence as
    # roster discovery. Missing crops abstain from cross-ID recovery.
    candidates = [(key, features[f'{key[0]}:{key[1]}']) for key, p in rows.items()
                  if not p.get('player_id') and f'{key[0]}:{key[1]}' in features]
    if candidates and ids:
        vectors = np.asarray([v for _, v in candidates], dtype=np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        values = []
        for pid in ids:
            sims = vectors @ gallery[pid].T
            n = min(3, sims.shape[1])
            values.append(np.partition(sims, -n, axis=1)[:, -n:].mean(axis=1))
        scores = {key: value for (key, _), value in zip(candidates, np.asarray(values).T)}

    proposed = {}
    for tracklet in pose_tracklets(annotations, fps):
        trusted = [(frame, rows[(frame, i)]['player_id']) for frame, _, i in tracklet
                   if rows[(frame, i)].get('player_id') in gallery]
        times = [frame for frame, _ in trusted]
        for frame, _, i in tracklet:
            key = (frame, i)
            if rows[key].get('player_id') or key in blocked:
                continue
            lo, hi = bisect_left(times, frame - fps), bisect_right(times, frame + fps)
            nearby = Counter(pid for _, pid in trusted[lo:hi])
            if len(nearby) != 1 or sum(nearby.values()) < 3:
                continue
            pid = next(iter(nearby))
            value = scores.get(key)
            # A pose/turn can change appearance sharply during a jump. Brief
            # continuous motion bridges that uncertainty, while distinct
            # evidence for another person still blocks the transfer.
            if value is not None and value.max() >= .85 and value.max() - value[ids.index(pid)] >= .10:
                continue
            proposed[key] = (pid, 'short_continuous_roster_membership', 1.)
        votes = Counter(rows[(frame, i)].get('player_id') for frame, _, i in tracklet
                        if rows[(frame, i)].get('player_id') in gallery)
        if not votes:
            continue
        pid, count = votes.most_common(1)[0]
        if count < 3 or count < .9 * sum(votes.values()):
            continue
        for frame, _, i in tracklet:
            key, pose = (frame, i), rows[(frame, i)]
            if pose.get('player_id') or key in proposed or key in blocked:
                continue
            value = scores.get(key)
            if value is not None:
                own = value[ids.index(pid)]
                if own < .55 or value.max() - own > .08:
                    continue
            proposed[key] = (pid, 'continuous_roster_membership', float(value.max()) if value is not None else 0.)

    anchors = defaultdict(list)
    for (frame, i), pose in rows.items():
        pid = pose.get('player_id')
        if pid in gallery:
            anchors[pid].append((frame, pose['bb']))
    for key, value in scores.items():
        if key in proposed or not len(value) or key in blocked or tentative_roster_side(rows[key]):
            continue
        best = int(value.argmax())
        pid, similarity = ids[best], float(value[best])
        if similarity < .85 or similarity - np.max(np.delete(value, best), initial=-1) < .10:
            continue
        box = rows[key]['bb']
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        for frame, old in anchors[pid]:
            gap = abs(frame - key[0]) / fps
            if gap == 0 or gap > 1:
                continue
            old_center = np.array([(old[0] + old[2]) / 2, (old[1] + old[3]) / 2])
            height, old_height = max(box[3] - box[1], 1), max(old[3] - old[1], 1)
            if (np.linalg.norm(center - old_center) <= (.5 + gap) * min(height, old_height)
                    and abs(np.log(height / old_height)) <= .5):
                proposed[key] = (pid, 'appearance_motion_roster_recovery', similarity)
                break

    by_frame = defaultdict(list)
    for (frame, index), value in proposed.items():
        by_frame[frame].append((index, value))
    recovered = changed = 0
    for ann in annotations:
        frame = ann['f']
        occupied = {p.get('player_id') for p in ann.get('poses') or [] if p.get('player_id')}
        for i, (pid, reason, score) in sorted(by_frame[frame], key=lambda row: -row[1][2]):
            if pid in occupied:
                continue
            pose = rows[(frame, i)]
            pose['player_id'] = pid
            pose['identity'] = {'method': report.get('method'), 'reason': reason,
                                'label_source': 'model', 'verified': False}
            occupied.add(pid)
            recovered += 1
    from src.tracking.player_roster_handoff import recover_recent_roster_handoffs
    handoffs = recover_recent_roster_handoffs(annotations, report, features, gallery, fps, target)
    recovered += handoffs.get('recovered_ids', 0)
    for ann, old_points in zip(annotations, points):
        frame = ann['f']
        mapping = team_map_at_frame(frame, identity_map, segments)
        for pose in ann.get('poses') or []:
            if not pose.get('player_id') and tentative_roster_side(pose):
                pose['team'] = pose.get('team_geometry', pose['team'])[-1]
            side = 'team_' + pose['team'][-1]
            handoff = pose.get('team_source') == 'recent_roster_track_handoff'
            if pose.get('player_id') in gallery or handoff:
                new_side = next(s for s, team in mapping.items() if team == target)
                if side != new_side:
                    pose.setdefault('team_geometry', pose['team'])
                    pose['team'] = new_side[-1]
                    changed += 1
                pose['team_id'] = target
                pose['team_source'] = 'recent_roster_track_handoff' if handoff else 'roster_membership'
            else:
                pose['team_id'] = mapping[side]
                pose.setdefault('team_source', 'frame_geometry')
        sync_centres(ann, old_points)
    report['team_id'] = target
    report['team_membership'] = {'method': 'continuous_roster_appearance_motion_v2',
                               'recovered_ids': recovered, 'corrected_sides': changed,
                               'recent_handoffs': handoffs,
                               'side_prior_rejected_observations': rejected,
                               'label_source': 'model', 'verified': False}
    report['person_observations'] = dict(Counter(p['player_id'] for p in rows.values() if p.get('player_id')))
    report['assigned'] = sum(report['person_observations'].values())
    report['observations'] = sum(p['team_id'] == target for p in rows.values())
    report['coverage'] = report['assigned'] / max(report['observations'], 1)
    return report


def cap_saved_players(annotations, projector, maximum, identity_report=None):
    """Apply the player cap only after roster membership has been resolved."""
    if maximum is None:
        return
    key_map = {}
    for ann in annotations:
        points = _centres(ann)
        poses = ann.get('poses') or []
        selected = set()
        for side in ('a', 'b'):
            members = [p for p in poses if p['team'][-1] == side]
            def rank(pose):
                score = projector.inside_score(((pose['bb'][0] + pose['bb'][2]) / 2, pose['bb'][3])) if projector else 0
                handoff = pose.get('team_source') == 'recent_roster_track_handoff'
                member = pose.get('team_source') in ('continuous_track_membership', 'roster_anchor_tracklet', 'roster_membership')
                return (handoff, bool(pose.get('player_id')), member, score)
            selected.update(id(p) for p in sorted(members, key=rank, reverse=True)[:maximum])
        ann['poses'] = [p for p in poses if id(p) in selected]
        indices = {id(p): i for i, p in enumerate(ann['poses'])}
        key_map.update((f'{ann["f"]}:{i}', f'{ann["f"]}:{indices[id(p)]}')
                       for i, p in enumerate(poses) if id(p) in selected)
        sync_centres(ann, points)
    if identity_report and identity_report.get('status') == 'completed':
        seed = identity_report.get('seed') or {}
        seed['assignments'] = {key_map[k]: v for k, v in seed.get('assignments', {}).items()}
        for anchor in seed.get('introductions') or []:
            anchor['key'] = key_map[anchor['key']]
        identity_report['gallery_keys'] = {pid: [key_map[k] for k in keys if k in key_map]
                                           for pid, keys in identity_report.get('gallery_keys', {}).items()}
        for other in identity_report.get('roster_discovery', {}).get('other_people', []):
            key = other.get('example_key')
            other['example_key'] = key_map.get(key)
        target = identity_report.get('team_id')
        poses = [p for ann in annotations for p in ann.get('poses') or []]
        identity_report['assigned'] = sum(bool(p.get('player_id')) for p in poses)
        identity_report['person_observations'] = dict(Counter(p['player_id'] for p in poses if p.get('player_id')))
        identity_report['observations'] = sum(p.get('team_id') == target for p in poses)
        identity_report['coverage'] = identity_report['assigned'] / max(identity_report['observations'], 1)
