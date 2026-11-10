"""Score roster identities against human movement review (``human_player_movement_v1``).

Human labels are joined to saved poses by frame and pose index. One pose per
verified player and frame is scored; flagged duplicate poses are left out.
Nothing here feeds labels back into association.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA = 'human_player_movement_v1'


def human_labels(movement: dict, video: str | None = None) -> dict:
    """Return ``{'players': {key: player_id}, 'excluded': {key: decision}, ...}``."""
    if movement.get('schema') != SCHEMA:
        raise ValueError(f'Expected {SCHEMA}')
    videos = movement['videos']
    if video is None:
        if len(videos) != 1:
            raise ValueError('Movement file covers several videos; name one')
        video = next(iter(videos))
    spans = sorted(videos[video]['spans'], key=lambda s: (s.get('revision') or 0, s['start']))
    players, excluded, duplicates = {}, {}, {}
    seen = set()
    for span in spans:
        frames = range(span['start'], span['end']+1)
        if len(span['pose_indices']) != len(frames):
            raise ValueError('Span pose indices do not cover its frames')
        skipped = set(span.get('duplicate_frames') or [])
        for frame, index in zip(frames, span['pose_indices']):
            key = f'{frame}:{index}'
            if span['decision'] != 'player':
                excluded[key] = span['decision']
                continue
            if span.get('verified') is not True or span.get('label_source') != 'human':
                continue
            pid = span['player_id']
            if frame in skipped or (pid, frame) in seen:
                duplicates[key] = pid
                continue
            seen.add((pid, frame))
            players[key] = pid
    references = {p['player_id']: p['references'] for p in movement['roster']['players']}
    return {'video': video, 'players': players, 'excluded': excluded, 'duplicates': duplicates,
            'references': references,
            'window': [min(s['start'] for s in spans), max(s['end'] for s in spans)]}


def best_mapping(predictions: dict, players: dict) -> dict:
    """One-to-one model-ID to human-ID naming that maximises agreement.

    Roster IDs are arbitrary names, so a permutation is not an accuracy claim.
    An ID left over when the model knows more people than the reviewer stays
    unmapped and counts as wrong.
    """
    pairs = Counter((predictions[key], pid) for key, pid in players.items()
                    if predictions.get(key) is not None)
    model = sorted({m for m, _ in pairs})
    human = sorted({h for _, h in pairs})
    if not model:
        return {}
    table = np.asarray([[pairs[m, h] for h in human] for m in model])
    rows, columns = linear_sum_assignment(-table)
    return {model[i]: human[j] for i, j in zip(rows, columns) if table[i, j] > 0}


def score(predictions: dict, labels: dict, fps: float, mapping: dict | None = None) -> dict:
    """``predictions`` maps ``frame:pose_index`` to a model ID or None."""
    players = labels['players']
    missing = [key for key in players if key not in predictions]
    if missing:
        raise ValueError(f'{len(missing)} reviewed poses are absent from predictions, e.g. {missing[0]}')
    mapping = best_mapping(predictions, players) if mapping is None else mapping
    totals = Counter()
    per_player = defaultdict(Counter)
    confusion = defaultdict(Counter)
    timeline = defaultdict(list)
    for key, pid in players.items():
        raw = predictions[key]
        named = mapping.get(raw, f'unmapped:{raw}') if raw is not None else None
        outcome = 'unknown' if named is None else 'correct' if named == pid else 'wrong'
        totals[outcome] += 1
        per_player[pid][outcome] += 1
        confusion[pid][named or 'P?'] += 1
        timeline[pid].append((int(key.split(':')[0]), named))
    switches = returns = returned = 0
    for pid, items in timeline.items():
        items.sort()
        for (f0, a), (f1, b) in zip(items, items[1:]):
            if f1-f0 == 1 and a is not None and b is not None and a != b:
                switches += 1
            if (f1-f0)/fps >= 1. and a == pid:
                returns += 1
                returned += b == pid
    total = sum(totals.values())
    excluded = Counter()
    for key, decision in labels['excluded'].items():
        if predictions.get(key) is not None:
            excluded[decision] += 1
    return {
        'observations': total, **{k: totals[k] for k in ('correct', 'wrong', 'unknown')},
        'correct_rate': totals['correct']/max(total, 1),
        'wrong_rate': totals['wrong']/max(total, 1),
        'unknown_rate': totals['unknown']/max(total, 1),
        'precision_when_assigned': totals['correct']/max(totals['correct']+totals['wrong'], 1),
        'mapping_model_to_human': mapping,
        'per_player': {pid: {**{k: c[k] for k in ('correct', 'wrong', 'unknown')},
                             'total': sum(c.values()),
                             'correct_rate': c['correct']/max(sum(c.values()), 1),
                             'predicted': dict(confusion[pid])}
                       for pid, c in sorted(per_player.items())},
        'adjacent_id_switches': switches,
        'returns_after_1s': {'total': returns, 'correct_on_first_frame': returned},
        'ids_on_excluded_poses': dict(excluded),
        'excluded_poses': dict(Counter(labels['excluded'].values())),
    }
