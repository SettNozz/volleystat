"""Reject contradictory roster labels on detector tracklets; never invent IDs."""
from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right


def reject_tracklet_conflicts(frames, predictions, anchors, *, dominance=.8,
                             min_clean_votes=3, clean_overlap=.3, fps=30., max_gap_seconds=3.,
                             appearance_limit=.35,appearance_max=.55,appearance_weight=.75):
    """Raw tracks are weak consistency evidence, not identity ground truth.

    A raw track may genuinely switch people. Reject only a locally bracketed
    contradiction, or an ID borrowed while its stronger established track is
    visible but ambiguous. No label is copied to a different observation.
    """
    groups = defaultdict(list)
    source = dict(predictions)
    for frame in frames:
        for obs in frame['players']:
            if obs.get('raw_id') is not None:
                groups[obs['raw_id']].append((frame['f'],obs))
    anchor_keys = {a['key'] for a in anchors}
    reliable = {}
    for raw, observations in groups.items():
        def identity_evidence(obs):
            prediction=source[obs['key']]
            distance=prediction.get('appearance_distance')
            if distance is None:
                # Legacy costs give an upper bound: their motion term is >=0.
                distance=((prediction.get('cost') or 0.)*appearance_max/appearance_weight
                          if appearance_weight > 0 else float('inf'))
            return obs['occlusion'] < clean_overlap or distance <= appearance_limit or obs['key'] in anchor_keys
        clean = [(f,source[o['key']]['player_id']) for f,o in observations
                 if identity_evidence(o)
                 and source[o['key']]['player_id'] is not None
                 and (source[o['key']].get('cost') or 0.) <= .55]
        votes = Counter(pid for _,pid in clean)
        if not votes:
            continue
        pid,count = votes.most_common(1)[0]
        if count >= min_clean_votes and count/sum(votes.values()) >= dominance:
            reliable[raw] = (pid,count,[f for f,_ in clean],clean)

    def bracket(raw,frame):
        if raw not in reliable:
            return None
        pid,_,times,clean = reliable[raw]
        left,right = bisect_left(times,frame)-1,bisect_right(times,frame)
        if left < 0 or right >= len(clean):
            return None
        before,after = clean[left],clean[right]
        if (before[1] == after[1] == pid and (frame-before[0])/fps <= max_gap_seconds
                and (after[0]-frame)/fps <= max_gap_seconds):
            return pid
        return None

    rejected = 0
    conflicts = set()
    for frame in frames:
        owners = defaultdict(list)
        for obs in frame['players']:
            raw = obs.get('raw_id')
            pid = bracket(raw,frame['f'])
            if pid is not None and source[obs['key']]['player_id'] is None:
                owners[pid].append((raw,reliable[raw][1]))
        for obs in frame['players']:
            key,raw = obs['key'],obs.get('raw_id')
            previous = source[key]
            pid = previous['player_id']
            if key in anchor_keys or pid is None:
                continue
            own_bracket = bracket(raw,frame['f'])
            own_count = reliable[raw][1] if raw in reliable and reliable[raw][0] == pid else 0
            conflict = own_bracket is not None and own_bracket != pid
            conflict |= any(other != raw and (own_count == 0 or count >= 2*own_count)
                            for other,count in owners.get(pid,[]))
            if conflict:
                predictions[key] = {**previous,'player_id':None,'candidate_player_id':pid,
                                    'unfiltered_reason':previous['reason'],
                                    'reason':'tracklet_identity_conflict'}
                rejected += 1
                if raw is not None:
                    conflicts.add(raw)
    return {'enabled':True, 'conflicting_tracks':len(conflicts), 'rejected_observations':rejected,
            'dominance':dominance, 'min_clean_votes':min_clean_votes,
            'max_gap_seconds':max_gap_seconds,
            'operation':'local rejection only; raw tracks are not trusted person identities'}
