"""Find the constant players of one court side from the whole video, then name every box.

A single early frame is a weak roster: one bystander on court or one missing
player gives a wrong identity for the whole match, and eight crops from half
a second cannot describe a person from every angle. This module instead:

1. cuts detector tracks into short clean tracklets and averages their frozen
   OSNet embeddings;
2. merges tracklets by appearance (average linkage). Two tracklets that are
   cleanly visible in the same frames are different people and never merge;
3. keeps the people who are present longest as the roster, or the people
   named by confirmed anchors;
4. scores every saved box against all tracklets of each roster person,
   averages the scores along its detector track and assigns IDs one-to-one
   per frame. Weak or ambiguous boxes stay unknown, and so does a box that
   looks more like a known other person (an opponent at the net, a referee).

No human check labels enter this module. Nothing is trained, no pose is
created, and an unknown box never becomes a new person.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RosterDiscoveryParameters:
    # Tracklets: at most this long, split at detector gaps, clean boxes only.
    chunk_seconds: float = 1.
    chunk_gap_seconds: float = .17
    clean_overlap_max: float = .3
    chunk_min_clean: int = 5
    chunk_coherence_min: float = .6
    max_chunks: int = 3000
    # Merging: cosine distance limit and the co-visibility that forbids a merge.
    merge_distance_max: float = .3
    cannot_link_frames: int = 3
    # A roster person must be present at least this share of the longest one.
    min_presence_ratio: float = .25
    min_presence_seconds: float = 3.
    gallery_max_views: int = 256
    gallery_top_views: int = 3
    # One stray duplicate box can forbid a merge of two halves of one person.
    # A group that looks like a larger one and shares at most this part of its
    # clean frames with it is a fragment of that person.
    fragment_covisible_max: float = .02
    # A non-roster group seen together with every roster person, for at least
    # this long, is a known other person (opponent at the net, referee).
    other_person_min_seconds: float = 1.
    # Assignment.
    smooth_seconds: float = .5
    occluded_weight: float = .3
    similarity_min: float = .6
    assignment_margin: float = .03

    def to_dict(self):
        return asdict(self)


class RosterNotFound(ValueError):
    """The video does not show enough separable people for a roster."""


def _unit(vector):
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 1e-8:
        raise ValueError('Missing/non-finite/zero appearance embedding')
    return value/norm


def tracklet_chunks(frames: list[dict], features: dict, fps: float,
                    parameters: RosterDiscoveryParameters) -> list[dict]:
    """Short runs of one detector track with a mean embedding of their clean boxes."""
    limit = max(1, round(parameters.chunk_seconds*fps))
    gap = max(1, round(parameters.chunk_gap_seconds*fps))
    tracks = defaultdict(list)
    for frame in frames:
        for obs in frame['players']:
            if obs.get('raw_id') is not None and not (obs.get('tentative_side') or obs.get('side_prior_only')):
                tracks[obs['raw_id']].append((frame['f'], obs))
    chunks = []

    def close(raw, run):
        clean = [(f, o) for f, o in run if o['occlusion'] < parameters.clean_overlap_max
                 and features.get(o['key']) is not None]
        if len(clean) < parameters.chunk_min_clean:
            return
        vectors = np.stack([_unit(features[o['key']]) for _, o in clean])
        mean = _unit(vectors.mean(0))
        coherence = float((vectors @ mean).min())
        if coherence < parameters.chunk_coherence_min:
            return  # A track that changes person inside one second is not evidence.
        chunks.append({'raw': raw, 'start': run[0][0], 'end': run[-1][0],
                       'frames': [f for f, _ in run], 'clean_frames': [f for f, _ in clean],
                       'keys': [o['key'] for _, o in clean], 'mean': mean, 'coherence': coherence})

    for raw, items in tracks.items():
        run = []
        for f, obs in items:
            if run and (f-run[-1][0] > gap or f-run[0][0] >= limit):
                close(raw, run)
                run = []
            run.append((f, obs))
        close(raw, run)
    chunks.sort(key=lambda c: (c['start'], str(c['raw'])))
    if len(chunks) > parameters.max_chunks:
        # Segments spread over the whole video, not its first minutes.
        keep = np.unique(np.linspace(0, len(chunks)-1, parameters.max_chunks).round().astype(int))
        chunks = [chunks[i] for i in keep]
    return chunks


def cluster_people(chunks: list[dict], parameters: RosterDiscoveryParameters) -> list[list[int]]:
    """Average-linkage merging; co-visible tracklets are different people."""
    count = len(chunks)
    if count == 0:
        return []
    means = np.stack([c['mean'] for c in chunks])
    distance = (1.-means @ means.T).astype(np.float32)
    frame_ids = sorted({f for c in chunks for f in c['clean_frames']})
    column = {f: i for i, f in enumerate(frame_ids)}
    rows = np.concatenate([np.full(len(c['clean_frames']), i) for i, c in enumerate(chunks)])
    columns = np.asarray([column[f] for c in chunks for f in c['clean_frames']])
    seen = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)),
                             shape=(count, len(frame_ids)))
    together = np.asarray((seen @ seen.T).todense()) >= parameters.cannot_link_frames
    raw = np.asarray([str(c['raw']) for c in chunks])
    distance[together & (raw[:, None] != raw[None, :])] = np.inf
    np.fill_diagonal(distance, np.inf)
    size = np.asarray([len(c['keys']) for c in chunks], dtype=np.float64)
    owner = np.arange(count)
    alive = np.ones(count, dtype=bool)
    nearest = distance.argmin(1)
    best = distance[np.arange(count), nearest]
    while True:
        i = int(np.argmin(best))
        if not np.isfinite(best[i]) or best[i] > parameters.merge_distance_max:
            break
        j = int(nearest[i])
        # An infinite (forbidden) pair stays forbidden for the merged person.
        merged = ((size[i]*distance[i]+size[j]*distance[j])/(size[i]+size[j])).astype(np.float32)
        merged[i] = merged[j] = np.inf
        distance[i, :] = merged
        distance[:, i] = merged
        distance[j, :] = np.inf
        distance[:, j] = np.inf
        size[i] += size[j]
        alive[j] = False
        owner[owner == j] = i
        best[j] = np.inf
        nearest[i] = int(np.argmin(merged))
        best[i] = merged[nearest[i]]
        stale = np.flatnonzero(alive & ((nearest == i) | (nearest == j)))
        for row in stale:
            if row != i:
                nearest[row] = int(np.argmin(distance[row]))
                best[row] = distance[row, nearest[row]]
        closer = alive & (merged < best)
        closer[i] = False
        nearest[closer] = i
        best[closer] = merged[closer]
    groups = defaultdict(list)
    for index, root in enumerate(owner):
        groups[int(root)].append(index)
    return list(groups.values())


def _presence(chunks, members):
    return len({f for i in members for f in chunks[i]['frames']})


def merge_fragments(chunks: list[dict], groups: list[list[int]],
                    parameters: RosterDiscoveryParameters) -> list[list[int]]:
    """Join a group to a larger look-alike it is (almost) never cleanly seen together with."""
    means = np.stack([c['mean'] for c in chunks])
    order = sorted(groups, key=lambda members: -sum(len(chunks[i]['clean_frames']) for i in members))
    merged, visible = [], []
    for members in order:
        frames = {f for i in members for f in chunks[i]['clean_frames']}
        best, best_distance = None, parameters.merge_distance_max
        for index, larger in enumerate(merged):
            distance = float(1.-(means[members] @ means[larger].T).mean())
            if (distance <= best_distance
                    and len(frames & visible[index]) <= parameters.fragment_covisible_max*len(frames)):
                best, best_distance = index, distance
        if best is None:
            merged.append(list(members))
            visible.append(frames)
        else:
            merged[best] += members
            visible[best] |= frames
    return merged


def discover_roster(frames: list[dict], features: dict, fps: float, roster_size: int = 6,
                    parameters: RosterDiscoveryParameters = RosterDiscoveryParameters(),
                    anchors: list[dict] | None = None) -> dict:
    """People of one court side, ordered by presence, with multi-segment galleries.

    ``anchors`` (confirmed ``{'key', 'player_id'}`` boxes) choose and name the
    roster people. Without them the ``roster_size`` longest-present people are
    kept and named later, left to right.
    """
    chunks = tracklet_chunks(frames, features, fps, parameters)
    groups = cluster_people(chunks, parameters)
    if not groups:
        raise RosterNotFound('No clean player tracklets')
    groups = sorted(merge_fragments(chunks, groups, parameters), key=lambda m: -_presence(chunks, m))
    presence = [_presence(chunks, m) for m in groups]
    owner = {key: g for g, members in enumerate(groups) for i in members for key in chunks[i]['keys']}

    def gallery(members):
        ordered = sorted(members, key=lambda i: chunks[i]['start'])
        if len(ordered) > parameters.gallery_max_views:
            keep = np.unique(np.linspace(0, len(ordered)-1, parameters.gallery_max_views).round().astype(int))
            ordered = [ordered[i] for i in keep]
        return ordered

    people = []
    if anchors:
        taken = {}
        for anchor in anchors:
            group = owner.get(anchor['key'])
            if group is None:
                vector = _unit(features[anchor['key']])
                scores = [float(np.sort(np.stack([chunks[i]['mean'] for i in gallery(m)]) @ vector)
                                [-parameters.gallery_top_views:].mean()) for m in groups]
                group = int(np.argmax(scores))
                if scores[group] < parameters.similarity_min:
                    raise RosterNotFound(f'Anchor {anchor["player_id"]} matches no discovered person')
            if group in taken:
                raise RosterNotFound(f'Anchors {taken[group]} and {anchor["player_id"]} look like one person')
            taken[group] = anchor['player_id']
            people.append((group, anchor['player_id']))
    else:
        floor = max(parameters.min_presence_ratio*presence[0], parameters.min_presence_seconds*fps)
        people = [(g, None) for g in range(len(groups)) if presence[g] >= floor][:roster_size]
        if len(people) < roster_size:
            raise RosterNotFound(f'Only {len(people)} of {roster_size} constant players found')
    chosen = [g for g, _ in people]
    clean = [{f for i in members for f in chunks[i]['clean_frames']} for members in groups]
    roster = []
    for group, pid in people:
        views = gallery(groups[group])
        roster.append({'player_id': pid, 'presence_frames': presence[group], 'tracklets': len(groups[group]),
                       'first_frame': min(chunks[i]['start'] for i in groups[group]),
                       'last_frame': max(chunks[i]['end'] for i in groups[group]),
                       'gallery': np.stack([chunks[i]['mean'] for i in views]),
                       'gallery_keys': [chunks[i]['keys'][len(chunks[i]['keys'])//2] for i in views],
                       'member_keys': {key for i in groups[group] for key in chunks[i]['keys']}})
    # Seen with every roster person, so certainly somebody else. A leftover
    # group without that proof may be a roster player from an unusual angle.
    known_others = [g for g in range(len(groups)) if g not in chosen
                    and len(clean[g]) >= parameters.other_person_min_seconds*fps
                    and all(len(clean[g] & clean[r]) >= parameters.cannot_link_frames for r in chosen)]
    others = [{'presence_frames': presence[g], 'tracklets': len(groups[g]),
               'first_frame': min(chunks[i]['start'] for i in groups[g]),
               'last_frame': max(chunks[i]['end'] for i in groups[g]),
               'example_key': chunks[groups[g][0]]['keys'][0],
               'role': 'other_person' if g in known_others else 'unresolved'}
              for g in range(len(groups)) if g not in chosen][:10]
    return {'roster': roster, 'other_people': others,
            'cohort': [np.stack([chunks[i]['mean'] for i in gallery(groups[g])]) for g in known_others],
            'tracklets': len(chunks), 'people': len(groups)}


def _track_average(scores, frame, raw, weight, window):
    """Weighted mean of each box's scores over nearby boxes of its detector track."""
    result = scores.copy()
    order = np.lexsort((frame, raw))
    raws, times = raw[order], frame[order]
    weighted = scores[order]*weight[order, None]
    weights = weight[order]
    start = 0
    for stop in range(1, len(order)+1):
        if stop < len(order) and raws[stop] == raws[start]:
            continue
        if raws[start] >= 0 and stop-start > 1:
            t = times[start:stop]
            total = np.vstack([np.zeros((1, scores.shape[1])), np.cumsum(weighted[start:stop], 0)])
            mass = np.r_[0., np.cumsum(weights[start:stop])]
            low, high = np.searchsorted(t, t-window, 'left'), np.searchsorted(t, t+window, 'right')
            result[order[start:stop]] = (total[high]-total[low])/np.maximum(mass[high]-mass[low], 1e-9)[:, None]
        start = stop
    return result


def assign_roster(frames: list[dict], features: dict, fps: float, roster: list[dict],
                  parameters: RosterDiscoveryParameters = RosterDiscoveryParameters(), *,
                  available_from: dict | None = None, forced: dict | None = None,
                  cohort: list | None = None) -> dict:
    """One-to-one roster IDs per frame from appearance; uncertain boxes stay unknown.

    ``cohort`` holds galleries of other people found on this side. A box that
    looks more like one of them than like any roster person stays unknown.
    """
    ids = [person['player_id'] for person in roster]
    if len(set(ids)) != len(ids) or any(not pid for pid in ids):
        raise ValueError('Roster people need distinct names before assignment')
    observations = [(frame['f'], obs) for frame in frames for obs in frame['players']]
    predictions = {obs['key']: {'player_id': None, 'reason': 'unmatched_or_appearance_gate', 'cost': None,
                               'label_source': 'model', 'verified': False} for _, obs in observations}
    usable = [(f, obs) for f, obs in observations if features.get(obs['key']) is not None]
    if not usable:
        return predictions
    frame = np.asarray([f for f, _ in usable])
    track_ids = {}
    raw = np.asarray([-1 if obs.get('raw_id') is None else track_ids.setdefault(obs['raw_id'], len(track_ids))
                      for _, obs in usable])
    weight = np.asarray([1. if obs['occlusion'] < parameters.clean_overlap_max else parameters.occluded_weight
                         for _, obs in usable])
    galleries = [person['gallery'] for person in roster]+list(cohort or [])
    own = np.empty((len(usable), len(galleries)), dtype=np.float32)
    top = parameters.gallery_top_views
    for start in range(0, len(usable), 16384):
        block = np.stack([_unit(features[obs['key']]) for _, obs in usable[start:start+16384]])
        for j, views in enumerate(galleries):
            similarity = block @ views.T
            k = min(top, similarity.shape[1])
            own[start:start+len(block), j] = np.partition(similarity, -k, axis=1)[:, -k:].mean(1)
    scores = _track_average(own.astype(np.float64), frame, raw, weight,
                            max(1, round(parameters.smooth_seconds*fps)))
    stranger = np.zeros(len(usable), dtype=bool)
    if len(galleries) > len(roster):
        stranger = scores[:, len(roster):].max(1) > scores[:, :len(roster)].max(1)
        scores = scores[:, :len(roster)]
        scores[stranger] = -np.inf
    for j, pid in enumerate(ids):
        first = (available_from or {}).get(pid)
        if first is not None:
            scores[frame < first, j] = -np.inf
    forced = forced or {}
    row = 0
    while row < len(usable):
        stop = row
        while stop < len(usable) and frame[stop] == frame[row]:
            stop += 1
        block = scores[row:stop].copy()
        for i in range(row, stop):
            pid = forced.get(usable[i][1]['key'])
            if pid is not None:
                j = ids.index(pid)
                block[:, j] = -np.inf
                block[i-row, :] = -np.inf
                block[i-row, j] = 2.
        finite = np.where(np.isfinite(block), block, -10.)
        augmented = np.concatenate([-finite, np.full((stop-row, stop-row), -parameters.similarity_min)], axis=1)
        chosen_rows, chosen_columns = linear_sum_assignment(augmented)
        for i, j in zip(chosen_rows, chosen_columns):
            obs = usable[row+i][1]
            entry = predictions[obs['key']]
            if j >= len(ids):
                if stranger[row+i]:
                    entry['reason'] = 'other_person'
                continue
            value = float(finite[i, j])
            entry['cost'] = float(1.-min(value, 1.))
            entry['appearance_distance'] = float(np.clip(1.-own[row+i, j], 0., 1.))
            if forced.get(obs['key']) == ids[j]:
                entry.update(player_id=ids[j], reason='roster_anchor', cost=0.)
            elif value-np.max(np.delete(finite[i], j), initial=-10.) < parameters.assignment_margin:
                entry['reason'] = 'ambiguous'
            else:
                entry.update(player_id=ids[j], reason='associated' if weight[row+i] == 1.
                             else 'occluded_appearance_only')
        row = stop
    return predictions


def name_left_to_right(frames: list[dict], roster: list[dict]) -> dict:
    """Name automatic roster people P1..Pn by image x on one clean reference frame.

    Returns the reference seed: the earliest frame that cleanly shows every
    roster person, else each person's own first clean box.
    """
    owner = {key: index for index, person in enumerate(roster) for key in person['member_keys']}
    reference = None
    first = {}
    for frame in frames:
        seen = {}
        for obs in frame['players']:
            index = owner.get(obs['key'])
            if index is not None:
                seen[index] = obs
                first.setdefault(index, (frame['f'], obs))
        if len(seen) == len(roster):
            reference = (frame['f'], seen)
            break
    if reference is None:
        if len(first) != len(roster):
            raise RosterNotFound('A roster person has no clean reference box')
        order = sorted(first, key=lambda index: (first[index][1]['bb'][0]+first[index][1]['bb'][2]))
        assignments = {first[index][1]['key']: f'P{rank+1}' for rank, index in enumerate(order)}
        frame_number = min(f for f, _ in first.values())
        complete = False
    else:
        frame_number, seen = reference
        order = sorted(seen, key=lambda index: seen[index]['bb'][0]+seen[index]['bb'][2])
        assignments = {seen[index]['key']: f'P{rank+1}' for rank, index in enumerate(order)}
        complete = True
    for rank, index in enumerate(order):
        roster[index]['player_id'] = f'P{rank+1}'
    roster.sort(key=lambda person: int(person['player_id'][1:]))
    return {'frame': frame_number, 'assignments': dict(sorted(assignments.items(), key=lambda item: item[1])),
            'label_source': 'automatic_initialisation', 'verified': False,
            'reference': 'one_frame_with_whole_roster' if complete else 'first_clean_box_per_person'}
