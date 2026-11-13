"""Find player substitutions on the roster side and give substitutes their own IDs.

A substitution changes who is on court: one player walks off, one walks on,
and the two usually cross at the sideline. It happens only while the ball is
dead, between one rally and the next serve. This module reads that from the
video itself:

1. Every saved box is scored against the roster people *and* the possible
   substitutes found by roster discovery (non-roster people present long
   enough to have played), one-to-one per frame as usual.
2. Each person's presence becomes on-court stints: runs of seconds with an
   assigned box, where short absences (occlusion, missed detections) do not
   break a stint.
3. A stint end (exit) and a stint start of another person (entry) close in
   time form one substitution, paired one-to-one by time. A pair whose whole
   crossing window lies inside one rally is rejected: play cannot pause for a
   substitution.
4. A substitute who takes part in a substitution gets the next free roster
   ID (P7, P8, ...). Everyone involved is limited to the frames between their
   own substitutions, so a player on the bench cannot lend their ID to the
   person who replaced them. Everybody else is assigned exactly as before.

A substitute who never pairs with an exit stays unknown; nothing is invented
for the frames nobody can see. Everything here is a model proposal
(``label_source: model``, ``verified: false``).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.player_roster_discovery import assign_roster

METHOD = 'presence_handoff_v1'


@dataclass(frozen=True)
class SubstitutionParameters:
    # A second counts as present when an assigned box is in this share of its frames.
    bin_seconds: float = 1.
    bin_presence_min: float = .25
    # Absences up to this long belong to one stint (occlusion, a missed detection,
    # a player chasing the ball off court).
    gap_merge_seconds: float = 20.
    # A stint must be this long and this dense; an opponent at the net or a
    # passer-by is seen in short scattered pieces.
    min_stint_seconds: float = 15.
    stint_density_min: float = .5
    # An exit and an entry at most this far apart form one substitution.
    pair_window_seconds: float = 15.
    # Rally boundaries are known to about a second.
    rally_tolerance_seconds: float = 1.
    # Both players are visible while they cross; each keeps their ID this long
    # around their own exit/entry.
    interval_padding_seconds: float = 3.

    def to_dict(self):
        return asdict(self)


def rally_spans(records) -> list[tuple[int, int]]:
    """(start, end) frames of rallies or service attempts that carry both."""
    spans = []
    for record in records or ():
        start, end = record.get('start_frame'), record.get('end_frame')
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            spans.append((int(start), int(end)))
    return sorted(spans)


def presence_stints(assigned: dict, frame_numbers, fps: float,
                    parameters: SubstitutionParameters = SubstitutionParameters()) -> dict:
    """``player_id -> [(first_frame, last_frame), ...]`` on-court stints.

    ``assigned`` maps a person to the frames that hold one of their boxes;
    ``frame_numbers`` are all processed frames (a skipped frame is not absence).
    """
    size = max(1, round(parameters.bin_seconds*fps))
    processed = np.asarray(sorted(frame_numbers), dtype=np.int64)
    if not len(processed):
        return {}
    bins = int(processed[-1]//size)+1
    available = np.bincount(processed//size, minlength=bins)
    merge = max(1, round(parameters.gap_merge_seconds/parameters.bin_seconds))
    stints = {}
    for pid, frames in assigned.items():
        frames = np.asarray(sorted(frames), dtype=np.int64)
        if not len(frames):
            stints[pid] = []
            continue
        counts = np.bincount(frames//size, minlength=bins)
        present = np.flatnonzero((available > 0) & (counts >= np.maximum(1., parameters.bin_presence_min*available)))
        runs = []
        for index in present:
            if runs and index-runs[-1][1] <= merge:
                runs[-1][1] = index
                runs[-1][2] += 1
            else:
                runs.append([index, index, 1])
        kept = []
        for low, high, count in runs:
            length = high-low+1
            if length*size < parameters.min_stint_seconds*fps or count/length < parameters.stint_density_min:
                continue
            inside = frames[(frames >= low*size) & (frames < (high+1)*size)]
            kept.append((int(inside[0]), int(inside[-1])))
        stints[pid] = kept
    return stints


def _inside_rally(low, high, spans, tolerance):
    """The rally that contains the whole window, if any."""
    for start, end in spans:
        if start+tolerance <= low and high <= end-tolerance:
            return start, end
    return None


def pair_lineup_changes(stints: dict, spans, fps: float,
                        parameters: SubstitutionParameters = SubstitutionParameters()):
    """Pair exits with entries of other people; returns (events, rejected)."""
    exits = [(pid, end) for pid, runs in stints.items() for _, end in runs]
    entries = [(pid, start) for pid, runs in stints.items() for start, _ in runs]
    window = parameters.pair_window_seconds*fps
    tolerance = round(parameters.rally_tolerance_seconds*fps)
    rejected = []
    if not exits or not entries:
        return [], rejected
    invalid = 1e9
    cost = np.full((len(exits), len(entries)), invalid)
    for i, (out_pid, exit_frame) in enumerate(exits):
        for j, (in_pid, entry_frame) in enumerate(entries):
            if out_pid == in_pid or abs(entry_frame-exit_frame) > window:
                continue
            low, high = min(exit_frame, entry_frame), max(exit_frame, entry_frame)
            rally = _inside_rally(low, high, spans, tolerance) if spans else None
            if rally is not None:
                rejected.append({'player_in': in_pid, 'player_out': out_pid, 'window': [low, high],
                                 'reason': 'during_rally', 'rally': list(rally)})
                continue
            cost[i, j] = abs(entry_frame-exit_frame)
    rows, columns = linear_sum_assignment(cost)
    events = []
    for i, j in zip(rows, columns):
        if cost[i, j] >= invalid:
            continue
        (out_pid, exit_frame), (in_pid, entry_frame) = exits[i], entries[j]
        low, high = min(exit_frame, entry_frame), max(exit_frame, entry_frame)
        dead = [f for f in range(low, high+1)
                if not any(start+tolerance <= f <= end-tolerance for start, end in spans)]
        frame = dead[len(dead)//2] if dead else (low+high)//2
        before = [end for start, end in spans if end <= frame+tolerance]
        after = [start for start, end in spans if start >= frame-tolerance]
        events.append({'player_in': in_pid, 'player_out': out_pid, 'frame': int(frame),
                       'time': round(frame/fps, 3), 'window': [int(low), int(high)],
                       'exit_frame': int(exit_frame), 'entry_frame': int(entry_frame),
                       'rally_check': 'between_rallies' if spans else 'no_rallies',
                       'previous_rally_end': max(before) if before else None,
                       'next_rally_start': min(after) if after else None,
                       'label_source': 'model', 'verified': False})
    events.sort(key=lambda event: event['frame'])
    return _consistent(events, rejected), rejected


def _consistent(events, rejected):
    """Each person alternates between leaving and entering the court."""
    state = {}
    kept = []
    for event in events:
        out_pid, in_pid = event['player_out'], event['player_in']
        if state.get(out_pid) == 'off' or state.get(in_pid) == 'on':
            rejected.append({'player_in': in_pid, 'player_out': out_pid, 'window': event['window'],
                             'reason': 'inconsistent_lineup'})
            continue
        state[out_pid], state[in_pid] = 'off', 'on'
        kept.append(event)
    return kept


def active_intervals(events, first_frame: int, last_frame: int, padding: int) -> dict:
    """On-court frame ranges of everyone who takes part in a substitution."""
    timeline = {}
    for event in events:
        low, high = event['window']
        timeline.setdefault(event['player_out'], []).append((low, high, 'out'))
        timeline.setdefault(event['player_in'], []).append((low, high, 'in'))
    intervals = {}
    for pid, changes in timeline.items():
        changes.sort()
        start = first_frame if changes[0][2] == 'out' else None
        runs = []
        for low, high, kind in changes:
            if kind == 'out' and start is not None:
                runs.append((start, high+padding))
                start = None
            elif kind == 'in' and start is None:
                start = max(first_frame, low-padding)
        if start is not None:
            runs.append((start, last_frame))
        intervals[pid] = [[int(a), int(b)] for a, b in runs]
    return intervals


def _next_number(ids):
    numbers = [int(match.group(1)) for pid in ids if (match := re.fullmatch(r'P(\d+)', str(pid)))]
    return max(numbers, default=len(ids))+1


def _assigned_frames(predictions, ids):
    frames = {pid: [] for pid in ids}
    for key, prediction in predictions.items():
        pid = prediction['player_id']
        if pid in frames:
            frames[pid].append(int(key.split(':', 1)[0]))
    return frames


def detect_substitutions(context: dict, spans, fps: float,
                         parameters: SubstitutionParameters = SubstitutionParameters()) -> dict:
    """Substitution events and the extended roster; nothing is written."""
    frames, features = context['frames'], context['features']
    found, roster = context['found'], context['roster']
    discovery = context['parameters']
    candidates = found.get('candidates') or []
    result = {'events': [], 'rejected': [], 'roster': roster, 'intervals': {}, 'substitutes': [],
              'candidates': len(candidates)}
    if not candidates:
        return result
    taken = {person['player_id'] for person in roster}
    temporary = []
    for index in range(len(candidates)):
        name = f'candidate_{index+1}'
        while name in taken:
            name += '_'
        temporary.append(name)
    people = list(roster)+[{'player_id': name, 'gallery': candidate['gallery']}
                           for name, candidate in zip(temporary, candidates)]
    excluded = {c['cohort_index'] for c in candidates if c.get('cohort_index') is not None}
    cohort = [g for i, g in enumerate(found.get('cohort') or []) if i not in excluded]
    provisional = assign_roster(frames, features, fps, people, discovery,
                                available_from=context.get('available'), forced=context.get('forced'),
                                cohort=cohort)
    ids = [person['player_id'] for person in people]
    frame_numbers = [frame['f'] for frame in frames]
    stints = presence_stints(_assigned_frames(provisional, ids), frame_numbers, fps, parameters)
    events, rejected = pair_lineup_changes(stints, spans, fps, parameters)
    result['rejected'] = rejected
    result['stints'] = stints
    if not events:
        return result
    names = {}
    number = _next_number(taken)
    for event in events:
        for key in ('player_out', 'player_in'):
            pid = event[key]
            if pid in temporary and pid not in names:
                names[pid] = f'P{number}'
                number += 1
    for event in events:
        event['player_in'] = names.get(event['player_in'], event['player_in'])
        event['player_out'] = names.get(event['player_out'], event['player_out'])
    for item in rejected:
        item['player_in'] = names.get(item['player_in'], item['player_in'])
        item['player_out'] = names.get(item['player_out'], item['player_out'])
    substitutes = []
    for name, candidate in zip(temporary, candidates):
        if name not in names:
            continue
        person = dict(candidate, player_id=names[name])
        substitutes.append(person)
    result['stints'] = {names.get(pid, pid): runs for pid, runs in stints.items()
                        if pid not in temporary or pid in names}
    used = {c['cohort_index'] for c in substitutes if c.get('cohort_index') is not None}
    result.update(events=events, substitutes=substitutes, roster=list(roster)+substitutes,
                  cohort=[g for i, g in enumerate(found.get('cohort') or []) if i not in used],
                  intervals=active_intervals(events, min(frame_numbers), max(frame_numbers),
                                             round(parameters.interval_padding_seconds*fps)))
    for index, event in enumerate(events, 1):
        event['index'] = index
    return result


def _inside(frame, intervals):
    return any(low <= frame <= high for low, high in intervals)


def apply_substitutions(annotations: list[dict], report: dict, context: dict, rallies, fps: float,
                        parameters: SubstitutionParameters = SubstitutionParameters()) -> dict:
    """Detect substitutions and rewrite roster IDs on the saved poses in place.

    ``context`` is what the whole-video roster pass kept (frames, features,
    discovered people, roster, its predictions). Only poses whose ID changes
    are touched; IDs recovered later by team membership are kept unless they
    name a player who is off court at that frame.
    """
    spans = rally_spans(rallies)
    found = detect_substitutions(context, spans, fps, parameters)
    block = {'method': METHOD, 'parameters': parameters.to_dict(), 'rally_spans': len(spans),
             'candidates': found['candidates'], 'events': found['events'], 'rejected': found['rejected'][:50],
             'label_source': 'model', 'verified': False}
    if not found['events']:
        block['status'] = 'none_found' if found['candidates'] else 'no_candidates'
        report['substitutions'] = block
        return report
    method = report.get('method')
    old = context['predictions']
    predictions = assign_roster(context['frames'], context['features'], fps, found['roster'], context['parameters'],
                                available_from=context.get('available'), forced=context.get('forced'),
                                cohort=found['cohort'], active_intervals=found['intervals'])
    by_key = {f'{ann["f"]}:{index}': (ann, pose) for ann in annotations
              for index, pose in enumerate(ann.get('poses') or [])}
    changed = cleared = 0
    for key, new in predictions.items():
        previous = old.get(key)
        if previous is None or key not in by_key:
            continue
        _, pose = by_key[key]
        if pose.get('player_id') != previous['player_id']:
            continue  # set later by team membership; that pass owns it
        if new['player_id'] == previous['player_id'] and new['reason'] == previous['reason']:
            continue
        pose['player_id'] = new['player_id']
        pose['identity'] = {'method': method, **{k: v for k, v in new.items() if k != 'player_id'}}
        changed += 1
    intervals = found['intervals']
    for key, (ann, pose) in by_key.items():
        pid = pose.get('player_id')
        if pid in intervals and not _inside(ann['f'], intervals[pid]):
            pose['player_id'] = None
            pose['identity'] = {**(pose.get('identity') or {}), 'reason': 'off_court_after_substitution',
                                'label_source': 'model', 'verified': False}
            cleared += 1
    # A recovered ID can now collide with a fresh assignment in the same frame.
    for ann in annotations:
        seen = {}
        for index, pose in enumerate(ann.get('poses') or []):
            pid = pose.get('player_id')
            if pid is None:
                continue
            key = f'{ann["f"]}:{index}'
            fresh = predictions.get(key, {}).get('player_id') == pid
            if pid in seen:
                other_index, other_fresh = seen[pid]
                loser = index if other_fresh or not fresh else other_index
                target = ann['poses'][loser]
                target['player_id'] = None
                target['identity'] = {**(target.get('identity') or {}), 'reason': 'duplicate_after_substitution'}
                cleared += 1
                if loser == other_index:
                    seen[pid] = (index, fresh)
            else:
                seen[pid] = (index, fresh)
    counts, observed = {}, 0
    for key in predictions:
        if key not in by_key:
            continue
        observed += 1
        pid = by_key[key][1].get('player_id')
        if pid is not None:
            counts[pid] = counts.get(pid, 0)+1
    assigned = sum(counts.values())
    gallery = dict(report.get('gallery_keys') or {})
    players = []
    for person in found['substitutes']:
        gallery[person['player_id']] = person['gallery_keys']
        players.append({'player_id': person['player_id'], 'reference_key': person['gallery_keys'][len(person['gallery_keys'])//2],
                        'presence_frames': person['presence_frames'], 'tracklets': person['tracklets'],
                        'first_frame': person['first_frame'], 'last_frame': person['last_frame'],
                        'source': 'automatic_substitute', 'label_source': 'model', 'verified': False})
    discovery = report.get('roster_discovery')
    if isinstance(discovery, dict) and isinstance(discovery.get('players'), list):
        discovery['players'] = discovery['players']+[
            {k: p[k] for k in ('player_id', 'presence_frames', 'tracklets', 'first_frame', 'last_frame')}
            | {'role': 'substitute'} for p in players]
    report.update(gallery_keys=gallery, roster_size=len(found['roster']), assigned=assigned,
                  observations=observed or report.get('observations'),
                  coverage=assigned/max(observed, 1), person_observations=counts,
                  scope='one court side; roster people are found from tracklets of the whole video; '
                        'substitutes get new IDs from detected lineup changes between rallies')
    block.update(status='completed', players=players, on_court=intervals,
                 stints={pid: [list(run) for run in runs] for pid, runs in found.get('stints', {}).items()},
                 changed_observations=changed, cleared_observations=cleared)
    report['substitutions'] = block
    return report


def player_lineup(stats: dict) -> dict:
    """Per-player court time and substitutions for the statistics pages.

    Players who take part in no substitution are on court for the whole
    analysis. Returns ``{}`` for results without substitution data.
    """
    identity = stats.get('player_identity') or {}
    block = identity.get('substitutions') or {}
    events = block.get('events') or []
    if block.get('status') != 'completed' or not events:
        return {}
    fps = float(stats.get('fps') or 30.)
    last = max(int(stats.get('total_frames') or 0)-1, max(e['window'][1] for e in events))
    substitutes = {p['player_id'] for p in block.get('players') or []}
    on_court = block.get('on_court') or {}
    lineup = {}
    for pid in identity.get('gallery_keys') or {}:
        intervals = [[int(a), min(int(b), last)] for a, b in on_court.get(pid, [[0, last]])]
        changes = []
        for event in events:
            if pid in (event['player_in'], event['player_out']):
                entered = event['player_in'] == pid
                changes.append({'frame': event['frame'], 'time': event['frame']/fps,
                                'direction': 'in' if entered else 'out',
                                'other': event['player_out'] if entered else event['player_in']})
        lineup[pid] = {'role': 'substitute' if pid in substitutes else 'starter', 'intervals': intervals,
                       'on_court_seconds': sum(b-a+1 for a, b in intervals)/fps, 'changes': changes}
    return lineup
