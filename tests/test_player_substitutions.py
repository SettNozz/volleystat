"""Substitutions: a lineup change between rallies gives the substitute a new roster ID."""
from copy import deepcopy

import numpy as np

from src.tracking.contact_author_runtime import photo_roster
from src.tracking.player_identity_workflow import _write_predictions
from src.tracking.player_roster_discovery import (
    RosterDiscoveryParameters, assign_roster, discover_roster, name_left_to_right)
from src.tracking.player_substitutions import (
    SubstitutionParameters, active_intervals, apply_substitutions, pair_lineup_changes, presence_stints,
    rally_spans)

FPS = 5.
PARAMETERS = RosterDiscoveryParameters()
SUBS = SubstitutionParameters()


def looks(count, seed=0, dim=48):
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return basis[:count]


def on_court(person, f):
    """Person 0 is replaced by person 6 at ~100 s and returns at ~200 s."""
    if person == 0:
        return f < 500 or f >= 1000
    if person == 6:
        return 495 <= f < 1005
    return True


def scene(length=1500, seed=3):
    rng = np.random.default_rng(seed)
    base = looks(7)
    frames, features, truth, annotations = [], {}, {}, []
    for f in range(length):
        players, poses = [], []
        for person in range(7):
            if not on_court(person, f):
                continue
            slot = 0 if person == 6 else person
            x = 100+slot*300+(40 if person == 6 else 0)
            key = f'{f}:{len(players)}'
            box = [x, 200, x+100, 500]
            players.append({'key': key, 'pose_index': len(players), 'bb': box,
                            'raw_id': person*1000+f//100, 'occlusion': 0.})
            poses.append({'id': person*1000+f//100, 'team': 'b', 'bb': box})
            features[key] = base[person]+rng.normal(scale=.03, size=base.shape[1])
            truth[key] = person
        frames.append({'f': f, 'players': players})
        annotations.append({'f': f, 'poses': poses})
    return frames, features, truth, annotations


def identity(frames, features, annotations):
    found = discover_roster(frames, features, FPS, 6, PARAMETERS)
    roster = found['roster']
    seed = name_left_to_right(frames, roster)
    predictions = assign_roster(frames, features, FPS, roster, PARAMETERS, cohort=found['cohort'])
    counts, by_person = _write_predictions(annotations, predictions, 'b', 'test_method',
                                           {p['key'] for f in frames for p in f['players']})
    report = {'status': 'completed', 'method': 'test_method', 'side': 'b', 'seed': seed,
              'gallery_keys': {p['player_id']: p['gallery_keys'] for p in roster},
              'roster_discovery': {'players': [{'player_id': p['player_id'], 'presence_frames': 0, 'tracklets': 0,
                                                'first_frame': 0, 'last_frame': 0} for p in roster]},
              'observations': counts['observations'], 'assigned': counts['assigned']}
    context = {'frames': frames, 'features': features, 'found': found, 'roster': roster, 'forced': {},
               'available': {}, 'parameters': PARAMETERS, 'predictions': predictions, 'side': 'b'}
    return report, context


def test_presence_stints_merge_short_absences_and_drop_scattered_presence():
    frames = range(0, 1000)
    assigned = {
        # 60 s, a 10 s occlusion inside, then gone.
        'A': [f for f in range(0, 300) if not 100 <= f < 150],
        # Seen for one second every 30 s: an opponent at the net, not a player.
        'B': [f for f in range(0, 1000) if f % 150 < 5],
    }
    stints = presence_stints(assigned, frames, FPS, SUBS)
    assert stints['A'] == [(0, 299)]
    assert stints['B'] == []


def test_exit_and_entry_pair_only_between_rallies():
    stints = {'P1': [(0, 500), (1000, 1499)], 'S': [(495, 1004)], 'P2': [(0, 1499)]}
    events, rejected = pair_lineup_changes(stints, rally_spans([{'start_frame': 550, 'end_frame': 900}]), FPS, SUBS)
    assert [(e['player_out'], e['player_in']) for e in events] == [('P1', 'S'), ('S', 'P1')]
    assert events[0]['window'] == [495, 500] and events[0]['rally_check'] == 'between_rallies'
    assert events[0]['next_rally_start'] == 550 and events[1]['previous_rally_end'] == 900
    # Play does not stop for a substitution: a crossing inside a rally is not one.
    events, rejected = pair_lineup_changes(stints, [(400, 600)], FPS, SUBS)
    assert [(e['player_out'], e['player_in']) for e in events] == [('S', 'P1')]
    assert rejected[0]['reason'] == 'during_rally' and rejected[0]['player_in'] == 'S'


def test_active_intervals_follow_each_players_own_substitutions():
    events = [{'player_out': 'P1', 'player_in': 'P7', 'window': [495, 500]},
              {'player_out': 'P7', 'player_in': 'P1', 'window': [1000, 1004]}]
    assert active_intervals(events, 0, 1499, 10) == {'P1': [[0, 510], [990, 1499]], 'P7': [[485, 1014]]}


def test_substitute_gets_the_next_id_and_statistics_identity():
    frames, features, truth, annotations = scene()
    report, context = identity(frames, features, annotations)
    before = deepcopy(annotations)
    sub_keys = [key for key, person in truth.items() if person == 6]
    by_key = {f'{a["f"]}:{i}': p for a in annotations for i, p in enumerate(a['poses'])}
    assert all(by_key[key]['player_id'] is None for key in sub_keys)
    rallies = [{'start_frame': 100, 'end_frame': 400}, {'start_frame': 550, 'end_frame': 900},
               {'start_frame': 1100, 'end_frame': 1400}]
    apply_substitutions(annotations, report, context, rallies, FPS, SUBS)
    block = report['substitutions']
    assert block['status'] == 'completed' and block['verified'] is False
    starter = next(by_key[key]['player_id'] for key, person in truth.items() if person == 0)
    assert [(e['player_out'], e['player_in']) for e in block['events']] == [(starter, 'P7'), ('P7', starter)]
    assert all(e['rally_check'] == 'between_rallies' for e in block['events'])
    assert 400 < block['events'][0]['frame'] < 550 and 900 < block['events'][1]['frame'] < 1100
    assert block['players'][0]['player_id'] == 'P7' and 'P7' in report['gallery_keys']
    assert report['roster_size'] == 7 and report['person_observations']['P7'] > 400
    assert report['roster_discovery']['players'][-1]['role'] == 'substitute'
    ids = {by_key[key]['player_id'] for key in sub_keys}
    assert ids == {'P7'}
    # Raw fields are untouched; only identities change.
    for old, new in zip(before, annotations):
        for a, b in zip(old['poses'], new['poses']):
            assert {k: v for k, v in a.items() if k not in ('player_id', 'identity')} == \
                   {k: v for k, v in b.items() if k not in ('player_id', 'identity')}
    for ann in annotations:
        assigned = [p['player_id'] for p in ann['poses'] if p['player_id']]
        assert len(assigned) == len(set(assigned))


def test_without_lineup_change_nothing_is_rewritten():
    frames, features, truth, annotations = scene()
    # Person 6 stays by the bench in one short visit; nobody leaves the court.
    keep = {key for key, person in truth.items() if person != 6 or int(key.split(':')[0]) < 520}
    for frame in frames:
        frame['players'] = [p for p in frame['players'] if p['key'] in keep]
    report, context = identity(frames, features, annotations)
    snapshot = deepcopy(annotations)
    apply_substitutions(annotations, report, context, [], FPS, SUBS)
    assert report['substitutions']['status'] in ('none_found', 'no_candidates')
    assert annotations == snapshot and 'P7' not in report['gallery_keys']


def test_contact_photo_roster_includes_substitutes():
    poses = [{'bb': [10, 10, 50, 90], 'team': 'b'}, {'bb': [60, 10, 100, 90], 'team': 'b'}]
    annotations = {'frames': [{'f': 3, 'poses': poses}], 'player_identity': {
        'side': 'b', 'seed': {'assignments': {'3:0': 'P1'}}, 'gallery_keys': {'P1': ['3:0'], 'P7': ['3:1']},
        'substitutions': {'players': [{'player_id': 'P7', 'reference_key': '3:1'}]}}}
    roster = photo_roster(annotations)
    assert [p['player_id'] for p in roster['players']] == ['P1', 'P7']
    assert roster['players'][1]['references'] == [{'frame': 3, 'bbox': [60, 10, 100, 90]}]


def test_player_lineup_lists_court_time_and_changes_for_statistics():
    from src.tracking.player_substitutions import player_lineup
    stats = {'fps': 10, 'total_frames': 1000, 'player_identity': {
        'gallery_keys': {'P1': [], 'P2': [], 'P7': []},
        'substitutions': {'status': 'completed', 'players': [{'player_id': 'P7'}],
                          'on_court': {'P1': [[0, 299], [600, 999]], 'P7': [[300, 599]]},
                          'events': [{'player_in': 'P7', 'player_out': 'P1', 'frame': 300, 'window': [295, 305]},
                                     {'player_in': 'P1', 'player_out': 'P7', 'frame': 600, 'window': [595, 605]}]}}}
    lineup = player_lineup(stats)
    assert lineup['P7']['role'] == 'substitute' and lineup['P7']['on_court_seconds'] == 30
    assert lineup['P2'] == {'role': 'starter', 'intervals': [[0, 999]], 'on_court_seconds': 100, 'changes': []}
    assert [(c['direction'], c['other']) for c in lineup['P1']['changes']] == [('out', 'P7'), ('in', 'P7')]
    assert player_lineup({'player_identity': {}}) == {}
