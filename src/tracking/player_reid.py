"""Frozen seed galleries and explicit lost-roster recovery for offline pilots.

No human check labels enter this module. Galleries use only initial/confirmed
roster anchors and nearby clean detections associated by local raw track IDs.
Neither appearance gallery nor roster meaning is updated by predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.player_identity import centre
from src.tracking.player_motion import PlayerMotionKalman
from src.tracking.player_identity_consistency import reject_tracklet_conflicts


@dataclass(frozen=True)
class ReIDParameters:
    # Old experiment plans without this field retain their original baseline.
    motion_model: str = 'ema'
    local_gap_seconds: float = 3.
    appearance_max: float = .55
    recovery_appearance_max: float = .35
    assignment_margin: float = .04
    recovery_appearance_margin: float = .06
    appearance_weight: float = .75
    unknown_cost: float = .85
    recovery_confirmations: int = 3
    gallery_half_window_frames: int = 15
    gallery_max_views: int = 8
    gallery_overlap_max: float = .3
    kalman_chi2_gate: float = 9.21
    motion_update_overlap_max: float = .6
    tracklet_consistency: bool = False
    tracklet_dominance: float = .8
    tracklet_min_clean_votes: int = 3
    local_appearance_regret_max: float | None = None

    def to_dict(self):
        return asdict(self)


def unit(vector):
    value = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(value)
    if not np.isfinite(value).all() or norm <= 1e-8:
        raise ValueError('Missing/non-finite/zero appearance embedding')
    return value/norm


def identity_anchors(seed):
    anchors = [{'key':key,'player_id':pid,'frame':seed['frame'],'available_from_frame':None}
               for key,pid in seed['assignments'].items()]
    anchors += list(seed.get('introductions') or [])
    ids = [a['player_id'] for a in anchors]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Roster anchors must introduce distinct nonempty IDs')
    if len({a['key'] for a in anchors}) != len(anchors):
        raise ValueError('A roster anchor observation cannot name two people')
    for anchor in anchors:
        start = anchor.get('available_from_frame')
        if start is not None and (start < 0 or start > anchor['frame']):
            raise ValueError('Roster availability must precede its observed anchor')
    return anchors


def gallery_keys(frames: list[dict], seed: dict, parameters: ReIDParameters) -> dict[str, list[str]]:
    """Short initial context only; no use of evaluation checks or predictions."""
    by_key = {p['key']: p for f in frames for p in f['players']}
    galleries = {}
    for anchor in identity_anchors(seed):
        key, pid = anchor['key'], anchor['player_id']
        reference = by_key[key]
        candidates = []
        for frame in frames:
            if anchor.get('available_from_frame') is not None and frame['f'] < anchor['available_from_frame']:
                continue
            if abs(frame['f']-anchor['frame']) > parameters.gallery_half_window_frames:
                continue
            for obs in frame['players']:
                if obs['key'] == key or reference['raw_id'] is None or obs['raw_id'] != reference['raw_id']:
                    continue
                height = max(1., reference['bb'][3]-reference['bb'][1])
                if obs['occlusion'] >= parameters.gallery_overlap_max:
                    continue
                if np.linalg.norm(centre(obs)-centre(reference)) > .75*height:
                    continue
                candidates.append((abs(frame['f']-anchor['frame']), frame['f'], obs['key']))
        galleries[pid] = [key] + [p[2] for p in sorted(candidates)[:parameters.gallery_max_views-1]]
    return galleries


def associate(frames: list[dict], seed: dict, features: dict, fps: float,
              recover: bool, parameters: ReIDParameters = ReIDParameters()) -> dict:
    """One-to-one association with immutable appearance and reversible lost state.

    Lost recovery needs distinct appearance and three spatially consistent
    observations. Offline confirmation backfills those observations. A rejected
    detection never changes a gallery, and missing detections are not invented.
    """
    if fps <= 0 or parameters.recovery_confirmations < 1:
        raise ValueError('Invalid FPS/recovery confirmation count')
    if parameters.motion_model not in ('ema', 'kalman'):
        raise ValueError('Unknown identity motion model')
    by_key = {p['key']: p for f in frames for p in f['players']}
    anchors = identity_anchors(seed)
    available = {a['player_id']:a.get('available_from_frame') for a in anchors}
    introduced = {a['frame']:{b['key']:b['player_id'] for b in anchors
                             if b['frame']==a['frame']} for a in anchors}
    keys = gallery_keys(frames, seed, parameters)
    roster = list(keys)
    vectors = {key: unit(value) for key, value in features.items() if value is not None}
    galleries = {pid: np.stack([vectors[key] for key in chosen if key in vectors])
                 for pid, chosen in keys.items()}
    predictions = {p['key']: {'player_id': None, 'reason': 'not_processed', 'cost': None,
                             'label_source': 'model', 'verified': False}
                   for f in frames for p in f['players']}

    def record(obs, pid, reason, cost=None, appearance_distance=None):
        predictions[obs['key']] = {'player_id': pid, 'reason': reason, 'cost': cost,
                                   'label_source': 'model', 'verified': False}
        if appearance_distance is not None and parameters.motion_model=='kalman':
            predictions[obs['key']]['appearance_distance']=float(appearance_distance)

    for anchor in anchors:
        key, pid = anchor['key'], anchor['player_id']
        if key not in vectors:
            raise ValueError('Seed embedding missing')
        if anchor['frame'] == seed['frame']:
            record(by_key[key], pid, 'seed',appearance_distance=0.)
    for direction in (1, -1):
        states = {a['player_id']: {'obs':by_key[a['key']], 'f':a['frame'], 'velocity':np.zeros(2)}
                  for a in anchors}
        filters = {pid: PlayerMotionKalman(centre(state['obs']),
                    state['obs']['bb'][3]-state['obs']['bb'][1], state['f']/fps)
                   for pid, state in states.items()}
        pending = {}
        sequence = sorted((f for f in frames if (f['f']-seed['frame'])*direction > 0),
                          key=lambda f: f['f'], reverse=direction < 0)
        for frame in sequence:
            observations = frame['players']
            if not observations:
                pending.clear()
                continue
            costs = np.full((len(observations), len(roster)), 100.)
            app = np.full_like(costs, 1.)
            recovery = np.zeros_like(costs, dtype=bool)
            active = [j for j,pid in enumerate(roster)
                      if available[pid] is None or frame['f'] >= available[pid]]
            forced = introduced.get(frame['f'], {})
            for i, obs in enumerate(observations):
                if obs['key'] not in vectors:
                    continue
                for j, pid in enumerate(roster):
                    app[i, j] = float(np.clip(1.-np.max(galleries[pid] @ vectors[obs['key']]), 0., 1.))
                for j, pid in enumerate(roster):
                    if j not in active:
                        continue
                    if obs['key'] in forced or pid in forced.values():
                        if forced.get(obs['key']) == pid:
                            costs[i,j] = 0.
                        continue
                    state = states[pid]
                    dt = frame['f']-state['f']
                    gap = abs(dt)/fps
                    lost = gap > parameters.local_gap_seconds
                    if lost and not recover:
                        continue
                    if app[i, j] > (parameters.recovery_appearance_max if lost else parameters.appearance_max):
                        continue
                    # Motion cannot overrule clearly stronger immutable person
                    # evidence. A blocked best match must not lend its ID to a
                    # physically convenient but visually different candidate.
                    if (not lost and parameters.local_appearance_regret_max is not None
                            and app[i,j]-np.min(app[i,active]) > parameters.local_appearance_regret_max):
                        continue
                    if lost:
                        rivals = app[i, [k for k in active if k != j]]
                        if obs['occlusion'] >= .6 or np.min(rivals, initial=1.)-app[i,j] < parameters.recovery_appearance_margin:
                            continue
                        costs[i,j] = parameters.appearance_weight*app[i,j]/parameters.appearance_max
                        recovery[i,j] = True
                        continue
                    height = max(1., obs['bb'][3]-obs['bb'][1])
                    old_height = max(1., state['obs']['bb'][3]-state['obs']['bb'][1])
                    if parameters.motion_model == 'kalman':
                        # Contaminated crops need distinct appearance; do not let
                        # a misleading box centre win against a similar uniform.
                        rivals = app[i, [k for k in active if k != j]]
                        if (obs['occlusion'] >= parameters.motion_update_overlap_max
                                and np.min(rivals, initial=1.)-app[i,j]
                                < parameters.recovery_appearance_margin):
                            continue
                        distance, mean, _, _, _ = filters[pid].innovation(
                            centre(obs), height, frame['f']/fps, obs['occlusion'])
                        physical = np.linalg.norm(centre(obs)-mean[:2])/((height+old_height)/2)
                        if (distance > parameters.kalman_chi2_gate
                                or physical > .75+.7*min(gap, 2.)
                                or abs(np.log(height/old_height)) > .85):
                            continue
                        motion = np.sqrt(distance/parameters.kalman_chi2_gate)
                        costs[i,j] = (parameters.appearance_weight*app[i,j]/parameters.appearance_max
                                      + (1.-parameters.appearance_weight)*motion)
                        continue
                    predicted = centre(state['obs']) + state['velocity']*np.clip(dt, -.5*fps, .5*fps)
                    motion = np.linalg.norm(centre(obs)-predicted)/((height+old_height)/2)
                    gate = .75+.7*min(gap, 2.)
                    if motion > gate or abs(np.log(height/old_height)) > .85:
                        continue
                    costs[i,j] = parameters.appearance_weight*app[i,j]/parameters.appearance_max + (1.-parameters.appearance_weight)*motion/gate
            augmented = np.concatenate([costs, np.full((len(observations), len(observations)), parameters.unknown_cost)], axis=1)
            rows, columns = linear_sum_assignment(augmented)
            pending_this_frame = set()
            for i, j in zip(rows, columns):
                obs = observations[i]
                if j >= len(roster):
                    record(obs, None, 'unmatched_or_appearance_gate')
                    continue
                cost = float(costs[i,j])
                if np.min(np.delete(costs[i], j), initial=100.)-cost < parameters.assignment_margin:
                    record(obs, None, 'ambiguous', cost)
                    continue
                pid = roster[j]
                if forced.get(obs['key']) == pid:
                    record(obs,pid,'roster_anchor',cost,0.)
                    states[pid] = {'obs':obs,'f':frame['f'],'velocity':np.zeros(2)}
                    filters[pid] = PlayerMotionKalman(centre(obs),obs['bb'][3]-obs['bb'][1],frame['f']/fps)
                    pending.pop(pid,None)
                    continue
                if recovery[i,j]:
                    history = pending.get(pid, [])
                    if history:
                        last_frame, last_obs = history[-1][:2]
                        height = max(1., obs['bb'][3]-obs['bb'][1])
                        if abs(frame['f']-last_frame)/fps > .25 or np.linalg.norm(centre(obs)-centre(last_obs)) > height:
                            history = []
                    history.append((frame['f'], obs, cost,float(app[i,j])))
                    pending[pid] = history
                    pending_this_frame.add(pid)
                    record(obs, None, 'pending_recovery', cost)
                    if len(history) < parameters.recovery_confirmations:
                        continue
                    for _, candidate, candidate_cost, candidate_app in history:
                        record(candidate, pid, 'recovered', candidate_cost,candidate_app)
                    states[pid] = {'obs': obs, 'f': frame['f'], 'velocity': np.zeros(2)}
                    filters[pid] = PlayerMotionKalman(centre(obs),
                        obs['bb'][3]-obs['bb'][1], frame['f']/fps)
                    pending.pop(pid)
                else:
                    record(obs, pid, 'associated', cost,app[i,j])
                    state = states[pid]
                    if parameters.motion_model == 'kalman':
                        if obs['occlusion'] < parameters.motion_update_overlap_max:
                            filters[pid].update(centre(obs), obs['bb'][3]-obs['bb'][1],
                                                frame['f']/fps, obs['occlusion'])
                            state['obs'], state['f'] = obs, frame['f']
                        else:
                            record(obs, pid, 'occluded_appearance_only', cost,app[i,j])
                        pending.pop(pid, None)
                        continue
                    velocity = (centre(obs)-centre(state['obs']))/(frame['f']-state['f'])
                    state['velocity'] = .5*state['velocity']+.5*velocity
                    state['obs'], state['f'] = obs, frame['f']
                    pending.pop(pid, None)
            for pid in list(pending):
                if pid not in pending_this_frame:
                    pending.pop(pid)
    consistency = {'enabled':False}
    if parameters.tracklet_consistency:
        consistency = reject_tracklet_conflicts(
            frames,predictions,anchors,dominance=parameters.tracklet_dominance,
            min_clean_votes=parameters.tracklet_min_clean_votes,
            clean_overlap=parameters.gallery_overlap_max,fps=fps,
            max_gap_seconds=parameters.local_gap_seconds,
            appearance_limit=parameters.recovery_appearance_max,
            appearance_max=parameters.appearance_max,appearance_weight=parameters.appearance_weight)
    return {'predictions': predictions, 'gallery_keys': keys, 'tracklet_consistency':consistency,
            'parameters': {**parameters.to_dict(), 'recover': recover}}
