"""Jersey-number evidence for the tracked team's roster identities.

OCR reads single frames, and a single read is noisy. Evidence therefore comes
from short continuous raw tracklets: a tracklet carries a number only when
several confident reads agree. Roster numbers come from the tracklets of the
roster anchors and, optionally, from a first association pass, again only by
tracklet votes. The association uses them as constraints: a detection whose
tracklet shows another player's number cannot take this ID, and a matching
number may confirm a lost player's return when appearance alone is ambiguous.
Raw reads stay model output (``label_source='model'``, unverified).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from src.tracking.player_identity import centre


@dataclass(frozen=True)
class JerseyParameters:
    # 0.8: on the e2F6 checks 0.6 already gave one roster ID a misread number
    # (7 read as 1) and 0.4 lost 254 assignments; see docs/player_identity_workflow.md.
    min_prob: float = .8
    min_votes: int = 2
    dominance: float = .75
    tracklet_gap_seconds: float = 1.
    tracklet_jump_heights: float = 1.
    roster_min_tracklets: int = 2
    roster_dominance: float = .6

    def to_dict(self):
        return asdict(self)


def raw_tracklets(frames: list[dict], fps: float, parameters: JerseyParameters) -> dict[str, int]:
    """Split raw track IDs at long gaps or large jumps; return key -> tracklet."""
    last: dict = {}
    assignment: dict[str, int] = {}
    next_id = 0
    for frame in sorted(frames, key=lambda f: f['f']):
        for obs in frame['players']:
            raw = obs.get('raw_id')
            if raw is None:
                assignment[obs['key']] = next_id
                next_id += 1
                continue
            previous = last.get(raw)
            height = max(1., obs['bb'][3]-obs['bb'][1])
            if (previous is None or (frame['f']-previous[0])/fps > parameters.tracklet_gap_seconds
                    or np.linalg.norm(centre(obs)-centre(previous[1])) > parameters.tracklet_jump_heights*height
                    *max(1., (frame['f']-previous[0])/fps*4)):
                tracklet = next_id
                next_id += 1
            else:
                tracklet = previous[2]
            last[raw] = (frame['f'], obs, tracklet)
            assignment[obs['key']] = tracklet
    return assignment


def _vote(counter: Counter, min_votes: int, dominance: float):
    if not counter:
        return None, 0, 0
    number, votes = counter.most_common(1)[0]
    total = sum(counter.values())
    if votes >= min_votes and votes/total >= dominance:
        return number, votes, total
    return None, votes, total


def observation_numbers(frames: list[dict], reads: dict, fps: float,
                        parameters: JerseyParameters = JerseyParameters()) -> dict[str, dict]:
    """Tracklet-level number per observation key (only where the vote is clear)."""
    tracklet_of = raw_tracklets(frames, fps, parameters)
    votes = defaultdict(Counter)
    for key, read in reads.items():
        if key in tracklet_of and read.get('number') and read.get('prob', 0.) >= parameters.min_prob:
            votes[tracklet_of[key]][read['number']] += 1
    numbers = {}
    for tracklet, counter in votes.items():
        number, support, total = _vote(counter, parameters.min_votes, parameters.dominance)
        if number is not None:
            numbers[tracklet] = {'number': number, 'votes': support, 'reads': total}
    return {key: {**numbers[t], 'tracklet': t} for key, t in tracklet_of.items() if t in numbers}


def roster_numbers(evidence: dict[str, dict], owners: dict[str, str],
                   parameters: JerseyParameters = JerseyParameters()) -> dict[str, str]:
    """Number per roster ID from tracklets whose observations belong to it.

    ``owners`` maps observation keys to roster IDs (anchors, or a first pass).
    Each tracklet votes once. A number claimed by two IDs is dropped for both.
    """
    seen = set()
    votes = defaultdict(Counter)
    for key, pid in owners.items():
        row = evidence.get(key)
        if pid is None or row is None or (pid, row['tracklet']) in seen:
            continue
        seen.add((pid, row['tracklet']))
        votes[pid][row['number']] += 1
    chosen = {}
    for pid, counter in votes.items():
        number, _, _ = _vote(counter, parameters.roster_min_tracklets, parameters.roster_dominance)
        if number is not None:
            chosen[pid] = number
    duplicated = {n for n, c in Counter(chosen.values()).items() if c > 1}
    return {pid: n for pid, n in chosen.items() if n not in duplicated}


def anchor_owners(frames: list[dict], seed: dict, fps: float, parameters: JerseyParameters) -> dict[str, str]:
    """Observation keys in the anchors' own raw tracklets."""
    from src.tracking.player_reid import identity_anchors
    tracklet_of = raw_tracklets(frames, fps, parameters)
    owner_of_tracklet = {tracklet_of[a['key']]: a['player_id'] for a in identity_anchors(seed)
                         if a['key'] in tracklet_of}
    return {key: owner_of_tracklet[t] for key, t in tracklet_of.items() if t in owner_of_tracklet}


def jersey_constraints(frames, reads, fps, seed, first_pass=None,
                       parameters: JerseyParameters = JerseyParameters()) -> dict:
    """Bundle passed to :func:`src.tracking.player_reid.associate` as ``jersey``."""
    evidence = observation_numbers(frames, reads, fps, parameters)
    owners = anchor_owners(frames, seed, fps, parameters)
    source = 'roster_anchor_tracklets'
    if first_pass is not None:
        confident = {'seed', 'associated', 'recovered', 'roster_anchor'}
        owners = {**{k: p['player_id'] for k, p in first_pass.items()
                     if p.get('player_id') and p.get('reason') in confident}, **owners}
        source = 'roster_anchor_tracklets+first_pass_tracklets'
    roster = roster_numbers(evidence, owners, parameters)
    return {'observations': {k: v['number'] for k, v in evidence.items()}, 'roster': roster,
            'roster_source': source, 'parameters': parameters.to_dict(),
            'tracklets_with_number': len({v['tracklet'] for v in evidence.values()})}
