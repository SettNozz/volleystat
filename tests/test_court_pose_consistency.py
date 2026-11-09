import numpy as np
import pytest

from src.detection.player_pose import CourtProjector
from src.tracking.court_pose_consistency import reject_unsupported_sideline_poses
from src.tracking.player_identity_workflow import (
    apply_player_identities,
    identity_seed,
    observations_from_annotations,
)


PROJECTOR = CourtProjector([[50, 0, 300], [0, 10, 100], [0, 0, 1]])


def pose(track_id, x, y=160, *, team='b', ankles=True):
    kp = [[0, 0, 0] for _ in range(17)]
    kp[15:] = [[x - 4, y, int(ankles)], [x + 4, y, int(ankles)]]
    return {'id': track_id, 'team': team, 'bb': [x - 12, y - 130, x + 12, y], 'kp': kp}


def test_attack_runup_keeps_same_roster_id_through_sideline_excursion():
    # A bench pose shares the player's earlier raw ID after a large jump.
    outward = [505, 507, 510, 514, 519, 525, 531, 537, 542, 546, 549, 551, 552]
    runup = outward + outward[::-1]
    frames = [{'f': f, 'poses': [pose(1, x)]} for f, x in enumerate(runup)]
    frames.append({'f': len(runup), 'poses': [pose(1, 44)]})

    assert reject_unsupported_sideline_poses(frames, PROJECTOR, 30) == 1
    assert frames[-1]['poses'] == []
    observations = observations_from_annotations(frames)
    seed = identity_seed(observations, roster_size=1)
    embeddings = {f'{f}:0': np.array([1., 0.], dtype=np.float32) for f in range(len(runup))}
    apply_player_identities(frames, observations, seed, embeddings, 30)
    assert [row['poses'][0]['player_id'] for row in frames[:-1]] == ['P1'] * len(runup)


def test_future_court_observation_supports_plausible_return_after_occlusion():
    frames = [{'f': 0, 'poses': [pose(1, 555)]}, {'f': 26, 'poses': [pose(1, 517)]}]

    assert reject_unsupported_sideline_poses(frames, PROJECTOR, 30) == 0
    assert frames[0]['poses'][0]['id'] == 1


@pytest.mark.parametrize('gap', [1, 31])
def test_track_id_alone_cannot_transfer_membership_across_jump_or_gap(gap):
    off_court = 44 if gap == 1 else 551
    frames = [{'f': 0, 'poses': [pose(1, 520)]}, {'f': gap, 'poses': [pose(1, off_court)]}]

    assert reject_unsupported_sideline_poses(frames, PROJECTOR, 30) == 1
    assert frames[1]['poses'] == []


def test_unsupported_spectator_removed_without_disturbing_grouped_points():
    frames = [{
        'f': 0,
        'poses': [pose(1, 44), pose(2, 300, 65, team='a'),
                  pose(3, 300, 220), pose(4, 530), pose(5, 550, ankles=False)],
        'p': [[300, 0], [44, 95], [300, 155], [530, 95], [550, 95]],
    }]

    assert reject_unsupported_sideline_poses(frames, PROJECTOR, 30) == 1
    assert [p['id'] for p in frames[0]['poses']] == [2, 3, 4, 5]
    assert frames[0]['p'] == [[300, 0], [300, 155], [530, 95], [550, 95]]
