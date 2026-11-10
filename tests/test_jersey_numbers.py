import unittest

import numpy as np

from src.models.jersey_ocr import torso_box
from src.tracking.jersey_numbers import (
    JerseyParameters, jersey_constraints, observation_numbers, raw_tracklets, roster_numbers,
)
from src.tracking.player_identity_workflow import PRODUCTION_PARAMETERS
from src.tracking.player_reid import associate


def obs(f, i, x, raw, y=100.):
    return {'key': f'{f}:{i}', 'pose_index': i, 'bb': [x, y, x + 40, y + 100], 'raw_id': raw, 'occlusion': 0.}


class TrackletTests(unittest.TestCase):
    def test_split_at_long_gap(self):
        frames = [{'f': 0, 'players': [obs(0, 0, 100, 5)]}, {'f': 1, 'players': [obs(1, 0, 102, 5)]},
                  {'f': 90, 'players': [obs(90, 0, 104, 5)]}]
        t = raw_tracklets(frames, 30., JerseyParameters())
        self.assertEqual(t['0:0'], t['1:0'])
        self.assertNotEqual(t['1:0'], t['90:0'])

    def test_tracklet_number_needs_agreeing_confident_reads(self):
        frames = [{'f': f, 'players': [obs(f, 0, 100, 5), obs(f, 1, 400, 6)]} for f in range(4)]
        reads = {'0:0': {'number': '7', 'prob': .9}, '1:0': {'number': '7', 'prob': .8},
                 '2:0': {'number': '1', 'prob': .3},  # below min_prob: ignored
                 '0:1': {'number': '9', 'prob': .9}, '1:1': {'number': '4', 'prob': .9}}
        numbers = observation_numbers(frames, reads, 30.)
        self.assertEqual({k for k, v in numbers.items() if v['number'] == '7'}, {'0:0', '1:0', '2:0', '3:0'})
        self.assertFalse(any(k.endswith(':1') for k in numbers))  # 9 vs 4: no clear vote

    def test_roster_numbers_drop_shared_numbers(self):
        evidence = {'a': {'number': '7', 'tracklet': 1}, 'b': {'number': '7', 'tracklet': 2},
                    'c': {'number': '7', 'tracklet': 3}, 'd': {'number': '7', 'tracklet': 4},
                    'e': {'number': '5', 'tracklet': 5}, 'f': {'number': '5', 'tracklet': 6}}
        owners = {'a': 'P1', 'b': 'P1', 'c': 'P2', 'd': 'P2', 'e': 'P3', 'f': 'P3'}
        self.assertEqual(roster_numbers(evidence, owners), {'P3': '5'})


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.frames = [{'f': 0, 'players': [obs(0, 0, 100, 1), obs(0, 1, 300, 2)]},
                       {'f': 1, 'players': [obs(1, 0, 101, 3)]}]
        self.seed = {'frame': 0, 'assignments': {'0:0': 'P1', '0:1': 'P2'}}
        self.features = {'0:0': np.eye(4)[0], '0:1': np.eye(4)[1], '1:0': np.eye(4)[0]}

    def run_with(self, jersey):
        return associate(self.frames, self.seed, self.features, 30., True, PRODUCTION_PARAMETERS,
                         jersey=jersey)['predictions']['1:0']

    def test_without_numbers_appearance_and_motion_decide(self):
        self.assertEqual(self.run_with(None)['player_id'], 'P1')

    def test_another_players_number_vetoes_the_id(self):
        prediction = self.run_with({'observations': {'1:0': '7'}, 'roster': {'P2': '7'}})
        self.assertIsNone(prediction['player_id'])
        self.assertEqual(prediction['jersey_number'], '7')

    def test_matching_number_keeps_the_id(self):
        self.assertEqual(self.run_with({'observations': {'1:0': '7'}, 'roster': {'P1': '7'}})['player_id'], 'P1')

    def test_constraints_from_anchor_tracklets(self):
        frames = [{'f': f, 'players': [obs(f, 0, 100 + f, 1), obs(f, 1, 300, 2)]} for f in range(3)]
        reads = {'0:0': {'number': '7', 'prob': .9}, '1:0': {'number': '7', 'prob': .9}}
        seed = {'frame': 0, 'assignments': {'0:0': 'P1', '0:1': 'P2'}}
        bundle = jersey_constraints(frames, reads, 30., seed,
                                    parameters=JerseyParameters(roster_min_tracklets=1))
        self.assertEqual(bundle['roster'], {'P1': '7'})
        self.assertEqual(bundle['observations']['2:0'], '7')


class TorsoBoxTests(unittest.TestCase):
    def test_keypoint_torso_and_small_player(self):
        kp = [[0, 0, 0]] * 17
        kp[5], kp[6], kp[11], kp[12] = [100, 100, .9], [160, 100, .9], [105, 180, .9], [155, 180, .9]
        box = torso_box({'kp': kp})
        self.assertAlmostEqual(box[3], 180)
        self.assertLess(box[0], 100)
        self.assertGreater(box[2], 160)
        kp[11], kp[12] = [105, 110, .9], [155, 110, .9]
        self.assertIsNone(torso_box({'kp': kp}))


if __name__ == '__main__':
    unittest.main()
