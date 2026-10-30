import copy
import itertools
import unittest
import numpy as np
from src.detection.person_masks import (
    PersonMaskAnalyzer,assign,encode_mask,decode_mask,FEATURE_NAMES,inside_polygon,
)
from scripts.evaluation.sam3_people_masks import plan_frames


class PersonMaskTests(unittest.TestCase):
    def mask(self,x=2):
        m = np.zeros((10,20),bool)
        m[2:8,x:x+4] = True
        return m

    def analyzer(self,court=True):
        return PersonMaskAnalyzer(200,100,[[0,0],[200,0],[200,100],[0,100]] if court else [])

    def process(self,analyzer=None,f=0,annotation=None,masks=None):
        return (analyzer or self.analyzer()).process(f,annotation or {},
            np.array([self.mask()]) if masks is None else masks,
            np.full(1 if masks is None else len(masks),.8))

    def test_rle_roundtrip(self):
        rng = np.random.default_rng(1)
        for m in [np.zeros((3,7),bool),np.ones((3,7),bool),rng.random((13,19)) > .5,self.mask()]:
            np.testing.assert_array_equal(decode_mask(encode_mask(m)),m)

    def test_rle_rejects_bad_counts(self):
        with self.assertRaises(ValueError):
            decode_mask(dict(encoding='rle_row_major_v1',size=[2,2],counts=[3]))

    def test_global_assignment_and_unmatched(self):
        self.assertEqual(set(assign([[.8,.7],[.75,0]],.1)),{(0,1),(1,0)})
        self.assertEqual(assign([[0,-1],[0,0]],.1),[])
        self.assertEqual(assign(np.zeros((0,3)),.1),[])
        self.assertEqual(assign([[.8],[.7]],.1),[(0,0)])

    def test_assignment_matches_exhaustive(self):
        rng = np.random.default_rng(3)
        for n,m in [(1,4),(4,1),(3,4),(4,3)]:
            for _ in range(20):
                scores = rng.random((n,m))
                scores[scores < .3] = -1
                actual = sum(scores[i,j] for i,j in assign(scores,.2))
                # Enumerate every matching including unmatched rows.
                best = 0.
                for choices in itertools.product(range(-1,m),repeat=n):
                    selected = [j for j in choices if j >= 0]
                    if len(selected) != len(set(selected)):
                        continue
                    if any(scores[i,j] < .2 for i,j in enumerate(choices) if j >= 0):
                        continue
                    best = max(best,sum(scores[i,j] for i,j in enumerate(choices) if j >= 0))
                self.assertAlmostEqual(actual,best)

    def test_preserves_yolo_identity_and_source(self):
        source = dict(ball=[30,40],poses=[dict(bb=[20,20,60,80],id=9,team='a')])
        before = copy.deepcopy(source)
        r = self.process(annotation=source)
        self.assertEqual(source,before)
        self.assertEqual(r['persons'][0]['yolo_track_id'],9)
        self.assertEqual(r['persons'][0]['team'],'a')
        self.assertEqual(r['persons'][0]['court_status'],'existing_court_member')
        self.assertEqual(r['features'][5],1)

    def test_unmatched_candidate_has_no_fake_pose(self):
        p = self.process()['persons'][0]
        self.assertTrue(p['review_required'])
        self.assertIsNone(p['team'])
        self.assertIsNone(p['yolo_track_id'])

    def test_spectators_excluded(self):
        a = PersonMaskAnalyzer(200,100,[[100,0],[200,0],[200,100],[100,100]])
        r = self.process(a)
        self.assertEqual(r['persons'][0]['court_status'],'outside')
        self.assertEqual(r['features'][7],0)
        self.assertTrue(r['valid'][7])

    def test_existing_member_hysteresis_preserved(self):
        a = PersonMaskAnalyzer(200,100,[[100,0],[200,0],[200,100],[100,100]])
        r = self.process(a,annotation=dict(poses=[dict(bb=[20,20,60,80],id=9,team='a')]))
        self.assertEqual(r['persons'][0]['court_status'],'existing_court_member')

    def test_unknown_court_invalidates_count(self):
        r = self.process(self.analyzer(False))
        self.assertEqual(r['valid'][6:],[False,False])

    def test_temporal_adjacent_only(self):
        a = self.analyzer()
        p = self.process(a,f=3)['persons'][0]
        q = self.process(a,f=4)['persons'][0]
        self.assertEqual(p['instance_id'],q['instance_id'])
        self.assertEqual(q['previous_mask_iou'],1.)
        r = self.process(a,f=6)['persons'][0]
        self.assertNotEqual(q['instance_id'],r['instance_id'])
        self.assertIsNone(r['previous_mask_iou'])

    def test_empty_frame_breaks_identity(self):
        a = self.analyzer()
        p = self.process(a)['persons'][0]
        self.process(a,f=1,masks=np.zeros((0,10,20),bool))
        q = self.process(a,f=2)['persons'][0]
        self.assertNotEqual(p['instance_id'],q['instance_id'])

    def test_conflicting_yolo_ids_do_not_merge(self):
        a = self.analyzer()
        def annotation(idx):
            return dict(poses=[dict(bb=[20,20,60,80],id=idx,team='a')])
        p = self.process(a,annotation=annotation(1))['persons'][0]
        q = self.process(a,f=1,annotation=annotation(2))['persons'][0]
        self.assertNotEqual(p['instance_id'],q['instance_id'])

    def test_yolo_disappears_no_identity_fabrication(self):
        a = self.analyzer()
        p = self.process(a,annotation=dict(poses=[dict(bb=[20,20,60,80],id=1,team='a')]))['persons'][0]
        q = self.process(a,f=1)['persons'][0]
        self.assertEqual(p['instance_id'],q['instance_id'])
        self.assertIsNone(q['yolo_track_id'])
        self.assertIsNone(q['team'])

    def test_one_pose_cannot_match_two_masks(self):
        r = self.process(annotation=dict(poses=[dict(bb=[20,20,60,80],id=1)]),masks=np.stack([self.mask(),self.mask(3)]))
        self.assertEqual(sum(p['pose_index'] is not None for p in r['persons']),1)

    def test_ball_final_not_raw(self):
        r = self.process(annotation=dict(ball=[30,40],ball_raw=[180,90]))
        self.assertEqual(r['features'][0],0.)
        self.assertEqual(r['features'][1],1.)
        self.assertEqual(r['valid'][:4],[True]*4)
        self.assertFalse(r['valid'][4])

    def test_offcamera_invalidates_stale_ball(self):
        for annotation in [dict(ball=[30,40],ball_out_of_frame=True),dict(ball=None),dict(ball=[float('nan'),2]),dict(ball=[200,10])]:
            r = self.process(annotation=annotation)
            self.assertEqual(r['valid'][:5],[False]*5)
            self.assertEqual(r['features'][:5],[0]*5)

    def test_resolution_independent(self):
        r = self.process(annotation=dict(ball=[80,40]))
        a = PersonMaskAnalyzer(400,200,[[0,0],[400,0],[400,200],[0,200]])
        q = self.process(a,annotation=dict(ball=[160,80]))
        np.testing.assert_array_equal(r['features'],q['features'])

    def test_no_masks_zero_count_not_unavailable(self):
        r = self.process(masks=np.zeros((0,10,20),bool))
        self.assertTrue(r['mask_detection_available'])
        self.assertEqual(r['features'][6:],[0,0])
        self.assertEqual(r['valid'][6:],[True,True])

    def test_rejects_invalid_masks_and_scores(self):
        a = self.analyzer()
        for masks,scores in [(np.full((1,10,20),2),[.8]),(np.full((1,10,20),np.nan),[.8]),(np.array([self.mask()]),[1.1])]:
            with self.assertRaises(ValueError):
                a.process(0,{},masks,scores)

    def test_rejects_frame_or_grid_change(self):
        a = self.analyzer()
        self.process(a)
        with self.assertRaises(ValueError):
            self.process(a)
        with self.assertRaises(ValueError):
            self.process(a,f=1,masks=np.ones((1,20,40),bool))

    def test_polygon_boundary(self):
        court = [[0,0],[10,0],[10,10],[0,10]]
        self.assertTrue(inside_polygon((0,5),court))
        self.assertFalse(inside_polygon((-1,5),court))

    def test_interval_absolute_indices(self):
        self.assertEqual(plan_frames(30000/1001,15,30),(450,1349))
        self.assertEqual(plan_frames(30,15,30),(450,1350))
        for fps,start,duration in [(0,15,30),(30,-1,30),(30,15,0)]:
            with self.assertRaises(ValueError):
                plan_frames(fps,start,duration)
        self.assertEqual(len(FEATURE_NAMES),8)


if __name__ == '__main__':
    unittest.main()
