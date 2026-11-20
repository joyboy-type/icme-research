import unittest
import numpy as np
from core import crop_xyxy, edge_blocks, connected_groups, false_positive_threshold


class ExperimentIntegrity(unittest.TestCase):
    def test_crop_uses_pixels_and_includes_changed_support(self):
        a=np.zeros((40,60,3),np.uint8); b=a.copy(); b[10:13,20:24]=30
        self.assertEqual(crop_xyxy(a,b),(16,6,28,17))
        self.assertEqual(crop_xyxy(b,a),crop_xyxy(a,b))

    def test_identical_pair_has_no_crop(self):
        a=np.zeros((20,20,3),np.uint8)
        self.assertIsNone(crop_xyxy(a,a.copy()))

    def test_merged_and_symmetrized_edges_have_no_direction(self):
        a=np.zeros((20,20),bool); b=a.copy(); a[3:6,4]=1; b[5:12,7]=1
        for mode in ['merged','symmetric']:
            np.testing.assert_allclose(edge_blocks(a,b,mode),edge_blocks(b,a,mode))
        self.assertFalse(np.allclose(edge_blocks(a,b,'directional'),edge_blocks(b,a,'directional')))

    def test_one_direction_features_do_not_contain_other_direction(self):
        a=np.zeros((20,20),bool); b=a.copy(); a[3:8,4]=1; b[5:10,7]=1
        np.testing.assert_allclose(edge_blocks(a,b,'new_only'),edge_blocks(np.zeros_like(a),b,'directional'))
        np.testing.assert_allclose(edge_blocks(a,b,'old_only'),edge_blocks(a,np.zeros_like(a),'directional'))
        self.assertEqual(len(edge_blocks(a,b,'directional')),65)

    def test_grouping_is_transitive_and_deterministic(self):
        ids=['c','b','a','d']
        self.assertEqual(connected_groups(ids,[(0,1),(1,2)]),['a','a','a','d'])

    def test_frozen_threshold_controls_dev_false_positives(self):
        scores=np.arange(100)/100
        t=false_positive_threshold(scores,.05)
        self.assertLessEqual(np.mean(scores>t),.05)
        self.assertGreater(np.mean(scores>t),0)


if __name__=='__main__': unittest.main()
