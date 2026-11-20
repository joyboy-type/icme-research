import unittest
import cv2
import numpy as np
from registration import estimate_translation,translate,residual_scores,calibration_threshold


def textured_page():
    rng=np.random.default_rng(17)
    a=rng.integers(0,256,(256,256,3),dtype=np.uint8)
    return cv2.GaussianBlur(a,(5,5),1)


class RegistrationContracts(unittest.TestCase):
    def test_global_and_consensus_remove_known_integer_capture_shift(self):
        a=textured_page()
        # Independent transform: one pixel right and one pixel up.
        b=cv2.warpAffine(a,np.float32([[1,0,1],[0,1,-1]]),(256,256),borderMode=cv2.BORDER_REPLICATE)
        for method in ['global_phase','tile_consensus']:
            r=estimate_translation(a,b,method)
            self.assertEqual(r['shift'],[1,-1])
            aligned=translate(b,-r['shift'][0],-r['shift'][1])
            np.testing.assert_array_equal(a[4:-4,4:-4],aligned[4:-4,4:-4])

    def test_consensus_preserves_a_local_position_mutation(self):
        a=textured_page();b=a.copy();b[40:58,40:62]=np.roll(a[40:58,40:62],1,axis=1)
        r=estimate_translation(a,b,'tile_consensus')
        self.assertEqual(r['shift'],[0,0])
        self.assertGreater(residual_scores(a,translate(b,*[-v for v in r['shift']]))['pixel4'],0)

    def test_low_texture_does_not_invent_translation(self):
        a=np.full((256,256,3),245,np.uint8)
        for method in ['global_phase','tile_consensus']:
            self.assertEqual(estimate_translation(a,a,method)['shift'],[0,0])

    def test_out_of_range_shift_is_rejected(self):
        a=textured_page();b=cv2.warpAffine(a,np.float32([[1,0,8],[0,1,0]]),(256,256),borderMode=cv2.BORDER_REPLICATE)
        self.assertEqual(estimate_translation(a,b,'global_phase')['shift'],[0,0])

    def test_evaluation_support_is_shared_and_identity_scores_zero(self):
        a=np.zeros((20,20,3),np.uint8);b=a.copy();b[:3]=255
        self.assertEqual(residual_scores(a,a),{'pixel4':0.,'blur8':0.})
        self.assertEqual(residual_scores(a,b),{'pixel4':0.,'blur8':0.})
        b[10,10]=255
        self.assertGreater(residual_scores(a,b)['pixel4'],0)

    def test_threshold_controls_each_calibration_condition(self):
        by={'easy':np.zeros(100),'hard':np.arange(100)/100}
        t=calibration_threshold(by)
        self.assertEqual(t,.95)
        for scores in by.values():self.assertLessEqual(float(np.mean(scores>t)),.05)

    def test_bad_mode_and_mismatched_shapes_are_rejected(self):
        a=textured_page()
        with self.assertRaises(ValueError):estimate_translation(a,a,'affine')
        with self.assertRaises(ValueError):residual_scores(a,a[:20])


if __name__=='__main__':unittest.main()
