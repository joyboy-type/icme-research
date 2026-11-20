import unittest
import numpy as np
from local_features import describe

class LocalFeatureContracts(unittest.TestCase):
    def test_identity_is_finite_and_has_no_change_signal(self):
        a=np.full((128,160,3),150,np.uint8)
        x,names,scores=describe(a,a)
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(x),len(names))
        self.assertEqual(scores,{'area_blur8':0.,'max_local32':0.,'largest_component8':0.})

    def test_small_local_change_is_not_diluted_by_full_canvas(self):
        a=np.full((128,160,3),150,np.uint8);b=a.copy();b[50:54,70:74]=220
        x,names,s=describe(a,b)
        self.assertGreater(s['max_local32'],s['area_blur8'])
        self.assertGreater(s['largest_component8'],0)
        self.assertGreater(x[names.index('raw_max')],0)

    def test_same_changed_area_can_have_different_concentration(self):
        a=np.zeros((128,160,3),np.uint8);b=a.copy();c=a.copy()
        b[50:58,70:78]=255
        for y in range(8):
            for z in range(8):c[10+y*14,10+z*18]=255
        _,_,local=describe(a,b);_,_,scattered=describe(a,c)
        self.assertGreater(local['max_local32'],scattered['max_local32'])
        self.assertGreater(local['largest_component8'],scattered['largest_component8'])

    def test_common_border_and_invalid_shapes(self):
        a=np.zeros((128,160,3),np.uint8);b=a.copy();b[:3]=255
        _,_,s=describe(a,b);self.assertEqual(s['area_blur8'],0)
        with self.assertRaises(ValueError):describe(a,a[:40])

if __name__=='__main__':unittest.main()
