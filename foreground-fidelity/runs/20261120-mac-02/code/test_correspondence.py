import unittest
import numpy as np
from correspondence import mean_nearest, supports

class CorrespondenceTests(unittest.TestCase):
    def test_radius_rejects_remote_texture_shortcut(self):
        # 3x3 token grid: the counterfeit perfect match is two columns away.
        m=np.eye(9)*.6
        m[0,2]=1.
        self.assertAlmostEqual(mean_nearest(m,[0]),1.)
        self.assertAlmostEqual(mean_nearest(m,[0],radius=1),.6)

    def test_region_cannot_gain_from_excluded_background(self):
        m=np.eye(4);m[0,0]=.2
        self.assertAlmostEqual(mean_nearest(m,[0]),.2)
        self.assertAlmostEqual(mean_nearest(m,[0,1,2,3]),.8)

    def test_equal_area_controls_and_source_determinism(self):
        p=np.array([.9,.8,.1,.1,.1,.1,.1,.1,.1])
        tokens=np.eye(10)
        first=supports(p,tokens,13);second=supports(p,tokens,13)
        self.assertEqual(set(first),{'subject','salient','random','center'})
        for key in first:
            self.assertEqual(len(first[key]),2)
            np.testing.assert_array_equal(first[key],second[key])
        np.testing.assert_array_equal(first['subject'],[0,1])

if __name__=='__main__':unittest.main()
