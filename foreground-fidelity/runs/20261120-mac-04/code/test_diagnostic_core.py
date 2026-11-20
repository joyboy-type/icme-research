import unittest
import numpy as np
from diagnostic_core import reviewed_support,matched_controls,match_diagnostic

class DiagnosticTests(unittest.TestCase):
 def test_box_union_counts_centers_once_and_explicit_fallback(self):
  np.testing.assert_array_equal(reviewed_support([[0,0,2,1],[1,0,3,1]],True,4),[0,1,2])
  np.testing.assert_array_equal(reviewed_support([],False,4),np.arange(16))
  with self.assertRaises(ValueError):reviewed_support([],True,4)

 def test_matched_region_controls_preserve_area_and_source_determinism(self):
  subject=np.array([2,5,7]);prob=np.linspace(0,1,9);sal=np.arange(9)[::-1]
  a=matched_controls(subject,prob,sal,17);b=matched_controls(subject,prob,sal,17)
  for key in a:
   self.assertEqual(len(a[key]),3);self.assertEqual(len(set(a[key])),3)
   np.testing.assert_array_equal(a[key],b[key])
  np.testing.assert_array_equal(a['u2_topk'],[8,7,6])

 def test_displacement_reports_disallowed_best_match_and_score_penalty(self):
  m=np.eye(9)*.6;m[0,2]=1
  d=match_diagnostic(m,[0],1)
  self.assertAlmostEqual(d['outside_radius_fraction'],1.)
  self.assertAlmostEqual(d['score_unrestricted'],1.)
  self.assertAlmostEqual(d['score_restricted'],.6)
  self.assertAlmostEqual(d['score_penalty'],.4)

if __name__=='__main__':unittest.main()
