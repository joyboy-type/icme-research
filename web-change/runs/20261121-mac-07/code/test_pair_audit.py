import unittest
import numpy as np
try:from pair_audit import pixel_audit,git_blob_sha
except ImportError:pixel_audit=git_blob_sha=None
class PairAuditTests(unittest.TestCase):
 def test_counts_and_bbox_of_actual_changed_pixels(self):
  self.assertTrue(callable(pixel_audit),'Pixel audit missing');a=np.zeros((3,4,3),np.uint8);b=a.copy();b[1,2]=9
  r=pixel_audit(a,b);self.assertEqual(r['changed_pixels'],1);self.assertEqual(r['bbox_xyxy'],[2,1,3,2]);self.assertEqual(r['changed_gt4'],1)
 def test_different_dimensions_are_not_resized_into_comparability(self):
  self.assertTrue(callable(pixel_audit),'Geometry audit missing');r=pixel_audit(np.zeros((2,2,3),np.uint8),np.zeros((3,2,3),np.uint8));self.assertFalse(r['same_geometry']);self.assertIsNone(r['changed_pixels'])
 def test_git_hash_covers_original_bytes(self):
  self.assertTrue(callable(git_blob_sha),'Git byte verification missing');self.assertEqual(git_blob_sha(b'test content\n'),'d670460b4b4aece5915caf5c68d12f560a9fe3e4')
if __name__=='__main__':unittest.main()
