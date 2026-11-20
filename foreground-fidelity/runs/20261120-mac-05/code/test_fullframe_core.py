import unittest
import numpy as np
from PIL import Image
from fullframe_core import spatial,patch_content,source_supports,nearest

class FullFrameTests(unittest.TestCase):
 def test_letterbox_preserves_landscape_edges_and_marks_padding(self):
  x=np.zeros((4,8,3),dtype=np.uint8);x[:,0]=[255,0,0];x[:,-1]=[0,0,255]
  im,valid=spatial(Image.fromarray(x),'letterbox',8)
  self.assertEqual(im.size,(8,8));self.assertEqual(im.getpixel((0,2)),(255,0,0));self.assertEqual(im.getpixel((7,5)),(0,0,255))
  np.testing.assert_array_equal(np.asarray(valid).sum(axis=1),[0,0,8,8,8,8,0,0])

 def test_float_mask_and_portrait_content_use_identical_geometry(self):
  im,valid=spatial(Image.fromarray(np.ones((8,4),dtype=np.float32),mode='F'),'letterbox',8,mask=True)
  np.testing.assert_array_equal(np.asarray(im),np.asarray(valid))
  np.testing.assert_array_equal(patch_content(valid,2),[1,2,5,6,9,10,13,14])

 def test_square_warp_retains_full_frame_and_has_no_padding(self):
  im,valid=spatial(Image.new('RGB',(16,4),(255,0,0)),'square_warp',8)
  self.assertEqual(im.size,(8,8))
  self.assertEqual(im.getpixel((7,7)),(255,0,0));self.assertEqual(np.asarray(valid).sum(),64)

 def test_source_controls_never_select_padding_and_share_K(self):
  p=np.array([1,.1,.9,1,.8,.1,.7,1,1]);v=np.array([1,2,4,6]);sal=np.arange(9)
  x=source_supports(p,sal,v,17)
  np.testing.assert_array_equal(x['auto'],[2,4,6])
  for ids in x.values():self.assertEqual(len(ids),3);self.assertTrue(set(ids)<=set(v))
  self.assertEqual(x['auto'].tolist(),source_supports(p,sal,v,17)['auto'].tolist())

 def test_empty_region_falls_back_inside_content_only(self):
  x=source_supports(np.array([1,.1,.2,1]),np.arange(4),np.array([1,2]),17)
  np.testing.assert_array_equal(x['auto'],[2])

 def test_padding_match_excluded_and_no_local_target_is_invalid(self):
  m=np.array([[1.,.4],[.7,.8]])
  self.assertAlmostEqual(nearest(m,[0],[1])['score'],.4)
  grid=np.eye(4);z=nearest(grid,[0],[3],radius=0)
  self.assertFalse(z['valid']);self.assertIsNone(z['score']);self.assertEqual(z['unmatched_source_patches'],1)

if __name__=='__main__':unittest.main()
