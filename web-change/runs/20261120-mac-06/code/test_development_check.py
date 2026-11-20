import unittest
import numpy as np
try:from jpeg_check import select_development,pair_inputs
except ImportError:select_development=pair_inputs=None
class DevelopmentCheckTests(unittest.TestCase):
 def test_selection_excludes_confirmation_and_prefers_middle_level(self):
  self.assertTrue(callable(select_development),'Development selector missing')
  rows=[dict(id='a',baseline_id='v0',family='spectre',split='development',label='text',level=0),dict(id='b',baseline_id='v0',family='spectre',split='development',label='text',level=1),dict(id='c',baseline_id='v0',family='pico',split='confirmation',label='text',level=1)]
  self.assertEqual([x['id'] for x in select_development(rows)],['b'])
 def test_canonicalization_preserves_actual_changed_frame(self):
  self.assertTrue(callable(pair_inputs),'Canonical pair transform missing')
  a=np.zeros((40,40,3),np.uint8);b=np.full_like(a,255)
  x,y=pair_inputs(a,b,'jpeg95','canonical90');self.assertEqual(float(x.mean()),0.);self.assertEqual(float(y.mean()),255.)
 def test_matched_capture_identical_inputs_are_zero_difference(self):
  self.assertTrue(callable(pair_inputs),'Matched capture transform missing')
  a=np.tile(np.arange(40,dtype=np.uint8)[None,:,None],(40,1,3));x,y=pair_inputs(a,a,'jpeg95','matched_capture');self.assertTrue(np.array_equal(x,y))
if __name__=='__main__':unittest.main()
