import unittest,tempfile
from pathlib import Path
import numpy as np
try:
 from evaluation_utils import unique_baselines,role_masks,detection_metrics,check_lock,file_hashes
except ImportError:
 unique_baselines=role_masks=detection_metrics=check_lock=file_hashes=None

class EvaluationTests(unittest.TestCase):
 def test_negative_units_collapse_repeated_mutations(self):
  self.assertTrue(callable(unique_baselines),'Unique control builder missing')
  rows=[dict(baseline_id='a',family='f',split='train',before_path='a.png',before_sha256='one'),dict(baseline_id='a',family='f',split='train',before_path='a.png',before_sha256='one'),dict(baseline_id='b',family='g',split='confirmation',before_path='b.png',before_sha256='two')]
  self.assertEqual([x['id'] for x in unique_baselines(rows)],['a','b'])
  rows[1]['before_sha256']='wrong'
  with self.assertRaises(ValueError):unique_baselines(rows)
 def test_fit_and_calibration_never_take_confirmation_or_ood(self):
  self.assertTrue(callable(role_masks),'Role isolation missing')
  rows=[dict(split=s,positive=p,condition=c) for s,p,c in [('train',1,'clean'),('train',0,'jpeg90'),('development',0,'jpeg95'),('confirmation',1,'clean'),('development',1,'clean')]]
  fit,cal,confirm=role_masks(rows,['identity','jpeg95'],['clean'])
  self.assertEqual(fit.tolist(),[True,False,False,False,False]);self.assertEqual(cal.tolist(),[False,False,True,False,False]);self.assertEqual(confirm.tolist(),[False,False,False,True,False])
 def test_rates_separate_positive_from_negative(self):
  self.assertTrue(callable(detection_metrics),'Detection aggregation missing')
  rows=[dict(positive=p,condition=c,label=l,family='f') for p,c,l in [(1,'clean','text'),(1,'clean','text'),(0,'jpeg95','no_diff'),(0,'jpeg95','no_diff')]]
  r=detection_metrics(rows,np.array([True,False,False,False]))
  self.assertEqual(r['recall']['clean'],{'n':2,'detected':1,'rate':.5});self.assertEqual(r['fpr']['jpeg95'],{'n':2,'detected':0,'rate':0.})
 def test_lock_rejects_mutated_file(self):
  self.assertTrue(callable(check_lock),'Lock verification missing')
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'input';p.write_text('original');h=file_hashes([p],Path(t));check_lock(h,Path(t));p.write_text('changed')
   with self.assertRaises(ValueError):check_lock(h,Path(t))

if __name__=='__main__':unittest.main()
