import unittest
import numpy as np
from probe_metrics import class_mask
class MetricContracts(unittest.TestCase):
    def test_integer_binary_labels_do_not_turn_mask_into_integer_indices(self):
        records=[{'label':'text','positive':0,'condition':'identity'},{'label':'text','positive':1,'condition':'clean'},{'label':'color','positive':1,'condition':'clean'},{'label':'text','positive':1,'condition':'jpeg95'}]
        mask=class_mask(records,'text')
        self.assertEqual(mask.dtype,np.dtype(bool))
        np.testing.assert_array_equal(mask,[False,True,False,False])
        self.assertEqual(np.asarray([0,1,0,0])[mask].mean(),1)
if __name__=='__main__':unittest.main()
