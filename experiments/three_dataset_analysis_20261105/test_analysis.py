import unittest
import numpy as np
from fusion import *
from common import lp
class FusionTests(unittest.TestCase):
 def test_zero_and_identical(self):
  a=lp([2,1,0]);b=lp([0,2,1])
  self.assertTrue(np.allclose(fuse(a,b,'CAD_full',0),a))
  self.assertTrue(np.allclose(fuse(a,b,'CAD_masked',0),b))
  for k,p in [('CAD_full',4),('CAD_masked',4),('probability_average',.25)]:
   self.assertTrue(np.allclose(fuse(a,a,k,p),a))
 def test_different_wrong_can_correct(self):
  f=np.log([.45,.1,.44,.01]);m=np.log([.1,.45,.44,.01])
  self.assertEqual(int(np.argmax(fuse(f,m,'probability_average',.5))),2)
 def test_empty_bothwrong(self):
  d=[dict(id='x',split='test',labels=['A','B','C'],gold_idx=0,f=lp([2,1,0]),m=lp([2,1,0]))]
  r,_=evaluate(d,'probability_average',.5)
  self.assertEqual(r['B']['numerator'],0);self.assertIsNone(r['corrected']['value'])
 def test_variable_candidates_and_case_balance(self):
  d=[dict(id='x',split='dev',labels=['A','B','C'],gold_idx=0,f=lp([2,1,0]),m=lp([0,2,1])),
     dict(id='y',split='dev',labels=['A','B','C','D'],gold_idx=3,f=lp([0,1,2,3]),m=lp([0,1,2,4]))]
  out=fit(d);self.assertEqual(len(out['parameters']['learned']),2)
  self.assertEqual(balance_weights(d+[d[0]]).tolist(),[.25,.5,.25])
if __name__=='__main__':unittest.main()
