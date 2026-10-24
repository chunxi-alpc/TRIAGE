import unittest
from analysis import action,select,temper,metric,bootstrap,EVIDENCE
class Controls(unittest.TestCase):
 def setUp(self):
  self.c={'options':['a','b'],'rules':[{'rid':0,'action':'a','condition':'x'},{'rid':1,'action':'a','condition':'y'},{'rid':2,'action':'b','condition':'z'}]}
 def test_uniform_is_frequency(self):self.assertEqual(action(self.c,[1/3]*3),[2/3,1/3])
 def test_count_controls(self):
  self.assertEqual(action(self.c,[.3,.3,.4],'max'),action(self.c,[.3,.3,.4],'max',duplicate=0))
  self.assertEqual(action(self.c,[.3,.3,.4],'mean'),action(self.c,[.3,.3,.4],'mean',duplicate=0))
  self.assertNotEqual(action(self.c,[.3,.3,.4]),action(self.c,[.3,.3,.4],duplicate=0))
 def test_temperature(self):
  self.assertAlmostEqual(temper([.8,.2],1)[0],.8);self.assertGreater(temper([.8,.2],.5)[0],.8)
 def test_no_gold_input(self):
  class NoGold(dict):
   def __getitem__(self,k):
    if k not in ('rules','options'):raise AssertionError('Target access')
    return super().__getitem__(k)
  for cfg in EVIDENCE:select(NoGold(self.c),[.45,.45,.1],[[.95,.02,.03],[.95,.02,.03],[.1,.8,.1]],[.45,.45,.1],1,cfg)
 def test_uncertainty_fallback(self):
  for cfg in EVIDENCE:self.assertEqual(select(self.c,[.45,.45,.1],[[.3,.3,.4]]*3,[.45,.45,.1],1,cfg),(1,False))
 def test_four_cells(self):
  r=[{'none':a,'pred':b,'gold':0,'tree':'t'} for a,b in [(0,0),(0,1),(1,0),(1,1)]];m=metric(r)
  self.assertEqual([m[k] for k in ['both_correct','corrected','harmed','both_wrong']],[1]*4);self.assertEqual(m['harm_given_baseline_correct'],.5)
  self.assertEqual(bootstrap(r,100)['delta_ci95'],[0.,0.])
if __name__=='__main__':unittest.main()
