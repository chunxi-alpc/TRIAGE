import unittest
from report import metric,paired
class Reporting(unittest.TestCase):
 def test_paired_harm_denominator(self):
  rows=[{'tree':'t','gold_option':g,'answers':{'original':a,'new':b},'S':[.7,.3],'covered':True} for g,a,b in [(0,0,1),(1,0,1),(0,0,0),(1,0,0)]]
  m=metric(rows,'new');self.assertEqual(m['harmed'],1);self.assertEqual(m['corrected'],1)
  self.assertEqual(m['harm'],.25);self.assertEqual(m['harm_among_original_correct'],.5)
  self.assertEqual(paired(rows,'new','original')['accuracy_delta'],0.)
if __name__=='__main__':unittest.main()
