"""Small regression tests for observed v3 failure classes; no holdout data."""
import unittest
from grounding import locate,normalize_value,normalize
from logic import evaluate
class Repairs(unittest.TestCase):
 def test_formats(self):
  s=['No finding here.','The liver nodule measures 9 mm.']
  for obj in [2,'2',2.0,{'sentence_id':'2'},[{'sentence_id':2,'value':9}]]:self.assertEqual(locate(obj,s)[0],[2])
  self.assertEqual(locate(2.8,s)[0],[])
 def test_relocation(self):
  self.assertEqual(locate({'sentence_id':1,'quote':'liver nodule measures 9 mm'},['No finding here.','The liver nodule measures 9 mm.'],True)[0],[2])
 def test_wrong_quote(self):
  self.assertEqual(locate({'sentence_id':1,'quote':'The nodule measures 9 mm'},['The nodule measures 12 mm.'],True)[0],[])
 def test_negation_not_vetoed(self):
  s=['AFP positivity is excluded; after repeat assessment the result is unequivocally negative.']
  x=normalize({'facts':{'afp':'negative'},'evidence':{'afp':[{'sentence_id':1,'quote':s[0]}]},'status':{'afp':'known'}},s,['afp'],set(),True)
  self.assertEqual(x['facts']['afp'],'negative')
 def test_types(self):
  self.assertEqual(normalize_value('9',True),9);self.assertIsNone(normalize_value(-1,False));self.assertIsNone(normalize_value(320,False))
 def test_bounds(self):
  r={'mode':'all','conditions':[{'field':'t','op':'gt','value':50}]}
  self.assertEqual(evaluate(r,{'t':{'gt':50}}),'T');self.assertEqual(evaluate(r,{'t':{'ge':50}}),'U');self.assertEqual(evaluate(r,{'t':{'le':50}}),'F')
if __name__=='__main__':unittest.main()
