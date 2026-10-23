import unittest
from prepare import atom,evaluate,complementary,execute,values,build
class RulesTest(unittest.TestCase):
 def test_boundaries(self):
  self.assertFalse(evaluate('<',6,6));self.assertTrue(evaluate('≥',6,6));self.assertTrue(evaluate('≤',6,6));self.assertFalse(evaluate('>',6,6))
 def test_complement(self):
  self.assertTrue(complementary('<','≥'));self.assertFalse(complementary('<','>'))
 def test_scalar(self):
  a=atom('<6 mm');self.assertEqual(a['unit'],'mm');self.assertIsNone(atom('size <6 mm and ≥2 mm'))
 def test_event(self):
  a=atom('<6 mo from completion of primary therapy');h=values('A 65 year old relapsed 4 months after completing primary therapy.',a);self.assertEqual([x['value'] for x in h],[4]);self.assertEqual(values('CT every 4 months.',a),[])
 def test_measurement_ambiguity(self):
  self.assertEqual(len(values('A nodule measured 4 mm and a mass measured 7 mm.',atom('<6 mm'))),2)
 def test_flip_and_preserve(self):
  ps=[['root','<6 mm','A'],['root','≥6 mm','B']];pair=['<6 mm','≥6 mm']
  self.assertEqual(execute(ps,ps[0],1,pair,4,6)['action'],'A');self.assertEqual(execute(ps,ps[0],1,pair,4,3)['action'],'B');self.assertEqual(execute(ps,ps[0],1,pair,4,5)['action'],'A')
 def test_missing_downstream(self):
  ps=[['root','<6 mm','A'],['root','≥6 mm','unknown status','B']]
  self.assertIsNone(execute(ps,ps[0],1,['<6 mm','≥6 mm'],4,3)['action'])
 def test_preserve_same_action(self):
  ps=[['root','<6 mm','A'],['root','≥6 mm','A']]
  self.assertEqual(execute(ps,ps[0],1,['<6 mm','≥6 mm'],4,3)['action'],'A')
 def test_end_to_end(self):
  ps={'tree':[['root','<6 mm','A'],['root','≥6 mm','B']]};r={'disease':'tree','path':ps['tree'][0],'profile':'A lung nodule measuring 4 mm.','answer':'A','answer_text':'A','options':['A','B']}
  _,pairs,cs,audit=build([r],ps);self.assertEqual(len(pairs),1);self.assertEqual(len(cs),1);self.assertEqual(cs[0]['new_gold'],'B');self.assertTrue(cs[0]['label_flipped']);self.assertTrue(cs[0]['preserving_label_ok'])
if __name__=='__main__':unittest.main()
