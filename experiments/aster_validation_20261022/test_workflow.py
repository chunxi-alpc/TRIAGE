import sys,unittest,json,importlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run import prompt_spec
from analyze import wilson,paired,corrupt_ids,option_prediction
class Tests(unittest.TestCase):
 def test_legacy_option_prediction(self):
  self.assertEqual(option_prediction({'prediction':'B'}),'B')
  self.assertEqual(option_prediction({'prediction':'B','score_prediction':None}),'B')
  self.assertEqual(option_prediction({'prediction':'E','score_prediction':'D'}),'D')
 def test_wilson(self):
  self.assertEqual(wilson(0,0),[None,None]);lo,hi=wilson(7,20);self.assertAlmostEqual(lo,.1812,places=3);self.assertAlmostEqual(hi,.5671,places=3)
 def test_exact_corruption_nested_shared(self):
  ids=[str(i) for i in range(100)]
  self.assertEqual(len(corrupt_ids(ids,.1,11)),10)
  self.assertLessEqual(corrupt_ids(ids,.1,11),corrupt_ids(ids,.2,11))
  self.assertEqual(corrupt_ids(ids,.1,11),corrupt_ids(list(reversed(ids)),.1,11))
 def test_pairing_identity_and_mcnemar(self):
  a=[dict(id=str(i),disease='x' if i<5 else 'y',correct=False,margin_prior=0.,margin_sibling=0.) for i in range(10)]
  b=[dict(r,correct=True,margin_prior=1.) for r in reversed(a)]
  r=paired(a,b,reps=100);self.assertEqual(r['corrected'],10);self.assertAlmostEqual(r['mcnemar_exact_p'],2/1024);self.assertEqual(r['disease_bootstrap_ci'],[1,1]);self.assertEqual(r['margin_deltas']['margin_prior']['mean_delta'],1.)
  with self.assertRaises(ValueError):paired(a,b[:-1])
 def test_annotation_pruning_and_wrong_rule(self):
  r={'prompt_text':'IF the clinical sequence is: fit\nTHEN one\n\nIF the clinical sequence is: unfit\nTHEN two\n\nPatient: X','applicable_char_span':[0,39],'sibling_char_span':[41,82],'applicable_rule_text':'fit THEN one','sibling_rule_text':'unfit THEN two','closed_book_prompt':'Patient: X','variant':'B_applicable_rule_changed'}
  txt,_,_=prompt_spec(r,'annotate');self.assertTrue(txt.startswith(r['prompt_text']));self.assertIn('rule 1',txt)
  txt,_,_=prompt_spec(r,'annotate',True);self.assertIn('rule 2',txt)
  txt,_,_=prompt_spec(r,'prune',True);self.assertIn('unfit THEN two',txt);self.assertNotIn('fit THEN one',txt)
if __name__=='__main__':unittest.main()
