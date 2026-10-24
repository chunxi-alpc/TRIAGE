"""Regression checks for matched denominators and auditable token-prefix evidence."""
import copy
import unittest
from complete800_comparison import paired
from analyze_cad18_trace import case_evidence,digest

class CompletedResearchTests(unittest.TestCase):
 def test_pair_direction_with_shuffled_rows(self):
  rows=[dict(id='a',method='visible',correct=False,disease='aml_7'),dict(id='b',method='visible',correct=True,disease='aml_7'),
   dict(id='b',method='hidden',correct=True,disease='aml_7'),dict(id='a',method='hidden',correct=True,disease='aml_7')]
  r=paired(rows,'visible','hidden');self.assertEqual(r['n'],2);self.assertEqual(r['delta'],.5);self.assertEqual(r['wins'],1)
 def test_missing_pair_rejected(self):
  with self.assertRaises(ValueError):paired([dict(id='a',method='visible',correct=True,disease='aml_7')],'visible','hidden')
 def test_empty_pair_unknown_not_zero(self):
  self.assertIsNone(paired([],'visible','hidden')['delta'])
 def fixture(self):
  tracked={'1':dict(logc=-5.,logu=-20.,cad_score=10.,masked=True),'2':dict(logc=0.,logu=0.,cad_score=0.,masked=False),'3':dict(logc=-1.,logu=-1.,cad_score=-1.,masked=False)}
  first=dict(step=0,prefix_token_ids=[],candidate_ids=[2,3],unfiltered_choice=1,filtered_choice=2,conditional_logp_max=0.,threshold=-2.302585,tracked_tokens=tracked)
  def make(ids,pred,text):
   second=copy.deepcopy(first);second.update(step=1,prefix_token_ids=ids[:1],unfiltered_choice=3,filtered_choice=3)
   return dict(summary=dict(id='fixture',gold='A',prediction=pred),result=dict(token_ids=ids,completion=text,stop_reason='eos'),steps=[copy.deepcopy(first),second])
  a=make([1,3],'A','old');b=make([2,3],'B','filtered')
  expected=dict(outcome='lost',old_prediction='A',filtered_prediction='B',old_completion_sha256=digest('old'),filtered_completion_sha256=digest('filtered'))
  return a,b,expected
 def test_divergence_requires_same_prefix_and_reproduced_output(self):
  a,b,e=self.fixture();r=case_evidence(a,b,1,{3},e)
  self.assertEqual(r['first_divergence_step'],0);self.assertTrue(r['divergence_old_choice_masked']);self.assertTrue(r['historical_text_reproduced'])
  b['steps'][0]['conditional_logp_max']=.1
  with self.assertRaises(ValueError):case_evidence(a,b,1,{3},e)
 def test_prefix_and_mask_corruption_rejected(self):
  a,b,e=self.fixture();b['steps'][1]['prefix_token_ids']=[1]
  with self.assertRaises(ValueError):case_evidence(a,b,1,{3},e)
  a,b,e=self.fixture();b['steps'][0]['candidate_ids']=[1,2,3]
  with self.assertRaises(ValueError):case_evidence(a,b,1,{3},e)
 def test_historical_text_mismatch_rejected(self):
  a,b,e=self.fixture();b['result']['completion']='different'
  with self.assertRaises(ValueError):case_evidence(a,b,1,{3},e)
if __name__=='__main__':unittest.main()
