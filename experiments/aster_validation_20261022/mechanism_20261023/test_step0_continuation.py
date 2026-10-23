import json
import tempfile
import unittest
from pathlib import Path
from step0_continuation import cancer, overlap, pair, clinical_inventory

class ContinuationTests(unittest.TestCase):
 def test_verified_disease_codes_with_irregular_separators(self):
  self.assertEqual(cancer('cns28'),'cns')
  self.assertEqual(cancer('cns_19'),'cns')
  self.assertEqual(cancer('nscl_3'),'nscl')
  with self.assertRaises(ValueError):cancer('unknown_2')
 def test_overlap_counts_cases_not_only_unique_trees(self):
  old=[dict(disease='a')];new=[dict(disease='a'),dict(disease='a'),dict(disease='b')]
  self.assertEqual(overlap(old,new,'disease')['new_cases_with_shared_value'],2)
  self.assertEqual(overlap(old,new,'disease')['common_unique'],1)
 def test_pair_direction_and_one_source_interval(self):
  scores={('a','A','m'):{'correct':False},('a','D','m'):{'correct':True},
   ('b','A','m'):{'correct':True},('b','D','m'):{'correct':True}}
  x=pair(['a','b'],scores,('A','m'),('D','m'),{'a':'aml_7','b':'aml_14'})
  self.assertEqual(x['delta'],.5);self.assertEqual(x['wins'],1)
  self.assertIsNone(x['publisher_ci']);self.assertIsNone(x['cancer_ci_descriptive'])
  self.assertIsNone(x['leave_one_cancer_out']['aml']['delta'])
 def test_missing_pair_not_silently_removed(self):
  with self.assertRaises(KeyError):pair(['a'],{},('A','m'),('D','m'),{'a':'aml_7'})
 def test_empty_group_not_zero_effect(self):
  x=pair([],{},('A','m'),('D','m'),{})
  self.assertEqual(x['n'],0);self.assertIsNone(x['delta'])
 def test_empty_or_partial_ratings_not_expert_review(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)
   for j in (1,2):(p/f'reviewer{j}.jsonl').write_text(json.dumps({'ratings':{'a':None,'b':None}})+'\n'+json.dumps({'ratings':{'a':'uncertain','b':None}})+'\n')
   r=clinical_inventory(p)
   self.assertEqual(r['files']['reviewer1']['items_with_any_rating'],1)
   self.assertEqual(r['files']['reviewer1']['items_with_all_rating_fields'],0)
   self.assertFalse(r['expert_identity_or_qualification_verified'])
if __name__=='__main__':unittest.main()
