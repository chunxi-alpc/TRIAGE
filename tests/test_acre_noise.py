"""Corruption plans stay deterministic and isolated from ordinary inference."""
import unittest
from scripts.run_acre_noise import plans
class NoisePlanTests(unittest.TestCase):
 def test_fixed_seed_and_nested_sets(self):
  a=plans();self.assertEqual(a,plans())
  sets={n:{x['item']['case_id'] for x in a if x['name']==n} for n in ['flip_0.1','flip_0.2','flip_0.3']}
  self.assertEqual([len(v) for v in sets.values()],[8,16,24])
  self.assertLess(sets['flip_0.1'],sets['flip_0.2']);self.assertLess(sets['flip_0.2'],sets['flip_0.3'])
 def test_only_declared_state_changes(self):
  for x in plans():
   if not x['name'].startswith('flip'):continue
   r=next(r for r in x['rules'] if r['rule_id']==x['change']['rule_id'])
   self.assertNotEqual(x['change']['old'],r['applicability'])
   self.assertTrue(x['change']['evidence_unchanged'])
if __name__=='__main__':unittest.main()
