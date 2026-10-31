import copy,unittest
from types import SimpleNamespace
import torch
from .test_core import AttentionTests
from .runtime import cache_copy
from .module_one import first_error_depth,reference_gains,choose_config,validation_specs
from .module_one_diagnostic import apply_pair_patch
from src.aster.manuscript_core import rotate

class SelectionTests(unittest.TestCase):
 def test_first_error_keeps_every_case(self):
  c={'gold_rules':['R0Y','R1N','R2Y']}
  self.assertEqual(first_error_depth(c,{'score':{'path':['R0Y']}}),1)
  self.assertEqual(first_error_depth(c,{'score':{'path':c['gold_rules']}}),2)
 def test_reference_uses_each_case_and_correct_headset(self):
  rs=[{'result':{'trace':[{'layer':1,'kv_head':0,'theory_unit_square_mean':x},{'layer':2,'kv_head':0,'theory_unit_square_mean':100.}]}} for x in [1.,7.]]
  g=reference_gains(rs,{'one':[[1,0]]},2.)
  self.assertAlmostEqual(g['one']['gain'],1.)
 def test_freeze_ties_and_common_direction_controls(self):
  specs=[{'name':'a','kind':'mean','gain':2.,'direction_heads':[[1,0]],'unit_directions':True,'full_rule_keys':True,'deeper':True},
   {'name':'b','kind':'mean','gain':1.,'direction_heads':[[1,0]],'unit_directions':True,'full_rule_keys':True,'deeper':True}]
  best=choose_config({'a':{'path':5,'margin':2.},'b':{'path':5,'margin':2.}},specs)
  self.assertEqual(best['name'],'b');arms=validation_specs(best)
  self.assertEqual(len(arms),7)
  self.assertTrue(all(a.get('direction_heads')==[[1,0]] and abs(a['gain'])==1. for a in arms[1:]))
  self.assertEqual([a['random_seed'] for a in arms if 'random_seed' in a],[13,37,73])
  self.assertEqual(next(a for a in arms if a['name']=='reverse_mean')['kind'],'mean')
  self.assertTrue(all('deeper' not in a for a in arms))

class PatchTests(unittest.TestCase):
 def setUp(self):
  obj=AttentionTests();obj.setUp();self.prefix=obj.prefix;self.prompt=obj.prompt
  self.ids=[2,3];k=obj.prefix.cache.layers[0].keys
  pre=rotate(k[0,0,self.ids].double(),self.prefix.cos[self.ids],-self.prefix.sin[self.ids])
  value=obj.prefix.cache.layers[0].values[0,0,self.ids].clone()
  token_ids=obj.prompt.input_ids[0,self.ids].tolist()
  self.rec={'ids':token_ids,'heads':{(0,0):{'key':pre,'value':value}}}
  self.donor={'ids':token_ids,'heads':{(0,0):{'key':pre+.1,'value':value+.3}}}
  unit=torch.zeros(8,dtype=torch.double);unit[0]=1.;self.means={(0,0):unit}
 def run_mode(self,mode,prompt=None,prefix=None,recipient=None,donor=None):
  cache=cache_copy(self.prefix.cache)
  return apply_pair_patch(cache,prompt or self.prompt,prefix or self.prefix,'r',recipient or self.rec,donor or self.donor,self.means,mode,1)
 def test_k_v_separation_and_original_unchanged(self):
  before=self.prefix.cache.layers[0].keys.clone();before_v=self.prefix.cache.layers[0].values.clone()
  k=self.run_mode('key');v=self.run_mode('value');both=self.run_mode('kv')
  torch.testing.assert_close(k.layers[0].values,before_v)
  torch.testing.assert_close(v.layers[0].keys,before)
  torch.testing.assert_close(both.layers[0].keys,k.layers[0].keys)
  torch.testing.assert_close(both.layers[0].values,v.layers[0].values)
  torch.testing.assert_close(self.prefix.cache.layers[0].keys,before)
  untouched=[0,1,4,5];torch.testing.assert_close(k.layers[0].keys[:,:,untouched],before[:,:,untouched])
 def test_true_key_copy_is_exact_without_roundtrip(self):
  self.donor['heads'][0,0]['rotated_key']=torch.full_like(self.prefix.cache.layers[0].keys[0,0,self.ids],.123)
  self.donor['positions']=self.ids
  got=self.run_mode('key')
  self.assertTrue(torch.equal(got.layers[0].keys[0,0,self.ids],self.donor['heads'][0,0]['rotated_key']))
 def test_delta_is_rotated_at_online_positions(self):
  # Source keys are pre-RoPE; moving the target token positions must change rotation.
  prompt=copy.copy(self.prompt);prompt.positions={'r':(4,5)};prompt.input_ids=prompt.input_ids.clone()
  prompt.input_ids[0,4:6]=prompt.input_ids[0,2:4].clone()
  got=self.run_mode('delta',prompt=prompt)
  delta=self.donor['heads'][0,0]['key']-self.rec['heads'][0,0]['key']
  expected=self.prefix.cache.layers[0].keys[0,0,[4,5]].double()+rotate(delta,self.prefix.cos[[4,5]],self.prefix.sin[[4,5]])
  torch.testing.assert_close(got.layers[0].keys[0,0,[4,5]],expected.float(),atol=1e-6,rtol=1e-5)
 def test_mean_preserves_per_token_true_difference_norm(self):
  got=self.run_mode('mean');before=self.prefix.cache.layers[0].keys[0,0,self.ids].double()
  actual=got.layers[0].keys[0,0,self.ids].double()-before
  expected=(self.donor['heads'][0,0]['key']-self.rec['heads'][0,0]['key']).norm(dim=-1)
  torch.testing.assert_close(actual.norm(dim=-1),expected,atol=1e-6,rtol=1e-5)
if __name__=='__main__':unittest.main()
