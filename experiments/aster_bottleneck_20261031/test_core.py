import unittest,copy
from types import SimpleNamespace
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from src.aster.manuscript_runtime import ManuscriptBackbone,PreparedPrompt
from .routing import direct_prompt,route
from .advanced import corrupt_route
from .runtime import attention_control,selective_recompute,cache_copy
class LogicTests(unittest.TestCase):
 def test_direct_prompt_is_action_blind(self):
  case={'patient':'Age 61 years.','question':'secret original question','gold_rules':['R0Y']}
  group={'ancestor_conditions':[],'rules':[{'rule_id':'R0Y','condition':'Age at least 65?（判断为是）','action':'secret action A'},{'rule_id':'R0N','condition':'Age at least 65?（判断为否）','action':'secret action B'}]}
  before=direct_prompt(case,group)
  case['question']='changed';case['gold_rules']=['R0N'];group['rules'][0]['action']='changed action'
  self.assertEqual(before,direct_prompt(case,group));self.assertNotIn('secret',before)
  self.assertIn('Unknown',before);self.assertIn('Age 61',before)
 def test_unknown_retains_ancestor(self):
  c={'guideline':{'groups':[{'group_id':'G0'},{'group_id':'G1','parent_rule_id':'R0Y'}]}}
  path,_=route(c,{'G0':{'prediction':'R0Y','margin':1},'G1':{'prediction':'U'}})
  self.assertEqual(path,['R0Y'])
 def test_corruption(self):
  c={'case_id':'x','gold_rules':['R0Y','R1N']}
  self.assertEqual(corrupt_route(c,0,'wrong',13),c['gold_rules'])
  self.assertEqual(corrupt_route(c,1,'wrong',13),['R0N'])
  self.assertEqual(corrupt_route(c,1,'stop',13),[])
class AttentionTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(3)
  cfg=Qwen3Config(vocab_size=50,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,attention_dropout=0.,use_sliding_window=False)
  self.model=Qwen3ForCausalLM(cfg).eval();self.model.config._attn_implementation='eager'
  self.back=ManuscriptBackbone(self.model,model_id='tiny',model_revision='test')
  self.prompt=PreparedPrompt('test',torch.tensor([[1,2,3,4,5,6,7,8]]),{'r':(2,3)},6,'test')
  self.prefix=self.back.prefill(self.prompt)
 def test_zero_and_cache_isolation(self):
  expected=self.back._forward(self.prompt.input_ids[:,6:],cache=cache_copy(self.prefix.cache),start=6,use_cache=True).logits
  with attention_control(self.back):got=self.back._forward(self.prompt.input_ids[:,6:],cache=cache_copy(self.prefix.cache),start=6,use_cache=True).logits
  torch.testing.assert_close(got,expected,atol=1e-6,rtol=1e-5);self.assertEqual(self.prefix.cache.get_seq_length(),6)
 def test_all_token_recompute_matches(self):
  mask={'path':list(range(6)),'nonpath':[]}
  result,_=selective_recompute(self.back,self.prefix,mask,1.)
  for a,b in zip(self.prefix.cache.to_legacy_cache(),result.cache.to_legacy_cache()):
   for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=1e-6,rtol=1e-5)
 def test_hard_mask_zero_mass(self):
  mask={'path':[2],'siblings':[3],'subtree':[4],'nonpath':[3,4],'patient':[6,7],'sink':[0,1]}
  with attention_control(self.back,mask=mask,hard=True,capture_at=7) as trace:
   self.back._forward(self.prompt.input_ids[:,6:],cache=cache_copy(self.prefix.cache),start=6,use_cache=True)
  self.assertEqual(len(trace),4)
  self.assertTrue(all(t['mass_siblings']==0 and t['mass_subtree']==0 for t in trace))


class ExtendedTests(unittest.TestCase):
 def test_joint_beam_probabilities(self):
  from .routing import beam_routes
  c={'guideline':{'groups':[{'group_id':'G0'},{'group_id':'G1','parent_rule_id':'R0Y'}]}}
  ds={'G0':{'probs':{'R0Y':.6,'R0N':.3,'NONE':.1}},'G1':{'probs':{'R1Y':.8,'R1N':.2,'NONE':0}}}
  rs=beam_routes(c,ds,3);self.assertEqual(rs[0]['path'],['R0Y','R1Y']);self.assertAlmostEqual(rs[0]['weight'],.48/.9)
 def test_erasure_coordinate_and_original_cache(self):
  from .runtime import erase_cached_keys
  from src.aster.manuscript_core import rotate
  setup=AttentionTests();setup.setUp();back=setup.back;prefix=setup.prefix
  axis=torch.zeros(8,dtype=torch.double);axis[0]=1.
  engine=SimpleNamespace(leace_maps={(0,0):{'mean':torch.zeros(8,dtype=torch.double),'a':axis,'b':axis,'random':{}}})
  original=prefix.cache.layers[0].keys.clone();cache=cache_copy(prefix.cache)
  cache=erase_cached_keys(engine,cache,setup.prompt,prefix,{'erase_heads':[[0,0]]})
  ids=[2,3];pre=rotate(cache.layers[0].keys[0,0,ids].double(),prefix.cos[ids],-prefix.sin[ids])
  self.assertLess(float(pre[:,0].abs().max()),1e-6);torch.testing.assert_close(original,prefix.cache.layers[0].keys)

if __name__=='__main__':unittest.main()
