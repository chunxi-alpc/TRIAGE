import unittest,json,contextlib
from pathlib import Path
import torch
from types import SimpleNamespace
from run import branch_prompt,parse
from attention import block
class MechanismTest(unittest.TestCase):
 def test_real_pair_invariants(self):
  p=Path('/home/gcx/aster_validation_20261022/private/run500/mechanism');test=list(map(json.loads,(p/'test.jsonl').read_text().splitlines()));dev=list(map(json.loads,(p/'dev.jsonl').read_text().splitlines()));self.assertEqual(len(test),500);self.assertFalse({r['id'] for r in test}&{r['id'] for r in dev})
  for c in test+dev:
   a=c['variant_rules']['A'];d=c['variant_rules']['D'];self.assertEqual([r['condition'] for r in a],[r['condition'] for r in d]);i,j=c['app_rule'],c['sib_rule'];self.assertEqual(d[i]['action'],a[j]['action']);self.assertEqual(d[j]['action'],a[i]['action']);self.assertEqual(branch_prompt(c,'A','hidden'),branch_prompt(c,'D','hidden'));self.assertNotIn('\nActions:',branch_prompt(c,'A','hidden'))
   for k in set(range(len(a)))-{i,j}:self.assertEqual(a[k],d[k])
   if 'CS' in c['variant_rules']:
    x=c['cs_validation']['value_mm'];self.assertTrue(6<=x<12);self.assertEqual([r['action'] for r in a],[r['action'] for r in c['variant_rules']['CS']]);self.assertEqual(c['gold_branch']['CS'],c['sib_branch'])
 def test_parser(self):
  self.assertEqual(parse('**Answer: O**',list('ABCDEFGHIJKLMNO')),'O');self.assertIsNone(parse('A patient has a fever',list('ABC')));self.assertIsNone(parse('Z',list('ABC')))
 def test_attention_block_exact_head_query(self):
  captured={}
  def native(module,q,k,v,mask,**kw):captured['mask']=mask;return mask
  modeling=SimpleNamespace(eager_attention_forward=native);q=torch.zeros(1,2,3,4);mask=torch.zeros(1,1,3,3)
  with block(modeling,1,[0],torch.tensor([0.,1.,0.])):modeling.eager_attention_forward(SimpleNamespace(layer_idx=1),q,q,q,mask,scaling=1)
  out=captured['mask'];self.assertLess(out[0,0,2,1].item(),-1e20);self.assertEqual(out[0,1,2,1].item(),0);self.assertEqual(out[0,0,1,1].item(),0);self.assertIs(modeling.eager_attention_forward,native)
if __name__=='__main__':unittest.main()
