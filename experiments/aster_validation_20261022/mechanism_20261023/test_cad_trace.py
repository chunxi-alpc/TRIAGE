import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from step0_cad_trace import step_evidence, instrument

class TraceTests(unittest.TestCase):
 def test_filter_uses_global_vocabulary_maximum(self):
  c=torch.tensor([-0.1,-4.,-3.,-6.]);u=torch.tensor([-0.1,-20.,-5.,-7.])
  x=step_evidence(c,u,1,{3})
  self.assertEqual(x['unfiltered_choice'],1)
  self.assertEqual(x['filtered_choice'],0)
  self.assertTrue(x['tracked_tokens']['1']['masked'])
  self.assertEqual(x['candidate_ids'],[0])
 def test_original_loop_captures_prefix_multi_eos_and_free_decode(self):
  code=instrument(Path(__file__).resolve().parents[1].joinpath('run.py').read_text())
  class Model:
   device='cpu';calls=0
   def __call__(self,**kwargs):
    step=self.calls//2;self.calls+=1
    z=torch.zeros(1,1,5);z[0,0,4 if step==10 else 1]=10
    return SimpleNamespace(logits=z,past_key_values=None)
  for mode in (False,True):
   m=Model();trace=[];ns=dict(torch=torch,model=m,tok=SimpleNamespace(eos_token_id=3,decode=lambda ids,**kw:','.join(map(str,ids))),
    trace=trace,filtered_mode=mode,gold_id=1,eos_ids={3,4},step_evidence=step_evidence,
    encode_prompt=lambda *a:({'input_ids':torch.ones((1,1),dtype=torch.long)},[]),
    parse_answer=lambda *a:'A',tokenids={'A':[1]})
   exec(compile(code,'trace-test','exec'),ns)
   r=ns['cad_score']('context','closed',{'A':'option'})
   self.assertEqual(r['stop_reason'],'eos');self.assertEqual(len(trace),11)
   self.assertEqual(trace[0]['prefix_token_ids'],[])
   self.assertEqual(trace[-1]['prefix_token_ids'],[1]*10)
   self.assertEqual(r['token_ids'][-1],4);self.assertEqual(m.calls,22)
 def test_source_drift_not_silently_ignored(self):
  with self.assertRaises(ValueError):instrument('def cad_score():\n return None')
if __name__=='__main__':unittest.main()
