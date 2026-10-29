import unittest,torch
from unittest.mock import Mock
from types import SimpleNamespace
from engine import Engine
import test_engine
class Controls(unittest.TestCase):
 def test_answer_without_gold(self):
  e=Engine.__new__(Engine);e.render=Mock(return_value=([1],[None]));e.tok=SimpleNamespace(encode=lambda text,**kw:[2] if text==' A' else [3]);e.forward=Mock(return_value=torch.tensor([[0.,0.,3.,1.]]))
  self.assertEqual(e.answer({'options':['x','y']})[0],0)
  self.assertIsNone(e.render.call_args.args[2])
 def test_mapping_without_gold(self):
  c={'rules':[{'rid':0,'action':'x'},{'rid':1,'action':'y'},{'rid':2,'action':'x'}],'options':['y','x']}
  pred,p=Engine.direct_map(c,[.3,.4,.3]);self.assertEqual(pred,1);self.assertAlmostEqual(p[1],.6)
 def test_scope_and_blind_independent(self):
  t=test_engine.Tests();t.setUp();e=t.e
  roles=[None,{'kind':'condition','rules':[1]},{'kind':'action','rules':[1]},{'kind':'option','rules':[1]},None]
  q=torch.zeros(1,1,5,5);v=torch.eye(5).reshape(1,1,5,5)
  mask=torch.zeros(1,1,5,5).masked_fill(torch.ones(5,5,dtype=torch.bool).triu(1),float('-inf'))
  def run(**kw):
   with e.intervene(roles,4,'hard',[.99,.01],**kw):return e.mod.eager_attention_forward(SimpleNamespace(layer_idx=0),q,q,v,mask,scaling=1.)[0][0,:,0]
  a=run();self.assertGreater(float(a[3,2]),0);self.assertEqual(float(a[4,2]),0);self.assertGreater(float(a[4,3]),0)
  b=run(gate_phase='all',gate_options=True,gate_scope='rule');self.assertEqual(float(b[4,1:4].sum()),0)
  c=run(blind=True);self.assertEqual(float(c[4,2]),0)
if __name__=='__main__':unittest.main()
