import unittest,torch
from transformers import LlamaConfig,LlamaForCausalLM
from transformers.models.llama import modeling_llama
from engine import Engine
class Tests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(4);self.e=Engine.__new__(Engine);self.e.nl=3;self.e.mod=modeling_llama
  self.m=LlamaForCausalLM(LlamaConfig(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,attn_implementation='eager')).eval()
  self.x=torch.tensor([[1,2,3,4,5,6,7,8]]);self.roles=[None,None,[0],[0],None,[1],None,None]
 def run_model(self,x,mode,phase=8):
  with self.e.intervene(self.roles,phase,mode,[.99,.01]),torch.no_grad():return self.m(x,use_cache=False).logits
 def test_all_query_indirect_leak(self):
  x=self.x.clone();x[0,2:4]=torch.tensor([20,21]);x[0,5]=22
  a=self.run_model(self.x,'blind');b=self.run_model(x,'blind')
  self.assertTrue(torch.equal(a[:,[0,1,4,6,7]],b[:,[0,1,4,6,7]]))
 def test_identity_sdpa(self):
  with torch.no_grad():a=self.m(self.x).logits
  b=self.run_model(self.x,'none');self.assertTrue(torch.allclose(a,b,atol=1e-6))
 def test_phase_gate_finite(self):
  a=self.run_model(self.x,'hard',6);self.assertTrue(torch.isfinite(a).all())
 def test_restore_after_error(self):
  old=self.e.mod.eager_attention_forward
  with self.assertRaises(RuntimeError):
   with self.e.intervene(self.roles,8,'blind'):raise RuntimeError('test')
  self.assertIs(self.e.mod.eager_attention_forward,old)
 def test_deleted_tokens_preserved_positions(self):
  keep=[i for i,r in enumerate(self.roles) if r is None]
  a=self.run_model(self.x,'blind')[:,-1]
  with self.e.intervene([None]*len(keep),len(keep),'none'),torch.no_grad():
   b=self.m(self.x[:,keep],position_ids=torch.tensor([keep]),attention_mask=torch.ones(1,len(keep)),use_cache=False).logits[:,-1]
  self.assertTrue(torch.allclose(a,b,atol=1e-6))
 def test_log_gate_reduces_low_probability_attention(self):
  from types import SimpleNamespace
  roles=[None,[0],[1],None];q=torch.zeros(1,1,4,4);k=q.clone();v=torch.eye(4).reshape(1,1,4,4);mask=torch.zeros(1,1,4,4).masked_fill(torch.ones(4,4,dtype=torch.bool).triu(1),float('-inf'))
  values=[]
  for lam in [0.,1.,2.]:
   with self.e.intervene(roles,3,'gate',[.99,.01],lam):
    o,_=self.e.mod.eager_attention_forward(SimpleNamespace(layer_idx=0),q,k,v,mask,scaling=1.)
   values.append(float(o[0,3,0,2]))
  self.assertGreater(values[0],values[1]);self.assertGreater(values[1],values[2])
if __name__=='__main__':unittest.main()
