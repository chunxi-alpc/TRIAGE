import unittest, math
import torch
from src.aster.manuscript_core import edit_kv_layers, rotate
from .numeric import score_change, cluster_ci

class MechanismTests(unittest.TestCase):
 def test_rope_score_and_copy_isolation(self):
  torch.manual_seed(1)
  k=torch.randn(1,2,7,8,dtype=torch.float64);v=torch.randn_like(k)
  angles=torch.randn(7,4,dtype=torch.float64);angles=torch.cat((angles,angles),-1)
  cos,sin=angles.cos(),angles.sin();direction=torch.randn(8,dtype=torch.float64);direction/=direction.norm()
  original_k=k.clone();original_v=v.clone();positions={"yes":[1,2],"no":[4,5]};gains={"yes":.7,"no":-.7}
  edited,audit=edit_kv_layers([(k,v)],{(0,1):direction},positions,gains,cos,sin)
  q=torch.randn(4,8,dtype=torch.float64)
  for rid,ids in positions.items():
   observed=q@(edited[0][0][0,1,ids]-k[0,1,ids]).T/math.sqrt(8)
   expected=score_change(q,direction,cos[ids],sin[ids],gains[rid])
   self.assertLess(float((observed-expected).abs().max()),1e-10)
  self.assertTrue(torch.equal(k,original_k));self.assertTrue(torch.equal(v,original_v))
  self.assertTrue(torch.equal(edited[0][1],v));self.assertNotEqual(edited[0][1].data_ptr(),v.data_ptr())
  self.assertTrue(torch.equal(edited[0][0][0,0],k[0,0]))
  self.assertTrue(torch.equal(edited[0][0][0,1,[0,3,6]],k[0,1,[0,3,6]]))
  self.assertEqual(audit["changed_key_vectors"],4)
 def test_rope_uses_key_position(self):
  v=torch.tensor([1.,0.,0.,0.],dtype=torch.float64);q=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
  angles=torch.tensor([[0.,0.,0.,0.],[math.pi,math.pi,math.pi,math.pi]],dtype=torch.float64)
  x=score_change(q,v,angles.cos(),angles.sin(),1)
  self.assertTrue(torch.allclose(x,torch.tensor([[.5,-.5]],dtype=torch.float64)))
 def test_gqa_mapping(self):
  from transformers.models.qwen3.modeling_qwen3 import repeat_kv
  k=torch.arange(2,dtype=torch.float64).reshape(1,2,1,1)
  self.assertEqual(repeat_kv(k,4).reshape(-1).tolist(),[0,0,0,0,1,1,1,1])
 def test_cluster_denominator(self):
  r=cluster_ci([1.,-1.]);self.assertEqual(r["clusters"],2);self.assertEqual(r["mean"],0)
 def test_probe_matches_tiny_qwen3_eager(self):
  from transformers import Qwen3Config,Qwen3ForCausalLM
  from src.aster.manuscript_runtime import ManuscriptBackbone
  from .probe import probe
  cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,attention_dropout=0.)
  cfg._attn_implementation="eager"
  model=Qwen3ForCausalLM(cfg).eval()
  backend=ManuscriptBackbone(model,model_id="tiny-test",model_revision="synthetic-test-only")
  ids=torch.tensor([[1,3,5,7,9,11,13,15]])
  with backend.session():
   initial=backend._forward(ids[:,:5],use_cache=True)
   legacy=initial.past_key_values.to_legacy_cache()
   cos,sin=model.model.rotary_emb(torch.zeros(1,5,32),torch.arange(5)[None])
   unchanged,_=edit_kv_layers(legacy,{},{"r":[1,2]},{"r":0.},cos[0],sin[0])
   cache=type(initial.past_key_values).from_legacy_cache(tuple(unchanged))
   expected=backend._forward(ids[:,5:],cache=cache,start=5,use_cache=True).logits
   unchanged,_=edit_kv_layers(legacy,{},{"r":[1,2]},{"r":0.},cos[0],sin[0])
   cache=type(initial.past_key_values).from_legacy_cache(tuple(unchanged))
   with probe(backend,[(0,1)],{"r":[1,2]},{"r":0.},{},legacy,unchanged,{},cos[0],sin[0],5,7) as traces:
    actual=backend._forward(ids[:,5:],cache=cache,start=5,use_cache=True).logits
   self.assertTrue(torch.allclose(expected,actual,atol=1e-6,rtol=1e-6))
   self.assertEqual(len(traces),1)
   self.assertEqual(traces[0]["query_heads"],[2,3])
   self.assertTrue(torch.equal(traces[0]["rules"]["r"]["score_delta_actual"],torch.zeros(2,2)))
 def test_zero_edit(self):
  k=torch.randn(1,1,3,4,dtype=torch.float64);v=k.clone()
  e,a=edit_kv_layers([(k,v)],{},{"x":[1]},{"x":0},torch.ones(3,4),torch.zeros(3,4))
  self.assertTrue(torch.equal(e[0][0],k));self.assertTrue(a["skipped"])
if __name__=="__main__":unittest.main()
