import unittest
import torch
import torch.nn.functional as F
from mass_operator import attention,redistribute,action_scores,group_roles
class Tests(unittest.TestCase):
 def test_conservation_masks_orphans(self):
  torch.manual_seed(0)
  g=torch.tensor([0,0,1,1,-1,2,-1]);x=torch.randn(2,3,7,7)
  allowed=torch.ones_like(x,dtype=torch.bool).tril();allowed[...,0,:]=False
  p=x.masked_fill(~allowed,float('-inf')).softmax(-1).nan_to_num()
  for a in (.0,.25,.5,1.):
   out=redistribute(p,g,torch.tensor([.1,.8,.1]),a,allowed)
   torch.testing.assert_close(out.sum(-1),p.sum(-1))
   torch.testing.assert_close(out[...,g<0],p[...,g<0])
   torch.testing.assert_close(out[...,g>=0].sum(-1),p[...,g>=0].sum(-1))
   self.assertTrue((out[~allowed]==0).all());self.assertTrue(torch.isfinite(out).all())
   if a==1:
    ratio=out[...,6,g==1].sum(-1)/out[...,6,g>=0].sum(-1)
    torch.testing.assert_close(ratio,torch.full_like(ratio,.8))
 def test_zero_and_gqa_long(self):
  torch.manual_seed(0);n=513
  q=torch.randn(1,4,n,8);k=torch.randn(1,2,n,8);v=torch.randn_like(k)
  m=torch.zeros(1,1,n,n).masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),float('-inf'))
  m[...,9]=-float('inf');g=torch.arange(n)%4-1;s=[.2,.3,.5]
  out=attention(q,k,v,m,8**-.5,g,s)
  ref=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=m)
  torch.testing.assert_close(out,ref,atol=1e-6,rtol=1e-5)
  a=attention(q,k,v,m,8**-.5,g,s,alpha=.5,chunk=37)
  b=attention(q,k,v,m,8**-.5,g,s,alpha=.5,chunk=128)
  torch.testing.assert_close(a,b)
 def test_multi_rule_and_orphan(self):
  c={'rules':[{'rid':0,'action':'x'},{'rid':1,'action':'x'},{'rid':2,'action':'y'}]}
  a,s=action_scores(c,[.2,.4,.4]);self.assertEqual(a,['x','y']);self.assertAlmostEqual(s[0],.6)
  roles=[{'kind':'option','rules':[0,1]},{'kind':'option','rules':[]},{'kind':'condition','rules':[0]}]
  self.assertEqual(group_roles(c,roles),[0,-1,-1])
 def test_zero_signal(self):
  p=torch.tensor([[[.2,.3,.5]]]);g=torch.tensor([0,1,-1]);mask=torch.ones_like(p,dtype=torch.bool)
  torch.testing.assert_close(redistribute(p,g,torch.zeros(2),1.,mask),p)
 def test_cross_layer_indirect(self):
  torch.manual_seed(0);x=torch.randn(1,1,9,8);y=x.clone();y[...,2:4,:]+=7
  act=torch.zeros(9,dtype=torch.bool);act[2:4]=True
  causal=torch.ones(9,9,dtype=torch.bool).tril();blind=causal&~((~act)[:,None]&act[None,:])
  for _ in range(3):
   x=x+attention(x,x,x,blind[None,None],8**-.5,[-1]*9,[])
   y=y+attention(y,y,y,blind[None,None],8**-.5,[-1]*9,[])
   torch.testing.assert_close(x[...,~act,:],y[...,~act,:])
if __name__=='__main__':unittest.main()
