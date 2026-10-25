import unittest
import torch
import torch.nn.functional as F
from sparse_attention import sparse_attention, group_bias

class SparseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.q = torch.randn(1, 4, 9, 8)
        self.k = torch.randn(1, 2, 9, 8)
        self.v = torch.randn_like(self.k)
        self.groups = [-1, 0, 0, 1, -1, 1, 2, -1, -1]
    def run_attn(self, **kw):
        args = dict(mask=None, scale=8**-.5, groups=self.groups, support=[.1,.8,.1],
                    heads=[0,2], query_start=6)
        args.update(kw)
        return sparse_attention(self.q,self.k,self.v,**args)
    def test_zero_matches_float_reference(self):
        ref=F.scaled_dot_product_attention(self.q,self.k.repeat_interleave(2,1),self.v.repeat_interleave(2,1),is_causal=True)
        self.assertTrue(torch.allclose(self.run_attn(strength=0.),ref,atol=3e-7))
    def test_only_selected_heads_and_answer_queries_change(self):
        a=self.run_attn(strength=0.);b=self.run_attn()
        self.assertTrue(torch.equal(a[:,:,:6],b[:,:,:6]))
        self.assertTrue(torch.equal(a[:,[1,3]],b[:,[1,3]]))
        self.assertGreater(float((a-b).abs().max()),1e-3)
    def test_mask_causal_and_fully_blocked(self):
        mask=torch.ones(1,1,9,9,dtype=torch.bool);mask[...,1]=False;mask[...,7,:]=False
        out,d=self.run_attn(mask=mask,return_details=True)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(float(d['after'][~d['allowed']].sum()),0.)
        self.assertEqual(float(out[:,:,7].abs().max()),0.)
        self.assertEqual(float(d['after'][...,0,7:].sum()),0.)
    def test_additive_mask_and_bool_equivalent(self):
        mask=torch.ones(1,1,9,9,dtype=torch.bool);mask[...,2]=False
        additive=torch.zeros_like(mask,dtype=torch.float32).masked_fill(~mask,float('-inf'))
        self.assertTrue(torch.allclose(self.run_attn(mask=mask),self.run_attn(mask=additive),atol=1e-6))
    def test_bound_and_nonaction(self):
        _,d=self.run_attn(strength=-.5,return_details=True)
        self.assertLessEqual(float(d['bias'].abs().max()),1.)
        self.assertEqual(float(d['bias'][...,torch.tensor(self.groups)<0].abs().max()),0.)
    def test_zero_support_no_effect(self):
        self.assertTrue(torch.equal(self.run_attn(support=[0,0,0]),self.run_attn(strength=0.)))
    def test_future_groups_do_not_affect_earlier_query(self):
        groups=[-1,0,0,0,-1,0,0,-1,1]
        a=self.run_attn(groups=groups,support=[1,0]);b=self.run_attn(groups=groups,support=[1,100])
        self.assertTrue(torch.equal(a[:,:,:8],b[:,:,:8]))
    def test_gqa_matches_explicit_replication(self):
        a=self.run_attn();self.k=self.k.repeat_interleave(2,1);self.v=self.v.repeat_interleave(2,1)
        self.assertTrue(torch.equal(a,self.run_attn()))
    def test_group_permutation_invariance(self):
        a=self.run_attn();mapping={-1:-1,0:2,1:0,2:1}
        b=self.run_attn(groups=[mapping[g] for g in self.groups],support=[.8,.1,.1])
        self.assertTrue(torch.allclose(a,b,atol=1e-6))
    def test_many_to_one_rules_options_orphans(self):
        from sparse_engine import action_scores,group_roles
        c={'rules':[{'rid':0,'action':'A'},{'rid':1,'action':'B'},{'rid':2,'action':'A'}],'options':['A','B','A','orphan']}
        actions,s=action_scores(c,[.2,.3,.5]);self.assertEqual(actions,['A','B']);self.assertAlmostEqual(s[0],.7)
        roles=[{'kind':'action','rules':[0]},{'kind':'option','rules':[0,2]},{'kind':'option','rules':[]},{'kind':'condition','rules':[1]}]
        self.assertEqual(group_roles(c,roles),[0,0,-1,-1])
    def test_current_mass_controls_dynamic_bias(self):
        p=torch.tensor([[[[.05,.05,.8,.1]]]])
        allowed=torch.ones_like(p,dtype=torch.bool)
        b=group_bias(p,allowed,[0,0,1,-1],[.8,.2],1.)
        self.assertGreater(float(b[...,0]),0);self.assertLess(float(b[...,2]),0)
        self.assertEqual(float(b[...,0]),float(b[...,1]))
    def test_invalid_head_and_cached_input(self):
        with self.assertRaises(ValueError):self.run_attn(heads=[4])
        self.k=self.k[:,:,:8]
        with self.assertRaises(ValueError):self.run_attn()
    def test_long_sequence_finite(self):
        self.q=torch.randn(1,4,513,8);self.k=torch.randn(1,2,513,8);self.v=torch.randn_like(self.k)
        self.assertTrue(torch.isfinite(self.run_attn(groups=[i%3 for i in range(513)],query_start=510)).all())

if __name__=='__main__':unittest.main()
