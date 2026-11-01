import copy
import unittest
from types import SimpleNamespace

import torch

from .data import swap_actions, deployment
from .intervention import ConditionalIntervention, objective
from .engine import copy_cache, intervention_hook, fill_action
from experiments.aster_bottleneck_20261031.routing import direct_prompt


class InterventionTests(unittest.TestCase):
    def test_zero_strength_exact_identity(self):
        module = ConditionalIntervention(32)
        x = torch.randn(1,1,32)
        y, _, _ = module(x, torch.randn_like(x), force_zero=True)
        self.assertTrue(torch.equal(x, y))

    def test_swap_preserves_blind_input_and_graph(self):
        group = {'group_id':'G0','rules':[
            {'rule_id':'R0Y','condition':'Age above 60?（判断为是）','action':'L1：first action'},
            {'rule_id':'R0N','condition':'Age above 60?（判断为否）','action':'L2：second action'}]}
        case = {'case_id':'test','source_id':'source','patient':'Age 70.','question':'question',
                'guideline':{'groups':[group]},'gold_rules':['R0Y'],'gold_leaf':'L1'}
        swapped = swap_actions(case)
        self.assertEqual(direct_prompt(case,group),direct_prompt(swapped,swapped['guideline']['groups'][0]))
        self.assertEqual(swapped['guideline']['groups'][0]['rules'][0]['action'],'L1：second action')
        self.assertEqual(case['gold_rules'],swapped['gold_rules'])
        self.assertNotIn('gold_rules',deployment(case))
        self.assertEqual(case['guideline']['groups'][0]['rules'][0]['action'],'L1：first action')
        result = fill_action({'text':'PATH: R0Y\nFINAL: L1'},swapped['guideline'])
        self.assertEqual(result['action'],'second action')

    def test_backward_only_new_parameters_and_cache_isolation(self):
        from experiments.aster_bottleneck_20261031.test_core import AttentionTests
        fixture=AttentionTests();fixture.setUp()
        model=fixture.model;model.requires_grad_(False)
        back=fixture.back;ids=fixture.prompt.input_ids
        with torch.no_grad():
            prefix=back._forward(ids[:,:6],start=0,use_cache=True).past_key_values
        before=[(k.clone(),v.clone()) for k,v in prefix.to_legacy_cache()]
        module=ConditionalIntervention(32);diag={}
        with torch.no_grad():
            clean=back._forward(ids[:,6:7],cache=copy_cache(prefix),start=6,use_cache=True)
            with intervention_hook(model,0,module,torch.randn(32),{},force_zero=True):
                zero=back._forward(ids[:,6:7],cache=copy_cache(prefix),start=6,use_cache=True)
        torch.testing.assert_close(clean.logits,zero.logits,atol=0,rtol=0)
        with intervention_hook(model,0,module,torch.randn(32),diag):
            out=back._forward(ids[:,6:7],cache=copy_cache(prefix),start=6,use_cache=True)
        out.logits.float().square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for old,new in zip(before,prefix.to_legacy_cache()):
            for a,b in zip(old,new):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(prefix.get_seq_length(),6)

    def test_ordinary_has_no_blind_dependency_and_loss_is_finite(self):
        module=ConditionalIntervention(32,variant='ordinary');x=torch.randn(1,1,32)
        a,mag,_=module(x);b,_,_=module(x,torch.randn_like(x))
        torch.testing.assert_close(a,b)
        logits=[torch.tensor([1.,-1.],requires_grad=True),torch.tensor([.5,0.],requires_grad=True)]
        loss,parts=objective(logits,[torch.tensor([1.,0.]),torch.tensor([0.,1.])],torch.tensor(0),[mag,mag])
        self.assertTrue(torch.isfinite(loss));loss.backward();self.assertGreater(parts['js'],0)


if __name__=='__main__':unittest.main()
