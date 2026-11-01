import unittest
from types import SimpleNamespace
import torch

from .comparators import BlindReadout, lookup
from .data import deployment, swap_actions
from .engine import Engine
from .intervention import ConditionalIntervention


class ComparatorTests(unittest.TestCase):
    def case(self):
        return {'case_id':'test','source_id':'test','patient':'Age 70.','question':'Follow the guideline.',
                'guideline':{'groups':[{'group_id':'G0','rules':[
                    {'rule_id':'R0Y','condition':'Age above 60?','action':'L1：first action'},
                    {'rule_id':'R0N','condition':'Age at most 60?','action':'L2：second action'}]}]},
                'gold_rules':['R0N'],'gold_leaf':'L2'}

    def engine(self):
        e=Engine.__new__(Engine)
        e.probabilities=lambda case,gid:torch.tensor([.2,.1,.7])
        e.builder=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text,**kwargs:list(text)))
        e.old=SimpleNamespace(freeze={'max_new_tokens':512})
        return e

    def test_hint_and_lookup_use_no_gold_and_swaps_only_change_action(self):
        e=self.engine();case=self.case();public=deployment(case);swapped=deployment(swap_actions(case))
        self.assertEqual(e.with_hint(public,'G0')['question'],e.with_hint(swapped,'G0')['question'])
        self.assertNotIn('gold_rules',e.with_hint(public,'G0'))
        original=lookup(e,public);changed=lookup(e,swapped)
        self.assertEqual(original['path'],['R0Y'])
        self.assertEqual(original['path'],changed['path'])
        self.assertEqual(original['action'],'first action');self.assertEqual(changed['action'],'second action')

    def test_readout_starts_at_base_and_matches_parameter_count(self):
        module=BlindReadout(4096);full=ConditionalIntervention(4096)
        self.assertEqual(sum(p.numel() for p in module.parameters()),sum(p.numel() for p in full.parameters()))
        base=torch.tensor([1.,2.]);edited,magnitude=module(torch.randn(4096),base)
        torch.testing.assert_close(edited,base,rtol=0,atol=0)
        self.assertEqual(float(magnitude),0.)
        torch.nn.functional.cross_entropy(edited[None],torch.tensor([0])).backward()
        self.assertGreater(float(module.output.weight.grad.abs().sum()),0.)

    def test_context_adapter_does_not_take_hidden_blind_input(self):
        module=ConditionalIntervention(32,variant='context');x=torch.randn(1,1,32)
        a,_,_=module(x,None);b,_,_=module(x,torch.randn_like(x))
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
