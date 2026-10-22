"""CPU regression tests; tiny randomly initialized models are NOT pilot results."""
import importlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from prepare_dataset import build_example,VARIANTS
from run_pilot import signed_attention_patch,parse_answer,span_mask,encode_prompt

class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.paths={'d1':[['Stage I','Fit','Action one'],['Stage I','Unfit','Action two'],['Stage II','Fit','Action three']]}
        self.row={'disease':'d1','path':self.paths['d1'][0], 'profile':'A fit stage I patient.',
                  'options':['Action one','Action two','Action three'],'answer':'A','answer_text':'Action one',
                  'prompt':'DO NOT DUPLICATE ORIGINAL PROMPT'}
    def test_counterfactuals(self):
        a,b,c=build_example(self.row,self.paths,12,42)
        self.assertEqual([r['gold_answer'] for r in (a,b,c)],['A','C','A'])
        self.assertEqual(a['sibling_rule_text'],b['sibling_rule_text'])
        self.assertEqual(a['applicable_rule_text'],c['applicable_rule_text'])
        self.assertNotEqual(a['applicable_rule_text'],b['applicable_rule_text'])
        self.assertNotEqual(a['sibling_rule_text'],c['sibling_rule_text'])
        self.assertEqual([r['sibling_action_label'] for r in (a,b,c)],['B','B','C'])
        for r in (a,b,c):
            self.assertEqual(r['options'],a['options']);self.assertEqual(r['rule_count'],3)
            self.assertNotIn(self.row['prompt'],r['prompt_text'])
            for role in ('applicable','sibling'):
                s,e=r[role+'_char_span'];self.assertEqual(r['prompt_text'][s:e],r[role+'_rule_text'])
            self.assertIn('C: Action three',r['closed_book_prompt'])
            self.assertIn(r['applicable_rule_text'],r['direct_rule_prompt'])
            self.assertNotIn(r['sibling_rule_text'],r['direct_rule_prompt'])
    def test_reject_unaligned_gold(self):
        self.row['answer']='B'
        with self.assertRaisesRegex(ValueError,'gold'):build_example(self.row,self.paths,0,1)
    def test_reject_missing_source(self):
        with self.assertRaisesRegex(ValueError,'missing'):build_example(self.row,{},0,1)
    def test_no_invented_sibling(self):
        with self.assertRaisesRegex(ValueError,'sibling'):build_example(self.row,{'d1':[self.row['path']]},0,1)
    def test_no_collapsed_edit(self):
        self.row['options']=self.row['options'][:2]
        with self.assertRaisesRegex(ValueError,'third'):build_example(self.row,self.paths,0,1)
    def test_eight_options_and_parser(self):
        self.assertEqual(parse_answer('H','ABCDEFGH'),'H')
        for text in ('A patient','A or B','think A then C','Z',''):
            self.assertIsNone(parse_answer(text))
        self.assertIsNone(parse_answer('H','AB'))
        for text in ('Answer: B','**B**','**Answer: B**','B: treatment','B. treatment','B\n\nExplanation: text','The correct answer is B.'):
            self.assertEqual(parse_answer(text),'B')
    def test_span(self):
        self.assertEqual(span_mask([(0,0),(0,3),(3,5),(5,8)],(3,5)).tolist(),[0,0,1,0])

class AttentionTests(unittest.TestCase):
    def test_native_gqa_logits_mask_shape_and_restore(self):
        for arch in ('llama','qwen3'):
            with self.subTest(arch=arch):
                m=importlib.import_module(f'transformers.models.{arch}.modeling_{arch}')
                module=torch.nn.Module();module.num_key_value_groups=2;module.eval()
                q=torch.randn(1,4,3,8);k=torch.randn(1,2,3,8);v=torch.randn(1,2,3,8)
                mask=torch.zeros(1,1,3,3).masked_fill(torch.ones(3,3,dtype=torch.bool).triu(1),torch.finfo(q.dtype).min)
                app=torch.tensor([1.,0.,0.]);sib=torch.tensor([0.,1.,0.]);original=m.eager_attention_forward
                baseline=original(module,q,k,v,mask,scaling=8**-.5)
                with signed_attention_patch(m,0,0,app,sib):
                    zero=m.eager_attention_forward(module,q,k,v,mask,scaling=8**-.5)
                torch.testing.assert_close(zero[0],baseline[0],rtol=0,atol=0)
                with signed_attention_patch(m,1,-1,app,sib) as state:
                    output,weights=m.eager_attention_forward(module,q,k,v,mask,scaling=8**-.5)
                    key=m.repeat_kv(k,2);value=m.repeat_kv(v,2)
                    logits=q@key.transpose(-1,-2)*8**-.5+mask+torch.tensor([1.,-1.,0.])
                    expected=logits.softmax(-1)
                    torch.testing.assert_close(weights,expected)
                    torch.testing.assert_close(output,(expected@value).transpose(1,2).contiguous())
                    self.assertEqual(tuple(output.shape),(1,3,4,8));self.assertEqual(state['calls'],1)
                    self.assertEqual(float(weights[...,0,1:].sum()),0)
                self.assertIs(m.eager_attention_forward,original)
                try:
                    with signed_attention_patch(m,1,-1,app,sib):raise RuntimeError('test')
                except RuntimeError:pass
                self.assertIs(m.eager_attention_forward,original)
    def test_tiny_prefill_cached_decode_and_generation(self):
        from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
        for Config,Model,arch in ((LlamaConfig,LlamaForCausalLM,'llama'),(Qwen3Config,Qwen3ForCausalLM,'qwen3')):
            with self.subTest(arch=arch):
                torch.manual_seed(7)
                config=Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                              num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,
                              attention_dropout=0.,bos_token_id=1,eos_token_id=2,pad_token_id=0)
                config._attn_implementation='eager';model=Model(config).eval()
                m=importlib.import_module(f'transformers.models.{arch}.modeling_{arch}')
                ids=torch.tensor([[1,5,6,7]]);mask=torch.ones_like(ids)
                app=torch.tensor([0.,1.,0.,0.]);sib=torch.tensor([0.,0.,1.,0.])
                with torch.inference_mode():
                    baseline=model(ids,attention_mask=mask,use_cache=True)
                    with signed_attention_patch(m,0,0,app,sib):
                        zero=model(ids,attention_mask=mask,use_cache=True)
                    torch.testing.assert_close(zero.logits,baseline.logits,rtol=0,atol=0)
                    with signed_attention_patch(m,1,-1,app,sib) as state:
                        first=model(ids,attention_mask=mask,use_cache=True)
                        second=model(torch.tensor([[8]]),attention_mask=torch.ones(1,5,dtype=torch.long),
                                     past_key_values=first.past_key_values,use_cache=True)
                        self.assertEqual(tuple(second.logits.shape),(1,1,32))
                        self.assertGreater(float((first.logits-baseline.logits).abs().max()),0.)
                        model.generate(ids,attention_mask=mask,max_new_tokens=3,do_sample=False)
                        self.assertGreaterEqual(state['calls'],6)


class AnalysisTests(unittest.TestCase):
    def test_missing_invalid_and_current_sibling(self):
        from analyze_pilot import summarize
        row={'id':'x','variant':'C_sibling_rule_changed','condition':'baseline','prediction':'C',
             'gold_answer':'A','original_answer':'A','sibling_action_label':'C','closed_book_prediction':'A'}
        result=summarize([row],['x','missing'])
        arm=result['arms']['C_sibling_rule_changed/baseline']
        self.assertEqual(arm['n_expected'],2);self.assertEqual(arm['missing'],1)
        self.assertEqual(arm['sibling_branch_misuse'],1);self.assertEqual(arm['accuracy'],0)
        self.assertFalse(result['complete'])
        row['prediction']=None;row['gold_answer']=None
        self.assertEqual(summarize([row],['x'])['arms']['C_sibling_rule_changed/baseline']['correct'],0)
    def test_duplicate_rejected(self):
        from analyze_pilot import summarize
        row={'id':'x','variant':'A_original','condition':'baseline'}
        with self.assertRaisesRegex(ValueError,'duplicate'):summarize([row,row])

if __name__=='__main__':unittest.main(verbosity=2)
