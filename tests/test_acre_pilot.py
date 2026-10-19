"""CPU tests with a tiny real Qwen3 plus numerical operator tests."""
import unittest,json
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,AutoTokenizer
from src.acre_pilot.attention_bias import AttentionBias,biased_mask
from src.acre_pilot.span_mapping import locate,build_prompt,map_annotations,select_targets
from src.acre_pilot.annotator import parse,direct_decision
class TestACRE(unittest.TestCase):
    def test_span_ambiguity(self):
        with self.assertRaisesRegex(ValueError,"ambiguous"):locate("yes yes","yes")
        self.assertEqual(locate("yes yes",{"text":"yes","before":"yes "}),(4,7))
        with self.assertRaises(ValueError):locate("abc",{"text":"bc","start":0,"end":2})
        with self.assertRaises(ValueError):locate("abc","def")
    def test_operator_formula_causality_and_input_queries(self):
        torch.manual_seed(1);scores=torch.randn(1,4,6,6,dtype=torch.float64)
        targets=[1,2];pos=torch.arange(6)
        base=biased_mask(None,pos,6,targets,0,5,scores.dtype,scores.device)
        edit=biased_mask(None,pos,6,targets,1,5,scores.dtype,scores.device)
        p=(scores+base).softmax(-1);q=(scores+edit).softmax(-1)
        self.assertTrue(torch.equal(p[:,:,:5],q[:,:,:5]))
        self.assertEqual(float(q.triu(1).max()),0)
        mass=p[0,0,5,targets].sum();actual=q[0,0,5,targets].sum();expected=mass*torch.exp(torch.tensor(1.))/(1+(torch.exp(torch.tensor(1.))-1)*mass)
        self.assertAlmostEqual(float(actual),float(expected),places=7)
        self.assertTrue((q[:,:,5,targets].sum(-1)>p[:,:,5,targets].sum(-1)).all())
    def test_real_qwen_gqa_identity_all_layers_and_cache(self):
        torch.manual_seed(1)
        c=Qwen3Config(vocab_size=80,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        c._attn_implementation="sdpa";m=Qwen3ForCausalLM(c).eval();x=torch.tensor([[1,2,3,4,5,6]])
        with torch.inference_mode():
            base=m(x,use_cache=False).logits
            with AttentionBias(m,[1],0,5):zero=m(x,use_cache=False).logits
            with AttentionBias(m,[],1,5):empty=m(x,use_cache=False).logits
            self.assertTrue(torch.equal(base,zero));self.assertTrue(torch.equal(base,empty))
            with AttentionBias(m,[1,2],1,5) as hook:modified=m(x,use_cache=False).logits
            self.assertEqual(len(hook.stats["layers"]),3)
            self.assertTrue(all(v["query_heads"]==4 and v["kv_heads"]==2 for v in hook.stats["layers"].values()))
            self.assertFalse(torch.equal(base[:,-1],modified[:,-1]))
            self.assertTrue(torch.equal(base[:,:5],modified[:,:5]))
            with AttentionBias(m,[1,2],1,5):
                pre=m(x[:,:5],use_cache=True)
                dec=m(x[:,5:],past_key_values=pre.past_key_values,use_cache=True).logits
            self.assertTrue(torch.allclose(dec[:,-1],modified[:,-1],atol=1e-6))
    def test_exact_template_mapping_and_failed_span_no_target(self):
        from pathlib import Path
        config=json.loads(Path("configs/models.json").read_text());tok=AutoTokenizer.from_pretrained(config["models"]["qwen3_8b"]["path"],local_files_only=True)
        item={"guideline":{"rules":[{"rule_id":"r","text":"If ill: treat. Otherwise wait.","action":"treat."}]},"patient":"Not ill. Not ill.","question":"管理？"}
        prompt,encoded,mapping=build_prompt(tok,item)
        a={"rule_id":"r","relevant":True,"applicability":"inapplicable","action_spans":["treat."],"supporting_evidence":[],"refuting_evidence":["Not ill."],"exception_spans":[]}
        rows=map_annotations(item,[a],mapping);targets,_=select_targets(rows,mapping,"B3")
        self.assertEqual(targets,[]);self.assertEqual(len(rows[0]["mapping_errors"]),1)
        a["refuting_evidence"]=[{"text":"Not ill.","start":9,"end":17}]
        rows=map_annotations(item,[a],mapping);targets,_=select_targets(rows,mapping,"B3")
        self.assertTrue(targets)
        self.assertFalse(set(targets)&set(rows[0]["mapped"]["action_spans"][0]["tokens"]))
        for span in rows[0]["mapped"]["refuting_evidence"]:self.assertEqual(prompt[span["prompt_start"]:span["prompt_end"]],span["text"])
        for method in ["B1","B2","B3"]:self.assertEqual(build_prompt(tok,item)[0],prompt)
    def test_direct_decision_uses_original_action_without_gold(self):
        from src.acre_pilot.annotator import direct_decision,identity
        rows=[{'rule_id':'r','relevant':True,'applicability':'applicable','mapped':{'action_spans':[{'text':'short excerpt'}]}}]
        options=[{'label':'A','action':'complete recommendation'},{'label':'B','action':'insufficient evidence'}]
        self.assertEqual(direct_decision(rows,options,[{'rule_id':'r','action':'complete recommendation'}]),'A')
        item={'guideline':{'rules':[{'rule_id':'r','text':'If sick: treat.'}]},'patient':'Sick.','question':'What action?','gold_action':'secret reference','reference_path':'secret path'}
        class Backend: model_path='model'
        payload,key=identity(item,Backend(),{'annotation_max_new_tokens':20})
        self.assertNotIn('secret',json.dumps(payload))
    def test_offline_metrics_keep_failed_annotations_and_fixed_competitor(self):
        from src.acre_pilot.evaluator import evaluate_cases
        case={'case_id':'c','guideline_id':1,'pair_id':None,'options':[{'label':'A','action':'correct'},{'label':'B','action':'wrong'}],'annotations':[],'annotation_errors':['failure'],'variants':{'B0':{'prediction':'A','option_logits':[2,1]},'B3@1.0':{'prediction':'B','option_logits':[1,3],'attention':{'enabled':False}}}}
        metrics,ann,details=evaluate_cases([case],{'c':{'gold_action':'correct'}})
        self.assertEqual(metrics['B3@1.0']['accuracy']['denominator'],1)
        self.assertEqual(metrics['B3@1.0']['retention']['numerator'],0)
        self.assertEqual(ann['annotation_failure_cases'],1)
        self.assertEqual(next(r for r in details if r['method']=='B3@1.0')['delta_margin'],-3)
    def test_schema_no_answers(self):
        with self.assertRaises(ValueError):parse('{"rules":[],"answer":"A"}',[])
        with self.assertRaises(ValueError):parse('{"rules":[]}',[{"rule_id":"r"}])
if __name__=="__main__":unittest.main()
