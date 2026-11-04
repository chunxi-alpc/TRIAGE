import copy, tempfile, unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig, Gemma3ForConditionalGeneration
from experiments.gemma3_head_pilot.core import HeadIntervention, group_indices, rank_heads, prediction_summary, selection_key
from experiments.gemma3_head_pilot.engine import Engine, Query, fork_cache, common_prefix

class PilotTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);torch.manual_seed(13)
  text=Gemma3TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
   num_attention_heads=4,num_key_value_heads=2,head_dim=8,query_pre_attn_scalar=8,
   max_position_embeddings=128,sliding_window=4,
   layer_types=["sliding_attention","full_attention","sliding_attention","full_attention"],
   pad_token_id=0,bos_token_id=1,eos_token_id=2)
  vision=SiglipVisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
                           image_size=16,patch_size=8)
  cfg=Gemma3Config(text_config=text.to_dict(),vision_config=vision.to_dict(),
                   image_token_index=120,boi_token_index=121,eoi_token_index=122)
  cls.model=Gemma3ForConditionalGeneration(cfg).eval()
  cls.tmp=tempfile.TemporaryDirectory()
  cls.engine=Engine(cls.model,None,cls.tmp.name,chunk=3)

 def query(self):
  with torch.no_grad():cache=self.engine.prefill(list(range(3,14)))
  return Query(14,11,cache,[[17],[18]])

 def test_native_sliding_cache_and_fork(self):
  q=self.query();before=[l.get_seq_length() for l in q.cache.layers]
  keys=[l.keys.clone() for l in q.cache.layers]
  with torch.no_grad():
   self.engine.scores(q)
   whole=self.engine.forward(list(range(3,15))).logits
   chunk_cache=self.engine.prefill(list(range(3,14)))
   split=self.engine.forward([14],chunk_cache,11).logits
  self.assertTrue(torch.allclose(whole,split,atol=1e-6,rtol=1e-5))
  self.assertEqual(before,[l.get_seq_length() for l in q.cache.layers])
  for a,b in zip(keys,q.cache.layers):self.assertTrue(torch.equal(a,b.keys))
  self.assertEqual(q.cache.layers[0].keys.shape[-2],3)
  self.assertEqual(q.cache.layers[1].keys.shape[-2],11)

 def test_head_mask_projection_and_gradient(self):
  cfg=self.engine.config;module=HeadIntervention(cfg,[1],rank=4)
  projection=self.engine.layers[1].self_attn.o_proj
  x=torch.randn(1,1,32);ref=torch.randn_like(x)
  changed,mag,_=module(x,ref,projection)
  self.assertTrue(torch.equal(changed[...,:16],x[...,:16]))
  self.assertFalse(torch.equal(changed[...,16:],x[...,16:]))
  actual=(projection(changed).float()-projection(x).float()).square().sum()/projection(x).float().square().sum()
  self.assertTrue(torch.allclose(actual,mag,atol=1e-8,rtol=1e-3))
  (changed.square().sum()+mag).backward()
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
  self.assertTrue(all(p.grad is None for p in self.model.parameters()))

 def test_zero_hook_native_and_backbone_frozen(self):
  q=self.query();reference={}
  with torch.no_grad():base,_=self.engine.scores(q,capture=reference)
  module=HeadIntervention(self.engine.config,[0],rank=4)
  with torch.no_grad():same,_=self.engine.scores(q,layer=1,module=module,reference=reference,force_zero=True)
  self.assertTrue(torch.equal(base,same))
  reference["pre"][1]=reference["pre"][1]+torch.randn(32)
  values,diag=self.engine.scores(q,layer=1,module=module,reference=reference)
  (-values.log_softmax(-1)[0]+diag["magnitude"]).backward()
  self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
  self.assertTrue(all(p.grad is None for p in self.model.parameters()))

 def test_multitoken_candidate_scoring(self):
  q=self.query();q.suffixes=[[17,19],[18,20]]
  with torch.no_grad():values,_=self.engine.scores(q)
  self.assertEqual(values.shape,(2,));self.assertTrue(torch.isfinite(values).all())
  self.assertEqual(common_prefix([[1,2,3],[1,2,4]]),([1,2],[[3],[4]]))

 def test_bf16_probe_edit_preserves_dtype_and_unselected_heads(self):
  projection=torch.nn.Linear(32,32,bias=False,dtype=torch.bfloat16)
  fake=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection))])
  x=torch.randn(1,1,32).to(torch.bfloat16);reference={"pre":{0:torch.randn(32)}}
  for strength in (.25,-.25,0.):
   captured=[]
   with Engine.edit(fake,0,reference=reference,groups=list(range(16)),strength=strength):
    handle=projection.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
    output=projection(x);handle.remove()
   self.assertEqual(captured[0].dtype,torch.bfloat16)
   self.assertTrue(torch.equal(captured[0][...,16:],x[...,16:]))
   self.assertTrue(torch.isfinite(output).all())
   if strength==0:self.assertTrue(torch.equal(captured[0],x))

 def test_development_baselines_and_paired_table(self):
  from experiments.gemma3_head_pilot.pipeline import development_comparison
  import json
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/"data").mkdir();(root/"development_baselines").mkdir()
   job=root/"jobs"/"heads_layer1";job.mkdir(parents=True)
   cases=[dict(case_id="a",gold_rules=["R0Y"],gold_leaf="L1"),
          dict(case_id="b",gold_rules=["R0N"],gold_leaf="L2")]
   (root/"data/development.jsonl").write_text("".join(json.dumps(c)+"\n" for c in cases))
   def records(correct):
    return [dict(case_id=c["case_id"],view=v,status="complete",
                 path=c["gold_rules"] if c["case_id"] in correct else ["R0Y","wrong"],
                 leaf=c["gold_leaf"] if c["case_id"] in correct else None,magnitudes=[])
            for c in cases for v in ("original","swapped")]
   for path,values in [(root/"development_baselines/zero.jsonl",records({"a"})),
                       (root/"development_baselines/blind.jsonl",records({"a","b"})),
                       (job/"development_epoch1.jsonl",records({"b"}))]:
    path.write_text("".join(json.dumps(r)+"\n" for r in values))
   summary=dict(name="heads_layer1",kind="heads",layer=1,groups=[0],epoch=1,
                original_correct=1,both_views_correct=1,mean_magnitude=0.,failures=0,parameters=137)
   (job/"development_epoch1.json").write_text(json.dumps(summary))
   result=development_comparison(root)
   self.assertEqual(result["summaries"]["zero"]["original_correct"],1)
   self.assertEqual(result["summaries"]["blind"]["original_correct"],2)
   self.assertEqual(result["paired_vs_zero"]["heads_layer1"]["corrections"],1)
   self.assertEqual(result["paired_vs_zero"]["heads_layer1"]["regressions"],1)
   table=(root/"DEVELOPMENT_COMPARISON.zh.md").read_text()
   self.assertIn("原模型／零干预",table);self.assertIn("直接盲读",table)
   self.assertFalse(result["confirmation_result"])

 def test_group_mapping(self):
  self.assertEqual(group_indices([1],self.engine.config),list(range(16,32)))

 def test_ranking_negative_scores_and_ties(self):
  records=[]
  for l in range(3):
   for g in range(5):
    records.append(dict(case_id="a",layer=l,group=g,status="complete",base_margin=0.,
                        plus_margin=-g-l,minus_margin=g+l))
  ranked=rank_heads(records,[[1.]*5 for _ in range(3)],["full_attention"]*3,5)
  self.assertEqual([x["layer"] for x in ranked["candidates"]],[0,1,2])
  self.assertEqual(ranked["candidates"][0]["groups"],[0,1,2,3])

 def test_selection_prefers_path_correctness(self):
  a=dict(original_correct=9,both_views_correct=1,mean_magnitude=.5)
  b=dict(original_correct=8,both_views_correct=8,mean_magnitude=0.)
  self.assertGreater(selection_key(a,1,1),selection_key(b,1,1))

 def test_failures_in_denominator(self):
  cases=[dict(case_id="a",gold_rules=["R0Y"],gold_leaf="L1"),
         dict(case_id="b",gold_rules=["R0N"],gold_leaf="L2")]
  records=[dict(case_id="a",view=v,status="complete",path=["R0Y"],leaf="L1",magnitudes=[])
           for v in ("original","swapped")]
  records += [dict(case_id="b",view=v,status="failed",path=[],leaf=None) for v in ("original","swapped")]
  summary=prediction_summary(cases,records)
  self.assertEqual(summary["path_accuracy"],.5);self.assertEqual(summary["failures"],2)

if __name__=="__main__":unittest.main()
