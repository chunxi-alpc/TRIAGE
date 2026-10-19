#!/usr/bin/env python3
import sys,json,time,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.module1.backend import LocalBackend
from src.acre_pilot.attention_bias import AttentionBias,biased_mask
from src.acre_pilot.span_mapping import build_prompt
from scripts.run_acre_pilot import logits,write,run_case
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
out=ROOT/"results/acre_pilot/oracle_switch_diagnostic";config=json.loads((ROOT/"configs/acre_pilot.json").read_text());torch.manual_seed(config["seed"]);torch.set_num_threads(4)
b=LocalBackend();start=time.time();b._load_qwen()
item={"case_id":"engineering","guideline_id":"synthetic","guideline":{"rules":[{"rule_id":"r","text":"When infection is present, treat infection. Otherwise observe.","action":"treat infection"}]},"patient":"No infection is present.","question":"What action is appropriate?"}
prompt,enc,mapping=build_prompt(b.tokenizer,item);enc=enc.to(b.model.device);target=[20,21];last=enc["input_ids"].shape[1]-1
base=logits(b,enc)
with AttentionBias(b.model,target,0,last):zero=logits(b,enc)
with AttentionBias(b.model,[],1,last):empty=logits(b,enc)
capture={}
def capture_qkv(module,args,kwargs):
 h=kwargs["hidden_states"];shape=(*h.shape[:-1],-1,module.head_dim)
 q=module.q_norm(module.q_proj(h).view(shape)).transpose(1,2);k=module.k_norm(module.k_proj(h).view(shape)).transpose(1,2)
 q,k=apply_rotary_pos_emb(q,k,*kwargs["position_embeddings"]);k=repeat_kv(k,module.num_key_value_groups)
 s=(q[:,:,:].float()@k.transpose(2,3).float())*module.scaling
 mask=kwargs.get("attention_mask");pos=torch.arange(h.shape[1],device=h.device)
 c0=biased_mask(mask,pos,h.shape[1],target,0,last,s.dtype,s.device)
 c1=biased_mask(mask,pos,h.shape[1],target,1,last,s.dtype,s.device)
 p=(s+c0).softmax(-1);r=(s+c1).softmax(-1);mass=p[0,0,last,target].sum();expected=mass*torch.exp(torch.tensor(1.,device=h.device))/(1+(torch.exp(torch.tensor(1.,device=h.device))-1)*mass)
 capture.update({"head0_target_mass_before":float(mass),"head0_target_mass_after":float(r[0,0,last,target].sum()),"formula_abs_error":float((r[0,0,last,target].sum()-expected).abs()),"all_heads_mass_increased":bool((r[:,:,last,target].sum(-1)>p[:,:,last,target].sum(-1)).all()),"future_attention_max":float(r.triu(1).max()),"input_query_attention_exact":torch.equal(p[:,:,:last],r[:,:,:last])})
h=b.model.model.layers[0].self_attn.register_forward_pre_hook(capture_qkv,with_kwargs=True)
_ = logits(b,enc);h.remove()
with AttentionBias(b.model,target,1,last) as bias:modified=logits(b,enc)
with torch.inference_mode():
 full_base=b.model(**enc,use_cache=False).logits[0,:last].float().cpu()
 with AttentionBias(b.model,target,1,last):full_edit=b.model(**enc,use_cache=False).logits[0,:last].float().cpu()
checks={"beta_zero_exact":torch.equal(base,zero),"empty_exact":torch.equal(base,empty),"all_layers":len(bias.stats["layers"])==b.model.config.num_hidden_layers,"all_heads":all(v["query_heads"]==b.model.config.num_attention_heads for v in bias.stats["layers"].values()),"gqa_groups":b.model.config.num_attention_heads//b.model.config.num_key_value_heads,"nonzero_logit_change":float((base-modified).abs().max()),"input_logits_max_delta":float((full_base-full_edit).abs().max()),**capture,"bias_stats":bias.stats,"gpu_peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,"operator_seconds":time.time()-start}
write(out/"operator_gpu_checks.json",checks)
assert checks["input_logits_max_delta"]==0
assert checks["beta_zero_exact"] and checks["empty_exact"] and checks["all_layers"] and checks["all_heads"] and checks["formula_abs_error"]<1e-6 and checks["future_attention_max"]==0 and checks["input_query_attention_exact"]
