"""HF implementation of paper final-prompt-token residual addition (not repository broadcast hook)."""
import contextlib,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[1]
MODEL="/home/gcx/TRIAGE/models/modelscope/Qwen3-8B"
SEED=261019

def append(path,obj):
 with open(path,"a") as f: f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+"\n");f.flush()

def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))

def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

class Engine:
 def __init__(self):
  random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
  self.tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
  start=time.perf_counter()
  self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
  self.load_seconds=time.perf_counter()-start;self.direction=None;self.layer=None
  dump(ROOT/"results/engine.json",dict(model=MODEL,seed=SEED,load_seconds=self.load_seconds,torch=torch.__version__,transformers=__import__("transformers").__version__,device=torch.cuda.get_device_name(),dtype=str(self.model.dtype),attention="sdpa",chat_template="enable_thinking=False",gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),config=self.model.config.to_dict()))
 def text(self,prompt):
  return self.tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 def encode(self,prompt):
  return self.tokenizer(self.text(prompt),return_tensors="pt").to("cuda")
 @contextlib.contextmanager
 def steer(self,alpha,prompt_length):
  handles=[];trace={"calls":0,"edited_positions":0,"requested_delta_norm":0.0,"realized_delta_norm":0.0}
  if alpha and self.direction is not None:
   def hook(module,args,out):
    h=out[0] if isinstance(out,tuple) else out
    # Cached decode has length 1; only the original prefill is modified.
    if h.shape[1]!=prompt_length: return out
    z=h.clone();delta=alpha*self.direction.to(device=h.device,dtype=torch.float32)
    z[:,-1,:]=(h[:,-1,:].float()+delta).to(h.dtype)
    trace.update(calls=trace["calls"]+1,edited_positions=trace["edited_positions"]+h.shape[0],requested_delta_norm=float(delta.norm()),realized_delta_norm=float((z[:,-1,:].float()-h[:,-1,:].float()).norm()))
    return (z,)+out[1:] if isinstance(out,tuple) else z
   handles.append(self.model.model.layers[self.layer].register_forward_hook(hook))
  try: yield trace
  finally:
   for h in handles:h.remove()
 @torch.inference_mode()
 def generate(self,prompt,alpha=0,max_new_tokens=24):
  x=self.encode(prompt);torch.cuda.synchronize();start=time.perf_counter()
  with self.steer(alpha,x.input_ids.shape[1]) as trace:
   y=self.model.generate(**x,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
  torch.cuda.synchronize();seconds=time.perf_counter()-start
  ids=y[0,x.input_ids.shape[1]:]
  return dict(raw_output=self.tokenizer.decode(ids,skip_special_tokens=True),generated_token_ids=ids.tolist(),total_seconds=seconds,input_tokens=x.input_ids.shape[1],output_tokens=len(ids),steering=trace,peak_memory_bytes=torch.cuda.max_memory_allocated())
 @torch.inference_mode()
 def capture(self,prompt):
  x=self.encode(prompt);values={};handles=[]
  for i,layer in enumerate(self.model.model.layers):
   def hook(m,a,o,i=i):
    v=o[0] if isinstance(o,tuple) else o;values[i]=v[0,-1].detach().float().cpu()
   handles.append(layer.register_forward_hook(hook))
  try:self.model(**x,use_cache=False,logits_to_keep=1)
  finally:
   for h in handles:h.remove()
  return torch.stack([values[i] for i in range(len(values))])
 @torch.inference_mode()
 def score(self,prompt,labels,alpha=0,key_operator=None,key_span=None,key_epsilon=0,key_layer=18):
  x=self.encode(prompt);handles=[];key_trace={}
  if key_operator is not None and key_epsilon:
   full=self.text(prompt);start,end=key_span;offset=full.index(prompt);start+=offset;end+=offset
   toks=self.tokenizer(full,return_offsets_mapping=True)
   idx=[i for i,(a,b) in enumerate(toks.offset_mapping) if b>start and a<end]
   if not idx: raise ValueError("No action tokens matched")
   def keyhook(m,a,o):
    z=o.clone();k=o[0,idx].float();P=key_operator.to(o.device)
    # One independent projector per KV head; coordinates are after k_norm, before RoPE.
    u=torch.einsum("thd,hde->the",k,P);norm=u.norm();delta=key_epsilon*k.norm()/norm.clamp_min(1e-12)*u
    z[0,idx]=(k+delta).to(o.dtype)
    key_trace.update(tokens=len(idx),relative_norm=float((z[0,idx].float()-k).norm()/k.norm()),coordinate="post_k_norm_pre_RoPE")
    return z
   handles.append(self.model.model.layers[key_layer].self_attn.k_norm.register_forward_hook(keyhook))
  torch.cuda.synchronize();start=time.perf_counter()
  try:
   with self.steer(alpha,x.input_ids.shape[1]) as trace: y=self.model(**x,use_cache=False,logits_to_keep=1).logits[0,-1].float()
   probs=y.log_softmax(-1);ids=[]
   for label in labels:
    v=self.tokenizer.encode(label,add_special_tokens=False)
    if len(v)!=1:raise ValueError(f"Label is not single token: {label}")
    ids.append(v[0])
   scores={label:float(probs[i]) for label,i in zip(labels,ids)}
   pred=max(scores,key=scores.get);raw_id=int(y.argmax())
  finally:
   for h in handles:h.remove()
  torch.cuda.synchronize()
  return dict(prediction=pred,scores=scores,raw_top_token_id=raw_id,raw_output=self.tokenizer.decode([raw_id]),total_seconds=time.perf_counter()-start,input_tokens=x.input_ids.shape[1],output_tokens=1,steering=trace,key_edit=key_trace,peak_memory_bytes=torch.cuda.max_memory_allocated(),scoring="constrained_single_token_choice")
 @torch.inference_mode()
 def capture_keys(self,prompt,span,layer=18):
  full=self.text(prompt);off=full.index(prompt);start,end=span;start+=off;end+=off
  tokens=self.tokenizer(full,return_offsets_mapping=True)
  idx=[i for i,(a,b) in enumerate(tokens.offset_mapping) if b>start and a<end]
  values=[]
  def hook(m,a,o):values.append(o[0,idx].float().mean(0).detach().cpu())
  h=self.model.model.layers[layer].self_attn.k_norm.register_forward_hook(hook)
  try:self.model(**self.encode(prompt),use_cache=False,logits_to_keep=1)
  finally:h.remove()
  return values[0]
