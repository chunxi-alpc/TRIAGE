"""End-to-end decision timing; no feature extraction or CPU microbenchmarks."""
import argparse,gc,platform,subprocess,time
from contextlib import nullcontext
import numpy as np
import torch
from experiments.aster_cr.engine import Engine
from experiments.aster_cr.baselines import Adapter
from experiments.aster_cr.comparators import reference
from .common import *
def tensor_bytes(x):
 if torch.is_tensor(x):return x.numel()*x.element_size()
 if isinstance(x,dict):return sum(tensor_bytes(v) for v in x.values())
 if isinstance(x,(list,tuple)):return sum(tensor_bytes(v) for v in x)
 return 0
class Method:
 def __init__(self,name,eng):
  self.name=name;self.eng=eng;self.handles=[];self.active=False;self.captured={};self.adapter=None
  self.info=dict(method=name,trainable_parameters=0,inference_trainable_parameters=0,
    storage_bytes=0,deployment_tensor_bytes=0,checkpoint=None,logical_forwards=2 if name=='cad' else 1)
  if name=='aster':
   self.rd=Readout(eng.device);self.by_layer={}
   for h in self.rd.heads:self.by_layer.setdefault(h//eng.n_heads,[]).append(h%eng.n_heads)
   for layer,js in self.by_layer.items():
    def hook(m,args,layer=layer,js=js):
     if self.active:self.captured[layer]=args[0][0,-1].view(eng.n_heads,eng.head_dim)[js].clone()
    self.handles.append(eng.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook))
   self.info.update(trainable_parameters=17,checkpoint=str(MODEL),storage_bytes=sum((MODEL/n).stat().st_size for n in ('m2.npz','m3.npz')),
    deployment_tensor_bytes=sum(tensor_bytes(v) for v in (self.rd.mu,self.rd.scale,self.rd.N,self.rd.w))+8+len(self.rd.heads)*8,
    storage_note='m2/m3 checkpoint includes fit metadata; deployment only needs heads, mu, scale, N, w, b')
  elif name=='leace':
   p=OUT/'checkpoints/leace.pt' if (OUT/'checkpoints/leace.pt').exists() else HIST/'comparators/leace.pt';self.ops=torch.load(p,map_location=eng.device,weights_only=False)
   by={}
   for h,op in self.ops.items():by.setdefault(int(h)//eng.n_heads,[]).append((int(h)%eng.n_heads,op))
   for layer,ops in by.items():
    def hook(m,args,ops=ops):
     if not self.active:return
     raw=args[0].view(1,1,eng.n_heads,eng.head_dim);work=raw.clone()
     for h,op in ops:
      z=raw[0,0,h].float();work[0,0,h]=(z-op['N']@(op['B']@(z-op['mu']))).to(raw.dtype)
     return (work.reshape_as(args[0]),)+args[1:]
    self.handles.append(eng.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook))
   self.info.update(checkpoint=str(p),storage_bytes=p.stat().st_size,
     deployment_tensor_bytes=sum(tensor_bytes(op[k]) for op in self.ops.values() for k in ('N','B','mu'))+len(self.ops)*8,
     parameters_note='closed-form fit; no gradient-trained parameters')
  elif name in ('lofit','lora'):
   original=HIST/'baselines'/(name+'_lr0.001.pt')
   p=OUT/'checkpoints'/(name+'.pt') if (OUT/'checkpoints'/(name+'.pt')).exists() else original;obj=torch.load(p,map_location='cpu',weights_only=False)
   self.adapter=Adapter(eng,name,obj['budget'],load=p)
   eng.model.eval().requires_grad_(False)
   if name=='lofit':self.adapter.offset.requires_grad_(False)
   self.info.update(checkpoint=str(p),storage_bytes=p.stat().st_size,
      trainable_parameters=read(OUT/'configs'/(name+'_lr0.001_training.json'))['parameters'],
      deployment_tensor_bytes=tensor_bytes(obj['state'])+(len(obj['state']['heads'])*8 if name=='lofit' else 0),
      selection_parameters=obj['state'].get('selection_parameters',0))
  elif name=='cad':
   self.info.update(reference_forwards=1,CAD_alpha=2.,storage_note='fixed scalar coefficient; no checkpoint')
 def final(self,ctx):
  eng=self.eng;self.active=True;self.captured={}
  try:
   output=eng.model.model(input_ids=ctx.input_ids[:,-1:],attention_mask=torch.ones_like(ctx.input_ids),
    past_key_values=eng._cache(ctx),use_cache=True,
    cache_position=torch.tensor([ctx.length-1],device=eng.device),
    position_ids=torch.tensor([[ctx.length-1]],device=eng.device))
   ell=eng._score_hidden(output.last_hidden_state[0,-1])
   cache=output.past_key_values
   kv=sum(tensor_bytes(l.keys)+tensor_bytes(l.values) for l in cache.layers)
   if self.name=='aster':
    values={layer*self.eng.n_heads+j:self.captured[layer][i] for layer,js in self.by_layer.items() for i,j in enumerate(js)}
    h=torch.stack([values[j] for j in self.rd.heads]).flatten().double()
    ell=ell.double()+self.rd.b+(((h-self.rd.mu)/self.rd.scale)@self.rd.N)@self.rd.w
   score=float(ell);del output,cache
   return score,kv
  finally:self.active=False;self.captured={}
 def one(self,prompt,length):
  from experiments.m1_diagnostics.engine import Context
  eng=self.eng
  ids=eng.tokenizer(eng.render(prompt),return_tensors='pt',add_special_tokens=False)['input_ids'][:,-length:].to(eng.device)
  torch.cuda.synchronize();t=time.perf_counter()
  cache=eng._prefill(ids,ids.shape[1]-1)
  torch.cuda.synchronize();prefill=time.perf_counter()-t
  ctx=Context(prompt,ids,cache,int(ids.shape[1]))
  ell,kv=self.final(ctx)
  return ell,prefill,kv,int(ids.shape[1])
 def measure(self,prompt,length):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  ell,prefill,kv,tokens=self.one(prompt,length);ref_tokens=0;ref_kv=0;ref_ell=None
  if self.name=='cad':
   ref_ell,rp,ref_kv,ref_tokens=self.one(reference(prompt),length)
   prefill+=rp;ell=3*ell-2*ref_ell
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  return dict(score=ell,reference_score=ref_ell,prefill_ms=prefill*1000,latency_ms=elapsed*1000,
    throughput_samples_s=1/elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,kv_cache_bytes=kv,
    reference_kv_cache_bytes=ref_kv,max_live_logical_kv_bytes=max(kv,ref_kv),
    primary_tokens=tokens,reference_tokens=ref_tokens,logical_forwards=self.info['logical_forwards'],
    model_forward_calls=2*self.info['logical_forwards'])
 def close(self):
  for h in self.handles:h.remove()
  if self.adapter:self.adapter.close()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--methods',nargs='+',default=['raw','aster','leace','lofit','lora','cad'])
 args=ap.parse_args();torch.set_num_threads(4);torch.set_grad_enabled(False)
 cfg=read(OUT/'configs/e11.json');examples=rows(OUT/'configs/e11_inputs.jsonl')
 write(OUT/'e11_environment.json',dict(torch=torch.__version__,python=platform.python_version(),
  cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
  hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv'],text=True)))
 target=OUT/'e11_measurements.jsonl'
 existing=rows(target) if target.exists() else []
 done={(r['method'],r['length'],r['repeat'],r['input_id']) for r in existing}
 infos=read(OUT/'e11_method_info.json') if (OUT/'e11_method_info.json').exists() else {}
 started=time.time()
 for name in args.methods:
  if all((name,l,r,e['input_id']) in done for l in cfg['lengths'] for r in range(cfg['repeats']) for e in examples):continue
  eng=Engine('qwen');method=Method(name,eng);infos[name]=method.info
  write(OUT/'e11_method_info.json',infos)
  for length in cfg['lengths']:
   for w in range(cfg['warmup']):method.measure(examples[w%len(examples)]['prompt'],length)
   torch.cuda.empty_cache()
   for repeat in range(cfg['repeats']):
    order=np.random.default_rng(cfg['seed']+repeat).permutation(len(examples))
    for ix in order:
     ex=examples[int(ix)];key=(name,length,repeat,ex['input_id'])
     if key in done:continue
     append(target,dict(method=name,length=length,repeat=repeat,input_id=ex['input_id'],
       **method.measure(ex['prompt'],length)))
  method.close();del method,eng;gc.collect();torch.cuda.empty_cache()
  write(OUT/'e11_status.json',dict(state='running',completed_methods=list(infos),seconds=time.time()-started))
 write(OUT/'e11_status.json',dict(state='available_methods_complete',completed_methods=args.methods,
  pending=['audited_ITI','completed_new_representative'],seconds=time.time()-started))
if __name__=='__main__':main()
