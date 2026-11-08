"""Official LoReFT component, architecture-adapted two-stage LoFiT, PEFT LoRA."""
import argparse,sys,time,pickle,copy
import numpy as np
import torch
import torch.nn.functional as F
from .common import *
sys.path.insert(0,str(ROOT/"vendor/python"))
from .engine import Engine
from .features import Features
from .paths import execute
from .statistics import pair_metrics
from experiments.aster_closed_form.data import canonical_guide,prompt
class Adapter:
 def __init__(self,eng,kind,budget,load=None):
  self.eng=eng;self.kind=kind;self.active=True;self.handles=[];self.stage="tune"
  self.layer=len(eng.layers)//2;self.module=None;self.heads=[]
  d=int(eng.model.config.hidden_size);hd=eng.head_dim
  self.rank=max(1,round(max(budget-1,1)/(2*d)));self.budget=budget
  if kind=="loreft":
   from pyreft import LoreftIntervention
   self.module=LoreftIntervention(embed_dim=d,low_rank_dimension=self.rank,dtype=torch.float32).to(eng.device)
   self.handles.append(eng.layers[self.layer].register_forward_hook(self.residual_hook))
  elif kind=="last_layer":
   self.layer=len(eng.layers)-1;self.module=torch.nn.Parameter(torch.zeros(d,device=eng.device))
   self.handles.append(eng.layers[self.layer].register_forward_hook(self.residual_hook))
  elif kind=="lofit":
   self.stage="select"
   self.alpha=torch.nn.Parameter(torch.zeros(len(eng.layers),eng.n_heads,hd,device=eng.device))
   self.offset=None;self.k=max(1,min(eng.num_heads,round(budget/hd)))
   for i,l in enumerate(eng.layers):self.handles.append(l.self_attn.o_proj.register_forward_pre_hook(self.head_hook(i)))
  elif kind=="lora":
   from peft import LoraConfig,get_peft_model
   target=f"model.layers.{self.layer}.self_attn.q_proj"
   self.peft=get_peft_model(eng.model,LoraConfig(r=self.rank,lora_alpha=self.rank,target_modules=[target],lora_dropout=0,bias="none",task_type="CAUSAL_LM"))
   eng.model=self.peft.get_base_model()
  else:raise ValueError(kind)
  if load:self.restore(load)
 def parameters(self):
  if self.kind=="lora":return [p for p in self.eng.model.parameters() if p.requires_grad]
  if self.kind=="lofit":return [self.alpha] if self.stage=="select" else [self.offset]
  return [self.module] if self.kind=="last_layer" else list(self.module.parameters())
 def residual_hook(self,module,args,out):
  if not self.active:return out
  value=out[0] if isinstance(out,tuple) else out;work=value.clone()
  h=value[:,-1].float()
  work[:,-1]=(self.module(h) if self.kind=="loreft" else h+self.module).to(value.dtype)
  return (work,)+out[1:] if isinstance(out,tuple) else work
 def head_hook(self,layer):
  def hook(module,args):
   if not self.active:return args
   value=args[0];shape=value.shape;raw=value.reshape(*shape[:-1],self.eng.n_heads,self.eng.head_dim)
   if self.stage=="select":work=raw*(1+self.alpha[layer].to(raw.dtype))
   else:
    work=raw.clone()
    for i,h in enumerate(self.heads):
     if h//self.eng.n_heads==layer:work[...,h%self.eng.n_heads,:]=raw[...,h%self.eng.n_heads,:]+self.offset[i].to(raw.dtype)
   return (work.reshape_as(value),)+args[1:]
  return hook
 def select(self):
  importance=self.alpha.detach().norm(dim=-1).flatten()
  self.heads=torch.argsort(importance,descending=True)[:self.k].cpu().tolist()
  self.selection_parameters=self.alpha.numel();self.alpha.requires_grad_(False)
  self.offset=torch.nn.Parameter(torch.zeros(len(self.heads),self.eng.head_dim,device=self.eng.device));self.stage="tune"
 def score_train(self,text):
  eng=self.eng;ids=eng.ids(text)
  out=eng.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
  return eng._score_hidden(out.last_hidden_state[0,-1])
 def score(self,text):
  # LoReFT and final-layer control edit only the judgment token, never the prefix endpoint.
  self.active=self.kind not in ("loreft","last_layer")
  context=self.eng.prepare(text);self.active=True
  with torch.no_grad():result=self.eng.run(context,capture=False)
  return result["ell"]
 def dump(self,path):
  if self.kind=="lora":
   from peft import get_peft_model_state_dict
   state=get_peft_model_state_dict(self.peft)
  elif self.kind=="lofit":state=dict(heads=self.heads,offset=self.offset.detach().cpu(),selection_parameters=self.selection_parameters)
  elif self.kind=="last_layer":state=self.module.detach().cpu()
  else:state=self.module.state_dict()
  torch.save(dict(kind=self.kind,budget=self.budget,rank=self.rank,layer=self.layer,state=state),path)
 def restore(self,path):
  obj=torch.load(path,map_location=self.eng.device,weights_only=False);state=obj["state"]
  if self.kind=="lora":
   from peft import set_peft_model_state_dict
   set_peft_model_state_dict(self.peft,state)
  elif self.kind=="lofit":
   self.heads=state["heads"];self.offset=torch.nn.Parameter(state["offset"].to(self.eng.device));self.stage="tune"
   self.selection_parameters=state["selection_parameters"];self.alpha.requires_grad_(False)
  elif self.kind=="last_layer":self.module.data.copy_(state)
  else:self.module.load_state_dict(state)
 def close(self):
  for h in self.handles:h.remove()
def budget(model):
 cand=read(ROOT/model/"readout_candidates.json")
 # Match the selected main readout if available, otherwise the registered rank-8 full feature comparison.
 selected=read(ROOT/model/"development_selection.json").get("aster_cr")
 chosen=next((r for r in cand if r["name"]==selected),None) or next((r for r in cand if r["name"]=="complete_exploratory"),next(r for r in cand if r["name"].startswith("cr_")))
 return chosen["parameters"]
def train(model,kind,lr):
 name=f"{kind}_lr{lr}";existing=ROOT/model/"baselines"
 if (existing/(name+".pt")).exists() and (existing/(name+"_training.json")).exists():return
 torch.set_num_threads(4);torch.manual_seed(SEED);start=time.time()
 f=Features(model);train=[v for v in f.views if v["split"]=="fit" and v["kind"] in TASK_KINDS]
 eng=Engine(model);adapter=Adapter(eng,kind,budget(model))
 eng.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
 eng.model.enable_input_require_grads();eng.model.train()
 rng=np.random.default_rng(SEED)
 cost=[];root=ROOT/model/"baselines";root.mkdir(exist_ok=True)
 for stage in (["select","tune"] if kind=="lofit" else ["tune"]):
  params=adapter.parameters();opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
  losses=[];stage_start=time.time()
  # One complete epoch per stage; identical fit examples and three learning rates per family.
  for step,i in enumerate(rng.permutation(len(train))):
   v=train[int(i)];opt.zero_grad(set_to_none=True);s=adapter.score_train(v["prompt"])
   loss=F.binary_cross_entropy_with_logits(s,torch.tensor(float(v["label"]),device=s.device))
   if kind=="lofit" and stage=="select":loss=loss+1e-3*adapter.alpha.abs().mean()
   if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
   loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();losses.append(float(loss.detach()))
   if step%100==0:status(f"{model}_{name}",state="training",training_stage=stage,done=step+1,total=len(train),seconds=time.time()-stage_start)
  cost.append(dict(stage=stage,steps=len(train),parameters=sum(p.numel() for p in params),seconds=time.time()-stage_start,mean_loss=float(np.mean(losses))))
  if kind=="lofit" and stage=="select":adapter.select()
 name=f"{kind}_lr{lr}";adapter.dump(root/(name+".pt"));eng.model.eval();eng.model.gradient_checkpointing_disable()
 write(root/(name+"_training.json"),dict(kind=kind,lr=lr,fit_views=len(train),stages=cost,parameters=sum(p.numel() for p in adapter.parameters()),
   target_budget=adapter.budget,seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,
   implementation="official pyreft.LoreftIntervention" if kind=="loreft" else "official PEFT LoRA" if kind=="lora" else
    "Qwen3/Gemma3 hooks port official LoFiT per-coordinate (1+A)*head+v; select A then discard A and tune selected v" if kind=="lofit" else "learned final-layer additive residual vector"))
 adapter.close()
def evaluate(model,kind,lr,split):
 if split=="test" and not read(ROOT/model/"frozen.json").get("frozen"):raise RuntimeError("Test must be frozen")
 torch.set_num_threads(4);start=time.time();name=f"{kind}_lr{lr}";root=ROOT/model/"baselines"
 checkpoint=root/(name+".pt");obj=torch.load(checkpoint,map_location="cpu",weights_only=False)
 eng=Engine(model);ad=Adapter(eng,kind,obj["budget"],load=checkpoint);eng.model.eval()
 target=root/(name+"_"+split+".jsonl");saved={r["view_id"]:r["ell"] for r in rows(target)}
 views=[v for v in rows(ROOT/"data/views.jsonl") if v["split"]==split and v["kind"] in TASK_KINDS]
 for v in views:
  if v["view_id"] in saved:continue
  score=ad.score(v["prompt"]);saved[v["view_id"]]=score;append(target,dict(view_id=v["view_id"],ell=score))
 ps=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]==split]
 metrics=pair_metrics(views,ps,saved,ci=split=="test")
 path_target=root/(name+"_paths_"+split+".jsonl");done={r["case_id"] for r in rows(path_target)}
 for c in rows(ROOT/"data/path_cases.jsonl"):
  if c["split"]!=split or c["case_id"] in done:continue
  tree=canonical_guide(c["guideline"]);nodes={n["id"]:n for n in tree["nodes"]};gid=tree["nodes"][0]["id"];scores={}
  while gid in nodes and gid not in scores:
   s=ad.score(prompt(c["guideline"],c["definitions"],c["patient"],gid));scores[gid]=s;gid=nodes[gid]["yes" if s>0 else "no"]
  append(path_target,dict(case_id=c["case_id"],source_id=c["source_id"],**execute(c,scores)))
 rs=rows(path_target);info=read(root/(name+"_training.json"))
 write(root/(name+"_"+split+"_metrics.json"),dict(name=name,pairs=metrics,paths=dict(n=len(rs),correct=sum(r["correct"] for r in rs),
   accuracy=float(np.mean([r["correct"] for r in rs])) if rs else None),parameters=info["parameters"],seconds=time.time()-start,
   peak_gib=torch.cuda.max_memory_allocated()/2**30,counts=eng.counts))
 ad.close()
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");p.add_argument("--kind",required=True,choices=["loreft","lofit","lora","last_layer"])
 p.add_argument("--lr",type=float,default=.001);p.add_argument("--split",default="dev");p.add_argument("--evaluate",action="store_true")
 a=p.parse_args()
 if a.evaluate:evaluate(a.model,a.kind,a.lr,a.split)
 else:train(a.model,a.kind,a.lr);evaluate(a.model,a.kind,a.lr,"dev")
