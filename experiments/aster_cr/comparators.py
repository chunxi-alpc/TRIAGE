"""Fixed historical settings; refit on new fit sources without a new search."""
import argparse,ast,pickle,time,sys
from collections import defaultdict
import numpy as np
import torch
from .common import *
from .features import Features
from .engine import Engine
from .paths import execute
from .statistics import pair_metrics
from experiments.aster_closed_form.data import canonical_guide,prompt
def reference(text):
 # Patient and current target are supplied; no gold label, path or test annotation is used.
 return "请根据病历判断当前条件。\n"+text.split("病历：",1)[-1]
def format_prompt(text,name):
 tail={
 "full":'请判断当前条件。只输出“是”或“否”。\n答案：',
 "order_no_yes":'请判断当前条件。只输出“否”或“是”。\n答案：',
 "map_ab":'请判断当前条件。A 表示条件成立，B 表示条件不成立。只输出 A 或 B。\n答案：',
 "map_ba":'请判断当前条件。B 表示条件成立，A 表示条件不成立。只输出 A 或 B。\n答案：',
 "equivalent":'根据给定事实，当前节点的完整条件是否满足？只输出“是”或“否”。\n答案：'}[name]
 base=text.rsplit("请判断当前条件。",1)[0]
 return base+tail,["B","A"] if name=="map_ba" else ["A","B"] if name=="map_ab" else None
def sea_basis(x,y):
 q,r=np.linalg.qr(x.T,mode="reduced")
 u,s,_=np.linalg.svd(r@y/len(x),full_matrices=False)
 if s.sum()==0:return np.empty((x.shape[1],0),np.float32)
 k=int(np.searchsorted(np.cumsum(s*s)/np.sum(s*s),.998))+1
 return (q@u[:,:k]).astype(np.float32)
def fit(model):
 from sklearn.linear_model import LogisticRegression
 from experiments.aster_closed_form.operators import fit_operators
 torch.set_num_threads(4);f=Features(model);root=ROOT/model/"comparators";root.mkdir(exist_ok=True)
 selection=read(ROOT/model/"selection.json");heads=selection["old_heads"] or selection["heads"]
 features={v["view_id"]:dict(f.get(v["view_id"]),ell=f.ell(v["view_id"]),split="fit",original=v["original"]) for v in f.views if v["split"]=="fit" and v["kind"] in {"original","A","P","C","grid"}}
 pairs=[p for p in f.pairs if p["split"]=="fit" and p["left"] in features and p["right"] in features]
 for name,variant in [("closed_form","aster"),("leace","leace")]:
  ops,meta=fit_operators(features,pairs,heads,{h:["A","P"] for h in heads},eta=.95,kmax=4,theta=45,variant=variant)
  torch.save(ops,root/(name+".pt"));write(root/(name+".json"),meta)
 del features
 eng=Engine(model);base=[v for v in f.views if v["split"]=="fit" and v["variant"]=="full"]
 pos=[];neg=[];xp=[];xn=[];xb=[];refs=[];full=[];y=[]
 truthfile=root/"truth_features.pt"
 if truthfile.exists():
  z=torch.load(truthfile,map_location="cpu",weights_only=False);pos,neg,xp,xn,xb,refs,full,y=[z[k] for k in ("pos","neg","xp","xn","xb","refs","full","y")]
 else:
  original_render=eng.render
  for v in base:
   rr={}
   for label in (0,1):
    token="是" if label else "否"
    eng.render=lambda text,task="binary",t=token:original_render(text,task)+t
    rr[label]=eng.score(v["prompt"],capture=True)
   eng.render=original_render
   correct=rr[v["label"]];wrong=rr[1-v["label"]]
   pos.append(correct["heads"].flatten(0,1).numpy());neg.append(wrong["heads"].flatten(0,1).numpy())
   xp.append(correct["residual"].numpy());xn.append(wrong["residual"].numpy());xb.append(f.get(v["view_id"])["residual"])
   refs.append(eng.score(reference(v["prompt"]),capture=False)["ell"]);full.append(f.ell(v["view_id"]));y.append(v["label"])
  torch.save(dict(pos=pos,neg=neg,xp=xp,xn=xn,xb=xb,refs=refs,full=full,y=y),truthfile)
 pos=np.stack(pos);neg=np.stack(neg)
 # ITI: fit-only source-held-out probes rank heads; direction uses correct-minus-incorrect answer representations.
 sources=sorted({v["source_id"] for v in base});held=set(sources[::5]);mask=np.array([v["source_id"] not in held for v in base]);acc=[]
 labels=np.r_[np.ones(mask.sum()),np.zeros(mask.sum())]
 for h in range(eng.num_heads):
  clf=LogisticRegression(C=1.,max_iter=100).fit(np.r_[pos[mask,h],neg[mask,h]],labels)
  xv=np.r_[pos[~mask,h],neg[~mask,h]];yv=np.r_[np.ones((~mask).sum()),np.zeros((~mask).sum())]
  acc.append(float(clf.score(xv,yv)))
 chosen=np.argsort(-np.array(acc))[:len(heads)]
 directions={}
 for h in chosen:
  d=pos[:,h].mean(0)-neg[:,h].mean(0);d/=max(np.linalg.norm(d),1e-8)
  sigma=np.std(np.r_[pos[:,h],neg[:,h]]@d);directions[int(h)]=torch.tensor(15*sigma*d)
 torch.save(directions,root/"iti.pt")
 sp=sea_basis(np.stack(xb),np.stack(xp));sn=sea_basis(np.stack(xb),np.stack(xn));d=len(xb[0])
 torch.save(dict(positive=torch.tensor(sp@sp.T),negative=torch.tensor(np.eye(d,dtype=np.float32)-sn@sn.T)),root/"sea.pt")
 fusion=LogisticRegression(C=1.,max_iter=1000).fit(np.column_stack([full,refs]),y)
 with (root/"reference_fusion.pkl").open("wb") as file:pickle.dump(fusion,file)
 # Output template choice uses only development original cases and no test outputs.
 values=[]
 for form in ("full","order_no_yes","map_ab","map_ba","equivalent"):
  vs=[v for v in f.views if v["split"]=="dev" and v["variant"]==form]
  values.append((float(np.mean([(f.ell(v["view_id"])>0)==bool(v["label"]) for v in vs])) if vs else -1,form))
 best=max(values,key=lambda x:(x[0],x[1]=="full"))[1]
 write(root/"settings.json",dict(best_template=best,template_dev_original_accuracy=dict((name,score) for score,name in values),
   CAD_alpha=2.,reference_forward_count=1,ITI_alpha=15.,ITI_heads=len(heads),ITI_fit_originals=len(base),
   SEA_positive_ratio=.998,SEA_negative_ratio=.998,SEA_layer="final residual",SEA_combination="official l2_norm",
   sources={"LoReFT":"https://github.com/stanfordnlp/pyreft","LoFiT":"https://github.com/fc2869/lo-fit",
    "ITI":"https://github.com/likenneth/honest_llama","SEA":"https://github.com/yfqiu-nlp/sea-llm"},
   note="ITI/SEA binary-task adaptation uses correct and incorrect fit-answer completions. Existing closed-form settings retained. No new hyperparameter search."))
def sea_class():
 # Import the official standalone class without its obsolete generation imports.
 source=(ROOT/"vendor/sea/src/decoding_algorithm/inference.py").read_text();module=ast.parse(source)
 cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=="AdapterLayer")
 scope={"torch":torch};exec(compile(ast.Module(body=[cls],type_ignores=[]),"<official SEA AdapterLayer>","exec"),scope)
 return scope["AdapterLayer"]
class Comparators:
 def __init__(self,model,all_forms=False):
  self.all_forms=all_forms;self.prompt_cache={}
  from experiments.aster_learnable_m3.m3 import LearnableM3
  self.eng=Engine(model);self.model=model;self.root=ROOT/model/"comparators";self.cfg=read(self.root/"settings.json")
  self.ops={name:torch.load(self.root/(name+".pt"),map_location=self.eng.device,weights_only=False) for name in ("closed_form","leace","iti")}
  sea=torch.load(self.root/"sea.pt",map_location=self.eng.device,weights_only=False)
  self.sea=sea_class()(sea["positive"],sea["negative"],"l2_norm",None);self.sea_zero_norm_fallbacks=0
  with (self.root/"reference_fusion.pkl").open("rb") as f:self.fusion=pickle.load(f)
  self.m3=None
  if model=="qwen":
   old=Path("/home/gcx/TRIAGE/results/aster_learnable_m3_20261106")
   self.m3=LearnableM3(torch.load(old/"frozen_m1_m2.pt",map_location="cpu",weights_only=False),self.eng.n_heads).to(self.eng.device)
   self.m3.load_state_dict(torch.load(old/"reference/selected.pt",map_location=self.eng.device,weights_only=False)["state_dict"]);self.m3.eval()
 def edited(self,context,name,refheads=None):
  handles=[]
  def head_hook(layer):
   def hook(module,args):
    value=args[0];raw=value.reshape(1,1,self.eng.n_heads,self.eng.head_dim);work=raw.clone()
    if name=="legacy_m3":
     for h in self.m3.heads_by_layer.get(layer,[]):
      z=raw[0,0,h%self.eng.n_heads];edited,_=self.m3.edit(h,z,refheads[h]);work[0,0,h%self.eng.n_heads]=edited
    else:
     for h,op in self.ops[name].items():
      h=int(h)
      if h//self.eng.n_heads!=layer:continue
      z=raw[0,0,h%self.eng.n_heads].float()
      if name=="iti":edited=z+op
      else:edited=z-op["N"]@(op["B"]@(z-op["mu"]))
      work[0,0,h%self.eng.n_heads]=edited.to(value.dtype)
    return (work.reshape_as(value),)+args[1:]
   return hook
  if name=="sea":
   def hook(module,args,out):
    value=out[0] if isinstance(out,tuple) else out
    edited=self.sea(value)
    bad=~torch.isfinite(edited);self.sea_zero_norm_fallbacks+=int(bad.sum())
    edited=torch.where(bad,value,edited)
    return (edited,)+out[1:] if isinstance(out,tuple) else edited
   handles.append(self.eng.layers[-1].register_forward_hook(hook))
  else:
   for i,l in enumerate(self.eng.layers):handles.append(l.self_attn.o_proj.register_forward_pre_hook(head_hook(i)))
  try:return self.eng.run(context,capture=False)["ell"]
  finally:
   for h in handles:h.remove()
 def score(self,text,cached=None,cached_forms=None):
  cached_forms=cached_forms or {}
  if text in self.prompt_cache:return self.prompt_cache[text]
  eng=self.eng;ref=eng.score(reference(text),capture=self.m3 is not None);context=eng.prepare(text)
  full=eng.run(context,capture=False)["ell"] if cached is None else cached
  values=dict(full=full,direct_reference=ref["ell"],CAD=3*full-2*ref["ell"],
   reference_fusion=float(self.fusion.decision_function([[full,ref["ell"]]])[0]))
  for name in ("closed_form","leace","iti","sea"):values[name]=self.edited(context,name)
  if self.m3 is not None:values["legacy_m3"]=self.edited(context,"legacy_m3",ref["heads"].flatten(0,1).to(eng.device))
  del context
  best=self.cfg["best_template"];forms=dict(cached_forms,full=full)
  needed=("order_no_yes","map_ab","map_ba","equivalent") if self.all_forms else (best,)
  for form in needed:
   if form not in forms:
    txt,labels=format_prompt(text,form);forms[form]=eng.score(txt,labels,capture=False)["ell"]
  values["best_template"]=forms[best]
  if self.all_forms:
   values.update({"template_"+name:value for name,value in forms.items()})
  self.prompt_cache[text]=values
  return values
def evaluate(model,split,shard=0,shards=3):
 if split=="test" and not read(ROOT/model/"frozen.json").get("frozen"):raise RuntimeError("Test must be frozen")
 torch.set_num_threads(4);start=time.time();comp=Comparators(model,all_forms=split=="dev");root=ROOT/model/"comparators";target=root/(split+"_"+str(shard)+".jsonl")
 saved={r["view_id"]:r["scores"] for r in rows(target)}
 f=Features(model,(split,));views=[v for v in rows(ROOT/"data/views.jsonl") if v["split"]==split and v["kind"] in {"original","A","P","C","grid"}][shard::shards]
 for v in views:
  if v["view_id"] in saved:continue
  cached_forms={form:f.ell(v["group_id"]+":"+form) for form in ("full","order_no_yes","map_ab","map_ba","equivalent")
   if v["variant"]=="full" and v["group_id"]+":"+form in f.scores}
  scores=comp.score(v["prompt"],f.ell(v["view_id"]) if v["view_id"] in f.scores else None,cached_forms)
  saved[v["view_id"]]=scores;append(target,dict(view_id=v["view_id"],scores=scores))
 names=list(next(iter(saved.values())))
 ps=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"]==split]
 metrics={"scope":"shard scores only; aggregate paired metrics computed by report after all shards finish"}
 pathfile=root/("paths_"+split+"_"+str(shard)+".jsonl");done={r["case_id"] for r in rows(pathfile)}
 for case in [c for c in rows(ROOT/"data/path_cases.jsonl") if c["split"]==split][shard::shards]:
  if case["case_id"] in done:continue
  tree=canonical_guide(case["guideline"]);nodes={n["id"]:n for n in tree["nodes"]}
  current={name:tree["nodes"][0]["id"] for name in names};scores={name:{} for name in names};cache={}
  while True:
   active={g for n,g in current.items() if g in nodes and g not in scores[n]}
   if not active:break
   for gid in active:
    if gid not in cache:cache[gid]=comp.score(prompt(case["guideline"],case["definitions"],case["patient"],gid))
    for name in names:
     if current[name]==gid:
      value=cache[gid][name];scores[name][gid]=value;current[name]=nodes[gid]["yes" if value>0 else "no"]
  append(pathfile,dict(case_id=case["case_id"],source_id=case["source_id"],methods={n:execute(case,scores[n]) for n in names}))
 rs=rows(pathfile)
 write(root/(split+"_"+str(shard)+"_metrics.json"),dict(pairs=metrics,paths={n:dict(n=len(rs),correct=sum(r["methods"][n]["correct"] for r in rs)) for n in names},
   seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,counts=comp.eng.counts,SEA_zero_norm_fallback_positions=comp.sea_zero_norm_fallbacks,
   reference_test_labels=False,forwards=dict(full=1,CAD=2,reference_fusion=2,direct_reference=1,legacy_m3=2,best_template=1,symmetric_yes_no=2,symmetric_AB=2,iti=1,sea=1,closed_form=1,leace=1)))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--model",default="qwen");p.add_argument("--split",default="dev");p.add_argument("--fit",action="store_true")
 p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=3)
 a=p.parse_args();fit(a.model) if a.fit else evaluate(a.model,a.split,a.shard,a.shards)
