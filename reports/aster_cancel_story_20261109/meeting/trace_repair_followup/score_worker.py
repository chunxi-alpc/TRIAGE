"""Resumable canonical Qwen3-8B scoring. No model fitting; final split requires freeze."""
import os,sys,json,time,argparse,pickle,traceback
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
REPO=H.parents[3]
sys.path.insert(0,str(REPO))
CACHE=Path("/home/gcx/aster_trace_repair_20261110")
OLD=Path("/home/gcx/aster_cr_repair_v2_run_20261108")
def rows(p):
 p=Path(p)
 if not p.exists():return []
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 temp=p.with_suffix(p.suffix+".tmp")
 temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n");temp.replace(p)
def append(p,x):
 with Path(p).open("a") as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+"\n");f.flush()
def union_config():
 with np.load(OLD/"qwen/supports_r16.npz") as a:main=a["heads"].astype(int).tolist()
 hh=set(main)
 for fold in range(5):
  p=REPO/"reports/aster_cancel_story_20261109/meeting/trace_validation/exp3/full"/("fold"+str(fold))/"configuration.json"
  hh.update(json.loads(p.read_text())["heads"])
 return dict(main_heads=main,head_ids=sorted(hh))
def build_reuse():
 # Exact prompt matching only. No file hashes, no cached edited-to-original substitution.
 vv={v["view_id"]:v for v in rows(OLD/"data/views.jsonl")}
 out={}
 for p in sorted((OLD/"qwen/scores").glob("*.jsonl")):
  for r in rows(p):
   v=vv.get(r.get("view_id"))
   if v and r.get("status")=="ok" and Path(r.get("feature","")).is_file():out[v["prompt"]]=dict(ell=r["ell"],feature=r["feature"],origin=str(p))
 CACHE.mkdir(parents=True,exist_ok=True)
 with (CACHE/"reuse_index.pkl").open("wb") as f:pickle.dump(out,f)
 dump(H/"reuse_manifest.json",dict(exact_prompt_records=len(out),source_root=str(OLD),policy="Reuse exact input text and canonical score protocol; only full-head caches containing the requested head union are accepted."))
def valid_cached(r,head_ids):
 try:
  with np.load(r["feature"]) as a:
   if "heads" not in a or "residual" not in a:return False
   z=a["heads"]
   if z.ndim==3:return z.reshape(-1,z.shape[-1]).shape[0]>max(head_ids)
   if "head_ids" in a:return set(head_ids)<=set(a["head_ids"].tolist())
   return False
 except (OSError,ValueError):return False
def score_selected(engine,text,selected):
 import torch
 started=time.perf_counter();context=engine.prepare(text)
 bylayer={};captured={};residual={};handles=[]
 for h in selected:bylayer.setdefault(int(h)//engine.n_heads,[]).append(int(h)%engine.n_heads)
 def head_hook(layer,indices):
  def hook(module,args):
   value=args[0]
   if value.shape!=(1,1,engine.n_heads*engine.head_dim):raise RuntimeError("Unexpected final-token shape")
   captured[layer]=value[0,0].reshape(engine.n_heads,engine.head_dim)[indices].detach()
  return hook
 def residual_hook(module,args,out):
  v=out[0] if isinstance(out,tuple) else out
  residual["h"]=v[0,-1].detach()
 before=time.perf_counter();engine._active=True
 try:
  for layer,indices in bylayer.items():handles.append(engine.layers[layer].self_attn.o_proj.register_forward_pre_hook(head_hook(layer,indices)))
  handles.append(engine.layers[-1].register_forward_hook(residual_hook))
  with torch.no_grad():
   embeds=engine.model.get_input_embeddings()(context.input_ids[:,-1:]).detach()
   output=engine.model.model(inputs_embeds=embeds,attention_mask=torch.ones_like(context.input_ids),
      past_key_values=engine._cache(context),use_cache=True,cache_position=torch.tensor([context.length-1],device=engine.device),
      position_ids=torch.tensor([[context.length-1]],device=engine.device))
   ell=float(engine._score_hidden(output.last_hidden_state[0,-1]))
   positions={(layer,head):j for layer,ii in bylayer.items() for j,head in enumerate(ii)}
   value=torch.stack([captured[int(h)//engine.n_heads][positions[int(h)//engine.n_heads,int(h)%engine.n_heads]] for h in selected])
   raw=value.float().cpu().numpy()
   res=residual["h"].float().cpu().numpy()
  engine.counts["final_forwards"]+=1;engine.counts["final_seconds"]+=time.perf_counter()-before
  return dict(ell=ell,tokens=context.length,inference_seconds=time.perf_counter()-started),raw,res
 finally:
  for handle in handles:handle.remove()
  engine._active=False
def run(args):
 if args.split=="final" and not (H/"evaluation/frozen.json").exists():raise RuntimeError("Final scoring is locked until evaluation/frozen.json exists")
 import torch
 torch.set_num_threads(2);torch.cuda.set_device(args.gpu)
 cfg=union_config();hh=cfg["head_ids"];main=np.array([hh.index(h) for h in cfg["main_heads"]])
 out=CACHE/"scoring"/args.split/("shard"+str(args.shard));out.mkdir(parents=True,exist_ok=True)
 (out/"features").mkdir(exist_ok=True)
 taskfile=CACHE/"data"/("tasks_"+("diagnostic_" if args.diagnostic else "")+args.split+".jsonl")
 tasks=[r for i,r in enumerate(rows(taskfile)) if i%args.shards==args.shard]
 if args.limit is not None:tasks=tasks[:args.limit]
 scores=out/("pilot_scores.jsonl" if args.pilot else "scores.jsonl")
 done={r["id"]:r for r in rows(scores)}
 # Pilot outputs can be reused by the complete run without re-scoring.
 if not args.pilot:
  for r in rows(out/"pilot_scores.jsonl"):
   if r["id"] not in done:append(scores,r);done[r["id"]]=r
 reuse={}
 if (CACHE/"reuse_index.pkl").exists():
  with (CACHE/"reuse_index.pkl").open("rb") as f:reuse=pickle.load(f)
 from experiments.aster_cr.engine import Engine
 engine=None;started=time.time();nnew=0;nreuse=0
 for i,t in enumerate(tasks):
  if t["id"] in done:continue
  rec={k:t.get(k) for k in ("id","source_id","case_id","gid","variant","split","label","kind")}
  rec["task_id"]=t["id"]
  cached=reuse.get(t["prompt"])
  if cached and valid_cached(cached,hh):
   rec.update(ell=cached["ell"],feature=cached["feature"],status="ok",reused_from=cached["origin"])
   nreuse+=1
  else:
   if engine is None:engine=Engine("qwen",device="cuda:"+str(args.gpu))
   result,raw,res=score_selected(engine,t["prompt"],hh)
   p=out/"features"/(("diagnostic_" if args.diagnostic else "")+str(i).zfill(7)+".npz")
   tmp=p.with_suffix(".npz.tmp")
   with tmp.open("wb") as f:np.savez_compressed(f,heads=raw,head_ids=np.array(hh),selected_heads=raw[main].reshape(-1),residual=res)
   tmp.replace(p)
   rec.update(result,feature=str(p),status="ok");nnew+=1
  append(scores,rec);done[t["id"]]=rec
  if (i+1)%50==0:
   dump(out/("pilot_status.json" if args.pilot else "diagnostic_status.json" if args.diagnostic else "status.json"),dict(state="running",split=args.split,gpu=args.gpu,shard=args.shard,
    completed=i+1,expected=len(tasks),new_forwards=nnew,reused=nreuse,seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(args.gpu)/2**30))
 dump(out/("pilot_done.json" if args.pilot else "diagnostic_done.json" if args.diagnostic else "done.json"),dict(state="complete",split=args.split,gpu=args.gpu,shard=args.shard,
  expected=len(tasks),complete=sum(t["id"] in done for t in tasks),new_forwards=nnew,reused=nreuse,
  seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated(args.gpu)/2**30,
  counts=engine.counts if engine else {},head_ids=hh,main_heads=cfg["main_heads"]))
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true")
 p.add_argument("--split",choices=["fit","dev","final"]);p.add_argument("--gpu",type=int,default=0)
 p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=3)
 p.add_argument("--diagnostic",action="store_true");p.add_argument("--limit",type=int);p.add_argument("--pilot",action="store_true")
 args=p.parse_args()
 if args.prepare:dump(H/"scoring_config.json",union_config());build_reuse()
 else:
  try:run(args)
  except Exception as exc:
   out=CACHE/"scoring"/str(args.split)/("shard"+str(args.shard))
   dump(out/"failed.json",dict(error=str(exc),traceback=traceback.format_exc()));raise
if __name__=="__main__":main()

