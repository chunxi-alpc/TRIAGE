"""Shared loader: real repair-v2 Qwen3-8B caches; never writes original caches."""
import os,sys,json,time
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO))
ROOT=Path(os.environ.get("TRACE_DATA_ROOT","/home/gcx/aster_cr_repair_v2_run_20261108"))
CACHE=Path(os.environ.get("TRACE_CACHE_ROOT","/home/gcx/aster_trace_validation_cache_20261109"))
OUT=Path(__file__).resolve().parent
from experiments.aster_cr.features import basis,orth_nuisance
def rows(p):
 p=Path(p)
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def dump(p,obj):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
class Data:
 def __init__(self,splits=("fit","dev","test")):
  self.root=ROOT; self.scores={}
  for p in sorted((ROOT/"qwen/scores").glob("*.jsonl")):
   for r in rows(p):
    if r.get("status")=="ok" and r.get("split") in splits:self.scores[r["view_id"]]=r
  vv={v["view_id"]:v for v in rows(ROOT/"data/views.jsonl") if v["split"] in splits and v["view_id"] in self.scores}
  self.ids=sorted(vv);self.ix={v:i for i,v in enumerate(self.ids)}
  self.views=[vv[v] for v in self.ids]
  self.pairs=[p for p in rows(ROOT/"data/pairs.jsonl") if p["split"] in splits and p["left"] in vv and p["right"] in vv]
  self.ell=np.array([self.scores[v]["ell"] for v in self.ids],float)
  self.labels=np.array([v["label"] for v in self.views])
  self.heads=json.loads((ROOT/"qwen/selection.json").read_text())["heads"]
  self._all=None;self._raw={}
 def all_raw(self):
  if self._all is not None:return self._all
  CACHE.mkdir(parents=True,exist_ok=True)
  dest=CACHE/"all_heads.npy";meta=CACHE/"all_heads_ids.json"
  if not dest.exists() or not meta.exists() or json.loads(meta.read_text())!=self.ids:
   byfile={}
   for i,v in enumerate(self.ids):byfile.setdefault(self.scores[v]["feature"],[]).append(i)
   with np.load(next(iter(byfile))) as a:dim=a["heads"].size
   arr=np.lib.format.open_memmap(dest,mode="w+",dtype="float32",shape=(len(self.ids),dim))
   for j,(file,ii) in enumerate(byfile.items()):
    with np.load(file) as a:arr[ii]=a["heads"].reshape(-1).astype(np.float32)
    if (j+1)%2000==0:print("cache files",j+1,"/",len(byfile),flush=True)
   arr.flush();del arr
   dump(meta,self.ids)
  self._all=np.load(dest,mmap_mode="r")
  return self._all
 def raw(self,heads=None):
  a=self.all_raw()
  if heads is None:return a
  key=tuple(map(int,heads))
  if key not in self._raw:
   # Retain only current and selected matrices, bounding memory in random controls.
   for k in list(self._raw):
    if k!=tuple(self.heads):del self._raw[k]
   cols=(np.array(key)[:,None]*128+np.arange(128)[None,:]).ravel()
   self._raw[key]=np.asarray(a[:,cols],dtype=np.float32)
  return self._raw[key]
 def support(self,heads,rank,train_sources=None):
  raw=self.raw(heads)
  ii=np.array([i for i,v in enumerate(self.views) if v["split"]=="fit" and v["variant"]=="full" and (train_sources is None or v["source_id"] in train_sources)])
  x=raw[ii];mu=x.mean(0);scale=np.maximum(x.std(0),1e-5)
  nd=[];sd=[]
  for p in self.pairs:
   if p["split"]!="fit" or p["kind"] not in ("A","P","C","F") or (train_sources is not None and p["source_id"] not in train_sources):continue
   d=(raw[self.ix[p["right"]]]-raw[self.ix[p["left"]]])/scale
   (nd if p["kind"] in ("A","P") else sd).append(d)
  S=basis(np.asarray(sd),rank);U=basis(np.asarray(nd),rank);N=orth_nuisance(S,U)
  return dict(heads=np.array(heads),mu=mu,scale=scale,S=S,N=N,U=U)
 def pair_arrays(self,kinds,split,features):
  pp=[p for p in self.pairs if p["kind"] in kinds and p["split"]==split]
  a=np.array([self.ix[p["left"]] for p in pp],int);b=np.array([self.ix[p["right"]] for p in pp],int)
  return pp,features[b]-features[a],self.ell[b]-self.ell[a]
def prepare():
 import subprocess
 d=Data();d.all_raw()
 manifest=rows(ROOT/"data/views.jsonl")
 src={s:sorted({v["source_id"] for v in manifest if v["split"]==s}) for s in ("fit","dev","test")}
 bad=[]
 vv={v["view_id"]:v for v in manifest}
 for p in rows(ROOT/"data/pairs.jsonl"):
  a,b=vv[p["left"]],vv[p["right"]]
  if a["split"]!=b["split"] or a["split"]!=p["split"] or a["source_id"]!=b["source_id"] or a["source_id"]!=p["source_id"]:bad.append(p["pair_id"])
 from collections import Counter
 dump(OUT/"data_audit.json",dict(data_kind="real_cache_recomputation",root=str(ROOT),repo_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
  dirty_at_start=subprocess.check_output(["git","status","--short"],cwd=REPO,text=True).splitlines(),source_ids=src,
  source_overlaps={a+"_"+b:sorted(set(src[a])&set(src[b])) for a,b in (("fit","dev"),("fit","test"),("dev","test"))},
  inconsistent_pairs=bad,available_views=dict(Counter(v["split"] for v in d.views)),
  expected_views=dict(Counter(v["split"] for v in manifest)),
  available_pairs={s:dict(Counter(p["kind"] for p in d.pairs if p["split"]==s)) for s in src},
  selected_heads=d.heads,all_heads_shape=list(d.all_raw().shape),
  history="Development used in prior method selection and analysis; test used in historical and real_completion evaluations, not a new independent test.",
  preprocessing="Fit sources only for head selection, coordinate normalization, subspaces and coefficients; fold preprocessing refit within train sources. Historical frozen selector treated separately.",
  evidence="Score/features are real Qwen3-8B repair-v2 caches; no synthetic numbers or fresh model forwards in CPU stage."))
if __name__=="__main__":prepare()

