"""Automatic preliminary experiment; reuse existing annotations and historical results."""
import json,sys,csv
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
ROOT=Path('/home/gcx/three_dataset_analysis_20261105')
INVENTORY_PATH=Path('/home/gcx/experiment_data_reuse_20261105/data_inventory.json')
INVENTORY=json.loads(INVENTORY_PATH.read_text())
REPO=Path(next(g['root'] for g in INVENTORY['groups'] if g['dataset']=='MedPAIR' and g['experiment']=='PEB S8 and PCD S6'))
HOME=Path('/home/gcx')
MODEL=HOME/'TRIAGE/models/modelscope/Qwen3-8B'
for d in ('data','results','logs'): (ROOT/d).mkdir(parents=True,exist_ok=True)
def rows(p):
 p=Path(p)
 if not p.exists(): return []
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def write_json(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_rows(p,x):
 with Path(p).open('w') as f:
  for r in x: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def lp(vals):
 a=np.asarray(vals,dtype=float); assert np.isfinite(a).all()
 return a-logsumexp(a)
def score_row(raw,labels=None):
 labels=labels or sorted(raw['option_logits'])
 z=[raw['option_logits'][k] for k in labels]
 l=lp(z); return labels,l,labels[int(np.argmax(l))]
def bootstrap(num,den=None,groups=None):
 num=np.asarray(num,dtype=float)
 den=np.ones(len(num)) if den is None else np.asarray(den,dtype=float)
 if not len(num) or den.sum()==0: return dict(value=None,numerator=float(num.sum()),denominator=float(den.sum()),ci95=None)
 gs=list(range(len(num))) if groups is None else groups
 uniq,inv=np.unique(gs,return_inverse=True); n=len(uniq)
 if groups is not None and n==1:
  return dict(value=float(num.sum()/den.sum()),numerator=float(num.sum()),denominator=float(den.sum()),ci95=None,resampling_units=1,ci_note='Only one guideline tree; across-tree bootstrap interval unavailable')
 ns=np.bincount(inv,weights=num,minlength=n); ds=np.bincount(inv,weights=den,minlength=n)
 rng=np.random.default_rng(42); idx=rng.integers(0,n,size=(2000,n))
 nd=ds[idx].sum(1); nn=ns[idx].sum(1); values=nn[nd>0]/nd[nd>0]
 return dict(value=float(num.sum()/den.sum()),numerator=float(num.sum()),denominator=float(den.sum()),ci95=np.quantile(values,[.025,.975]).tolist(),resampling_units=n)
def rate(flags,groups=None): return bootstrap(flags,groups=groups)
def mean_ci(values,groups=None): return bootstrap(values,groups=groups)
def csv_table(path,data):
 if not data: return
 keys=list(dict.fromkeys(k for row in data for k in row))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for row in data:
   w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
