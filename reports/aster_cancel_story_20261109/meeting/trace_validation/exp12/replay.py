"""Replay saved fitted parameters against saved real development predictions."""
import csv,gzip
import numpy as np
import torch
from torch import nn
from run import HERE,Data,project,arr_pairs,metrics,dump
torch.set_num_threads(8)
d=Data()
dev,dl,dr,dy=arr_pairs(d,("A","P"),"dev")
am=np.array([p["kind"]=="A" for p in dev])
lookup={}
with gzip.open(HERE/"pair_predictions.csv.gz","rt",newline="") as f:
 for r in csv.DictReader(f):
  if r["split"]=="dev":
   lookup[(r["method"],r["seed"],r["pair_id"])]=float(r["prediction"])
raw=d.raw(d.heads)
with np.load(HERE/"selected_support_r16.npz") as a:sup={k:a[k] for k in a.files}
z=project(raw,sup)
results=[]
def check(method,seed,pd):
 saved=np.array([lookup[(method,seed,p["pair_id"])] for p in dev])
 err=float(np.max(np.abs(saved-pd)))
 np.testing.assert_allclose(pd,saved,rtol=1e-5,atol=1e-5)
 results.append(dict(method=method,seed=seed,dev_pairs=len(dev),max_absolute_error=err,A=metrics(dy[am],pd[am])))
with np.load(HERE/"selected_N16_model.npz") as a:beta=a["coef"]
check("selected_N16","",(z[dr]-z[dl])@beta)
rng=np.random.default_rng(11);q,_=np.linalg.qr(rng.normal(size=(raw.shape[1],16)))
rz=(((raw-sup["mu"])/sup["scale"])@q.astype(np.float32)).astype(np.float64)
with np.load(HERE/"random_subspace16_seed11_model.npz") as a:beta=a["coef"]
check("random_subspace16","11",(rz[dr]-rz[dl])@beta)
with np.load(HERE/"random_heads51_seed11_support.npz") as a:rs={k:a[k] for k in a.files}
rz=project(d.raw(rs["heads"]),rs)
with np.load(HERE/"random_heads51_seed11_model.npz") as a:beta=a["coef"]
check("random_heads51","11",(rz[dr]-rz[dl])@beta)
for seed in (11,22,33):
 with np.load(HERE/("mlp16_seed"+str(seed)+"_scaler.npz")) as a:sc={k:a[k] for k in a.files}
 mod=nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Linear(32,1))
 mod.load_state_dict(torch.load(HERE/("mlp16_seed"+str(seed)+".pt"),map_location="cpu",weights_only=True));mod.eval()
 with torch.no_grad():
  pd=mod(torch.tensor(((z[dr]-z[dl])-sc["x_mean"])/sc["x_scale"],dtype=torch.float32)).squeeze(1).numpy()*sc["y_scale"]+sc["y_mean"]
 check("mlp16",str(seed),pd)
dump("replay_check.json",dict(status="passed",data_origin="real_cache_parameter_replay",checks=results,
 requirement="Saved chosen16, random-projection seed11, random-head seed11 and all three neural models reproduce saved development pair predictions.",
 tolerance=dict(relative=1e-5,absolute=1e-5),refit_required=False))

