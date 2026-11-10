"""Small interface check on real legacy caches; not an experiment or model selection."""
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train as m
oldpairs=m.rows(m.LEGACY/"data/pairs.jsonl")
ap=[p for p in oldpairs if p["split"]=="fit" and p["kind"]=="A"][:20]
cf=[p for p in oldpairs if p["split"]=="fit" and p["kind"] in ("C","F")][:6]
want={p[k] for p in ap+cf for k in ("left","right")}
vv={v["view_id"]:v for v in m.iterrows(m.LEGACY/"data/views.jsonl") if v["view_id"] in want}
ss={}
for f in (m.LEGACY/"qwen/scores").glob("fit_*.jsonl"):
 for r in m.iterrows(f):
  if r.get("status")=="ok" and r["view_id"] in want:ss[r["view_id"]]=r
ids=sorted(want);ix={v:i for i,v in enumerate(ids)}
d=m.Dataset.__new__(m.Dataset)
d.ids=ids;d.ix=ix;d.pairs=ap+cf
natural={p["left"] for p in ap};inter={p["right"] for p in ap}
d.tasks=[dict(id=v,source_id=vv[v]["source_id"],case_id=vv[v].get("case_id",vv[v]["group_id"]),
 variant="original" if v in natural else "A" if v in inter else "evidence") for v in ids]
d.sources=np.array([t["source_id"] for t in d.tasks])
d.ell=np.array([ss[v]["ell"] for v in ids]);d.labels=np.array([vv[v]["label"] for v in ids])
d.natural=np.array([v in natural for v in ids]);d.interference=np.array([v in inter for v in ids])
sup=m.support();raw=[];res=[]
for v in ids:
 with np.load(ss[v]["feature"]) as a:
  raw.append(m.select_raw(a["heads"],None,sup["heads"]))
  res.append(a["residual"].reshape(-1))
raw=np.stack(raw);res=np.stack(res)
t=((raw-sup["mu"])/sup["scale"])@sup["N"]
x=np.column_stack([t,d.ell]);g=m.index_groups(d,set(d.sources))
root=m.ROOT/"training_cache/interface_smoke";root.mkdir(parents=True,exist_ok=True)
np.savez_compressed(root/"support.npz",**sup)
q=np.linalg.qr(np.random.default_rng(11).normal(size=(len(sup["mu"]),16)),mode="reduced")[0].astype("float32")
np.savez_compressed(root/"random_basis.npz",Q=q)
c=m.tagged_config("label_linear",.1,1.,.1)
lin,ls,li=m.fit_linear(d,t.astype(float),g,c);m.save_model(root/"linear.npz",lin)
c=m.tagged_config("trace_nn",.1,1.,.1)
net,ns,ni=m.fit_nn(d,x,g,c,11,epochs=50);m.save_model(root/"network.pt",net)
m.dump(root/"models.json",dict(models=[
 dict(name="lin",family="trace",architecture="label_linear",model_file=str(root/"linear.npz"),main=True),
 dict(name="net",family="trace",architecture="trace_nn",model_file=str(root/"network.pt"),main=True),
 dict(name="frozen_strict",family="baseline",architecture="frozen_optimized",frozen_name="optimized_strict_fit_calibration",main=True)]))
loaded=m.load_models(root)
out=loaded.predict_all(d.ell,raw.reshape(-1,51,128),sup["heads"],res)
linear_error=float(np.max(np.abs(out["lin"]-ls)))
nn_error=float(np.max(np.abs(out["net"]-ns)))
np.testing.assert_allclose(out["lin"],ls,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(out["net"],ns,rtol=1e-6,atol=1e-6)
frozen_errors=[]
for i in range(min(3,len(ids))):
 old=loaded.frozen.predict(float(d.ell[i]),raw[i],res[i])["optimized_strict_fit_calibration"]
 frozen_errors.append(abs(old-out["frozen_strict"][i]))
np.testing.assert_allclose(frozen_errors,0,atol=3e-5)
check=dict(status="passed",data_origin="real_legacy_cache_interface_check_not_scientific_evaluation",rows=len(ids),
 linear_export_replay_max_error=linear_error,NN_export_replay_max_error=nn_error,frozen_vectorized_max_error=max(frozen_errors),
 linear_max_CF=float(np.max(np.abs((ls-d.ell)[g["cr"]]-(ls-d.ell)[g["cl"]]))),
 neural_max_CF=float(np.max(np.abs((ns-d.ell)[g["cr"]]-(ns-d.ell)[g["cl"]]))),
 evidence_limit=.1*g["scale"],nominal_network_parameters=609,effective_score_only_network_parameters=97,
 loss_averaging="equal source -> case -> variant -> node",CV_selection="gold path exact success, not node accuracy")
m.dump(m.OUT/"interface_check.json",check)

