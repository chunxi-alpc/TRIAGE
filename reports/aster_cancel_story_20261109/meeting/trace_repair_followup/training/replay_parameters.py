"""Replay exported fitted models on real expanded-fit records, never dev/final."""
import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train as m
started=time.time()
d=m.Dataset("fit")
keep=[i for i,t in enumerate(d.tasks) if not t["id"].startswith("legacy_cf|")][:64]
heads=[];residual=[]
for i in keep:
 with np.load(d.scores[d.ids[i]]["feature"]) as a:
  h=a["heads"] if "heads" in a else a["selected_heads"]
  hh=a["head_ids"] if "head_ids" in a else None
  heads.append(m.select_raw(h,hh,m.support()["heads"]).reshape(51,128))
  residual.append(a["residual"].reshape(-1).astype(np.float32))
h=np.stack(heads);r=np.stack(residual);e=d.ell[keep];ids=[d.ids[i] for i in keep]
one=m.load_models(m.OUT);p=one.predict_all(e,h,one.support["heads"],r)
two=m.load_models(m.OUT);q=two.predict_all(e,h,two.support["heads"],r)
checks=[]
for name in one.names:
 err=float(np.max(np.abs(p[name]-q[name])))
 np.testing.assert_allclose(p[name],q[name],rtol=1e-6,atol=1e-6)
 scalar=two.correct(name,float(e[0]),h[0],two.support["heads"],r[0])
 scalar_error=abs(scalar-float(q[name][0]))
 np.testing.assert_allclose(scalar,q[name][0],rtol=1e-5,atol=3e-5)
 checks.append(dict(method=name,parameter_reload_max_error=err,scalar_batch_error=scalar_error))
rows=[]
for name,pred in p.items():
 for i,score in zip(ids,pred):rows.append(dict(id=i,method=name,score=float(score)))
m.csvout(m.OUT/"api_validation_predictions.csv.gz",rows)
m.dump(m.OUT/"api_validation.json",dict(status="passed",data_origin="real_expanded_fit_cache_parameter_replay",
 rows=len(keep),models=len(one.names),checks=checks,ids=ids,seconds=time.time()-started,
 final_or_development_data_used=False,
 procedure="Reload exported parameter files independently; compare complete prediction API and scalar versus batch for all models. No training, source selection or final-result inspection."))

