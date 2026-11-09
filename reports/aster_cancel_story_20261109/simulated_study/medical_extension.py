"""CPU-only prespecified score simulations; no model/data loading or fitting.
Run: python medical_extension.py. Existing simulations are never overwritten.
"""
from pathlib import Path
import csv,json
import numpy as np

OUT=Path(__file__).resolve().parent
LABEL="预设模拟，未运行对应模型与数据集，待真实评测替换。"
CONFIG={
 "status":"prespecified_simulation", "notice":LABEL, "generation_version":2,
 "seeds":[11,22,33,44,55],
 "models":{
  "Qwen3-8B":{"evidence":1.0,"nuisance":1.0,"noise":1.0,"trace_quality":1.0},
  "Llama-3.1-8B-Instruct":{"evidence":0.94,"nuisance":1.08,"noise":1.06,"trace_quality":0.85},
  "Gemma-2-9B-IT":{"evidence":1.04,"nuisance":0.92,"noise":1.08,"trace_quality":0.90},
  "Gemma-3-27B-IT":{"evidence":1.10,"nuisance":0.97,"noise":0.98,"trace_quality":1.05},
  "MedGemma-27B":{"evidence":1.12,"nuisance":1.02,"noise":1.04,"trace_quality":0.98}},
 "methods":{
  "raw":{"name":"原模型","g":1.0,"a":1.0,"c":0.0,"d":0.0,"b":0.35},
  "calibration":{"name":"仅校准","g":1.0,"a":1.0,"c":0.0,"d":0.0,"b":0.0},
  "unconstrained":{"name":"无证据约束","g":0.75,"a":0.30,"c":0.25,"d":0.15,"b":0.0},
  "linear":{"name":"线性 ASTER-CR","g":0.995,"a":0.95,"c":-0.10,"d":0.015,"b":0.0},
  "balanced_mlp":{"name":"平衡 MLP","g":0.97,"a":0.72,"c":-0.30,"d":0.10,"b":0.0},
  "protected_mlp":{"name":"强保护 MLP","g":0.99,"a":0.86,"c":-0.16,"d":0.06,"b":0.0}},
 "tasks":{
  "MedPIC":{"n":467,"options":5,"evidence_pairs":92,"margin":1.6,"nuisance":0.9},
  "MedPAIR":{"n":898,"options":5,"margin":2.5,"nuisance":0.8},
  "MedEinst":{"n":5383,"options":49,"evidence_pairs":5383,"margin":3.2,"trap_margin":2.1,"nuisance":0.8},
  "MedGUIDE":{"n":7747,"options":5,"groups":17,"margin":2.3,"nuisance":0.9},
  "TrialGPT":{"n":1000,"unknown":250,"positive_probability":0.65,"margin":1.6,"nuisance":1.0}},
 "equation":"s_m=g_m E + a_eff N + c_m (T+H*E) + d_m (X+J*E) + B + b_m; a_eff=1-(1-a_m)*trace_quality",
 "evidence_trace_coupling":{"H":"N(0,1)","J":"N(0,1)","rng_version":2,"sharing":"Same H/J across paired views and all methods; separate RNG keeps v1 latent inputs unchanged.","interpretation":"Assumed evidence dependence of score-correction features; not an estimated or trained method."},
 "noise":{"evidence_sd":0.6,"unresolved_sd":0.35,"trace_sd":1.0,"extra_sd":1.0,"addback_sd":0.08},
 "semantics":{
  "methods":"Condition names only, not trained algorithms; no optimizer, search, tuning or ranking filter.",
  "models":"Model labels index assigned coefficients, not empirical estimates.",
  "draws":"All six conditions share latent evidence/nuisance/noise/labels and H/J within a model-task-seed.",
  "direction":"Count original-positive signed responses becoming nonpositive; rate denominator is original-positive responses. MedPIC uses changed option pairs, not case pairs. Zero eligible denominator is undefined.",
  "placeholder":"All generated values are provisional demonstration values, not predicted measurements. Replace from corresponding real paired scores before submission.",
  "v1":"Original positive-g-only pairing made sign degradation identically zero; archived in medical_extension_v1 and superseded, not evidence for protection.",
  "multiselect":"Independent yes/no score per option; predicted set is scores>0; exact set equality.",
  "single_choice":"argmax; one shared intercept for all options, hence calibration cannot change ranking.",
  "MedPIC":"92 synthetic paired cases, option identities aligned; 1-2 labels change; unchanged labels are not nuisance interventions.",
  "MedPAIR":"Low deletion changes nuisance only by assumption. Key deletion attenuates evidence to 25%. Addback restores evidence/nuisance plus shared N(0,.08) context-order noise. Deletion/addback report margin changes, never post-deletion accuracy.",
  "MedEinst":"Shared nuisance favours control label in both sides; trap changes gold class. BTR=P(trap predicts control label | control correct).",
  "TrialGPT":"1000 generated labels include 250 unknown; exclude unknown, retain included/not included and exclusion polarity as criterion satisfaction, not trial eligibility.",
  "no_pairs":"MedGUIDE/TrialGPT intervention metrics and every task path metric are undefined.",
  "uncertainty":"Mean and sample SD over five generation seeds only, not model training uncertainty or real confidence intervals. Undefined direction rates are omitted, defined seed count is recorded; SD needs at least two defined seeds."}}
def write_csv(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def rms(x):return float(np.sqrt(np.mean(x*x)))
def metric_acc(pred,y):return float(np.mean(pred==y)*100)
def balanced_f1(pred,y):
 ba=np.mean([np.mean(pred[y==c]==c) for c in [0,1]])*100
 f1=[]
 for c in [0,1]:
  tp=np.sum((pred==c)&(y==c));fp=np.sum((pred==c)&(y!=c));fn=np.sum((pred!=c)&(y==c))
  f1.append(2*tp/max(2*tp+fp+fn,1))
 return float(ba),float(np.mean(f1)*100)
def direction_metrics(d0,d,sgn):
 eligible=sgn*d0>0
 count=int(np.sum(eligible & (sgn*d<=0)))
 denom=int(np.sum(eligible))
 return {"direction_degradation":count,"direction_eligible_n":denom,
         "direction_pair_n":int(np.size(d0)),
         "direction_degradation_rate":count/denom*100 if denom else None}
def generate():
 # A deterministic counterexample: average ER=100% can conceal a reversed response.
 counter=direction_metrics(np.array([1.,1.]),np.array([3.,-1.]),np.ones(2))
 assert counter["direction_degradation"]==1 and counter["direction_degradation_rate"]==50
 assert direction_metrics(np.array([-1.]),np.array([-2.]),np.ones(1))["direction_degradation_rate"] is None
 rows=[];checks={"raw_vr_zero":True,"raw_er_100":True,"intercept_differences":True,
 "shared_intercept_argmax":True,"paired_accuracy_bounds":True,"unknown_excluded":True,
 "no_invented_interventions":True,"direction_counterexample":True,
 "zero_direction_denominator_undefined":True,"direction_rate_consistency":True,
 "raw_calibration_direction_zero":True,"shared_evidence_feature_potential_outcomes":True}
 for mi,(model,mc) in enumerate(CONFIG["models"].items()):
  for ti,(task,tc) in enumerate(CONFIG["tasks"].items()):
   for seed in CONFIG["seeds"]:
    rng=np.random.default_rng(np.random.SeedSequence([seed,mi,ti,20261109]))
    n=tc["n"];k=tc.get("options",1);shape=(n,k) if k>1 else (n,)
    if task=="MedPIC":
     y=rng.random(shape)<0.40;y[:,0]|=~y.any(axis=1)
     E=(2*y-1)*rng.uniform(.4,2.8,shape)
    elif k>1:
     y=rng.integers(k,size=n);E=rng.normal(0,.6,shape);E[np.arange(n),y]+=tc["margin"]
    else:
     y=(rng.random(n)<tc["positive_probability"]).astype(int)
     E=(2*y-1)*rng.uniform(.3,2.9,n)
    E=E*mc["evidence"];N=rng.normal(0,tc["nuisance"]*mc["nuisance"],shape)
    T=rng.normal(size=shape);X=rng.normal(size=shape);B=rng.normal(0,.35*mc["noise"],shape)
    addlow=rng.normal(0,.08,shape);addkey=rng.normal(0,.08,shape)
    Nlow=rng.normal(0,tc["nuisance"]*mc["nuisance"],shape)
    group=rng.choice(17,size=n,p=np.arange(1,18)/153) if task=="MedGUIDE" else None
    pair=None
    if task=="MedPIC":
     q=tc["evidence_pairs"];yp=y[:q].copy()
     for j in range(q):yp[j,rng.choice(k,size=1+j%2,replace=False)]^=True
     # Exact paired potential outcomes differ only in evidence on changed option labels.
     Ep=E[:q].copy();changed=yp!=y[:q];Ep[changed]*=-1
     pair=(q,yp,Ep)
    elif task=="MedEinst":
     q=n;yp=(y+rng.integers(1,k,size=n))%k
     Ep=rng.normal(0,.6,shape)*mc["evidence"]
     Ep[np.arange(n),yp]+=tc["trap_margin"]*mc["evidence"]
     N[np.arange(n),y]+=.8 # same nuisance on control/trap, no label leakage into a real input
     pair=(q,yp,Ep)
    # Draw coupling once from a separate stream; never choose it from observed outcomes.
    coupling_rng=np.random.default_rng(np.random.SeedSequence([seed,mi,ti,20261109,2]))
    H=coupling_rng.normal(size=shape);J=coupling_rng.normal(size=shape)
    def scores(m,ev=E,nuis=N,sl=None):
     g=m["g"];a=1-(1-m["a"])*mc["trace_quality"]
     if sl is None:return g*ev+a*nuis+m["c"]*(T+H*ev)+m["d"]*(X+J*ev)+B+m["b"]
     return g*ev+a*nuis+m["c"]*(T[:sl]+H[:sl]*ev)+m["d"]*(X[:sl]+J[:sl]*ev)+B[:sl]+m["b"]
    outputs={key:scores(m) for key,m in CONFIG["methods"].items()}
    assert np.allclose(outputs["raw"]-outputs["calibration"],.35)
    if k>1 and task!="MedPIC":assert np.array_equal(outputs["raw"].argmax(1),outputs["calibration"].argmax(1))
    for key,m in CONFIG["methods"].items():
     s=outputs[key];raw=outputs["raw"]
     r={"status":"prespecified_simulation","model":model,"task":task,"method":key,"method_name":m["name"],
        "seed":seed,"generated_n":n,"scored_n":n}
     if task=="MedPIC":
      correct=np.all((s>0)==y,axis=1)
      r.update(answer_set_exact=float(correct.mean()*100),option_accuracy=float(((s>0)==y).mean()*100))
     elif k>1:
      pred=s.argmax(1);correct=pred==y
      r["accuracy"]=float(correct.mean()*100)
      if task=="MedGUIDE":
       # Unequal disease group sizes; the same 17 labels are shared by all conditions.
       r["disease_macro_accuracy"]=float(np.mean([correct[group==g].mean() for g in range(17)])*100)
     else:
      known=np.arange(n)>=tc["unknown"];pred=(s>0).astype(int)
      ba,f1=balanced_f1(pred[known],y[known])
      r.update(scored_n=int(known.sum()),unknown_n=tc["unknown"],balanced_accuracy=ba,macro_f1=f1)
      assert r["scored_n"]+r["unknown_n"]==n
     if pair:
      q,yp,Ep=pair
      rawpair=scores(CONFIG["methods"]["raw"],Ep,N[:q],q)
      calpair=scores(CONFIG["methods"]["calibration"],Ep,N[:q],q)
      assert np.allclose(rawpair-outputs["raw"][:q],calpair-outputs["calibration"][:q])
      sp=scores(m,Ep,N[:q],q);rp=scores(CONFIG["methods"]["raw"],Ep,N[:q],q)
      if task=="MedPIC":
       ok1=correct[:q];ok2=np.all((sp>0)==yp,axis=1)
       direction=yp.astype(int)-y[:q].astype(int)
       d0=(rp-raw[:q])[direction!=0];d=(sp-s[:q])[direction!=0];sgn=direction[direction!=0]
      else:
       ok1=correct;ok2=sp.argmax(1)==yp;ix=np.arange(q)
       # One fixed contrast for both members: trap gold versus control gold.
       d0=(rp[ix,yp]-rp[ix,y])-(raw[ix,yp]-raw[ix,y])
       d=(sp[ix,yp]-sp[ix,y])-(s[ix,yp]-s[ix,y]);sgn=np.ones(q)
       btr=np.sum(ok1&(sp.argmax(1)==y))/max(np.sum(ok1),1)*100
       r.update(control_accuracy=float(ok1.mean()*100),trap_accuracy=float(ok2.mean()*100),bias_trap_rate=float(btr))
      joint=float((ok1&ok2).mean()*100);left=float(ok1.mean()*100);right=float(ok2.mean()*100)
      assert joint<=min(left,right)+1e-10
      r.update(evidence_pair_n=q,pair_left_accuracy=left,pair_right_accuracy=right,pair_exact=joint,
       evidence_er=float(np.sum(sgn*d)/np.sum(sgn*d0)*100),evidence_correction_rms=rms(d-d0),
       **direction_metrics(d0,d,sgn))
      if key=="raw":assert np.isclose(r["evidence_er"],100)
     if task=="MedPAIR":
      # Gold-vs-fixed-alternative margin is fixed before deleting any sentences.
      ix=np.arange(n);alt=(y+1)%k
      def margin(v):return v[ix,y]-v[ix,alt]
      full=margin(s);rf=margin(raw)
      low=margin(scores(m,E,Nlow));rlow=margin(scores(CONFIG["methods"]["raw"],E,Nlow))
      keydel=margin(scores(m,E*.25,N));rkeydel=margin(scores(CONFIG["methods"]["raw"],E*.25,N))
      lowback=margin(scores(m,E,N)+m["g"]*addlow)
      keyback=margin(scores(m,E,N)+m["g"]*addkey)
      di=low-full;d0i=rlow-rf;de=full-keydel;d0e=rf-rkeydel
      r.update(nuisance_pair_n=n,nuisance_rms=rms(di),vr=float((1-np.var(di)/np.var(d0i))*100),
       low_delete_signed=float(np.mean(di)),low_delete_rms=rms(di),
       low_addback_signed=float(np.mean(lowback-low)),low_addback_rms=rms(lowback-low),
       low_restore_error_rms=rms(lowback-full),
       key_delete_signed=float(np.mean(keydel-full)),key_delete_rms=rms(keydel-full),
       key_addback_signed=float(np.mean(keyback-keydel)),key_addback_rms=rms(keyback-keydel),
       key_restore_error_rms=rms(keyback-full),
       evidence_er=float(np.sum(de)/np.sum(d0e)*100),evidence_correction_rms=rms(de-d0e),
       **direction_metrics(d0e,de,np.ones(n)))
      if key=="raw":assert np.isclose(r["vr"],0) and np.isclose(r["evidence_er"],100)
      if key=="calibration":
       assert np.allclose(di,rlow-rf) and np.allclose(de,rf-rkeydel)
     if task in ["MedGUIDE","TrialGPT"]:
      assert not any(z in r for z in ["vr","evidence_er","pair_exact"])
     if "direction_degradation" in r:
      assert 0<=r["direction_degradation"]<=r["direction_eligible_n"]<=r["direction_pair_n"]
      if r["direction_eligible_n"]:
       assert np.isclose(r["direction_degradation_rate"],100*r["direction_degradation"]/r["direction_eligible_n"])
      if key in ["raw","calibration"]:assert r["direction_degradation"]==0
     rows.append(r)
 groups={}
 excluded={"seed","generated_n","scored_n","unknown_n","evidence_pair_n","nuisance_pair_n"}
 for r in rows:groups.setdefault((r["model"],r["task"],r["method"]),[]).append(r)
 summary=[]
 for (model,task,method),rr in groups.items():
  out={k:v for k,v in rr[0].items() if not isinstance(v,(int,float)) and v is not None and k!="seed"}
  out["generation_seeds"]=len(rr)
  if "direction_degradation_rate" in rr[0]:
   out["direction_defined_seeds"]=sum(x["direction_degradation_rate"] is not None for x in rr)
  keys=dict.fromkeys(k for x in rr for k,v in x.items() if isinstance(v,(int,float)) or v is None)
  for k in keys:
   if k=="seed":continue
   if k in excluded:out[k]=rr[0][k]
   else:
    vals=[x[k] for x in rr if x.get(k) is not None]
    out[k+"_mean"]=float(np.mean(vals)) if vals else None
    out[k+"_sd"]=float(np.std(vals,ddof=1)) if len(vals)>1 else None
  summary.append(out)
 write_csv(OUT/"medical_extension_seed_metrics.csv",rows)
 write_csv(OUT/"medical_extension_summary.csv",summary)
 (OUT/"medical_extension_parameters.json").write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (OUT/"medical_extension_checks.json").write_text(json.dumps({"conditions":len(groups),"seed_rows":len(rows),"checks":checks},indent=2)+"\n",encoding="utf-8")
 return summary
if __name__=="__main__":generate()
