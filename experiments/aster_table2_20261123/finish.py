"""Fit the approved Table 2 diagnostics and export frozen evaluations."""
import csv,json,sqlite3,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.utils.extmath import randomized_svd
from .common import REPO,OUT,RUN,Features,datasets,node_type_map,read,write
from .metrics import pair_weights,fit_ridge,separable_predictability,evidence_coverage
from experiments.aster_cr_retrain_paired_v2_20261111 import m1_plan

def save_csv(path,records):
    if not records:return
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

def select_heads(features):
    scores=defaultdict(dict)
    for path in [RUN/"reused_scores.sqlite",*sorted(RUN.glob("new_scores_*.sqlite"))]:
        c=sqlite3.connect("file:"+str(path)+"?mode=ro",uri=True)
        for a,b,h,v in c.execute("SELECT target,donor,head,value FROM scores"):
            scores[a,b][int(h)]=float(v)
        c.close()
    def exact(pair,heads):
        a,b=pair["left"],pair["right"]
        return [(h,scores[a,b][h],scores[b,a][h]) for h in heads]
    private=RUN/"m1"
    plan=read(OUT/"m1/m1_plan.json")
    try:
        heads,result=m1_plan.finish_plan(plan,features,exact,private)
    except RuntimeError as exc:
        result=read(private/"m1.json")
        if result.get("heads")!=[] or "selected no heads" not in str(exc):raise
        heads=[]
    # Preserve full exact records privately and publish compact numerical evidence.
    selected={k:result[k] for k in ("method_version","heads","selected_by_kind","nuisance_heads","evidence_heads","shared_heads",
        "selected_counts_by_kind","candidate_counts_by_kind","fit_pair_counts","fit_source_count","exact_validation_pair_count","exact_forward_requests","seconds")}
    selected["weighting"]=plan["weighting"]
    write(OUT/"m1/selection.json",selected)
    head_rows=[]
    source_rows=[]
    for kind,stats in result["selection_metrics"].items():
        for head,stat in stats.items():
            row=dict(kind=kind,head=int(head),layer=int(head)//32,local_head=int(head)%32,**stat)
            ci=row.pop("ci95");row["ci95_low"]=None if ci is None else ci[0];row["ci95_high"]=None if ci is None else ci[1]
            head_rows.append(row)
        for head,rows in result["source_effects"][kind].items():
            source_rows.extend(dict(kind=kind,head=int(head),**r) for r in rows)
    save_csv(OUT/"m1/head_statistics.csv",head_rows)
    save_csv(OUT/"m1/source_effects.csv",source_rows)
    return heads,selected

def basis(matrix,rank,seed):
    if min(matrix.shape)==0:return np.empty((matrix.shape[1],0),np.float32),np.empty(0)
    _,s,vt=randomized_svd(matrix.astype(np.float32),n_components=min(rank,min(matrix.shape)),random_state=seed)
    keep=s>max(1e-7,1e-6*float(s.max()))
    return vt[keep].T.astype(np.float32),s

def fit_subspaces(pairs,features,heads,config):
    weights=pair_weights(pairs);endpoint_mass=defaultdict(float)
    for p,w in zip(pairs,weights):
        endpoint_mass[p["left"]]+=.5*w;endpoint_mass[p["right"]]+=.5*w
    ids=sorted(endpoint_mass);ew=np.array([endpoint_mass[u] for u in ids],np.float64)
    x=np.stack([features.heads(u,heads) for u in ids]).astype(np.float32)
    mu=(ew@x).astype(np.float32)
    scale=np.maximum(np.sqrt(ew@np.square(x.astype(np.float64)-mu)).astype(np.float32),1e-5)
    index={u:i for i,u in enumerate(ids)}
    diff=np.stack([(x[index[p["right"]]]-x[index[p["left"]]])/scale for p in pairs]).astype(np.float32)
    ap=np.array([p["kind"] in ("A","P") for p in pairs])
    cf=~ap
    s,sv_s=basis(diff[cf]*np.sqrt(weights[cf]/weights[cf].sum()).astype(np.float32)[:,None],16,config["seed"])
    u,sv_u=basis(diff[ap]*np.sqrt(weights[ap]/weights[ap].sum()).astype(np.float32)[:,None],16,config["seed"])
    residual=u.astype(np.float64)-s.astype(np.float64)@(s.astype(np.float64).T@u.astype(np.float64))
    residual-=s.astype(np.float64)@(s.astype(np.float64).T@residual)
    if residual.shape[1]:
        n,singulars,_=np.linalg.svd(residual,full_matrices=False)
        keep=singulars>max(1e-7,1e-6*float(singulars.max()))
        n=n[:,keep].astype(np.float32)
    else:n=np.empty((len(mu),0),np.float32);singulars=np.empty(0)
    model=dict(heads=np.asarray(heads,dtype=int),mu=mu,scale=scale,S=s,U=u,N=n,
        S_singular_values=sv_s,U_singular_values=sv_u,N_singular_values=singulars,
        standardization_ids=np.asarray(ids),standardization_weights=ew)
    np.savez_compressed(OUT/"m2.npz",**model)
    write(OUT/"m2.json",dict(head_count=len(heads),dimension=len(mu),S_rank=s.shape[1],U_rank=u.shape[1],N_rank=n.shape[1],
        endpoint_count=len(ids),orthogonality_max=float(np.max(np.abs(s.T@n))) if s.size and n.size else 0.0,
        weighting=config["weighting"],seed=config["seed"]))
    return model

def pair_arrays(pairs,features,model):
    coords={};scoords={}
    for uid in sorted({p[e] for p in pairs for e in ("left","right")}):
        z=(features.heads(uid,model["heads"])-model["mu"])/model["scale"]
        coords[uid]=(z@model["N"]).astype(np.float64)
        scoords[uid]=(z@model["S"]).astype(np.float64)
    x=np.stack([coords[p["right"]]-coords[p["left"]] for p in pairs])
    ds=np.stack([scoords[p["right"]]-scoords[p["left"]] for p in pairs])
    coverage=x-ds@(model["S"].astype(np.float64).T@model["N"].astype(np.float64))
    y=np.array([features.ell(p["right"])-features.ell(p["left"]) for p in pairs],np.float64)
    return x,y,coverage,pair_weights(pairs)

def intervals(pairs,weights,mask,stat_numerator,stat_denominator,mode,config):
    # One set of source draws for every cell in a split; empty resamples are omitted.
    sources=sorted({p["source_id"] for p in pairs});source_index={s:i for i,s in enumerate(sources)}
    num=np.zeros(len(sources));den=np.zeros(len(sources))
    for i in np.flatnonzero(mask):
        j=source_index[pairs[i]["source_id"]]
        num[j]+=weights[i]*stat_numerator[i];den[j]+=weights[i]*stat_denominator[i]
    rng=np.random.default_rng(config["bootstrap"]["seed"])
    counts=rng.multinomial(len(sources),np.ones(len(sources))/len(sources),size=config["bootstrap"]["draws"])
    numer=counts@num;denom=counts@den;valid=denom>0
    if not valid.any():return None,0
    vals=1-numer[valid]/denom[valid] if mode=="r2" else np.sqrt(numer[valid]/denom[valid])
    return [float(v) for v in np.quantile(vals,[.025,.975])],int(valid.sum())

def rows_for_split(split,pairs,arrays,w,config):
    x,y,cov,weights=arrays
    residual=y+x@w
    rms2=np.sum(cov*cov,axis=1)
    types=node_type_map()
    pairtypes=np.array([types.get(p.get("group_id"),"未标注") for p in pairs])
    definitions=[("指南","A","C"),("患者","P","F"),("总体","AP","CF")]
    results=[]
    for layer in ["全部",*sorted(set(pairtypes))]:
        layer_mask=np.ones(len(pairs),bool) if layer=="全部" else pairtypes==layer
        for label,ap_kind,cf_kind in definitions:
            ap=np.array([p["kind"] in ap_kind for p in pairs])&layer_mask
            cf=np.array([p["kind"] in cf_kind for p in pairs])&layer_mask
            prediction=separable_predictability(x[ap],y[ap],w,weights[ap])
            coverage=evidence_coverage(cov[cf],weights[cf])
            ci_r2,valid_r2=intervals(pairs,weights,ap,residual**2,y**2,"r2",config)
            ci_cov,valid_cov=intervals(pairs,weights,cf,rms2,np.ones(len(pairs)),"cov",config)
            results.append(dict(split=split,node_type=layer,information=label,preserving_kind=ap_kind,flipping_kind=cf_kind,
                preserving_pairs=int(ap.sum()),flipping_pairs=int(cf.sum()),
                preserving_sources=len({p["source_id"] for p,m in zip(pairs,ap) if m}),
                flipping_sources=len({p["source_id"] for p,m in zip(pairs,cf) if m}),
                r2_sep=prediction["r2_sep"],r2_ci95=ci_r2,r2_bootstrap_valid=valid_r2,
                r_cov=coverage["r_cov"],r_cov_ci95=ci_cov,r_cov_bootstrap_valid=valid_cov,
                baseline_mse=prediction["baseline_mse"],residual_mse=prediction["residual_mse"]))
    return results

def run():
    started=time.monotonic();config=read(OUT/"configuration.json");data=datasets();features=Features()
    heads,selection=select_heads(features)
    write(OUT/"status.json",dict(state="fitting_subspaces",selected_heads=len(heads),gpus_released=True))
    model=fit_subspaces(data["fit"],features,heads,config)
    arrays={s:pair_arrays(data[s],features,model) for s in ("fit","dev")}
    fitx,fity,_,fitweights=arrays["fit"]
    mask=np.array([p["kind"] in ("A","P") for p in data["fit"]])
    devx,devy,_,devweights=arrays["dev"]
    devmask=np.array([p["kind"] in ("A","P") for p in data["dev"]])
    grid=[];fits={}
    for lam in config["ridge"]["candidates"]:
        w=fit_ridge(fitx[mask],fity[mask],fitweights[mask]/fitweights[mask].sum(),lam)
        fits[lam]=w
        dev=separable_predictability(devx[devmask],devy[devmask],w,devweights[devmask])
        grid.append(dict(regularization=lam,dev_r2_sep=dev["r2_sep"],dev_residual_mse=dev["residual_mse"],
            dev_baseline_mse=dev["baseline_mse"],coefficient_norm=float(np.linalg.norm(w))))
    best=min(r["dev_residual_mse"] for r in grid)
    lam=max(r["regularization"] for r in grid if np.isclose(r["dev_residual_mse"],best,rtol=1e-12,atol=1e-12))
    w=fits[lam]
    save_csv(OUT/"regularization_selection.csv",grid)
    np.savez_compressed(OUT/"ridge.npz",w=w,regularization=np.asarray(lam))
    frozen=dict(regularization=lam,selection_split="dev",selection_criterion=config["ridge"]["selection"],
       coefficient_norm=float(np.linalg.norm(w)),head_count=len(heads),feature_rank=int(model["N"].shape[1]),
       test_used_for_selection=False,configuration="Section 3 adaptive M1 and hierarchical weighting",
       feature_status="nonempty" if model["N"].shape[1] else "zero-dimensional")
    write(OUT/"frozen.json",frozen)
    # Held-out outputs are computed after the chosen coefficient vector is fixed.
    arrays["test"]=pair_arrays(data["test"],features,model)
    table=[]
    for s in ("dev","test"):
        table.extend(rows_for_split(s,data[s],arrays[s],w,config))
        x,y,cov,weights=arrays[s]
        export=[]
        types=node_type_map()
        for i,p in enumerate(data[s]):
            export.append(dict(pair_id=p["pair_id"],source_id=p["source_id"],case_id=p["case_id"],kind=p["kind"],
               node_type=types.get(p.get("group_id"),"未标注"),weight=float(weights[i]),original_delta=float(y[i]),
               correction_delta=float(x[i]@w),residual_delta=float(y[i]+x[i]@w),
               evidence_feature_residual_norm=float(np.linalg.norm(cov[i]))))
        save_csv(OUT/(s+"_pair_metrics.csv"),export)
    write(OUT/"table2.json",table)
    flat=[]
    for row in table:
        v=dict(row)
        for key in ("r2_ci95","r_cov_ci95"):
            ci=v.pop(key);v[key+"_low"]=None if ci is None else ci[0];v[key+"_high"]=None if ci is None else ci[1]
        flat.append(v)
    save_csv(OUT/"table2_all_rows.csv",flat)
    from .report import render
    render(config,selection,frozen,table,grid)
    write(OUT/"completion.json",dict(complete=True,seconds=time.monotonic()-started,selected_heads=len(heads),
        regularization=lam,feature_rank=int(model["N"].shape[1]),main_rows=sum(r["node_type"]=="全部" for r in table),
        stratified_rows=sum(r["node_type"]!="全部" for r in table)))
    write(OUT/"status.json",dict(state="complete",manuscript_updated=True,main_rows=6,report="REPORT.zh.md"))
if __name__=="__main__":run()
