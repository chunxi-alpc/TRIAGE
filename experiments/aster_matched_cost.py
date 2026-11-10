"""Retrospective Qwen matched-evidence-cost analysis; real cached forwards only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path
import sys, json, csv, gzip, time, importlib.util
import numpy as np
from scipy.optimize import minimize, LinearConstraint, brentq
from scipy.special import expit
REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "reports/aster_cancel_story_20261109/meeting"
OLD = BASE / "trace_validation"
OUT = BASE / "matched_cost_feature_analysis"
ROOT = Path("/home/gcx/aster_cr_repair_v2_run_20261108")
CACHE = Path("/home/gcx/aster_trace_validation_cache_20261109")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(OLD))
from shared import Data, rows
from experiments.aster_closed_form.data import canonical_guide
OUT.mkdir(parents=True, exist_ok=True)
TARGET = .20
ALPHA = .01
BUDGETS = [0., .025, .05, .1, .25, .5, 1., 2., 4.]
BOOT = 2000

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
def csvout(name, data):
    if not data: return
    opener = gzip.open if name.endswith(".gz") else open
    with opener(OUT/name, "wt", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
        w.writeheader(); w.writerows(data)
def rms(x): return float(np.sqrt(np.mean(np.square(x))))
def calibrate(raw, labels):
    fun=lambda b: float(np.mean(expit(raw+b)-labels))
    bound=1.
    while fun(-bound)*fun(bound)>0: bound*=2
    return float(brentq(fun,-bound,bound,xtol=1e-12))
def metric(y, dc, direction=None, delta=None):
    new=y+dc
    if direction is None:
        den=float(np.var(y))
        return dict(n=len(y), raw_rms=rms(y), corrected_rms=rms(new),
                    correction_rms=rms(dc), VR=1-float(np.var(new))/den,
                    repair_R2=1-float(np.mean(new*new))/den, repair_RMSE=rms(new))
    signed=direction*y; newsigned=direction*new
    return dict(n=len(y), raw_rms=rms(y), corrected_rms=rms(new),
                correction_rms=rms(dc), ER=float(np.sum(newsigned)/np.sum(signed)),
                direction_degraded=int(np.sum((signed>0)&(newsigned<=0))),
                direction_improved=int(np.sum((signed<=0)&(newsigned>0))),
                exceeds_fit_budget=int(np.sum(np.abs(dc)>delta+1e-7)) if delta is not None else None)

def execute(case, scores):
    tree=canonical_guide(case["guideline"]); nodes={n["id"]:n for n in tree["nodes"]}
    gid=tree["nodes"][0]["id"]; path=[]; error=None
    while gid in nodes:
        if len(path)>len(nodes): error="cycle"; break
        val=scores.get(gid)
        if val is None: error="unavailable_score"; break
        label=int(val>0); path.append(dict(gid=gid,label=label))
        gid=nodes[gid]["yes" if label else "no"]
    return dict(correct=not error and path==case["gold_path"] and gid==case["gold_leaf"],
                path=path, leaf=gid, error=error)

def main():
    start=time.time()
    if not (CACHE/"all_heads.npy").exists(): raise RuntimeError("Missing all_heads.npy; no forwards will be run")
    d=Data()
    sup=dict(np.load(OLD/"exp12/selected_support_r16.npz"))
    names=["N16","U16","S16"]+["random16_seed"+str(s) for s in range(11,61)]
    bases=[sup["N"],sup["U"],sup["S"]]
    for seed in range(11,61):
        q=np.linalg.qr(np.random.default_rng(seed).normal(size=(len(sup["mu"]),16)),mode="reduced")[0]
        bases.append(q.astype(np.float32))
    q=np.concatenate(bases,axis=1)
    print("Loading cached selected heads",flush=True)
    raw=d.raw(d.heads)
    z=(((raw-sup["mu"])/sup["scale"])@q).astype(np.float64)
    del raw
    pair={}
    for split in ("fit","dev","test"):
        for group,kinds in (("AP",("A","P")),("CF",("C","F"))):
            pp=[p for p in d.pairs if p["split"]==split and p["kind"] in kinds]
            l=np.array([d.ix[p["left"]] for p in pp],int);r=np.array([d.ix[p["right"]] for p in pp],int)
            pair[(split,group)]=dict(pp=pp,l=l,r=r,y=d.ell[r]-d.ell[l],
                direction=np.array([p["right_label"]-p["left_label"] for p in pp]))
    cal=np.array([i for i,v in enumerate(d.views) if v["split"]=="fit" and v["variant"]=="full" and v.get("original",True)],int)
    scale=rms(pair[("fit","AP")]["y"])
    models={}; frontier=[]; freeze={}
    train_log=[]
    # Budgets are fixed before any test comparison. Only dev CF is used to choose a budget.
    for j,name in enumerate(names):
        t=z[:,j*16:(j+1)*16]
        fa=pair[("fit","AP")];fc=pair[("fit","CF")];dc=pair[("dev","CF")]
        X=t[fa["r"]]-t[fa["l"]]; E=t[fc["r"]]-t[fc["l"]]; ED=t[dc["r"]]-t[dc["l"]]
        y=fa["y"]; Q=X.T@X+ALPHA*np.eye(16); qv=X.T@y
        wu=-np.linalg.solve(Q,qv)
        normalization=len(y)*max(float(np.mean(y*y)),1.)
        Qn=Q/normalization;qn=qv/normalization
        solutions={}
        def solve(tau):
            key=round(float(tau),12)
            if key in solutions: return solutions[key]
            if tau==0: w=np.zeros(16); info=dict(iterations=0,seconds=0.)
            elif tau*scale>=float(np.max(np.abs(E@wu))):
                w=wu.copy(); info=dict(iterations=0,seconds=0.,unconstrained_feasible=True)
            else:
                tt=time.time(); delta=tau*scale
                initial=wu*min(1.,delta/max(float(np.max(np.abs(E@wu))),1e-12))
                res=minimize(lambda v:float(.5*v@Qn@v+qn@v),initial,
                    jac=lambda v:Qn@v+qn,method="SLSQP",
                    constraints=[LinearConstraint(E,-delta,delta)],
                    options=dict(ftol=1e-12,maxiter=1500,disp=False))
                if not res.success: raise RuntimeError(name+" budget "+str(tau)+": "+res.message)
                w=res.x;info=dict(iterations=int(res.nit),seconds=time.time()-tt)
            actual=rms(ED@w)
            solutions[key]=dict(w=w,tau=float(tau),actual=actual,**info)
            train_log.append(dict(method=name,tau=float(tau),dev_CF_RMS=actual,**info))
            return solutions[key]
        for tau in BUDGETS: solve(tau)
        uncon_rms=rms(ED@wu)
        crossing=None
        for lo,hi in zip(BUDGETS[:-1],BUDGETS[1:]):
            if (solve(lo)["actual"]-TARGET)*(solve(hi)["actual"]-TARGET)<=0:
                crossing=(lo,hi);break
        if crossing:
            tau=float(brentq(lambda v:solve(v)["actual"]-TARGET,*crossing,xtol=1e-6))
            choice=solve(tau)
            match="dev_target_matched"
        elif uncon_rms>=TARGET:
            # Every eligible candidate is fit-only. This adds a scale of the unconstrained fit
            # with the same C/F maximum-difference constraint and is selected using dev CF only.
            f=TARGET/uncon_rms;w=wu*f
            tau=float(np.max(np.abs(E@w))/scale)
            choice=dict(w=w,tau=tau,actual=rms(ED@w),iterations=0,seconds=0.)
            match="scaled_unconstrained_fit_dev_target_matched"
        else:
            tau=float(np.max(np.abs(E@wu))/scale)
            choice=dict(w=wu.copy(),tau=tau,actual=uncon_rms,iterations=0,seconds=0.)
            match="unconstrained_below_target"
        w=choice["w"]; bias=calibrate((d.ell+t@w)[cal],d.labels[cal])
        models[name]=dict(w=w,b=bias,tau=choice["tau"],probe_w=wu,col=j*16)
        freeze[name]=dict(tau=choice["tau"],dev_CF_RMS=choice["actual"],status=match,bias=bias,
            fit_CF_bound=choice["tau"]*scale,fit_CF_max_correction=float(np.max(np.abs(E@w))))
        # The full fixed family and selected budget are exported. Test is descriptive only.
        candidates=[solve(tau) for tau in BUDGETS]+[dict(w=wu,tau=None,actual=uncon_rms),choice]
        for ci,candidate in enumerate(candidates):
            for split in ("dev","test"):
                a=pair[(split,"AP")];c=pair[(split,"CF")];ww=candidate["w"]
                am=metric(a["y"],(t[a["r"]]-t[a["l"]])@ww)
                cm=metric(c["y"],(t[c["r"]]-t[c["l"]])@ww,c["direction"],
                    candidate["tau"]*scale if candidate["tau"] is not None else None)
                frontier.append(dict(method=name,split=split,tau=candidate["tau"],
                    point="selected" if ci==len(candidates)-1 else "unconstrained" if ci==len(candidates)-2 else "grid",
                    VR=am["VR"],evidence_RMS=cm["correction_rms"],ER=cm["ER"],
                    direction_degraded=cm["direction_degraded"],AP_RMS=am["corrected_rms"]))
        print("Frozen",name,"tau",choice["tau"],"dev CF RMS",choice["actual"],flush=True)
    # Freeze file is written before the selected test evaluation.
    dump("selection_freeze.json",dict(target_CF_RMS=TARGET,alpha=ALPHA,grid=BUDGETS,fit_AP_RMS=scale,
        selection="Only dev CF RMS chooses tau by the first crossing of the fixed budget grid; no dev or test AP metric selects tau. Test never rescales weights.",
        models=freeze))
    np.savez_compressed(OUT/"fitted_models.npz",heads=sup["heads"],mu=sup["mu"],scale=sup["scale"],bases=q,
        **{name+"_"+k:m[k] for name,m in models.items() for k in ("w","b","tau","probe_w")})
    csvout("training.csv",train_log);csvout("pareto.csv",frontier)
    print("All dev selections frozen; evaluating selected configurations",flush=True)
    # Raw and bias-only are convenience references; all feature comparisons use identical calibration.
    models={"raw":dict(w=np.zeros(16),b=0.,tau=None,probe_w=np.zeros(16),col=0),
            "bias_only":dict(w=np.zeros(16),b=calibrate(d.ell[cal],d.labels[cal]),tau=None,probe_w=np.zeros(16),col=0),**models}
    summaries=[];pairrows=[];noderows=[];sourcerows=[];bootrows=[];draws={};arrays={}
    for split in ("dev","test"):
        sources=sorted({v["source_id"] for v in d.views if v["split"]==split})
        si={s:i for i,s in enumerate(sources)}
        weights=np.random.default_rng(20261111).multinomial(len(sources),np.ones(len(sources))/len(sources),size=BOOT)
        draws[split]={}
        for name,m in models.items():
            t=z[:,m["col"]:m["col"]+16];corr=t@m["w"];score=d.ell+corr+m["b"]
            a=pair[(split,"AP")];c=pair[(split,"CF")]
            ac=corr[a["r"]]-corr[a["l"]];cc=corr[c["r"]]-corr[c["l"]]
            ap=metric(a["y"],ac);cf=metric(c["y"],cc,c["direction"],m["tau"]*scale if m["tau"] is not None else None)
            probe=-(t[a["r"]]-t[a["l"]])@m["probe_w"]
            probe_r2=1-float(np.mean((a["y"]-probe)**2))/float(np.var(a["y"]))
            natural=np.array([i for i,v in enumerate(d.views) if v["split"]==split and v["variant"]=="full" and v.get("original",True)],int)
            natural_correct=(score[natural]>0)==d.labels[natural]
            summary=dict(method=name,split=split,tau=m["tau"],AP_n=ap["n"],CF_n=cf["n"],
                VR=ap["VR"],AP_raw_RMS=ap["raw_rms"],AP_corrected_RMS=ap["corrected_rms"],
                repair_R2=ap["repair_R2"],repair_RMSE=ap["repair_RMSE"],probe_R2=probe_r2,probe_RMSE=rms(a["y"]-probe),
                evidence_RMS=cf["correction_rms"],ER=cf["ER"],direction_degraded=cf["direction_degraded"],
                exceeds_fit_budget=cf["exceeds_fit_budget"],natural_n=len(natural),natural_correct=int(np.sum(natural_correct)),
                natural_accuracy=float(np.mean(natural_correct)))
            summaries.append(summary)
            ai=np.array([si[p["source_id"]] for p in a["pp"]]);ci=np.array([si[p["source_id"]] for p in c["pp"]])
            agg=lambda idx,val:np.bincount(idx,weights=val,minlength=len(sources)).astype(float)
            # Additive sufficient statistics allow exact source-cluster resampling of nonlinear VR/RMS/ER.
            raw=a["y"];new=raw+ac;sign=c["direction"];signed=sign*c["y"]
            stats=dict(AP_n=agg(ai,np.ones(len(ai))),raw_sum=agg(ai,raw),raw_sq=agg(ai,raw*raw),
                new_sum=agg(ai,new),new_sq=agg(ai,new*new),CF_n=agg(ci,np.ones(len(ci))),CF_corr_sq=agg(ci,cc*cc),
                CF_signed_raw=agg(ci,signed),CF_signed_new=agg(ci,sign*(c["y"]+cc)))
            totals={k:weights@v for k,v in stats.items()}
            with np.errstate(divide="ignore",invalid="ignore"):
                rv=totals["raw_sq"]/totals["AP_n"]-(totals["raw_sum"]/totals["AP_n"])**2
                nv=totals["new_sq"]/totals["AP_n"]-(totals["new_sum"]/totals["AP_n"])**2
                bd=dict(VR=1-nv/rv,AP_RMS=np.sqrt(totals["new_sq"]/totals["AP_n"]),
                    evidence_RMS=np.sqrt(totals["CF_corr_sq"]/totals["CF_n"]),
                    ER=totals["CF_signed_new"]/totals["CF_signed_raw"])
            draws[split][name]=bd
            arrays[(split,name)]=dict(ac=ac,cc=cc,probe=probe)
            for key,vals in bd.items():
                good=vals[np.isfinite(vals)]
                lo,hi=np.quantile(good,[.025,.975])
                point=summary[{"AP_RMS":"AP_corrected_RMS"}.get(key,key)]
                bootrows.append(dict(split=split,method=name,metric=key,estimate=point,ci_low=float(lo),ci_high=float(hi),valid_draws=len(good)))
            for source,idx in si.items():
                am=ai==idx;cm=ci==idx
                rr=dict(method=name,split=split,source_id=source)
                if am.any(): rr.update({"AP_"+k:v for k,v in metric(raw[am],ac[am]).items()})
                if cm.any(): rr.update({"CF_"+k:v for k,v in metric(c["y"][cm],cc[cm],sign[cm],m["tau"]*scale if m["tau"] is not None else None).items()})
                sourcerows.append(rr)
            for group,pp,dcorr in (("AP",a,ac),("CF",c,cc)):
                for k,p in enumerate(pp["pp"]):
                    l,r=pp["l"][k],pp["r"][k]
                    pairrows.append(dict(method=name,split=split,pair_id=p["pair_id"],kind=p["kind"],source_id=p["source_id"],
                        raw_left=d.ell[l],raw_right=d.ell[r],corrected_left=score[l],corrected_right=score[r],
                        raw_difference=pp["y"][k],correction_difference=dcorr[k],corrected_difference=pp["y"][k]+dcorr[k],
                        left_label=p["left_label"],right_label=p["right_label"],probe_prediction=float(probe[k]) if group=="AP" else None))
            for i in natural:
                v=d.views[i]
                noderows.append(dict(method=name,split=split,view_id=v["view_id"],source_id=v["source_id"],label=v["label"],
                    raw_score=d.ell[i],correction=corr[i],bias=m["b"],corrected_score=score[i],correct=bool((score[i]>0)==v["label"])))
    csvout("summary.csv",summaries);csvout("source_metrics.csv",sourcerows);csvout("bootstrap_intervals.csv",bootrows)
    csvout("pair_predictions.csv.gz",pairrows);csvout("node_predictions.csv.gz",noderows)
    print("Selected-pair summary ready",flush=True)
    comparisons=[]
    for split in ("dev","test"):
        randoms=[n for n in names if n.startswith("random")]
        lookup={s["method"]:s for s in summaries if s["split"]==split}
        for control in ("U16","S16","random_mean"):
            for metric_name in ("VR","evidence_RMS","ER"):
                n=draws[split]["N16"][metric_name]
                other=np.mean([draws[split][s][metric_name] for s in randoms],axis=0) if control=="random_mean" else draws[split][control][metric_name]
                delta=n-other;delta=delta[np.isfinite(delta)]
                lo,hi=np.quantile(delta,[.025,.975])
                estimate=lookup["N16"][metric_name]-(float(np.mean([lookup[s][metric_name] for s in randoms])) if control=="random_mean" else lookup[control][metric_name])
                comparisons.append(dict(split=split,comparison="N16-minus-"+control,metric=metric_name,estimate=estimate,ci_low=float(lo),ci_high=float(hi)))
    csvout("paired_comparisons.csv",comparisons)
    np.savez_compressed(OUT/"bootstrap_draws.npz",**{split+"__"+name+"__"+key:val for split,dd in draws.items() for name,bd in dd.items() for key,val in bd.items()})
    # Existing path caches provide scores for all reachable nodes. New feature projections are CPU only.
    paths=[];path_summary=[];path_source=[]
    for split in ("dev","test"):
        records={}
        for file in sorted((ROOT/"qwen/paths").glob("features_"+split+"_*.jsonl")):
            for r in rows(file):
                if r.get("status")=="ok": records[r["id"]]=r
        rr=[records[k] for k in sorted(records)]
        cases=[c for c in rows(ROOT/"data/path_cases.jsonl") if c["split"]==split]
        zz=np.empty((len(rr),q.shape[1]),np.float64)
        byfile={}
        for i,r in enumerate(rr): byfile.setdefault(r["feature"],[]).append(i)
        pending=list(byfile.items())
        for batchstart in range(0,len(pending),128):
            batch=pending[batchstart:batchstart+128];raw=[]
            for file,ii in batch:
                with np.load(file) as a:
                    hh=a["heads"];raw.append(hh.reshape(-1,hh.shape[-1])[sup["heads"]].reshape(-1).astype(np.float32))
            temp=((np.asarray(raw)-sup["mu"])/sup["scale"])@q
            for val,(_,ii) in zip(temp,batch): zz[ii]=val
        print("Projected cached paths",split,len(rr),flush=True)
        rawell=np.array([r["ell"] for r in rr])
        for name,m in models.items():
            corr=zz[:,m["col"]:m["col"]+16]@m["w"]
            bycase={}
            for i,r in enumerate(rr):
                bycase.setdefault(r["case_id"],{})[r["gid"]]=float(rawell[i]+corr[i]+m["b"])
            successes=[];source_counts={}
            for c in cases:
                result=execute(c,bycase.get(c["case_id"],{}));successes.append(int(result["correct"]))
                source_counts.setdefault(c["source_id"],[0,0]);source_counts[c["source_id"]][0]+=int(result["correct"]);source_counts[c["source_id"]][1]+=1
                paths.append(dict(method=name,split=split,case_id=c["case_id"],source_id=c["source_id"],
                    correct=result["correct"],path=json.dumps(result["path"]),leaf=result["leaf"],error=result["error"]))
            path_summary.append(dict(method=name,split=split,n=len(cases),correct=sum(successes),accuracy=float(np.mean(successes))))
            for source,(correct,n) in source_counts.items():
                path_source.append(dict(method=name,split=split,source_id=source,n=n,correct=correct,accuracy=correct/n))
    csvout("path_predictions.csv.gz",paths);csvout("path_summary.csv",path_summary);csvout("path_source_metrics.csv",path_source)
    pboot=[]
    for split in ("dev","test"):
        sources=sorted({r["source_id"] for r in path_source if r["split"]==split})
        weights=np.random.default_rng(20261111).multinomial(len(sources),np.ones(len(sources))/len(sources),size=BOOT)
        stats={}
        for name in models:
            rr={r["source_id"]:r for r in path_source if r["split"]==split and r["method"]==name}
            good=np.array([rr[s]["correct"] for s in sources]);count=np.array([rr[s]["n"] for s in sources])
            stats[name]=(weights@good)/(weights@count)
        for name in models:
            delta=stats[name]-stats["raw"];lo,hi=np.quantile(delta,[.025,.975])
            a=next(r for r in path_summary if r["split"]==split and r["method"]==name)
            b=next(r for r in path_summary if r["split"]==split and r["method"]=="raw")
            pboot.append(dict(method=name,split=split,delta_accuracy=a["accuracy"]-b["accuracy"],ci_low=float(lo),ci_high=float(hi)))
    csvout("path_bootstrap.csv",pboot)
    # Cost mismatch is itself a result. No test adjustment is made.
    random_summary=[]
    for split in ("dev","test"):
        rr=[r for r in summaries if r["split"]==split and r["method"].startswith("random")]
        nn=next(r for r in summaries if r["split"]==split and r["method"]=="N16")
        for key in ("VR","evidence_RMS","ER","probe_R2"):
            vals=np.array([r[key] for r in rr])
            random_summary.append(dict(split=split,metric=key,n=50,mean=float(vals.mean()),std=float(vals.std()),min=float(vals.min()),max=float(vals.max()),
                q025=float(np.quantile(vals,.025)),q975=float(np.quantile(vals,.975)),N16=nn[key],N16_greater_than_random=int(np.sum(nn[key]>vals))))
    csvout("random_distribution.csv",random_summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,split in zip(axes,("dev","test")):
        for name,color in (("N16","#2166ac"),("U16","#d95f02"),("S16","#1b7837")):
            rr=[r for r in frontier if r["split"]==split and r["method"]==name and r["point"]=="grid"]
            ax.plot([r["evidence_RMS"] for r in rr],[100*r["VR"] for r in rr],"-o",ms=3,label=name,c=color)
            ss=next(r for r in summaries if r["split"]==split and r["method"]==name)
            ax.scatter(ss["evidence_RMS"],100*ss["VR"],c=color,s=75,marker="*",zorder=5)
        for name in names[3:]:
            rr=[r for r in frontier if r["split"]==split and r["method"]==name and r["point"]=="grid"]
            ax.plot([r["evidence_RMS"] for r in rr],[100*r["VR"] for r in rr],c="gray",alpha=.15,lw=.6)
        ax.axvline(TARGET,c="black",ls=":",lw=.8);ax.set_xlim(0,1.)
        ax.set_xlabel("Actual C/F correction RMS (logit)");ax.set_ylabel("A/P variance reduction (%)")
        ax.set_title(split+" (retrospective)");ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/"pareto.png",dpi=180);fig.savefig(OUT/"pareto.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,("dev","test")):
        rr=[r for r in summaries if r["split"]==split and r["method"].startswith("random")]
        ax.hist([100*r["VR"] for r in rr],bins=12,color="#bdbdbd",label="50 random subspaces")
        for name,color in (("N16","#2166ac"),("U16","#d95f02"),("S16","#1b7837")):
            ss=next(r for r in summaries if r["split"]==split and r["method"]==name)
            ax.axvline(100*ss["VR"],c=color,label=name,lw=2)
        ax.set_xlabel("A/P variance reduction (%)");ax.set_ylabel("Number of random subspaces");ax.set_title(split+" (frozen dev selection)")
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/"random_distribution.png",dpi=180);fig.savefig(OUT/"random_distribution.pdf");plt.close(fig)
    dump("config.json",dict(status="complete",analysis="retrospective/exploratory; existing test previously analyzed",data_kind="real Qwen3-8B cached scores and activations",
        new_model_forwards=0,GPUs_used=[],source_root=str(ROOT),cache_root=str(CACHE),support_file=str(OLD/"exp12/selected_support_r16.npz"),
        rank=16,selected_heads=sup["heads"].tolist(),random_seeds=list(range(11,61)),alpha=ALPHA,
        fit_sources=80,dev_sources=20,test_sources=40,fit_AP=len(pair[("fit","AP")]["pp"]),fit_CF=len(pair[("fit","CF")]["pp"]),
        objective="min sum_fit_AP(raw_delta + delta_feature @ w)^2 + 0.01 ||w||^2; no pair intercept",
        constraint="For each fit C/F pair, abs(delta_feature @ w) <= tau * fit AP raw RMS",
        selection="Fixed tau grid, first dev CF RMS=0.20 crossing, root solve; only dev C/F RMS chooses tau; same procedure for N/U/S/50 random",
        calibration="Fit-only original natural nodes, logistic intercept; no effect on pair deltas",
        bootstrap=dict(draws=BOOT,unit="source",paired=True,seed=20261111,random_control="mean of 50 separate VR statistics, not an ensemble",
            interpretation="Source uncertainty conditional on fixed 50 random draws; random-draw distribution reported separately"),
        metrics=dict(probe_R2="Unconstrained saved AP ridge predictor on same feature family; distinct from bounded repair_R2",
            VR="1-var(corrected AP delta)/var(raw AP delta)",ER="sum(label_direction*corrected CF delta)/sum(label_direction*raw CF delta)",
            evidence_RMS="RMS(CF correction-right minus correction-left)",direction_degraded="raw correct label-signed evidence response >0 and corrected <=0",
            exceeds_fit_budget="Heldout CF abs correction difference exceeds chosen tau*fit_AP_RMS"),
        test_policy="All dev-selected weights frozen. No test rescaling or budget selection. Full grid mapped to test descriptively only.",
        seconds=time.time()-start))
    lines=["# 同证据代价特征比较（回顾性、探索性）","",
        "全部使用真实 Qwen3-8B 缓存；新增模型前向为 0。复用原 fit/dev/test 的 80/20/40 来源及既有测试集。采用相同 A/P 岭回归目标、alpha=0.01、fit C/F 逐对最大改动约束和自然节点截距校准。每类特征仅根据 dev C/F 实际修正 RMS 选择预算，目标 0.20 logit；测试不缩放、不调参。","",
        "| 特征 | 划分 | 实际证据 RMS | A/P VR | 证据 ER | 无约束预测 R² | 方向退化 | 超 fit 预算 | 动态路径正确数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ("dev","test"):
        for name in ("raw","bias_only","N16","U16","S16"):
            ss=next(r for r in summaries if r["split"]==split and r["method"]==name)
            ps=next(r for r in path_summary if r["split"]==split and r["method"]==name)
            lines.append(f"| {name} | {split} | {ss['evidence_RMS']:.4f} | {ss['VR']:.2%} | {ss['ER']:.2%} | {ss['probe_R2']:.4f} | {ss['direction_degraded']} | {ss['exceeds_fit_budget']} | {ps['correct']}/{ps['n']} |")
        rr=[r for r in summaries if r["split"]==split and r["method"].startswith("random")]
        lines.append(f"| 50 随机均值 | {split} | {np.mean([r['evidence_RMS'] for r in rr]):.4f} | {np.mean([r['VR'] for r in rr]):.2%} | {np.mean([r['ER'] for r in rr]):.2%} | {np.mean([r['probe_R2'] for r in rr]):.4f} | 见逐配置表 | 见逐配置表 | 见逐配置表 |")
    lines+=["","## 来源簇配对置信区间","","| 比较 | 划分 | VR 差值 | 来源 bootstrap 95% CI |","|---|---|---:|---:|"]
    for r in comparisons:
        if r["metric"]=="VR": lines.append(f"| {r['comparison']} | {r['split']} | {r['estimate']:.2%} | [{r['ci_low']:.2%}, {r['ci_high']:.2%}] |")
    lines+=["","开发集匹配 0.20 不保证测试集实际代价匹配。测试结论必须结合上表实际 RMS：若各方法证据改动明显不同，不能把 VR 差异解释为严格同代价优势。pareto.csv 与图保留所有预设预算及其冻结后的 test 对应点，不用 test 重新选点。",
        "","无约束预测 R²衡量特征对 A/P 分数变化的可预测性；受约束修复残差 R²另列 summary.csv 的 repair_R2，不混用两者。自然节点标签只用于统一截距校准；配对线性修正使用 A/P 差分监督。",
        "","随机对照的来源置信区间针对固定 50 个种子的统计量均值，随机种子分布独立保存在 random_distribution.csv。每个来源整簇重采样 2,000 次，保持同一次重采样中的方法配对。",
        "","![证据代价与干扰抑制](pareto.png)","","![50 个随机子空间的分布](random_distribution.png)","",
        "逐例配对结果：pair_predictions.csv.gz；逐节点：node_predictions.csv.gz；动态路径：path_predictions.csv.gz；来源统计：source_metrics.csv、path_source_metrics.csv；参数：fitted_models.npz；开发集冻结配置：selection_freeze.json。"]
    (OUT/"REPORT.zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    from aster_matched_cost_report import render
    render()
    print(json.dumps(dict(status="complete",seconds=time.time()-start,summary=[r for r in summaries if r["method"] in ("N16","U16","S16")]),ensure_ascii=False),flush=True)

if __name__=="__main__": main()
