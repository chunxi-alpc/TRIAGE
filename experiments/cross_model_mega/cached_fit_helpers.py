"""CPU-only paired regression with evidence constraints and saved predictors."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
import argparse,csv,json,time
from collections import Counter
import numpy as np
from scipy.optimize import minimize,LinearConstraint,brentq
from scipy.special import expit
from sklearn.utils.extmath import randomized_svd
ALPHA=.01
TAU=.1

def rows(path):
    path=Path(path)
    if not path.exists():return []
    out=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:out.append(json.loads(line))
                except json.JSONDecodeError:continue
    return out
def dump(path,obj):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");temp.replace(path)
def rms(x):return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
def basis(x,rank):
    if not x.size:return np.empty((x.shape[1],0),np.float32)
    _,s,vt=randomized_svd(x,n_components=min(rank,*x.shape),random_state=20261109)
    return vt[s>max(1e-7,float(s.max())*1e-6)].T.astype(np.float32)
def orthogonalize(signal,nuisance):
    if nuisance.shape[1]==0:return nuisance.copy()
    x=nuisance.astype(np.float64)
    if signal.shape[1]:
        q=np.linalg.qr(signal.astype(np.float64),mode="reduced")[0]
        x=x-q@(q.T@x);x=x-q@(q.T@x)
    u,s,_=np.linalg.svd(x,full_matrices=False)
    return u[:,s>max(1e-10,float(s.max())*1e-6)].astype(np.float32)
def calibrate(base,labels):
    if not len(labels):return 0.
    if np.all(labels==labels[0]):return float((40. if labels[0] else -40.)-base.mean())
    def derivative(b):return float(np.mean(expit(base+b)-labels))
    lo,hi=-1.,1.
    while derivative(lo)>0:lo*=2
    while derivative(hi)<0:hi*=2
    return float(brentq(derivative,lo,hi,xtol=1e-12))
def solve(x,y,evidence,delta,constrained=True):
    dim=x.shape[1]
    if not dim:return np.empty(0),dict(solver="empty_support",success=True,iterations=0)
    gram=x.T@x+ALPHA*np.eye(dim);linear=x.T@y
    unconstrained=-np.linalg.solve(gram,linear)
    largest=float(np.max(np.abs(evidence@unconstrained))) if len(evidence) else 0.
    if not constrained or largest<=delta+1e-12:
        return unconstrained,dict(solver="closed_form",success=True,iterations=0)
    initial=unconstrained*min(1.,delta/max(largest,1e-15))
    factor=len(y)*max(float(np.mean(y*y)),1.)
    q,c=gram/factor,linear/factor
    result=minimize(lambda w:float(.5*w@q@w+c@w),initial,jac=lambda w:q@w+c,
                    method="SLSQP",constraints=[LinearConstraint(evidence,-delta,delta)],
                    options=dict(ftol=1e-12,maxiter=2000,disp=False))
    w=result.x;largest=float(np.max(np.abs(evidence@w)))
    if largest>delta and largest>0:w=w*delta/largest
    return w,dict(solver="SLSQP",success=bool(result.success),message=str(result.message),
                  iterations=int(result.nit),fit_max_evidence_correction=float(np.max(np.abs(evidence@w))))
def selected_feature(feature,heads):
    if "selected_heads" in feature:return np.asarray(feature["selected_heads"],np.float32).reshape(-1)
    value=feature["heads"]
    return value.reshape(-1,value.shape[-1])[heads].reshape(-1).astype(np.float32)
def load_available(out_root,data_root,heads):
    allviews=rows(data_root/"views.jsonl");scores={}
    for file in sorted((out_root/"scores").glob("*.jsonl")):
        for r in rows(file):
            if r.get("status","ok")=="ok" and r.get("feature"):
                p=Path(r["feature"]);p=p if p.is_absolute() else out_root/p
                if p.exists():scores[r["view_id"]]=dict(r,feature=str(p))
    views=[v for v in allviews if v["view_id"] in scores]
    pairs=[p for p in rows(data_root/"pairs.jsonl") if p["left"] in scores and p["right"] in scores]
    needed={v["view_id"] for v in views if v.get("variant")=="full" or v.get("original",False)}
    needed.update(p[end] for p in pairs for end in ("left","right"))
    views=[v for v in views if v["view_id"] in needed];ids=[v["view_id"] for v in views];ix={v:i for i,v in enumerate(ids)}
    pairs=[p for p in pairs if p["left"] in ix and p["right"] in ix]
    byfile={}
    for vid in ids:byfile.setdefault(scores[vid]["feature"],[]).append(ix[vid])
    raw=None
    for file,positions in byfile.items():
        with np.load(file,allow_pickle=False) as a:value=selected_feature(a,heads)
        if raw is None:raw=np.empty((len(ids),len(value)),np.float32)
        raw[positions]=value
    if raw is None:raise ValueError("No available head features")
    return dict(all_views=allviews,views=views,pairs=pairs,raw=raw,index=ix,ids=ids,
                ell=np.array([scores[v]["ell"] for v in ids],np.float64),
                labels=np.array([v["label"] for v in views],np.int8),unique_features=len(byfile))
def pair_indices(data,split,kinds):
    pp=[p for p in data["pairs"] if p["split"]==split and p["kind"] in kinds]
    return pp,np.array([data["index"][p["left"]] for p in pp],int),np.array([data["index"][p["right"]] for p in pp],int)
def natural_indices(data,split):
    return np.array([i for i,v in enumerate(data["views"]) if v["split"]==split and v.get("variant")=="full" and v.get("original",True)],int)

def fit(data,out_root,heads):
    raw,ell,labels=data["raw"],data["ell"],data["labels"];fi=natural_indices(data,"fit")
    if not len(fi):
        fi=np.array([i for i,v in enumerate(data["views"]) if v["split"]=="fit" and v.get("original",False)],int)
    if not len(fi):raise ValueError("Natural fit examples are required for calibration")
    mu=raw[fi].mean(0);scale=np.maximum(raw[fi].std(0),1e-5)
    ap,al,ar=pair_indices(data,"fit",("A","P"));cf,cl,cr=pair_indices(data,"fit",("C","F"))
    if not ap or not cf:raise ValueError("Both fit nuisance A/P and evidence C/F pairs are required")
    apx=((raw[ar]-raw[al])/scale).astype(np.float32);cfx=((raw[cr]-raw[cl])/scale).astype(np.float32)
    y=ell[ar]-ell[al];ap_rms=rms(y);delta=TAU*ap_rms
    s,u=basis(cfx,16),basis(apx,16);n=orthogonalize(s,u)
    s8,u8=basis(cfx,8),basis(apx,8);n8=orthogonalize(s8,u8)
    supports=dict(r16=n,U=u,S=s,r8=n8,r32=np.concatenate([n,s],axis=1))
    saved=dict(heads=np.array(heads),mu=mu,scale=scale,S=s,U=u,N=n,S8=s8,N8=n8,
               alpha=ALPHA,tau=TAU,delta=delta)
    details={};natural=(raw[fi]-mu)/scale;base=ell[fi];yy=labels[fi]
    for name,key,constrained in [("aster_cr","r16",True),("unconstrained_AP","r16",False),
        ("no_orthogonal","U",True),("signal_only","S",True),("rank8","r8",True),("r32","r32",True)]:
        support=supports[key]
        w,info=solve((apx@support).astype(np.float64),y,(cfx@support).astype(np.float64),delta,constrained)
        coef=support@w;b=calibrate(base+natural@coef,yy)
        saved.update({name+"_w":w,name+"_coef":coef,name+"_b":b})
        details[name]=dict(info,rank=len(w),b=b,constrained=constrained)
    for name,coef,b in [("raw",np.zeros(len(mu)),0.),("bias_only",np.zeros(len(mu)),calibrate(base,yy)),
                        ("no_bias",saved["aster_cr_coef"],0.)]:
        saved[name+"_coef"]=coef;saved[name+"_b"]=b
        details[name]=dict(rank=0 if name!="no_bias" else n.shape[1],b=b,solver="baseline")
    saved["w"]=saved["aster_cr_w"];saved["b"]=saved["aster_cr_b"]
    np.savez_compressed(out_root/"fitted.npz",**saved)
    frozen=dict(frozen=True,created_utc=time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()),
                protocol="fit A/P difference ridge; fit C/F per-pair correction bound; natural fit bias calibration",
                alpha=ALPHA,tau=TAU,delta=delta,fit_ap_raw_rms=ap_rms,selected_heads=len(heads),
                fit_ap=len(ap),fit_cf=len(cf),fit_natural=len(fi),methods=details,
                fit_ap_by_kind=dict(Counter(p["kind"] for p in ap)),fit_cf_by_kind=dict(Counter(p["kind"] for p in cf)),
                basis_fitted_split="fit",coefficient_fitted_split="fit",calibration_fitted_split="fit",
                test_used_for_fitting=False)
    dump(out_root/"selection-freeze.json",frozen)
    return saved,frozen

def group_metrics(data,scores,split,kinds,delta):
    pp,li,ri=pair_indices(data,split,kinds)
    if not pp:return dict(n=0,sources=0)
    ell=data["ell"];rawdy=ell[ri]-ell[li];dy=scores[ri]-scores[li];dc=dy-rawdy
    yl=np.array([p["left_label"] for p in pp],dtype=np.int8);yr=np.array([p["right_label"] for p in pp],dtype=np.int8)
    cor=((scores[li]>0)==yl)&((scores[ri]>0)==yr)
    basecor=((ell[li]>0)==yl)&((ell[ri]>0)==yr)
    out=dict(n=len(pp),sources=len({p["source_id"] for p in pp}),by_kind=dict(Counter(p["kind"] for p in pp)),
             raw_difference_rms=rms(rawdy),difference_rms=rms(dy),correction_rms=rms(dc),
             correction_abs_max=float(np.max(np.abs(dc))),paired_correct=int(cor.sum()),paired_accuracy=float(cor.mean()),
             corrected_pairs_vs_raw=int(np.sum(cor&~basecor)),new_error_pairs_vs_raw=int(np.sum(~cor&basecor)),
             prediction_flip_fraction=float(np.mean((scores[li]>0)!=(scores[ri]>0))))
    if set(kinds).issubset({"A","P","F0"}):
        var=float(np.var(rawdy));total=float(np.sum((rawdy-rawdy.mean())**2))
        out.update(variance_reduction=1-float(np.var(dy))/var if var else None,
                   prediction_r2=1-float(dy@dy)/total if total else None)
    else:
        d=yr-yl;rawsigned=d*rawdy;signed=d*dy;den=float(rawsigned.sum())
        out.update(evidence_response_retention=float(signed.sum())/den if den else None,
                   raw_response_mean=float(rawsigned.mean()),response_mean=float(signed.mean()),
                   response_direction_degraded=int(np.sum((rawsigned>0)&(signed<=0))),
                   response_direction_improved=int(np.sum((rawsigned<=0)&(signed>0))),
                   raw_response_positive=int(np.sum(rawsigned>0)),corrected_response_positive=int(np.sum(signed>0)),
                   exceeds_fit_budget_count=int(np.sum(np.abs(dc)>delta+1e-8)))
    return out
def evaluate(data,params,frozen,out_root):
    z=(data["raw"]-params["mu"])/params["scale"];ell=data["ell"];labels=data["labels"]
    names=list(frozen["methods"]);splits=[s for s in ("fit","dev","test") if any(v["split"]==s for v in data["views"])]
    scores={name:ell+z@params[name+"_coef"]+float(params[name+"_b"]) for name in names}
    result=dict(protocol=frozen["protocol"],selected_heads=len(params["heads"]),fit=frozen,
                available_views=len(data["views"]),unique_features=data["unique_features"],
                evaluated_utc=time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()),methods={},
                coverage={s:dict(expected_views=sum(v["split"]==s for v in data["all_views"]),
                    available_views=sum(v["split"]==s for v in data["views"]),
                    pair_counts=dict(Counter(p["kind"] for p in data["pairs"] if p["split"]==s))) for s in splits})
    summary=[];source_rows=[]
    for name,ss in scores.items():
        method=dict(splits={})
        for split in splits:
            groups={}
            for group,kinds in [("A",("A",)),("P",("P",)),("AP",("A","P")),("C",("C",)),("F",("F",)),("CF",("C","F")),("F0",("F0",))]:
                groups[group]=group_metrics(data,ss,split,kinds,frozen["delta"])
                summary.append(dict(method=name,split=split,group=group,**{k:v for k,v in groups[group].items() if not isinstance(v,dict)}))
            for group,predicate in [("natural",lambda v:v.get("original",True)),
                                     ("constructed",lambda v:not v.get("original",True)),("all_full",lambda v:True)]:
                inds=np.array([i for i,v in enumerate(data["views"]) if v["split"]==split and v.get("variant")=="full" and predicate(v)],int)
                cor=(ss[inds]>0)==labels[inds];basecor=(ell[inds]>0)==labels[inds]
                groups[group]=dict(n=len(inds),correct=int(cor.sum()),accuracy=float(cor.mean()) if len(inds) else None,
                    corrected_vs_raw=int(np.sum(cor&~basecor)),new_errors_vs_raw=int(np.sum(~cor&basecor)))
                summary.append(dict(method=name,split=split,group=group,**groups[group]))
            method["splits"][split]=groups
            pp=[p for p in data["pairs"] if p["split"]==split]
            for source in sorted({p["source_id"] for p in pp},key=str):
                for family,kinds in [("AP",("A","P")),("CF",("C","F"))]:
                    subset=[p for p in pp if p["source_id"]==source and p["kind"] in kinds]
                    if not subset:continue
                    li=np.array([data["index"][p["left"]] for p in subset],int);ri=np.array([data["index"][p["right"]] for p in subset],int)
                    rawdy=ell[ri]-ell[li];dy=ss[ri]-ss[li];sign=np.array([int(p["right_label"])-int(p["left_label"]) for p in subset])
                    source_rows.append(dict(method=name,split=split,source_id=source,family=family,n=len(subset),
                        raw_sum=float(rawdy.sum()),sum=float(dy.sum()),raw_squared_sum=float(rawdy@rawdy),squared_sum=float(dy@dy),
                        correction_squared_sum=float((dy-rawdy)@(dy-rawdy)),raw_signed_sum=float(sign@rawdy),signed_sum=float(sign@dy)))
        result["methods"][name]=method
    dump(out_root/"metrics.json",result)
    for filename,records in [("summary.csv",summary),("source_metrics.csv",source_rows)]:
        if records:
            fields=list(dict.fromkeys(k for r in records for k in r))
            with (out_root/filename).open("w",encoding="utf-8",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    with (out_root/"predictions.jsonl").open("w",encoding="utf-8") as f:
        for i,v in enumerate(data["views"]):
            r={k:v.get(k) for k in ("view_id","source_id","group_id","split","kind","variant","original","label")}
            r["scores"]={name:float(ss[i]) for name,ss in scores.items()};f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
    lines=["# Five-model paired correction: measured features","",
           "Supports, coefficients and calibration use fit only. Primary tau=0.1; ridge=0.01.",
           "","| Split | Method | Heads | AP n | VR % | CF n | ER % | CF correction RMS | Natural |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x,scale=1.):return "NA" if x is None else format(x*scale,".4f")
    for split in splits:
        for name in names:
            m=result["methods"][name]["splits"][split];ap,cf,nn=m["AP"],m["CF"],m["natural"]
            lines.append("| "+" | ".join([split,name,str(len(params["heads"])),str(ap["n"]),fmt(ap.get("variance_reduction"),100),
                str(cf["n"]),fmt(cf.get("evidence_response_retention"),100),fmt(cf.get("correction_rms")),str(nn["correct"])+"/"+str(nn["n"])])+" |")
    lines+=["","Per-kind counts and response direction changes: metrics.json. Source-level statistics: source_metrics.csv."]
    (out_root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result

class Predictor:
    """One-node scorer: predict(ell, heads=all_head_array) or predict_file(ell, feature)."""
    def __init__(self,out_root):
        self.out_root=Path(out_root)
        with np.load(self.out_root/"fitted.npz",allow_pickle=False) as f:self.params={k:f[k] for k in f.files}
        self.heads=self.params["heads"].astype(int)
        self.names=list(json.loads((self.out_root/"selection-freeze.json").read_text())["methods"])
    def predict(self,ell,heads=None,selected_heads=None):
        if selected_heads is None:
            value=np.asarray(heads);selected_heads=value.reshape(-1,value.shape[-1])[self.heads].reshape(-1)
        raw=np.asarray(selected_heads,np.float32).reshape(-1)
        z=(raw-self.params["mu"])/self.params["scale"]
        return {name:float(ell+z@self.params[name+"_coef"]+self.params[name+"_b"]) for name in self.names}
    def predict_file(self,ell,feature):
        with np.load(feature,allow_pickle=False) as f:
            return self.predict(ell,selected_heads=selected_feature(f,self.heads))

def fit_evaluate(out_root,data_root,evaluate_only=False):
    started=time.time();out_root,data_root=Path(out_root),Path(data_root);out_root.mkdir(parents=True,exist_ok=True)
    selection=json.loads((out_root/"selection.json").read_text(encoding="utf-8"));heads=selection.get("heads",selection.get("H"))
    data=load_available(out_root,data_root,heads)
    if evaluate_only:
        with np.load(out_root/"fitted.npz",allow_pickle=False) as f:params={k:f[k] for k in f.files}
        frozen=json.loads((out_root/"selection-freeze.json").read_text(encoding="utf-8"))
    else:params,frozen=fit(data,out_root,heads)
    result=evaluate(data,params,frozen,out_root)
    dump(out_root/"fit_evaluate_status.json",dict(state="complete",evaluate_only=evaluate_only,available_views=len(data["views"]),
         splits=list(result["coverage"]),seconds=time.time()-started))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out-root",required=True);parser.add_argument("--data-root",required=True)
    parser.add_argument("--evaluate-only",action="store_true");args=parser.parse_args()
    fit_evaluate(args.out_root,args.data_root,args.evaluate_only)

