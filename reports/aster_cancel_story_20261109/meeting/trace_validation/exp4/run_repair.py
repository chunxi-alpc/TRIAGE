"""Real-cache score repair, fixed evidence budgets; run on server, CPU only."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import sys,json,csv,time,argparse,pickle,gzip
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq,minimize,LinearConstraint
import torch
from collections import Counter
torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
sys.path.insert(0,str(REPO))
from experiments.aster_closed_form.data import canonical_guide

def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def dump(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def csvout(p,rr):
    if not rr:return
    p=Path(p)
    compressed=p.name in {"pair_predictions.csv","node_predictions.csv","path_results.csv","path_node_predictions.csv"}
    target=Path(str(p)+".gz") if compressed else p
    opener=gzip.open if compressed else open
    with opener(target,"wt",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)))
        w.writeheader();w.writerows(rr)
def rms(x):return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None

def execute(case,scores):
    tree=canonical_guide(case["guideline"]);nodes={n["id"]:n for n in tree["nodes"]}
    gid=tree["nodes"][0]["id"];actual=[];error=None
    while gid in nodes:
        if len(actual)>len(nodes):error="cycle";break
        value=scores.get(gid)
        if value is None:error="unavailable_score";break
        label=int(value>0);actual.append(dict(gid=gid,label=label))
        gid=nodes[gid]["yes" if label else "no"]
    return dict(correct=not error and actual==case["gold_path"] and gid==case["gold_leaf"],
                path=actual,leaf=gid,error=error)

class Repair:
    def __init__(self,args):
        self.args=args;self.root=Path(args.root);self.out=Path(args.out)
        self.out.mkdir(parents=True,exist_ok=True)
        self.cache=Path(args.cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.started=time.time()
        self.sup=dict(np.load(self.root/"qwen/supports_r16.npz"))
        rng=np.random.default_rng(args.random_seed)
        self.random_q=np.linalg.qr(rng.normal(size=(len(self.sup["mu"]),16)),mode="reduced")[0].astype(np.float32)
        self.scores={}
        for p in sorted((self.root/"qwen/scores").glob("*.jsonl")):
            for r in rows(p):
                if r.get("status")=="ok" and r.get("split") in ("fit","dev","test"):self.scores[r["view_id"]]=r
        self.views=[v for v in rows(self.root/"data/views.jsonl") if v["view_id"] in self.scores and v["split"] in ("fit","dev","test")]
        self.byid={v["view_id"]:v for v in self.views}
        self.allpairs=[p for p in rows(self.root/"data/pairs.jsonl") if p["split"] in ("fit","dev","test")]
        self.pairs=[p for p in self.allpairs if p["left"] in self.byid and p["right"] in self.byid]
        needed={p[k] for p in self.pairs for k in ("left","right")}
        needed.update(v["view_id"] for v in self.views if v["variant"]=="full")
        self.ids=sorted(needed);self.ix={v:i for i,v in enumerate(self.ids)}
        self.ell=np.array([self.scores[v]["ell"] for v in self.ids],float)
        self.labels=np.array([self.byid[v]["label"] for v in self.ids])
        self.views=[v for v in self.views if v["view_id"] in self.ix]
        cp=self.cache/("exp4_coordinates_seed"+str(args.random_seed)+".npz")
        if cp.exists() and np.load(cp)["ids"].tolist()==self.ids:
            a=np.load(cp);self.t=a["t"];self.rt=a["rt"]
        else:
            self.t,self.rt=self.project_records([self.scores[v] for v in self.ids])
            np.savez_compressed(cp,ids=np.array(self.ids),t=self.t,rt=self.rt)
        self.coverage={s:{k:dict(expected=sum(p["split"]==s and p["kind"]==k for p in self.allpairs),
                        available=sum(p["split"]==s and p["kind"]==k for p in self.pairs))
                        for k in ("A","P","F0","C","F")} for s in ("fit","dev","test")}
        self.cal_idx=np.array([self.ix[v["view_id"]] for v in self.views if v["split"]=="fit" and v["variant"]=="full" and v.get("original",True)],int)
        self.models={};self.metadata={}
    def project_records(self,records):
        tt=np.empty((len(records),16),np.float64);rr=np.empty_like(tt)
        byfile={}
        for i,r in enumerate(records):byfile.setdefault(r["feature"],[]).append(i)
        q=np.concatenate([self.sup["N"],self.random_q],axis=1)
        for j,(file,positions) in enumerate(byfile.items()):
            with np.load(file) as a:
                h=a["heads"];raw=h.reshape(-1,h.shape[-1])[self.sup["heads"]].reshape(-1).astype(np.float32)
            val=((raw-self.sup["mu"])/self.sup["scale"])@q
            tt[positions]=val[:16];rr[positions]=val[16:]
            if (j+1)%2000==0:print("projected",j+1,"/",len(byfile),flush=True)
        return tt,rr
    def pp(self,split,kinds,sources=None):
        return [p for p in self.pairs if p["split"]==split and p["kind"] in kinds and (sources is None or p["source_id"] in sources)]
    def indices(self,pp):
        return np.array([self.ix[p["left"]] for p in pp],int),np.array([self.ix[p["right"]] for p in pp],int)
    def arrays(self,t,pp):
        l,r=self.indices(pp);return t[r]-t[l],self.ell[r]-self.ell[l]
    def calibrate(self,corr):
        x=(self.ell+corr)[self.cal_idx];y=self.labels[self.cal_idx]
        fun=lambda b:float(np.mean(expit(x+b)-y));bound=1.
        while fun(-bound)*fun(bound)>0:bound*=2
        return float(brentq(fun,-bound,bound,xtol=1e-12))
    def add_linear(self,name,t,feature,w,tau,info=None):
        corr=t@w;b=0. if name=="raw" else self.calibrate(corr)
        self.models[name]=dict(type="linear",feature=feature,w=w,b=b,tau=tau)
        self.metadata[name]=dict(type="linear",feature=feature,tau=tau,parameters=len(w)+1,
                                 bias=b,**(info or {}))
    def train_linear(self):
        ap=self.pp("fit",("A","P"));cf=self.pp("fit",("C","F"))
        _,y=self.arrays(self.t,ap);self.scale=rms(y)
        self.add_linear("raw",self.t,"trace",np.zeros(16),None)
        self.metadata["raw"]["parameters"]=0
        self.add_linear("bias_only",self.t,"trace",np.zeros(16),None)
        self.metadata["bias_only"]["parameters"]=1
        for prefix,t,feature in (("trace",self.t,"trace"),("random",self.rt,"random")):
            X,y=self.arrays(t,ap);E,_=self.arrays(t,cf)
            Q=X.T@X+self.args.alpha*np.eye(16);q=X.T@y
            w=-np.linalg.solve(Q,q)
            self.add_linear(prefix+"_unconstrained",t,feature,w,None)
            for tau in self.args.budgets:
                started=time.time();delta=tau*self.scale;normal=len(y)*max(np.mean(y*y),1)
                Qn=Q/normal;qn=q/normal
                res=minimize(lambda v:float(.5*v@Qn@v+qn@v),
                             w*min(1,delta/max(np.max(np.abs(E@w)),1e-12)),
                             jac=lambda v:Qn@v+qn,method="SLSQP",
                             constraints=[LinearConstraint(E,-delta,delta)],
                             options=dict(ftol=1e-12,maxiter=1500,disp=False))
                if not res.success:raise RuntimeError(prefix+" constrained fit: "+res.message)
                self.add_linear(prefix+"_bounded_"+str(tau),t,feature,res.x,tau,
                   dict(solver="SLSQP",success=True,iterations=int(res.nit),seconds=time.time()-started,
                        max_fit_cf_correction=float(np.max(np.abs(E@res.x))),
                        fit_constraint_violation=max(0,float(np.max(np.abs(E@res.x)))-delta)))
        np.savez_compressed(self.out/"linear_models.npz",random_q=self.random_q,
             **{name+"_"+k:v[k] for name,v in self.models.items() for k in ("w","b")})
    def fit_nn(self,seed,sources,config,tau,validation_sources=None):
        start=time.time();torch.manual_seed(seed)
        fit_full=np.array([self.ix[v["view_id"]] for v in self.views if v["split"]=="fit" and v["variant"]=="full" and v["source_id"] in sources])
        mu=self.t[fit_full].mean(0);sd=np.maximum(self.t[fit_full].std(0),1e-5)
        z=torch.tensor((self.t-mu)/sd,dtype=torch.float32)
        ap=self.pp("fit",("A","P"),sources);cf=self.pp("fit",("C","F"),sources)
        l,r=self.indices(ap);cl,cr=self.indices(cf)
        y=torch.tensor(self.ell[r]-self.ell[l],dtype=torch.float32)
        scale=float(torch.sqrt(torch.mean(y*y)));delta=tau*scale
        net=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
        opt=torch.optim.Adam(net.parameters(),lr=config["lr"],weight_decay=0.)
        best=None;bestloss=np.inf;bestepoch=config["epochs"]
        for epoch in range(config["epochs"]):
            opt.zero_grad()
            output=net(z).squeeze(1)
            pred=output[r]-output[l]
            ec=output[cr]-output[cl]
            loss=((y+pred).square().mean()+config["penalty"]*torch.relu(ec.abs()-delta).square().mean())/(scale*scale)
            loss=loss+self.args.alpha/len(ap)*sum(p.square().sum() for p in net.parameters())/(scale*scale)
            loss.backward();opt.step()
            if validation_sources is not None and (epoch+1)%50==0:
                with torch.no_grad():
                    out=net(z).squeeze(1).numpy().astype(float)
                    shrink=min(1,delta/max(np.max(np.abs(out[cr]-out[cl])),1e-12))
                    vp=self.pp("fit",("A","P"),validation_sources);vl,vr=self.indices(vp)
                    ep=self.pp("fit",("C","F"),validation_sources);el,er=self.indices(ep)
                    vdy=self.ell[vr]-self.ell[vl]+shrink*(out[vr]-out[vl])
                    ev=shrink*(out[er]-out[el])
                    vloss=float(np.mean(vdy*vdy)+config["penalty"]*np.mean(np.maximum(np.abs(ev)-delta,0)**2))
                    if vloss<bestloss:
                        bestloss=vloss;bestepoch=epoch+1
                        best={k:v.detach().clone() for k,v in net.state_dict().items()}
        if best is not None:net.load_state_dict(best)
        with torch.no_grad():out=net(z).squeeze(1).numpy().astype(float)
        shrink=min(1,delta/max(np.max(np.abs(out[cr]-out[cl])),1e-12))
        return dict(type="nn",net=net,mu=mu,sd=sd,shrink=shrink,b=0.,tau=tau),dict(
           seconds=time.time()-start,epochs=bestepoch,internal_validation_objective=None if not np.isfinite(bestloss) else bestloss,
           config=config,training_ap=len(ap),training_cf=len(cf),shrink=shrink,
           max_fit_cf_correction=float(np.max(np.abs(out[cr]-out[cl]))*shrink))
    def train_neural(self):
        sources=sorted({v["source_id"] for v in self.views if v["split"]=="fit"})
        rng=np.random.default_rng(20261109);shuffled=rng.permutation(sources).tolist()
        n=max(1,len(sources)//5);val=set(shuffled[:n]);train=set(shuffled[n:])
        configs=[dict(lr=.003,penalty=10.,epochs=500),dict(lr=.01,penalty=10.,epochs=500),dict(lr=.003,penalty=100.,epochs=500)]
        tuning=[]
        # Each budget is an independently prespecified configuration, never selected on dev/test.
        for seed in self.args.seeds:
            for tau in self.args.budgets:
                candidates=[]
                for cfg in configs:
                    _,info=self.fit_nn(seed,train,cfg,tau,val)
                    candidates.append(info)
                chosen=min(candidates,key=lambda c:c["internal_validation_objective"])
                config={**chosen["config"],"epochs":chosen["epochs"]}
                model,info=self.fit_nn(seed,set(sources),config,tau)
                corr=self.predict_correction(model,self.t,self.rt);model["b"]=self.calibrate(corr)
                name="nn_seed"+str(seed)+"_bounded_"+str(tau)
                self.models[name]=model
                self.metadata[name]=dict(type="nn",feature="trace",tau=tau,seed=seed,parameters=578,
                    bias=model["b"],selection="fit source internal holdout only",**info)
                tuning.append(dict(seed=seed,tau=tau,candidates=candidates,chosen=config))
                torch.save(dict(state_dict=model["net"].state_dict(),mu=model["mu"],sd=model["sd"],
                          shrink=model["shrink"],b=model["b"],tau=tau,seed=seed),self.out/(name+".pt"))
                print("trained",name,flush=True)
        dump(self.out/"neural_training_selection.json",dict(inner_train_sources=sorted(train),inner_validation_sources=sorted(val),
             protocol="Frozen external-fit support; neural input mean/std fitted on inner-training sources during tuning. Select validation loss on fit-only source holdout; refit full fit, no dev/test selection.",runs=tuning))
    def predict_correction(self,m,t,rt):
        if m["type"]=="linear":return (t if m["feature"]=="trace" else rt)@m["w"]
        with torch.no_grad():
            return m["net"](torch.tensor((t-m["mu"])/m["sd"],dtype=torch.float32)).squeeze(1).numpy().astype(float)*m["shrink"]
    def eval_nodes(self):
        summaries=[];pairrows=[];noderows=[]
        for name,m in self.models.items():
            corr=self.predict_correction(m,self.t,self.rt);scores=self.ell+corr+m["b"]
            for split in ("fit","dev","test"):
                for group,kinds in (("A",("A",)),("P",("P",)),("AP",("A","P")),("CF",("C","F")),("C",("C",)),("F",("F",)),("F0",("F0",))):
                    pp=self.pp(split,kinds)
                    if not pp:continue
                    l,r=self.indices(pp);dy=self.ell[r]-self.ell[l];dc=corr[r]-corr[l];new=dy+dc
                    mm=dict(method=name,tau=m["tau"],split=split,group=group,n=len(pp),
                        sources=len({p["source_id"] for p in pp}),raw_rms=rms(dy),corrected_rms=rms(new),correction_rms=rms(dc))
                    if group in ("A","P","AP","F0"):
                        mm.update(VR=1-float(np.var(new))/float(np.var(dy)),R2=1-float(np.sum(new**2))/float(np.sum((dy-dy.mean())**2)))
                    else:
                        direction=np.array([p["right_label"]-p["left_label"] for p in pp])
                        signed=direction*dy;newsigned=direction*new
                        mm.update(ER=float(np.mean(newsigned)/np.mean(signed)),direction_degraded=int(np.sum((signed>0)&(newsigned<=0))),
                            direction_improved=int(np.sum((signed<=0)&(newsigned>0))),
                            exceeds_fit_budget=int(np.sum(np.abs(dc)>m["tau"]*self.scale+1e-7)) if m["tau"] is not None else None)
                    summaries.append(mm)
                inds=np.array([self.ix[v["view_id"]] for v in self.views if v["split"]==split and v["variant"]=="full" and v.get("original",True)],int)
                if len(inds):
                    summaries.append(dict(method=name,tau=m["tau"],split=split,group="natural",n=len(inds),
                        accuracy=float(np.mean((scores[inds]>0)==self.labels[inds])),correct=int(np.sum((scores[inds]>0)==self.labels[inds]))))
            for p in self.pairs:
                if p["split"]=="fit":continue
                l,r=self.ix[p["left"]],self.ix[p["right"]]
                pairrows.append(dict(method=name,tau=m["tau"],split=p["split"],pair_id=p["pair_id"],kind=p["kind"],source_id=p["source_id"],
                    raw_left=self.ell[l],raw_right=self.ell[r],corrected_left=scores[l],corrected_right=scores[r],
                    correction_difference=corr[r]-corr[l],raw_difference=self.ell[r]-self.ell[l],
                    corrected_difference=scores[r]-scores[l],left_label=p["left_label"],right_label=p["right_label"]))
            for v in self.views:
                if v["split"]=="fit" or v["variant"]!="full":continue
                i=self.ix[v["view_id"]]
                noderows.append(dict(method=name,split=v["split"],view_id=v["view_id"],source_id=v["source_id"],
                       original=v.get("original",True),label=v["label"],raw_score=self.ell[i],correction=corr[i],bias=m["b"],
                       corrected_score=scores[i],correct=int((scores[i]>0)==v["label"])))
        csvout(self.out/"summary.csv",summaries);csvout(self.out/"pair_predictions.csv",pairrows);csvout(self.out/"node_predictions.csv",noderows)
        return summaries
    def eval_paths(self):
        pathrows=[];summaries=[];pathnodes=[]
        for split in ("dev","test"):
            records={}
            for f in sorted((self.root/"qwen/paths").glob("features_"+split+"_*.jsonl")):
                for r in rows(f):
                    if r.get("status")=="ok":records[r["id"]]=r
            rs=[records[k] for k in sorted(records)]
            cases=[c for c in rows(self.root/"data/path_cases.jsonl") if c["split"]==split]
            if not rs:
                summaries.extend(dict(method=n,split=split,n=len(cases),accuracy=None,status="missing path feature cache") for n in self.models)
                continue
            cp=self.cache/("exp4_path_coordinates_"+split+"_seed"+str(self.args.random_seed)+".npz")
            ids=[r["id"] for r in rs]
            if cp.exists() and np.load(cp)["ids"].tolist()==ids:
                a=np.load(cp);t,rt=a["t"],a["rt"]
            else:
                t,rt=self.project_records(rs);np.savez_compressed(cp,ids=np.array(ids),t=t,rt=rt)
            for name,m in self.models.items():
                corr=self.predict_correction(m,t,rt)
                bycase={}
                for i,r in enumerate(rs):
                    score=float(r["ell"]+corr[i]+m["b"])
                    bycase.setdefault(r["case_id"],{})[r["gid"]]=score
                    pathnodes.append(dict(method=name,split=split,id=r["id"],case_id=r["case_id"],source_id=r["source_id"],gid=r["gid"],
                           raw_score=r["ell"],correction=corr[i],bias=m["b"],corrected_score=score))
                results=[]
                for c in cases:
                    result=execute(c,bycase.get(c["case_id"],{}))
                    results.append(result)
                    pathrows.append(dict(method=name,split=split,case_id=c["case_id"],source_id=c["source_id"],
                          correct=result["correct"],path=json.dumps(result["path"]),leaf=result["leaf"],error=result["error"]))
                summaries.append(dict(method=name,split=split,n=len(results),correct=sum(r["correct"] for r in results),
                    accuracy=sum(r["correct"] for r in results)/len(results) if results else None,
                    unavailable=sum(r["error"]=="unavailable_score" for r in results),
                    status="real unedited node-cache autonomous path execution"))
        csvout(self.out/"path_summary.csv",summaries);csvout(self.out/"path_results.csv",pathrows);csvout(self.out/"path_node_predictions.csv",pathnodes)
        return summaries
    def finish(self,summaries,paths):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rowsmain=[]
        for split in ("dev","test"):
            for name,m in self.models.items():
                groups={s["group"]:s for s in summaries if s["method"]==name and s["split"]==split}
                if "AP" not in groups or "CF" not in groups:continue
                path=next((p for p in paths if p["method"]==name and p["split"]==split),{})
                rowsmain.append(dict(method=name,seed=self.metadata[name].get("seed"),tau=m["tau"],split=split,
                    VR=groups["AP"].get("VR"),ER=groups["CF"].get("ER"),evidence_RMS=groups["CF"]["correction_rms"],
                    direction_degraded=groups["CF"].get("direction_degraded"),natural_n=groups.get("natural",{}).get("n"),
                    natural_accuracy=groups.get("natural",{}).get("accuracy"),path_n=path.get("n"),path_accuracy=path.get("accuracy")))
        csvout(self.out/"main_results.csv",rowsmain)
        fig,axes=plt.subplots(1,2,figsize=(12,4.6))
        for ax,split in zip(axes,("dev","test")):
            rr=[r for r in rowsmain if r["split"]==split]
            for label,starts,color,marker in (("Original",("raw",),"black","o"),("Threshold",("bias_only",),"gray","x"),
                     ("Trace linear",("trace_",),"#2166ac","o"),("Random linear",("random_",),"#b2182b","s"),
                     ("Trace neural",("nn_",),"#1b7837","^")):
                subset=[r for r in rr if any(r["method"].startswith(p) for p in starts)]
                ax.scatter([r["evidence_RMS"] for r in subset],[100*r["VR"] for r in subset],label=label,c=color,marker=marker,s=42,alpha=.8)
                if label in ("Trace linear","Random linear"):
                    for r in subset:
                        ax.annotate("unconstrained" if r["tau"] is None else str(r["tau"]),
                             (r["evidence_RMS"],100*r["VR"]),fontsize=7,xytext=(3,3),textcoords="offset points")
            ax.axhline(0,c="gray",lw=.6);ax.set_title(split+" (existing holdout)")
            ax.set_xlabel("Evidence correction RMS (logit units)");ax.set_ylabel("Interference variance reduction (%)")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.tight_layout()
        fig.savefig(self.out/"repair_tradeoff.png",dpi=180);fig.savefig(self.out/"repair_tradeoff.pdf");plt.close(fig)
        src={s:sorted({v["source_id"] for v in self.views if v["split"]==s}) for s in ("fit","dev","test")}
        config=dict(root=str(self.root),seeds=self.args.seeds,random_seed=self.args.random_seed,alpha=self.args.alpha,budgets=self.args.budgets,
            provenance="new real-cache training and evaluation; no synthetic data",test_history="Previously used by older ASTER-CR evaluations; no new independent test claim",
            support_file=str(self.root/"qwen/supports_r16.npz"),support_selection="Existing frozen fit-only selected 51-head r16 nuisance support",
            objective="sum_AP (raw_delta + correction_right - correction_left)^2 + .01 ||w||^2; no pair regression intercept",
            constraint="Every fit C/F absolute correction difference <= tau * fit AP raw RMS; same maximum-difference budget for linear/random/neural",
            neural="16->32 tanh->1; pair-difference objective, penalty and final shrink enforce budget; bias calibrated only on fit natural node labels",
            calibration="One global node-score bias fitted with fit natural nodes after fixing pair correction",
            metrics=dict(VR="1-var(corrected AP delta)/var(raw AP delta), population variance",
                ER="mean(label_direction * corrected CF delta)/mean(label_direction * raw CF delta)",
                evidence_RMS="sqrt(mean((CF correction_right-correction_left)^2))",
                direction_degraded="count(raw label-signed CF response>0 and corrected signed response<=0)",
                natural_accuracy="full original=True nodes",
                dynamic_accuracy="Autonomous execution from first guide node until leaf using corrected score sign, exact gold path+leaf equality"),
            fit_ap_raw_rms=self.scale,coverage=self.coverage,sources=src,
            source_overlap={a+"_"+b:sorted(set(src[a])&set(src[b])) for a,b in (("fit","dev"),("fit","test"),("dev","test"))},
            pair_split_check=all(self.byid[p["left"]]["split"]==p["split"]==self.byid[p["right"]]["split"] for p in self.pairs),
            pair_source_check=all(self.byid[p["left"]]["source_id"]==p["source_id"]==self.byid[p["right"]]["source_id"] for p in self.pairs),
            methods=self.metadata,runtime_seconds=time.time()-self.started)
        dump(self.out/"config_and_metrics.json",config)
        print(json.dumps(dict(seconds=config["runtime_seconds"],models=len(self.models),main=rowsmain),ensure_ascii=True),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default="/home/gcx/aster_cr_repair_v2_run_20261108")
    p.add_argument("--out",default=str(HERE));p.add_argument("--cache",default="/home/gcx/aster_trace_validation_cache_20261109")
    p.add_argument("--alpha",type=float,default=.01);p.add_argument("--budgets",type=float,nargs="+",default=[.05,.1,.25])
    p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33]);p.add_argument("--random-seed",type=int,default=11)
    args=p.parse_args();r=Repair(args);r.train_linear();r.train_neural();summary=r.eval_nodes();paths=r.eval_paths();r.finish(summary,paths)
if __name__=="__main__":main()

