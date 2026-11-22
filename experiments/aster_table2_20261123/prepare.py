"""Prepare the approved hierarchy-weighted five-fold M1 computation."""
import json,time,sqlite3
from collections import Counter,defaultdict
import numpy as np
from .common import OUT,RUN,SOURCE,OLD_SCORES,Features,datasets,read,write
from .metrics import pair_weights
from experiments.aster_cr_retrain_paired_v2_20261111 import m1_plan

def run():
    started=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    data=datasets();features=Features()
    missing={}
    for s,ps in data.items():
        ids=sorted({p[e] for p in ps for e in ("left","right")})
        missing[s]=[u for u in ids if u not in features.records or not __import__("pathlib").Path(features.records[u]["feature"]).exists()]
    fit_uids={p[e] for p in data["fit"] for e in ("left","right")}
    missing_grads=[u for u in sorted(fit_uids) if u in features.records and not features.records[u].get("has_grads")]
    write(OUT/"cache_coverage.json",dict(missing_counts={s:len(v) for s,v in missing.items()},missing_fit_gradients=len(missing_grads),
         pair_counts={s:dict(Counter(p["kind"] for p in ps)) for s,ps in data.items()}))
    write(RUN/"missing_inputs.json",dict(by_split=missing,fit_gradients=missing_grads))
    if missing["fit"] or missing_grads:
        raise RuntimeError("Training endpoint activations/gradients need extraction before M1.")
    attr_path=RUN/"pair_attributions.npz"
    if attr_path.exists():
        a=np.load(attr_path);attribution=a["attribution"];pair_ids=a["pair_ids"].tolist()
    else:
        attribution=np.empty((len(data["fit"]),1152),dtype=np.float64)
        for i,p in enumerate(data["fit"]):
            l,r=features.get(p["left"]),features.get(p["right"])
            d=r["heads"]-l["heads"]
            attribution[i]=.5*(np.abs(np.sum(d*l["grads"],axis=-1,dtype=np.float32))+np.abs(np.sum(d*r["grads"],axis=-1,dtype=np.float32)))
            if (i+1)%100==0:
                write(OUT/"preparation_progress.json",dict(stage="attribution",completed=i+1,total=len(data["fit"]),seconds=time.monotonic()-started))
        pair_ids=[p["pair_id"] for p in data["fit"]]
        np.savez_compressed(attr_path,attribution=attribution,pair_ids=np.asarray(pair_ids))
    index={pid:i for i,pid in enumerate(pair_ids)}
    def gradient(ps,features,group,number_heads):
        return pair_weights(ps)@attribution[[index[p["pair_id"]] for p in ps]]
    old_representatives=m1_plan._validation_representatives
    def representatives(ps,sources,kind,fold):
        reps,diag=old_representatives(ps,sources,kind,fold)
        observed=[p for p in reps if p["m1_role"]=="observed"]
        weights={p["pair_id"]:float(w) for p,w in zip(observed,pair_weights(observed))} if observed else {}
        masses=defaultdict(float)
        for p in observed:masses[p["source_id"]]+=weights[p["pair_id"]]
        originals={(p["left"],p["source_id"]):p for p in observed}
        for p in reps:
            ref=p if p["m1_role"]=="observed" else originals[p["left"],p["source_id"]]
            p["m1_weight"]=weights[ref["pair_id"]]/masses[p["source_id"]]
        return reps,diag
    m1_plan._gradient_attribution=gradient
    m1_plan._validation_representatives=representatives
    plan=m1_plan.make_plan("E1",data["fit"],features,OUT/"m1")
    plan["weighting"]="class/source/case/variant hierarchy; normalized in each training fold"
    write(OUT/"m1/m1_plan.json",plan)
    needed=defaultdict(set)
    for kd in plan["kinds"].values():
        for fold in kd["folds"]:
            for p in fold["representatives"]:
                needed[p["left"],p["right"]].update(fold["candidates"])
                needed[p["right"],p["left"]].update(fold["candidates"])
    lookup=[sqlite3.connect("file:"+str(p)+"?mode=ro",uri=True) for p in [OLD_SCORES/"existing_scores.sqlite",*sorted(OLD_SCORES.glob("new_scores_*.sqlite"))] if p.exists()]
    cache=sqlite3.connect(RUN/"reused_scores.sqlite")
    cache.execute("CREATE TABLE IF NOT EXISTS scores (target TEXT,donor TEXT,head INTEGER,value REAL,PRIMARY KEY(target,donor,head)) WITHOUT ROWID")
    pending=defaultdict(list);reused=0
    for (a,b),heads in sorted(needed.items()):
        found={}
        for c in lookup:
            for h,v in c.execute("SELECT head,value FROM scores WHERE target=? AND donor=?",(a,b)):
                if h in heads:found[int(h)]=float(v)
        cache.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",[(a,b,h,v) for h,v in found.items()])
        reused+=len(found)
        for h in sorted(heads-set(found)):
            pending[a].append(dict(target=a,donor=b,head=h))
    cache.commit();cache.close()
    for c in lookup:c.close()
    shards=[[] for _ in range(4)];costs=[0]*4
    for target,requests in sorted(pending.items(),key=lambda kv:-len(kv[1])):
        w=int(np.argmin(costs));requests.sort(key=lambda r:(r["donor"],r["head"]))
        shards[w].extend(requests);costs[w]+=len(requests)
    for i,shard in enumerate(shards):write(RUN/("requests_"+str(i)+".json"),shard)
    summary=dict(complete=True,candidate_heads_by_kind={q:len({h for f in d["folds"] for h in f["candidates"]}) for q,d in plan["kinds"].items()},
        requested_directed_scores=sum(len(h) for h in needed.values()),reused_directed_scores=reused,
        missing_directed_scores=sum(costs),per_worker_scores=costs,targets=len(pending),seconds=time.monotonic()-started)
    write(OUT/"preparation.json",summary)
    print(json.dumps(summary))
if __name__=="__main__":run()
