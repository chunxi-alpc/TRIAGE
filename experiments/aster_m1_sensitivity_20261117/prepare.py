"""Prepare fixed source subsets, screen from cached gradients, and enumerate missing interventions."""
import json,sqlite3,time
from collections import defaultdict,Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from .audit import read,lines,write,table
from .statistics import KINDS,scenarios,candidates

REPO=Path(__file__).resolve().parents[2]
CODE=REPO/"experiments/aster_m1_sensitivity_20261117"
OUT=REPO/"reports/aster_m1_sensitivity_20261117"
RUN=Path("/home/gcx/aster-m1-sensitivity-runtime-20261117")
OLD=Path("/home/gcx/aster-m1-revision-20261117")
OLDQ=OLD/"reports/aster_m1_revision_20261117/models/qwen"
DATA=Path("/home/gcx/aster-cr-retrain-paired-v2-20261111/reports/aster_cr_retrain_paired_v2_20261111")
PRIVATE=Path("/home/gcx/aster-m1-revision-runtime-20261117/qwen")

def prepare():
    started=time.monotonic()
    if (OUT/"plan.json").exists():raise FileExistsError("Plan already exists; do not overwrite")
    OUT.mkdir(exist_ok=True);RUN.mkdir(exist_ok=True)
    cfg=read(CODE/"configuration.json")
    old=read(OLDQ/"selection_plan.json")
    pairs=list(lines(DATA/"data/E1_pairs.jsonl"))
    screen=old["screening_sources"];confirm=old["confirmation_sources"]
    assert not(set(screen)&set(confirm))
    rng=np.random.default_rng(cfg["seed"])
    order=rng.permutation(len(screen));folds=np.empty(len(screen),int)
    for f,ix in enumerate(np.array_split(order,5)):folds[ix]=f
    subsets=[]
    for i in range(cfg["source_subsets"]):
        subsets.append(dict(id=i,screen=sorted(rng.choice(len(screen),int(len(screen)*.8),replace=False).tolist()),
                            confirm=sorted(rng.choice(len(confirm),int(len(confirm)*.8),replace=False).tolist())))
    cache={r["uid"]:r for r in lines(PRIVATE/"cache/scores_fit.jsonl")}
    @lru_cache(maxsize=64)
    def feature(uid):
        with np.load(cache[uid]["feature"],allow_pickle=False) as z:
            return dict(heads=z["heads"].astype(float).reshape(1152,-1),
                        grads=z["grads"].astype(float).reshape(1152,-1))
    sums=np.zeros((4,len(screen),1152));counts=np.zeros((4,len(screen)),int)
    si={s:i for i,s in enumerate(screen)}
    for p in pairs:
        if p["source_id"] not in si:continue
        l,r=feature(p["left"]),feature(p["right"])
        diff=r["heads"]-l["heads"]
        score=.5*(np.abs((diff*l["grads"]).sum(1))+np.abs((diff*r["grads"]).sum(1)))
        k,s=KINDS.index(p["kind"]),si[p["source_id"]]
        sums[k,s]+=score;counts[k,s]+=1
    assert (counts>0).all()
    attribution=sums/counts[:,:,None]
    np.savez_compressed(OUT/"screening_source_attributions.npz",attribution=attribution,counts=counts,
                        folds=folds,sources=np.array(screen))
    masks=[];records=[]
    for scenario in scenarios(cfg):
        mask,freq=candidates(attribution,folds,np.arange(len(screen)),scenario)
        masks.append(mask)
        records.append(dict(name=scenario["name"],scope="full",
                            by_kind={k:np.flatnonzero(mask[j]).tolist() for j,k in enumerate(KINDS)}))
        for sub in subsets:
            mask,_=candidates(attribution,folds,sub["screen"],scenario)
            masks.append(mask)
            records.append(dict(name=scenario["name"],scope="subset_"+str(sub["id"]),
                                by_kind={k:np.flatnonzero(mask[j]).tolist() for j,k in enumerate(KINDS)}))
    measured=np.any(masks,axis=0)
    required={k:np.flatnonzero(measured[j]).tolist() for j,k in enumerate(KINDS)}
    ps=[p for p in pairs if p["source_id"] in set(confirm)]
    strata=defaultdict(list)
    for p in ps:strata[(p["source_id"],p["kind"],p["left_label"],p["right_label"])].append(p)
    tasks=[];coverage=Counter();donor_counts=Counter()
    for p in sorted(ps,key=lambda x:x["pair_id"]):
        pool=sorted([d for d in strata[(p["source_id"],p["kind"],p["left_label"],p["right_label"])]
                    if d["pair_id"]!=p["pair_id"] and (d["left"],d["right"])!=(p["left"],p["right"])],
                    key=lambda x:x["pair_id"])
        choices=rng.permutation(len(pool))[:cfg["matched_donors"]]
        donors=[pool[i] for i in choices]
        donor_counts[len(donors)]+=1
        tasks.append(dict(target=p,donors=donors,heads=required[p["kind"]],
                          eligible=bool(donors),worker=len(tasks)%4))
        if donors:coverage[p["kind"]]+=1
    write(OUT/"candidate_manifest.json",records)
    write(OUT/"source_subsets.json",subsets)
    write(OUT/"matched_donor_plan.json",tasks)
    import sqlite3,gzip
    db=RUN/"existing_scores.sqlite"
    if db.exists():raise FileExistsError("Existing score index already present; inspect before rebuilding")
    conn=sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=OFF");conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE scores (target TEXT, donor TEXT, head INTEGER, value REAL, PRIMARY KEY(target,donor,head)) WITHOUT ROWID")
    lookup={p["pair_id"]:p for p in ps}
    buffer=[]
    with gzip.open(OLDQ/"models/E1/full/paired_effects.jsonl.gz","rt") as f:
        for line in f:
            r=json.loads(line);p=lookup[r["pair_id"]];h=r["head"]
            buffer.extend([(p["left"],p["right"],h,r["paired_forward"]),
                           (p["right"],p["left"],h,r["paired_reverse"]),
                           (p["left"],p["left"],h,r["self_left"]),
                           (p["right"],p["right"],h,r["self_right"])])
            if len(buffer)>=20000:
                conn.executemany("INSERT OR IGNORE INTO scores VALUES (?,?,?,?)",buffer);buffer=[]
    conn.executemany("INSERT OR IGNORE INTO scores VALUES (?,?,?,?)",buffer);conn.commit()
    indexed=conn.execute("SELECT count(*) FROM scores").fetchone()[0]
    missing=set();all_requests=set();eligible_sources={k:set() for k in KINDS}
    for task in tasks:
        if not task["eligible"]:continue
        p=task["target"];eligible_sources[p["kind"]].add(p["source_id"])
        for h in task["heads"]:
            edges=[(p["left"],p["left"],h),(p["right"],p["right"],h),
                   (p["left"],p["right"],h),(p["right"],p["left"],h)]
            for d in task["donors"]:
                edges.extend([(p["left"],d["right"],h),(p["right"],d["left"],h)])
            for key in edges:
                if key in all_requests:continue
                all_requests.add(key)
                if conn.execute("SELECT 1 FROM scores WHERE target=? AND donor=? AND head=?",key).fetchone() is None:
                    missing.add(key)
    conn.close()
    # Assignment is by target input so every new target/donor/head score has exactly one owner.
    targets=sorted({a for a,b,h in missing})
    target_owner={uid:i%4 for i,uid in enumerate(targets)}
    missing_by_worker=[[] for _ in range(4)]
    for a,b,h in sorted(missing):
        missing_by_worker[target_owner[a]].append(dict(target=a,donor=b,head=h))
    for i,rows in enumerate(missing_by_worker):write(RUN/("requests_"+str(i)+".json"),rows)
    plan=dict(version=cfg["version"],configuration=cfg,source_model="Qwen3-8B",
              screening_sources=screen,confirmation_sources=confirm,source_subsets=len(subsets),
              configurations=len(scenarios(cfg)),required_heads=required,
              existing_score_index_rows=indexed,requested_unique_scores=len(all_requests),
              reused_unique_scores=len(all_requests)-len(missing),new_unique_patch_forwards=len(missing),
              new_baseline_forwards=len(targets),new_total_forwards=len(missing)+len(targets),
              worker_patch_forwards=[len(x) for x in missing_by_worker],
              matched_pairs_by_kind=dict(coverage),
              eligible_sources_by_kind={k:len(s) for k,s in eligible_sources.items()},
              donor_count_distribution=dict(donor_counts),private_runtime=str(RUN),
              no_matching_donor_action="exclude pair, retain missing coverage in results",
              pair_count=len(ps),fit_input_cache_reused=True,fit_gradients_reused=True,
              preparation_seconds=time.monotonic()-started,selected_using_task_accuracy=False,
              new_gpu_work_started=False,protocol_confirmation="user confirmed new statistical version; retain PR81 rule")
    write(OUT/"plan.json",plan)
    print(json.dumps({k:plan[k] for k in ["configurations","new_total_forwards","worker_patch_forwards",
                "matched_pairs_by_kind","eligible_sources_by_kind","donor_count_distribution","preparation_seconds"]}))
if __name__=="__main__":prepare()
