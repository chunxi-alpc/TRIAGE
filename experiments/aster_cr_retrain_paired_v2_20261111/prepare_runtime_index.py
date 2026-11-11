"""One-time CPU runtime indexes and incremental, lossless FP16 cache materialization."""
import argparse,collections,json,os,sqlite3,time
from pathlib import Path
import numpy as np
from .common import OUT,DATA,CACHE,read,rows,write,load_records

def build_inputs(out=OUT):
    out=Path(out);runtime=out/"runtime";runtime.mkdir(parents=True,exist_ok=True)
    source=out/"data/inputs.jsonl";st=source.stat();manifest=read(runtime/"input_index.json")
    if manifest.get("ready") and manifest.get("source_size")==st.st_size and manifest.get("source_mtime_ns")==st.st_mtime_ns:
        return manifest
    started=time.monotonic();temp=runtime/("inputs.sqlite.tmp."+str(os.getpid()))
    conn=sqlite3.connect(temp)
    conn.execute("PRAGMA journal_mode=OFF");conn.execute("PRAGMA synchronous=OFF")
    conn.execute("CREATE TABLE inputs(ordinal INTEGER PRIMARY KEY, uid TEXT UNIQUE, offset INTEGER, length INTEGER, fit INTEGER, metadata TEXT)")
    batch=[];count=0;offset=0
    with source.open("rb") as f:
        for raw in f:
            record=json.loads(raw)
            uid=record.get("uid",record["view_id"])
            metadata={k:v for k,v in record.items() if k not in ("prompt","aliases")}
            batch.append((count,uid,offset,len(raw),int(record.get("required_grads",False)),json.dumps(metadata,ensure_ascii=True,separators=(",",":"))))
            count+=1;offset+=len(raw)
            if len(batch)>=1000:
                conn.executemany("INSERT INTO inputs VALUES(?,?,?,?,?,?)",batch);batch=[]
    if batch:conn.executemany("INSERT INTO inputs VALUES(?,?,?,?,?,?)",batch)
    conn.execute("CREATE INDEX training_inputs ON inputs(fit)")
    conn.commit();conn.close();temp.replace(runtime/"inputs.sqlite")
    manifest=dict(ready=True,count=count,source=str(source),source_size=st.st_size,source_mtime_ns=st.st_mtime_ns,
                  seconds=time.monotonic()-started,read_policy="source text remains read-only; byte offsets and metadata are indexed; prompt and aliases load on demand")
    write(runtime/"input_index.json",manifest)
    return manifest

def prefix_representatives(out=OUT):
    out=Path(out);needed=set();pairs=0
    for group in ("E1","E2"):
        bykindsource={}
        for p in rows(out/"data"/(group+"_pairs.jsonl")):
            key=(p["kind"],p["source_id"]);old=bykindsource.get(key)
            if old is None or p["pair_id"]<old["pair_id"]:bykindsource[key]=p
        for p in bykindsource.values():needed.update((p["left"],p["right"]));pairs+=1
    result=dict(uids=sorted(needed),unique_inputs=len(needed),representative_pairs_before_cross_group_dedup=pairs,
       policy="All new-group exact M1 representatives: minimum pair_id per category/source in E1 and E2; E3 shares E2; source folds select subsets")
    write(out/"runtime/prefix_required_uids.json",result);return result

def build_features(out=OUT,follow=False,interval=30):
    out=Path(out);runtime=out/"runtime";build_inputs(out);started=time.monotonic()
    conn=sqlite3.connect("file:"+str(runtime/"inputs.sqlite")+"?mode=ro",uri=True)
    uids=[r[0] for r in conn.execute("SELECT uid FROM inputs WHERE fit=1 ORDER BY ordinal")];conn.close()
    if not (runtime/"fit_heads.npy").exists():
        heads=np.lib.format.open_memmap(runtime/"fit_heads.npy",mode="w+",dtype=np.float16,shape=(len(uids),1152,128))
        grads=np.lib.format.open_memmap(runtime/"fit_grads.npy",mode="w+",dtype=np.float16,shape=(len(uids),1152,128))
        present=np.lib.format.open_memmap(runtime/"fit_present.npy",mode="w+",dtype=np.uint8,shape=(len(uids),))
        present[:]=0;present.flush()
        write(runtime/"fit_uids.json",uids);sources=[""]*len(uids);write(runtime/"fit_sources.json",sources)
    else:
        if read(runtime/"fit_uids.json")!=uids:raise RuntimeError("Training UID order changed")
        heads=np.load(runtime/"fit_heads.npy",mmap_mode="r+");grads=np.load(runtime/"fit_grads.npy",mmap_mode="r+")
        present=np.load(runtime/"fit_present.npy",mmap_mode="r+");sources=read(runtime/"fit_sources.json")
    residual=np.lib.format.open_memmap(runtime/"fit_residual.npy",mode="w+",dtype=np.float16,shape=(len(uids),4096)) if not (runtime/"fit_residual.npy").exists() else np.load(runtime/"fit_residual.npy",mmap_mode="r+")
    total_converted=0
    while True:
        records=load_records(out);converted=0
        for i,uid in enumerate(uids):
            r=records.get(uid)
            if r is None:continue
            path=r["feature"]
            if sources[i]==path and ((int(present[i])&3)==3 or ((int(present[i])&1) and not r.get("has_grads"))):continue
            with np.load(path,allow_pickle=False) as a:
                h=a["heads"]
                if h.size!=1152*128 or h.dtype!=np.float16:continue
                if "grads" in a and a["grads"].dtype!=np.float16:continue
                present[i]=0
                heads[i]=h.reshape(1152,128)
                flag=1
                if "grads" in a:
                    grads[i]=a["grads"].reshape(1152,128);flag|=2
                if "residual" in a and a["residual"].size==4096 and a["residual"].dtype==np.float16:
                    residual[i]=a["residual"].reshape(4096);flag|=4
                sources[i]=path;present[i]=flag;converted+=1
        heads.flush();grads.flush();residual.flush();present.flush()
        total_converted+=converted;write(runtime/"fit_sources.json",sources)
        complete=int(np.count_nonzero((present&3)==3));head_rows=int(np.count_nonzero(present&1))
        manifest=dict(ready=True,training_inputs=len(uids),head_rows=head_rows,gradient_rows=complete,
          complete=complete==len(uids),converted=total_converted,seconds=time.monotonic()-started,
          dtype="float16 from existing npz arrays; no extra quantization",storage_bytes=int(heads.nbytes+grads.nbytes+residual.nbytes+present.nbytes),
          source_records="existing reused_scores plus newly written cache scores; no model forwards",
          fallback="Store uses original npz for absent or changed source paths")
        write(runtime/"features.json",manifest)
        if not follow or complete==len(uids):return manifest
        time.sleep(interval)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inputs",action="store_true");p.add_argument("--features",action="store_true")
    p.add_argument("--all",action="store_true");p.add_argument("--follow",action="store_true");p.add_argument("--interval",type=int,default=30)
    a=p.parse_args()
    if a.inputs or a.all:print(json.dumps(build_inputs()),flush=True);print(json.dumps(prefix_representatives()),flush=True)
    if a.features or a.all:print(json.dumps(build_features(follow=a.follow,interval=a.interval)),flush=True)
