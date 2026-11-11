"""Shared files and exact-input feature access for paired-data retraining."""
import json, os, sqlite3
from pathlib import Path
from functools import lru_cache
from collections.abc import Mapping, ValuesView, ItemsView
import numpy as np
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "aster_cr_retrain_paired_v2_20261111"
DATA = OUT / "data"
CACHE = OUT / "cache"
RUNTIME = OUT / "runtime"
HIST = Path("/home/gcx/aster_cr_repair_v2_run_20261108")
FROZEN = Path("/home/gcx/aster_cr_pair_dataset_20261111")
SEED = 20261111
def rows(path):
    path = Path(path)
    if not path.exists(): return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
def read(path, default=None):
    path=Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else ({} if default is None else default)
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)
def lines(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp."+str(os.getpid()))
    with tmp.open("w",encoding="utf-8") as f:
        for row in records: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    tmp.replace(path)
def append(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
CONFIG = read(OUT / "configuration.json")

class _InputRecord(Mapping):
    """The complete input mapping, with large prompt/alias values read only on access."""
    def __init__(self, table, uid, offset, size, metadata):
        self._table,self._uid,self._offset,self._size=table,uid,offset,size
        self._metadata=json.loads(metadata) if isinstance(metadata,str) else metadata
        self._keys=tuple(self._metadata)+tuple(k for k in ("prompt","aliases") if k not in self._metadata)
    def __getitem__(self,key):
        if key in self._metadata:return self._metadata[key]
        return self._table._raw(self._uid,self._offset,self._size)[key]
    def __iter__(self):return iter(self._keys)
    def __len__(self):return len(self._keys)

class _InputValues(ValuesView):
    def __iter__(self):
        for row in self._mapping.conn.execute("SELECT uid,offset,length,metadata FROM inputs ORDER BY ordinal"):
            yield _InputRecord(self._mapping,*row)
class _InputItems(ItemsView):
    def __iter__(self):
        for row in self._mapping.conn.execute("SELECT uid,offset,length,metadata FROM inputs ORDER BY ordinal"):
            yield row[0],_InputRecord(self._mapping,*row)

class InputTable(Mapping):
    """Insertion-ordered UID mapping backed by a read-only byte-offset index."""
    def __init__(self, source, database, manifest):
        self.source=Path(source);self.manifest=manifest
        self.conn=sqlite3.connect("file:"+str(database)+"?mode=ro&immutable=1",uri=True,check_same_thread=False)
        self.fd=os.open(str(self.source),os.O_RDONLY)
        self.count=int(manifest["count"])
        self._raw=lru_cache(maxsize=128)(self._read_raw)
    def _read_raw(self,uid,offset,size):
        return json.loads(os.pread(self.fd,size,offset))
    def __getitem__(self,uid):
        row=self.conn.execute("SELECT uid,offset,length,metadata FROM inputs WHERE uid=?",(uid,)).fetchone()
        if row is None:raise KeyError(uid)
        return _InputRecord(self,*row)
    def __iter__(self):
        for row in self.conn.execute("SELECT uid FROM inputs ORDER BY ordinal"):yield row[0]
    def __len__(self):return self.count
    def values(self):return _InputValues(self)
    def items(self):return _InputItems(self)
    def close(self):
        if getattr(self,"fd",None) is not None:os.close(self.fd);self.fd=None
        if getattr(self,"conn",None) is not None:self.conn.close();self.conn=None
    def __del__(self):
        try:self.close()
        except Exception:pass

def load_inputs(out):
    out=Path(out);source=out/"data/inputs.jsonl";runtime=out/"runtime"
    manifest=read(runtime/"input_index.json")
    if manifest.get("ready") and (runtime/"inputs.sqlite").exists() and source.exists():
        st=source.stat()
        if st.st_size==manifest.get("source_size") and st.st_mtime_ns==manifest.get("source_mtime_ns"):
            return InputTable(source,runtime/"inputs.sqlite",manifest)
    return {r.get("uid",r["view_id"]):r for r in rows(source)}

def load_records(out):
    out=Path(out);records={}
    for path in [out/"data/reused_scores.jsonl",*sorted((out/"cache").glob("scores_*.jsonl"))]:
        for r in rows(path):
            if r.get("status","ok")=="ok" and "feature" in r and "ell" in r:
                records[r.get("uid",r["view_id"])]=r
    return records

class RuntimeFeatures:
    """Read-only FP16 training features; source files and numeric values are unchanged."""
    def __init__(self,out):
        self.root=Path(out)/"runtime";self.manifest=read(self.root/"features.json")
        self.ready=False
        if not self.manifest.get("ready"):return
        self.uids=read(self.root/"fit_uids.json",[])
        self.index={uid:i for i,uid in enumerate(self.uids)}
        self.sources=read(self.root/"fit_sources.json",[])
        self.heads=np.load(self.root/"fit_heads.npy",mmap_mode="r")
        self.grads=np.load(self.root/"fit_grads.npy",mmap_mode="r")
        self.present=np.load(self.root/"fit_present.npy",mmap_mode="r")
        self.residual=np.load(self.root/"fit_residual.npy",mmap_mode="r") if (self.root/"fit_residual.npy").exists() else None
        self.ready=True
    def row(self,uid,record,need_grads=False):
        if not self.ready:return None
        i=self.index.get(uid)
        if i is None or i>=len(self.sources) or self.sources[i]!=record.get("feature"):return None
        flag=int(self.present[i]);needed=3 if need_grads else 1
        if flag&needed!=needed:return None
        return i
    def get(self,uid,record):
        i=self.row(uid,record,need_grads=bool(record.get("has_grads")))
        if i is None:return None
        result=dict(record);result["ell"]=float(record["ell"])
        result["heads"]=self.heads[i].astype(np.float32)
        if int(self.present[i])&2:result["grads"]=self.grads[i].astype(np.float32)
        if int(self.present[i])&4 and self.residual is not None:result["residual"]=self.residual[i].astype(np.float32)
        return result
    def selected(self,uid,record,selected):
        i=self.row(uid,record)
        if i is None:return None
        h=self.heads[i]
        return h.astype(np.float32) if selected is None else h[np.asarray(selected,dtype=int)].astype(np.float32).reshape(-1)

class Store:
    def __init__(self, out=None, runtime=True):
        self.out=Path(out) if out is not None else OUT
        self.inputs=load_inputs(self.out)
        self.runtime=RuntimeFeatures(self.out) if runtime else None
        self.records={}
        self.refresh()
    def refresh(self):
        self.records=load_records(self.out)
        self.scores=self.records
        self.get.cache_clear()
        return self
    def record(self,uid): return self.records[uid]
    def ell(self,uid): return float(self.record(uid)["ell"])
    @lru_cache(maxsize=24)
    def get(self,uid):
        r=self.record(uid)
        if self.runtime is not None:
            result=self.runtime.get(uid,r)
            if result is not None:return result
        result=dict(r);result["ell"]=float(r["ell"])
        with np.load(r["feature"],allow_pickle=False) as arrays:
            heads=arrays["heads"].astype(np.float32)
            if "head_ids" in arrays and heads.ndim==2 and heads.shape[0] != 1152:
                ids=arrays["head_ids"].astype(int).reshape(-1)
                full=np.full((1152,heads.shape[-1]),np.nan,np.float32)
                full[ids]=heads.reshape(len(ids),-1);heads=full
            else:heads=heads.reshape(-1,heads.shape[-1])
            result["heads"]=heads
            if "grads" in arrays:result["grads"]=arrays["grads"].astype(np.float32).reshape(heads.shape)
            if "residual" in arrays:result["residual"]=arrays["residual"].astype(np.float32)
        return result
    def heads(self,uid,selected=None):
        if self.runtime is not None:
            result=self.runtime.selected(uid,self.record(uid),selected)
            if result is not None:return result
        h=self.get(uid)["heads"]
        return h if selected is None else h[np.asarray(selected,dtype=int)].reshape(-1)
