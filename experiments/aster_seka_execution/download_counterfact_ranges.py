from pathlib import Path
import requests,concurrent.futures,time,hashlib,json
p=Path("/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/seka_data/pasta_bench")
parts=p/"range_parts";parts.mkdir(exist_ok=True)
u="https://rome.baulab.info/data/dsets/counterfact.json"
size=45108470; chunk=2*1024*1024;t=time.time()
def get(start):
 end=min(start+chunk,size)-1;f=parts/f"{start:012d}.part"
 if f.exists() and f.stat().st_size==end-start+1:return f
 for attempt in range(3):
  try:
   with requests.get(u,headers={"Range":f"bytes={start}-{end}"},timeout=(15,90)) as r:
    r.raise_for_status()
    if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}" or len(r.content)!=end-start+1:raise ValueError("range mismatch")
    f.write_bytes(r.content);print("chunk_complete",start,len(r.content),flush=True);return f
  except Exception:
   if attempt==2:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:files=list(ex.map(get,range(0,size,chunk)))
out=p/"counterfact.ranges.json"
with out.open("wb") as f:
 for part in files:f.write(part.read_bytes())
b=out.read_bytes();rows=json.loads(b);assert len(b)==size
canonical=p/"counterfact.json"
if canonical.exists():assert canonical.read_bytes()==b
else:out.rename(canonical)
meta={"url":u,"transport":"8 parallel HTTP byte ranges; Content-Range and byte counts checked","sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b),"rows":len(rows),"elapsed_seconds":time.time()-t}
(p/"download_manifest.json").write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
