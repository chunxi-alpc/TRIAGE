import requests,hashlib,json,time
from pathlib import Path
p=Path("/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/seka_data/pasta_bench")
p.mkdir(parents=True,exist_ok=True)
u="https://rome.baulab.info/data/dsets/counterfact.json"
t=time.time()
with requests.get(u,stream=True,timeout=(15,60)) as r:
 r.raise_for_status()
 with (p/"counterfact.json.part").open("wb") as f:
  for b in r.iter_content(1024*1024): f.write(b)
(p/"counterfact.json.part").rename(p/"counterfact.json")
b=(p/"counterfact.json").read_bytes()
x=json.loads(b)
meta={"url":u,"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b),"rows":len(x),"elapsed_seconds":time.time()-t}
(p/"download_manifest.json").write_text(json.dumps(meta,indent=2));print(meta,flush=True)
