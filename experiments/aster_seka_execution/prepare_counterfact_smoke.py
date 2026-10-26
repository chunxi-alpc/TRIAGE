from pathlib import Path
import json,sys,hashlib
s=Path("/home/gcx/TRIAGE/third_party/SEKA")
sys.path.insert(0,str(s))
from benchmarks.counterfact.download import _reformat_counterfact_sample
p=Path("/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/seka_data")
raw=p/"pasta_bench/counterfact.json"
if not raw.exists():raw=raw.with_suffix(".json.part")
text=raw.read_text()
dec=json.JSONDecoder();i=text.index("[")+1;rows=[]
for index in range(16):
 while text[i].isspace() or text[i]==",": i+=1
 row,j=dec.raw_decode(text,i);rows.append(_reformat_counterfact_sample(row));i=j
out=p/"counterfact_smoke16";out.mkdir(exist_ok=True)
f=out/"counterfact.jsonl"
with f.open("x") as h:
 for row in rows:h.write(json.dumps(row)+"\n")
meta={"source_url":"https://rome.baulab.info/data/dsets/counterfact.json","source_indices":[0,16],"rows":len(rows),"source_prefix_bytes_sha256":hashlib.sha256(text[:i].encode()).hexdigest(),"formatted_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"case_ids":[r["id"] for r in rows],"reformatter":"upstream _reformat_counterfact_sample; write one JSON object per line (upstream file writer omits newline)","full_download_complete":raw.name=="counterfact.json"}
(out/"manifest.json").write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
