from pathlib import Path
import json,sys,types,time,hashlib,spacy
s=Path("/home/gcx/TRIAGE/third_party/SEKA");sys.path.insert(0,str(s))
from benchmarks.biasbios import reformat_dataset as upstream
p=Path("/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/seka_data/hf_official/pasta_bench")
output=p/"biasbios.json"
if output.exists():raise FileExistsError(output)
original_dump=json.dump
def dump_jsonl(value, handle, *args, **kwargs):
 original_dump(value,handle,*args,**kwargs);handle.write("\n")
upstream.json=types.SimpleNamespace(dump=dump_jsonl)
t=time.time();upstream.main(types.SimpleNamespace(biasbios_raw_path=p/"BIOS.pkl",biasbios_save_file=output))
rows=0;ids=set();duplicates=0
with output.open() as f:
 for line in f:
  x=json.loads(line);rows+=1;duplicates+=int(x["id"] in ids);ids.add(x["id"])
record={"rows":rows,"unique_source_ids":len(ids),"duplicate_source_id_count":duplicates,"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"upstream_source_sha256":hashlib.sha256((s/"benchmarks/biasbios/reformat_dataset.py").read_bytes()).hexdigest(),"serialization_patch":"module json.dump wrapper appends one newline; original filtering/prompt/attribute routines unchanged","upstream_raw_limit":50000,"spacy":spacy.__version__,"nlp_model":"en_core_web_sm","elapsed_seconds":time.time()-t,"new_gpu_forwards":0}
(p/"biasbios_reformat_manifest.json").write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
