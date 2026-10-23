import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);result={}
for name,path in {'llama':'/home/gcx/TRIAGE/models/modelscope/Meta-Llama-3.1-8B-Instruct','qwen':'/home/gcx/TRIAGE/models/modelscope/Qwen3-8B'}.items():
 files={}
 for p in sorted(Path(path).iterdir()):
  if not p.is_file() or (p.suffix!='.safetensors' and p.name not in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json')):continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
  files[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
 result[name]={'path':path,'files':files}
(root/'model_fingerprints.json').write_text(json.dumps(result,indent=2))
