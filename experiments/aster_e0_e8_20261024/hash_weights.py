import hashlib,json,time
from pathlib import Path
start=time.time();out={}
for name in ['Meta-Llama-3.1-8B-Instruct','Qwen3-8B','medgemma-27b-text-it']:
 out[name]={}
 for p in sorted((Path('/home/gcx/TRIAGE/models/modelscope')/name).glob('*.safetensors')):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  out[name][p.name]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
Path(__file__).with_name('results').joinpath('weight_hashes.json').write_text(json.dumps({'models':out,'seconds':time.time()-start},indent=2))
