import hashlib,json,subprocess,sys,platform
from pathlib import Path
P=Path(__file__).resolve().parent
models={'phi4':Path('/home/gcx/models/care_module1_v5/phi-4'),'adapter':Path('/home/gcx/models/care_module1_v5/trialmatchai-phi4-reasoning-lora'),'qwen':Path('/home/gcx/care_20261016/CARE_Codex_Pilot/models/Qwen3-4B-Base')}
out={'python':sys.version,'platform':platform.platform(),'models':{}}
for name,p in models.items():
 files={}
 for f in sorted(p.iterdir()):
  if f.is_file() and (f.suffix in ['.json','.safetensors','.bin','.txt']):
   h=hashlib.sha256()
   with f.open('rb') as s:
    for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
   files[f.name]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
 out['models'][name]={'path':str(p),'files':files}
(P/'model_fingerprints.json').write_text(json.dumps(out,indent=2))
print('model fingerprints completed')
