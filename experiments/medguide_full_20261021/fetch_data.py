"""Download source files privately; verify recorded content hashes."""
import hashlib,json,os
from pathlib import Path
import requests
P=Path(__file__).resolve().parent
manifest=json.loads((P/'data_manifest.json').read_text());private=P/'private';private.mkdir(exist_ok=True)
endpoint=os.environ.get('HF_ENDPOINT','https://hf-mirror.com')
files=[('train-00000-of-00001.parquet',endpoint+'/datasets/'+manifest['dataset']+'/resolve/'+manifest['revision']+'/data/train-00000-of-00001.parquet',manifest['parquet_sha256']),('all_paths.json',manifest['source_guideline_url'],manifest['source_guideline_sha256'])]
for name,url,expected in files:
 p=private/name
 if not p.exists():
  r=requests.get(url,timeout=120);r.raise_for_status();p.write_bytes(r.content)
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 if actual!=expected:raise RuntimeError(name+' hash differs from fixed experiment source; do not substitute silently')
 print(name,actual)
