import hashlib,json,sys,shutil
from pathlib import Path
base,extra,out=map(Path,sys.argv[1:])
bm=json.loads(base.with_suffix('.manifest.json').read_text());em=json.loads(extra.with_suffix('.manifest.json').read_text())
if not bm['complete'] or not em['complete'] or bm['input_sha256']!=em['input_sha256']:raise ValueError('Incompatible/incomplete inputs')
if em['rows']!=bm['groups']*6:raise ValueError('Incomplete unsigned ablation')
if out.exists():raise FileExistsError(str(out))
with out.open('wb') as f:
 for p in (base,extra):
  with p.open('rb') as r:shutil.copyfileobj(r,f)
bm.update(rows=bm['rows']+em['rows'],rows_per_case=58,supplement_manifest=em,components_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base,extra)})
out.with_suffix('.manifest.json').write_text(json.dumps(bm,indent=2))
