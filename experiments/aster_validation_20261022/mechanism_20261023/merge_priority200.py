from pathlib import Path
import json,hashlib
r=Path('/home/gcx/aster_validation_20261022/private/run500/mechanism/priority200');ids={x['id'] for x in map(json.loads,(r/'test.jsonl').read_text().splitlines())}
for stage in ['branches','complete']:
 rows=[];manifests=[]
 for gpu in [0,1]:
  p=r/f'gpu{gpu}/qwen_{stage}.jsonl';m=json.loads(p.with_suffix('.manifest.json').read_text());part=list(map(json.loads,p.read_text().splitlines()));assert m['complete'] and m['cases']==100 and len(part)==m['rows'];rows+=part;manifests.append(m)
 assert {x['id'] for x in rows}==ids and len(ids)==200
 keys=[(x['id'],x['variant'],x['method']) for x in rows];assert len(keys)==len(set(keys))
 out=r/f'qwen_{stage}.jsonl';assert not out.exists();out.write_text(''.join(json.dumps(x)+'\n' for x in rows));m=dict(manifests[0]);m.update(cases=200,rows=len(rows),input_sha256=hashlib.sha256((r/'test.jsonl').read_bytes()).hexdigest(),gpu_shards=[0,1],shard_seconds=[x['seconds'] for x in manifests]);out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2))
