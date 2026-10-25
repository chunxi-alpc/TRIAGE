"""Merge preserved rows and disjoint dynamic GPU workers without overwriting runs."""
import hashlib,json,time
from pathlib import Path
old=Path('/home/gcx/aster_action_mass_seed0_20261025');root=old/'qwen_dual_v1';out=root/'qwen'
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
expected={(r['id'],r['variant']) for r in read(root/'dev.jsonl')}
rows=read(old/'qwen/predictions.jsonl');caches=read(old/'qwen/condition_cache.jsonl')
previous=json.loads((root/'previous_status.json').read_text());worker_info={}
for gpu in ('0','1','6'):
 d=root/('worker'+gpu);done=json.loads((d/'DONE.json').read_text());assert done['complete']
 rr=read(d/'predictions.jsonl');assert done['completed']==len(rr)
 rows+=rr;caches+=read(d/'condition_cache.jsonl')
 worker_info[gpu]={'done':done,'manifest':json.loads((d/'manifest.json').read_text()),'status':json.loads((d/'status.json').read_text()),'gpu_tests':json.loads((d/'gpu_tests.json').read_text())}
assert len(rows)==800 and len({(r['id'],r['variant']) for r in rows})==800
assert {(r['id'],r['variant']) for r in rows}==expected
out.mkdir(exist_ok=False);rows.sort(key=lambda r:(r['id'],r['variant']))
(out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(out/'condition_cache.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in caches))
status={'completed':800,'total':800,'complete':True,'elapsed_seconds':previous['elapsed_seconds']+max(w['done']['elapsed_seconds'] for w in worker_info.values()),'gpu_hours':previous['gpu_hours']+sum(w['done']['gpu_hours'] for w in worker_info.values()),'peak_GiB':max([previous['peak_GiB']]+[w['status']['peak_GiB'] for w in worker_info.values()]),'execution':'44 preserved rows plus dynamic workers on GPU0/1/6','forward_call_units':previous['forward_call_units']+sum(w['status']['forward_call_units'] for w in worker_info.values())}
(out/'status.json').write_text(json.dumps(status,indent=2))
manifest={'name':'qwen','seed':0,'rows':800,'previous_manifest':json.loads((old/'qwen/manifest.json').read_text()),'workers':worker_info,'data_sha256':hashlib.sha256((root/'dev.jsonl').read_bytes()).hexdigest(),'previous_predictions_sha256':hashlib.sha256((old/'qwen/predictions.jsonl').read_bytes()).hexdigest(),'note':'No previous row repeated. All 23 configurations retained. Same operator; scheduling only changed.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'gpu_tests.json').write_text(json.dumps({g:w['gpu_tests'] for g,w in worker_info.items()},indent=2))
print(json.dumps(status))
