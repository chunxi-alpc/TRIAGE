"""Publish only the requested Qwen structural-validity subset; no inference."""
from pathlib import Path
import json,shutil,datetime
base=Path(__file__).resolve().parent;r=Path('/home/gcx/aster_validation_20261022/private/run500/mechanism/priority200');out=base/'results/priority200';out.mkdir(exist_ok=True,parents=True)
shutil.copyfile(r/'selection.json',out/'selection.json')
for stage in ['branches','complete']:
 p=r/f'qwen_{stage}.manifest.json'
 if p.exists():shutil.copyfile(p,out/p.name)
a=r/'qwen_attention';dest=out/'qwen/attention'
if a.exists():
 dest.mkdir(exist_ok=True,parents=True)
 for n in ['summary.json','selection.json','status.json']:
  if (a/n).exists():shutil.copyfile(a/n,dest/n)
 if (a/'test.jsonl').exists():(dest/'scores.jsonl').write_text(''.join(json.dumps({k:v for k,v in x.items() if k!='completion'})+'\n' for x in map(json.loads,(a/'test.jsonl').read_text().splitlines())))
s=(r/'qwen.status').read_text().strip() if (r/'qwen.status').exists() else 'not_started';complete=s=='complete' and (out/'qwen/full/summary.json').exists() and (dest/'status.json').exists()
# Attention writes summary rather than skip-status when executed.
if s=='complete' and (out/'qwen/full/summary.json').exists() and (dest/'summary.json').exists():complete=True
(out/'STATUS.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'model':'Qwen3-8B','cases':200,'K':76,'U':124,'status':s,'execution_mode':(r/'execution_mode').read_text().strip() if (r/'execution_mode').exists() else 'previous_queued_shards','gpu_shards':{} if (r/'execution_mode').exists() else {str(g):(r/f'gpu{g}/status').read_text().strip() if (r/f'gpu{g}/status').exists() else 'not_started' for g in [0,1]},'complete':complete,'original_500_and_CAD':'Qwen unchanged; Llama saved and paused at user request','remaining_mechanism':'deferred at user request; not declared complete'},indent=2))
print(s)
