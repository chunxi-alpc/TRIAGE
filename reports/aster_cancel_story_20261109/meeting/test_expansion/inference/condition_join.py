
"""Complete the two fixed fit/dev prompt jobs; no GPU work or monitoring."""
import json,time
from pathlib import Path
ROOT=Path('/home/gcx/aster_next_20261109/condition_only_results')
tags=['fit_natural','dev_full_gold']
while True:
 failed=[str(ROOT/(tag+'.failed.json')) for tag in tags if (ROOT/(tag+'.failed.json')).exists()]
 if failed:
  (ROOT/'failed.json').write_text(json.dumps(dict(state='failed',files=failed),indent=2));break
 done=[ROOT/(tag+'.done.json') for tag in tags]
 if all(p.exists() for p in done):
  records={tag:json.loads(path.read_text()) for tag,path in zip(tags,done)}
  if all(r.get('state')=='complete' for r in records.values()):
   (ROOT/'done.json').write_text(json.dumps(dict(state='complete',jobs=records),indent=2));break
 time.sleep(5)

