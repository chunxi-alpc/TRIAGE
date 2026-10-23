"""Only future invocations: honor an explicit output ownership plan before loading a model."""
import os,sys,json,time,hashlib
from pathlib import Path
DEFAULT_PLAN=Path('/home/gcx/aster_validation_20261022/private/run500/parallel_qwen/plan.json')

def wait_if_delegated(plan_path=DEFAULT_PLAN,argv=None):
 argv=sys.argv[1:] if argv is None else argv
 if '--output' not in argv or not plan_path.exists():return
 output=Path(argv[argv.index('--output')+1]).resolve();plan=json.loads(plan_path.read_text());task=next((t for t in plan['tasks'] if Path(t['output']).resolve()==output),None)
 if task is None:return
 if plan.get('cancelled') and not task.get('selected200_only'):raise RuntimeError('Old task plan cancelled: '+plan['cancelled'])
 for flag,want in [('--input',task['input']),('--model',task['model'])]:
  assert flag in argv and Path(argv[argv.index(flag)+1]).resolve()==Path(want).resolve(),f'delegated task argument mismatch: {flag}'
 owner=os.environ.get('TRIAGE_JOB_OWNER')==task['owner'];manifest=output.with_suffix('.manifest.json');printed=False
 while True:
  if manifest.exists():
   try:m=json.loads(manifest.read_text())
   except json.JSONDecodeError:m={}
   if m.get('complete'):
    assert m['rows']==task['rows'] and m['input_sha256']==task['input_sha256'],'delegated completion manifest mismatch'
    barrier=task.get('consumer_barrier')
    if owner or not barrier or (Path(barrier).exists() and Path(barrier).read_text().strip()=='complete'):
     print('Reusing completed delegated output: '+str(output),flush=True);raise SystemExit(0)
  if owner:
   if output.exists() and task['kind']!='filtered':raise RuntimeError('Partial delegated output preserved; explicit recovery required: '+str(output))
   return
  status=Path(task['owner_status'])
  if status.exists() and status.read_text().startswith('failed'):raise RuntimeError('Delegated owner failed; output preserved: '+status.read_text())
  if not printed:print('Waiting for delegated output, without loading a model: '+str(output),flush=True);printed=True
  time.sleep(10)
