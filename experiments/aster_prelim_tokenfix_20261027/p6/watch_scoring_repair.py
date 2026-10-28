"""Wait for the preserved generation plus repaired scorer, then summarize once."""
import json,subprocess,sys,time,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
run=B/'private/p6_smoke_auto100_v1';status=B/'private/p6_smoke_auto100_v1.repair_coordinator.json'
def dump(x):status.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
dump({'status':'WAITING_FOR_EXISTING_GENERATION_AND_REPAIRED_SCORING','generation_run':str(run),'scoring_subdir':'scoring_v2','generation_restarted':False})
while True:
 if (run/'FAILED.json').exists():
  dump({'status':'GENERATION_FAILED','generation_restarted':False});break
 if (run/'DONE.json').exists() and (run/'scoring_v2/DONE.json').exists():
  with (B/'private/p6_smoke_auto100_v1.summary_v2.log').open('x') as f:
   result=subprocess.run([sys.executable,str(D/'summarize_auto_pilot_v2.py'),'--run-dir',str(run)],stdout=f,stderr=subprocess.STDOUT)
  dump({'status':'COMPLETED' if result.returncode==0 else 'SUMMARY_FAILED','exit_code':result.returncode,'scoring_subdir':'scoring_v2','generation_restarted':False});break
 pid=int((B/'private/p6_smoke_auto100_v1.scoring_v2.pid').read_text())
 try:os.kill(pid,0)
 except ProcessLookupError:
  if not (run/'scoring_v2/DONE.json').exists():dump({'status':'REPAIRED_SCORER_EXITED_WITHOUT_DONE','generation_restarted':False});break
 time.sleep(10)
