"""Wait for one annotation PID to exit; signal failure to the root pipeline."""
import os,sys,json,time,select,argparse,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
CACHE=Path('/home/gcx/aster_trace_repair_20261110')
FAILED=CACHE/'scoring/annotations/shard0/failed.json'
LOG=CACHE/'annotations/extraction.log'
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main(pid):
 started=time.strftime('%Y-%m-%d %H:%M:%S')
 meta=dict(guard_pid=os.getpid(),annotation_pid=pid,started_at=started,annotation_log=str(LOG),waiting_scope='process exit only; no GPU or experiment-progress monitoring')
 dump(HERE/'diagnostic_guard.json',dict(meta,state='waiting_for_process_exit'))
 try:
  try:fd=os.pidfd_open(pid,0)
  except ProcessLookupError:fd=None
  if fd is not None:
   try:
    poll=select.poll();poll.register(fd,select.POLLIN|select.POLLHUP|select.POLLERR);poll.poll()
   finally:os.close(fd)
  ready={}
  for split in ['dev','final']:
   file=HERE/f'diagnostic_{split}_ready.json'
   ready[split]=json.loads(file.read_text()) if file.exists() else None
  complete=HERE/'diagnostic_extraction_complete.json'
  success=complete.exists() and all(x and x.get('state')=='complete' and x.get('ready') for x in ready.values())
  if not success:
   raise RuntimeError('Annotation process exited before complete diagnostic data and ready markers were exported. See '+str(LOG))
  dump(HERE/'diagnostic_guard.json',dict(meta,state='complete',completed_at=time.strftime('%Y-%m-%d %H:%M:%S'),diagnostic_ready={k:v['state'] for k,v in ready.items()}))
 except Exception as e:
  error=dict(meta,state='failed',time=time.strftime('%Y-%m-%d %H:%M:%S'),error=str(e),traceback=traceback.format_exc(),failure_scope='source_grounded_annotation_or_diagnostic_export')
  dump(FAILED,error);dump(HERE/'diagnostic_guard.json',error)
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args();main(a.pid)
