"""Exactly one independent case at a time per worker, with dead-claim recovery."""
import errno,json,os,shutil,time
from pathlib import Path
from .data import write
from .engine import METHODS
def alive(pid):
 try:os.kill(int(pid),0);return True
 except ProcessLookupError:return False
def complete(path,judge):
 return all((path/(m+'.json')).exists() and (not judge or (path/'official_scores'/(m+'.json')).exists()) for m in METHODS)
def claim(root,i,shard):
 lock=root/(str(i)+'.lock');staging=root/(str(i)+'.new.'+str(os.getpid()))
 staging.mkdir(exist_ok=False);write(staging/'owner.json',{'pid':os.getpid(),'shard':shard,'case_index':i})
 try:
  try:staging.rename(lock);return lock
  except OSError as e:
   if e.errno not in [errno.EEXIST,errno.ENOTEMPTY]:raise
  guard=root/(str(i)+'.reclaim')
  try:guard.mkdir()
  except FileExistsError:return None
  try:
   try:owner=json.loads((lock/'owner.json').read_text())
   except FileNotFoundError:
    try:staging.rename(lock);return lock
    except (FileExistsError,OSError):return None
   if alive(owner['pid']):return None
   stale=root/(str(i)+'.stale.'+str(os.getpid()))
   try:lock.rename(stale);shutil.rmtree(stale);staging.rename(lock);return lock
   except FileNotFoundError:return None
  finally:guard.rmdir()
 finally:
  if staging.exists():shutil.rmtree(staging)
def assigned(cases,run,out,shard,dynamic=False,judge=False):
 if not dynamic:
  for i,c in enumerate(cases):
   if i%2==shard:yield i,c,out/'units'/c['case_id'].replace(':','_')
  return
 root=run/'official_test_queue';root.mkdir(exist_ok=True);units=run/'official_test_units';units.mkdir(exist_ok=True)
 for i,c in enumerate(cases):
  path=units/c['case_id'].replace(':','_')
  if complete(path,judge):continue
  lock=claim(root,i,shard)
  if lock is None:continue
  try:
   if not complete(path,judge):yield i,c,path
   if complete(path,judge):write(root/(str(i)+'.done.json'),{'case_index':i,'owner_shard':shard,'status':'complete'})
  finally:
   if lock.exists() and json.loads((lock/'owner.json').read_text())['pid']==os.getpid():shutil.rmtree(lock)

