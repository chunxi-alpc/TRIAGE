"""Cache one score per (case, exact output, reference, scoring configuration)."""
import fcntl,hashlib,json,os,tempfile
from pathlib import Path
def cache_key(case_id,output,reference,scoring_config):
 payload={'case_id':case_id,'output':output,'reference':reference,'scoring_config':scoring_config}
 encoded=json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
 return hashlib.sha256(encoded.encode()).hexdigest()
def get_or_score(directory,case_id,output,reference,scoring_config,scorer):
 root=Path(directory);root.mkdir(parents=True,exist_ok=True)
 key=cache_key(case_id,output,reference,scoring_config)
 target=root/(key+'.json')
 with (root/(key+'.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if target.exists():return json.loads(target.read_text())['score'],key,True
  score=scorer()
  record=json.dumps({'cache_key':key,'score':score},ensure_ascii=False,allow_nan=False)
  fd,tmp=tempfile.mkstemp(prefix=key+'.',suffix='.tmp',dir=root)
  try:
   with os.fdopen(fd,'w') as f:f.write(record+'\n');f.flush();os.fsync(f.fileno())
   os.replace(tmp,target)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
  return score,key,False

