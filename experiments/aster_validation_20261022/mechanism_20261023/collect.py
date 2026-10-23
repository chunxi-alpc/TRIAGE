import json,hashlib,shutil,datetime
from pathlib import Path
root=Path('/home/gcx/aster_validation_20261022/private/run500/mechanism');exp=Path(__file__).resolve().parent;out=exp/'results';out.mkdir(exist_ok=True)
shutil.copyfile(root/'manifest.json',out/'dataset_manifest.json');labels=json.loads((root/'condition_labels.json').read_text());(out/'condition_type_review.json').write_text(json.dumps({hashlib.sha256(k.encode()).hexdigest():v for k,v in labels.items()},indent=2))
status={'updated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'models':{}}
for model in ['llama','qwen']:
 dest=out/model;dest.mkdir(exist_ok=True)
 for name in ['smoke','branches','complete']:
  f=root/f'{model}_{name}.manifest.json'
  if f.exists():shutil.copyfile(f,dest/f.name)
 att=root/f'{model}_attention';attdest=dest/'attention'
 if att.exists():
  attdest.mkdir(exist_ok=True)
  for name in ['summary.json','selection.json','status.json']:
   if (att/name).exists():shutil.copyfile(att/name,attdest/name)
  if (att/'test.jsonl').exists():(attdest/'scores.jsonl').write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='completion'})+'\n' for r in map(json.loads,(att/'test.jsonl').read_text().splitlines())))
 f=root/f'{model}.status';status['models'][model]=f.read_text().strip() if f.exists() else 'not_started'
status['complete']=all(s=='complete' for s in status['models'].values());(out/'STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
