import json
from pathlib import Path
def rows(p):
 p=Path(p)
 return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def read(p):
 p=Path(p)
 return json.loads(p.read_text()) if p.exists() else {}
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lines(p,vv):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for v in vv:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n')
