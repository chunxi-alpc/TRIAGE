import json, os, time
from pathlib import Path
SEED=20261105
DEFAULT_ROOT=Path("/home/gcx/aster_closed_form_20261105")
MODEL="/home/gcx/TRIAGE/models/modelscope/Qwen3-8B"
MEDGEMMA="/home/gcx/TRIAGE/models/modelscope/medgemma-27b-text-it"
REPO=Path(__file__).resolve().parents[2]
def rows(path):
    path=Path(path)
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n");tmp.replace(path)
def write_rows(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as f:
        for v in values:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+"\n")
    tmp.replace(path)
def append(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
def status(root,stage,**kwargs):
    write_json(Path(root)/"status"/(stage+".json"),dict(stage=stage,time=time.strftime("%Y-%m-%d %H:%M:%S"),pid=os.getpid(),**kwargs))
def evaluate(e,f):
    op=e["op"]
    if op=="and": return all(evaluate(x,f) for x in e["args"])
    if op=="or": return any(evaluate(x,f) for x in e["args"])
    if op=="not": return not evaluate(e["arg"],f)
    a,b=f[e["field"]],e["value"]
    return {"<":lambda:a<b,"<=":lambda:a<=b,">":lambda:a>b,">=":lambda:a>=b,
            "==":lambda:a==b,"!=":lambda:a!=b,"in":lambda:a in b,"not_in":lambda:a not in b}[op]()
def fields(e):
    if e["op"] in ("and","or"):return set().union(*(fields(x) for x in e["args"]))
    if e["op"]=="not":return fields(e["arg"])
    return {e["field"]}
