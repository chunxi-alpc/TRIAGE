import json,sys,subprocess,os,time
from pathlib import Path
p=Path(sys.argv[1]);q=json.loads(p.read_text());results=[]
for item in q["jobs"]:
 with open(item["log"],"w") as log:r=subprocess.run(item["command"],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED="0"))
 results.append({"name":item["name"],"exit_code":r.returncode,"completed_at":time.time()});p.with_name(p.stem+"_results.json").write_text(json.dumps(results,indent=2))
 if r.returncode:sys.exit(r.returncode)
