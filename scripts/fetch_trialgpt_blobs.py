from pathlib import Path
import urllib.request,json,base64,subprocess,concurrent.futures,time
repo=Path("/home/gcx/TRIAGE/third_party/TrialGPT")
commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","FETCH_HEAD"],text=True).strip()
tree=subprocess.check_output(["git","-C",str(repo),"ls-tree","-r","FETCH_HEAD"],text=True)
wanted=[]
for line in tree.splitlines():
 info,path=line.split("\t",1);mode,kind,sha=info.split()
 if kind=="blob" and (path=="README.md" or path.startswith("trialgpt_matching/") or path=="dataset/sigir/retrieved_trials.json"):
  wanted.append((path,sha))
def fetch(item):
 path,sha=item
 # Populate the existing Git object's cache from the official immutable GitHub blob API.
 has=subprocess.run(["git","-C",str(repo),"cat-file","--batch-check"],input=sha+"\n",text=True,capture_output=True,env={"GIT_NO_LAZY_FETCH":"1","PATH":"/usr/bin:/bin"})
 if " missing" not in has.stdout:return path,"cached"
 req=urllib.request.Request(f"https://api.github.com/repos/ncbi-nlp/TrialGPT/git/blobs/{sha}",headers={"Accept":"application/vnd.github.raw+json","User-Agent":"TRIAGE-deployment"})
 with urllib.request.urlopen(req,timeout=90) as response:
  content=response.read();ctype=response.headers.get("Content-Type","")
 if "json" in ctype:
  obj=json.loads(content);content=base64.b64decode(obj["content"])
 actual=subprocess.check_output(["git","-C",str(repo),"hash-object","-w","--stdin"],input=content).decode().strip()
 if actual!=sha:raise RuntimeError(f"Git blob SHA mismatch for {path}")
 return path,len(content)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(fetch,wanted):print(result,flush=True)
exists=subprocess.run(["git","-C",str(repo),"show-ref","--verify","--quiet","refs/heads/main"]).returncode==0
subprocess.run(["git","-C",str(repo),"checkout","main"] if exists else ["git","-C",str(repo),"checkout","-b","main","FETCH_HEAD"],check=True)
print("Original official sparse checkout completed at",commit,flush=True)
