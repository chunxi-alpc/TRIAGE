import json, time, concurrent.futures as cf
from resume_requested_downloads import ROOT, RESULTS, session, valid, git_download
from pathlib import Path
meta=json.loads((RESULTS/"source_ClashEval_history.json").read_text())
items=[i for i in meta["tree"] if i["path"] in ["data/dataset/dataset.pqt","data/dataset/croissant.json","README.md"]]
jobs=[]
for item in items:
 target=ROOT/"data/ClashEval"/item["path"]
 if valid(target,item):continue
 target.parent.mkdir(parents=True,exist_ok=True)
 part=target.with_name(target.name+".incomplete")
 offset=part.stat().st_size if part.exists() else 0
 url="https://raw.githubusercontent.com/kevinwu23/StanfordClashEval/ba892d17f9b125264494689b7ce5d8ab22ed821d/"+item["path"]
 for start in range(offset,item["size"],256*1024):
  end=min(start+256*1024,item["size"])-1
  chunk=target.parent/(target.name+".chunks")/str(start);chunk.parent.mkdir(parents=True,exist_ok=True)
  jobs.append((url,start,end,chunk))
def work(j):
 url,start,end,path=j
 if path.exists() and path.stat().st_size==end-start+1:return
 for attempt in range(6):
  try:
   r=session().get(url,headers={"Range":f"bytes={start}-{end}"},timeout=(10,40));r.raise_for_status()
   if r.status_code==200 and start==0 and len(r.content)==end+1:data=r.content
   else:
    assert r.status_code==206 and r.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
    data=r.content
   assert len(data)==end-start+1
   path.write_bytes(data);return
  except Exception as e:
   if attempt==5:raise
   time.sleep(1+attempt)
with cf.ThreadPoolExecutor(max_workers=12) as ex:
 for f in cf.as_completed([ex.submit(work,j) for j in jobs]):f.result()
for item in items:
 target=ROOT/"data/ClashEval"/item["path"]
 if valid(target,item):continue
 part=target.with_name(target.name+".incomplete")
 offset=part.stat().st_size if part.exists() else 0
 with part.open("ab") as out:
  for start in range(offset,item["size"],256*1024):
   chunk=target.parent/(target.name+".chunks")/str(start);out.write(chunk.read_bytes())
 assert valid(part,item),target
 part.replace(target)
git_download("ClashEval","kevinwu23/StanfordClashEval","ba892d17f9b125264494689b7ce5d8ab22ed821d",["README.md","data/dataset/croissant.json","data/dataset/dataset.pqt"],metadata_file="source_ClashEval_history.json")
