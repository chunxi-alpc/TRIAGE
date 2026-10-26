import argparse,requests,concurrent.futures,time,hashlib,json
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument("--metadata",type=Path,required=True);a.add_argument("--output",type=Path,required=True);a=a.parse_args()
m=json.loads(a.metadata.read_text());h=m["headers"];size=int(h["X-Linked-Size"]);expected=h["X-Linked-ETag"].strip(chr(34));url=h["Location"]
a.output.parent.mkdir(parents=True,exist_ok=True);parts=a.output.with_name(a.output.name+".parts");parts.mkdir(exist_ok=True);chunk=4*1024*1024;t=time.time()
def get(start):
 end=min(start+chunk,size)-1;f=parts/f"{start:012d}"
 if f.exists() and f.stat().st_size==end-start+1:return f
 for attempt in range(3):
  try:
   with requests.get(url,headers={"Range":f"bytes={start}-{end}"},timeout=(15,90)) as r:
    r.raise_for_status();data=r.content
    if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}" or len(data)!=end-start+1:raise ValueError("range mismatch")
    f.write_bytes(data);print("completed_bytes",start,len(data),flush=True);return f
  except Exception:
   if attempt==2:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:files=list(ex.map(get,range(0,size,chunk)))
partial=a.output.with_name(a.output.name+".download")
sha=hashlib.sha256()
with partial.open("wb") as f:
 for part in files:
  b=part.read_bytes();sha.update(b);f.write(b)
assert sha.hexdigest()==expected, "SHA256 mismatch with official HF linked ETag"
assert partial.stat().st_size==size
if a.output.exists():assert hashlib.sha256(a.output.read_bytes()).hexdigest()==expected
else:partial.rename(a.output)
record={"repo_id":m["repo_id"],"revision":h["X-Repo-Commit"],"filename":m["filename"],"bytes":size,"sha256":expected,"verified_against_official_linked_etag":True,"transport":"Official public Hugging Face CDN; redirect resolved by client, all data downloaded on research server","elapsed_seconds":time.time()-t}
a.output.with_name(a.output.name+".download_manifest.json").write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
