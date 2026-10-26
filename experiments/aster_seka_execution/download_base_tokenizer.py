import requests,hashlib,json,time
from pathlib import Path
u="https://modelscope.cn/models/Qwen/Qwen3-8B-Base/resolve/master/tokenizer.json"
p=Path("/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/models/Qwen3-8B-Base/tokenizer.json")
t=time.time();r=requests.get(u,timeout=(15,90));r.raise_for_status();b=r.content;oid=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
expected="443909a61d429dff23010e5bddd28ff530edda00"
assert oid==expected,"ModelScope mirror does not match pinned official Hugging Face git object"
p.write_bytes(b);print(json.dumps({"bytes":len(b),"git_blob_oid":oid,"matched_pinned_official_object":True,"elapsed_seconds":time.time()-t}),flush=True)
