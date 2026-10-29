import os, concurrent.futures, json
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import snapshot_download, hf_hub_download
root=Path('/home/gcx/TRIAGE-worktrees/applicability-decision-20261029-assets')
def model():
 return snapshot_download('Qwen/Qwen3-0.6B-Base',revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd',local_dir=root/'model',allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json'],max_workers=3)
def data():
 return hf_hub_download('nchennnn/MEGA-CDP','train_val/decision_trees.jsonl',repo_type='dataset',revision='8f058c88ce1ed43693785b435ad0ff99907a2890',local_dir=root/'data')
with concurrent.futures.ThreadPoolExecutor(2) as ex:
 futures={ex.submit(f): n for n,f in [('model',model),('data',data)]}
 for f in concurrent.futures.as_completed(futures):
  n=futures[f]
  try: print(json.dumps({'asset':n,'path':str(f.result()),'status':'completed'}),flush=True)
  except Exception as err: print(json.dumps({'asset':n,'status':'failed','error':repr(err)}),flush=True)
