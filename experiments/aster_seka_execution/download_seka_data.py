import json,time,os
from huggingface_hub import hf_hub_download
from pathlib import Path
r=Path("/home/gcx/TRIAGE-worktrees/aster-seka-execution-20261025/experiments/aster_seka_execution/runtime/seka_data")
r.mkdir(exist_ok=True)
for name in ["README.md","pasta_bench/counterfact.jsonl","pasta_bench/BIOS.pkl"]:
 print("download_start",name,flush=True)
 p=hf_hub_download(repo_id="waylonli/SEKA-datasets",repo_type="dataset",revision="91bf3f67fb2044c7e81514e103a962d6ef76c230",filename=name,local_dir=r,endpoint="https://hf-mirror.com")
 print("downloaded",p,Path(p).stat().st_size,flush=True)
