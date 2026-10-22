"""Retrieve the official SEKA CounterFact release; no replacement synthetic benchmark."""
import json,shutil,sys
from pathlib import Path
from huggingface_hub import hf_hub_download
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
try:
 f=hf_hub_download('waylonli/SEKA-datasets','pasta_bench/counterfact.json',repo_type='dataset',revision='main')
 shutil.copyfile(f,out/'counterfact.json')
 (out/'positive_source.json').write_text(json.dumps({'path':f,'downloaded':True}))
except Exception as e:
 (out/'positive_source.json').write_text(json.dumps({'downloaded':False,'error':type(e).__name__+': '+str(e)[:500]}));raise
