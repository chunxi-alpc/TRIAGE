from pathlib import Path
import json
from transformers import AutoTokenizer
from mega_trainval_reader import iter_method_tasks,render_official_prompt
p=Path(__file__).resolve().parent;m=json.loads((p/'mega_dev16_input_lock.json').read_text());root=Path('/home/gcx/TRIAGE/data/.hf_cache/datasets--nchennnn--MEGA-CDP/snapshots/8f058c88ce1ed43693785b435ad0ff99907a2890');tokenizer=AutoTokenizer.from_pretrained('/home/gcx/TRIAGE/models/modelscope/Qwen3-8B',use_fast=True)
rows=[]
for identity,task in iter_method_tasks(root,m):
 text=render_official_prompt(task,p/'runtime/vendors/MEGA-CDP-ssh',m)
 ids=tokenizer.apply_chat_template([{'role':'system','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 rows.append({'sample_id':identity.sample_id,'input_tokens':len(ids)})
report={'scope':'CPU tokenizer preflight only; no clinical inference or labels','cases':len(rows),'min_input_tokens':min(x['input_tokens'] for x in rows),'max_input_tokens':max(x['input_tokens'] for x in rows),'rows':rows};(p/'mega_dev16_token_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
