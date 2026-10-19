#!/usr/bin/env python3
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.module1.backend import LocalBackend
from scripts.run_acre_pilot import run_case,write
DATA=ROOT/'results/acre_pilot/round2';OUT=DATA/'revision3';config=json.loads((ROOT/'configs/acre_pilot_v3.json').read_text());torch.set_num_threads(4);torch.manual_seed(config['seed']);backend=LocalBackend();backend._load_qwen();start=time.time()
manifest=json.loads((DATA/'development_manifest.json').read_text())['samples'];ids=['v2_39_3_original','v2_112_13_original','v2_68_17_original'];samples=[next(x for x in manifest if x['case_id']==cid) for cid in ids]
history=json.loads((ROOT/'results/acre_pilot/history_manifest.json').read_text())['samples'];samples.append(next(x for x in history if x['case_id']=='5:15'))
summary=[]
for item in samples:
 result=run_case(backend,item,config,OUT/'smoke',betas=[1.]);write(OUT/'smoke'/('case_'+item['case_id'].replace(':','_')+'.json'),result)
 assert not result['annotation_errors'] and all(result['identity_controls'].values())
 assert not any(row['mapping_errors'] for row in result['annotations'])
 summary.append({'case_id':item['case_id'],'seconds':result['elapsed_seconds'],'prompt_tokens':result['prompt_tokens'],'B4_tokens':result['variants']['B4']['prompt_tokens'],'peak_gib':result['gpu_peak_allocated_gib'],'annotation_errors':0,'mapping_errors':0})
write(OUT/'smoke/summary.json',{'cases':summary,'total_seconds':time.time()-start,'complete':True})
