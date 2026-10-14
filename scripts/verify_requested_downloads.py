#!/usr/bin/env python3
"""Check integration of the requested original files; no benchmark inference."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from src.data_loader import load_dataset
from resume_requested_downloads import atomic
cfg=json.loads((ROOT/'configs/datasets.json').read_text())
result={}
for name,split in [('MEGA-CDP','train_val'),('MedCalc-Bench','train'),('FaithEval-unanswerable','test'),('ConflictMedQA','original'),('ConflictMedQA','v2'),('MedChangeQA','test'),('ClashEval','test')]:
    sample=next(load_dataset(name,split=split))
    assert sample['question'] and sample['reference_answer'] is not None,(name,split)
    result[name+':'+split]={'status':'passed','raw_fields':list(sample['raw']),'question_type':type(sample['question']).__name__,'reference_answer_type':type(sample['reference_answer']).__name__}
completed=json.loads((ROOT/'results/completed_data_loader_validation.json').read_text())
for split in ['QA','MR','MC']:
    record=completed['ConFiQA:'+split]
    assert record['status']=='passed' and record['rows']==6000
    result['ConFiQA:'+split]=record
assert all(cfg[n]['status']=='downloaded' and cfg[n]['complete'] for n in ['MEGA-CDP','MedCalc-Bench','FaithEval-unanswerable','ConflictMedQA','ConFiQA','MedChangeQA','ClashEval'])
assert cfg['MEGA-CDP'].get('images_requested') is False
atomic(ROOT/'results/continuation_integration_validation.json',result)
print(json.dumps({k:v['status'] for k,v in result.items()}))

