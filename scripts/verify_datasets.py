#!/usr/bin/env python
"""Read one original sample per available dataset; do not run benchmarks."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import load_dataset

def main():
    config=json.loads((ROOT/'configs'/'datasets.json').read_text())
    status={}
    for name,entry in config.items():
        if entry.get('status')!='downloaded':
            status[name]={'status':entry.get('status'),'verification':'not_run','reason':entry.get('reason')}
            continue
        try:
            item=next(load_dataset(name))
            if not item['question'] or item['reference_answer'] is None:
                raise ValueError('Original sample lacks a mapped question or reference answer')
            status[name]={'status':'passed','verification':'passed','id':item.get('id'),'question_type':type(item['question']).__name__,
                          'reference_answer_type':type(item['reference_answer']).__name__,
                          'original_fields':list(item['raw'].keys()),'path':entry['path']}
        except Exception as exc:
            status[name]={'status':'failed','verification':'failed','reason':f'{type(exc).__name__}: {exc}'}
    (ROOT/'results'/'datasets_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps({n:v['status'] for n,v in status.items()},ensure_ascii=False))
    return int(any(v.get('verification')=='failed' for v in status.values()))

if __name__=='__main__':
    raise SystemExit(main())
