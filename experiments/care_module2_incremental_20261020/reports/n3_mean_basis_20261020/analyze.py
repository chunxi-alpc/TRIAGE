"""Summarize saved N3 dev outputs without model inference."""
import sys
from pathlib import Path
import numpy as np
from care_m2_inc.io import read_jsonl,write_json
p=Path(sys.argv[1]);grid=read_jsonl(p/'dev_grid_predictions.jsonl');reused=read_jsonl(p/'reused_comparisons.jsonl')
mean={r['case_id']:r for r in reused if r['method']=='mean'}
base={r['case_id']:r for r in reused if r['method']=='baseline'}
analysis=[]
for beta in (0.,.5,1.):
    rows=[r for r in grid if r['beta']==beta]
    record={'beta':beta,'correct':sum(r['correct'] for r in rows),'total':len(rows),
        'by_DQ':{s:{'correct':sum(r['correct'] for r in rows if r['DQ']==s),'total':sum(r['DQ']==s for r in rows)} for s in ('D','Q')},
        'Q_wrong_decision':sum(r['DQ']=='Q' and not r['predicted_query'] for r in rows),
        'D_unnecessary_query':sum(r['DQ']=='D' and r['predicted_query'] for r in rows)}
    for name,reference in [('mean',mean),('baseline',base)]:
        record['vs_'+name]={'corrected':[r['case_id'] for r in rows if r['correct'] and not reference[r['case_id']]['correct']],
            'damaged':[r['case_id'] for r in rows if not r['correct'] and reference[r['case_id']]['correct']]}
    edits=[e for r in rows for e in r['hook']['residual']]
    if edits:
        record['requested_norm_median']=float(np.median([e['requested_norm'] for e in edits]))
        record['actual_relative_norm_median']=float(np.median([e['relative_norm'] for e in edits]))
    analysis.append(record)
write_json(p/'strength_analysis.json',{'grid':analysis,'new_forward_count':0,'scope':'saved development predictions only'})
