"""Read saved tensors/predictions only. No tokenizer, model, GPU or new evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from collections import Counter
import json
import numpy as np
import torch
from care_m2_inc.io import read_json,read_jsonl,write_json,sha256_file

ROOT=Path('experiments/care_module2_incremental_20261020')
RUN=ROOT/'runs/v3_repair_20261020T144100Z'
OUT=Path(__file__).parent
before={str(p):sha256_file(p) for p in RUN.rglob('*') if p.is_file()}
result={'new_model_forwards':0,'new_predictions':0,'source_run':str(RUN),'tasks':{}}
for task,axis_name in [('N2','source'),('N3','DQ'),('N5','A1')]:
    axis=torch.load(RUN/task/'axes.pt',map_location='cpu',weights_only=True)[axis_name]
    d=axis['mean'];u=axis['basis'];projected=u@(u.T@d);sv=axis['singular_values']
    geometry={'mean_norm':float(d.norm()),'projected_mean_norm':float(projected.norm()),
        'mean_length_retained':float(projected.norm()/d.norm()),'mean_energy_retained':float(projected.square().sum()/d.square().sum()),
        'centered_difference_variance_top2':float(sv[:2].square().sum()/sv.square().sum()),
        'route_control_component_norm_beta1':float(projected.norm()/2)}
    dev=read_jsonl(RUN/task/'dev_predictions.jsonl');test=read_jsonl(RUN/task/'test/test_predictions.jsonl')
    baseline={(r['case_id'],json.dumps(r['control'],sort_keys=True)):r for r in dev if r['method']=='baseline'}
    grid=[]
    for method,beta in sorted({(r['method'],r['beta']) for r in dev if r['method'] in ('route','mean','direct_bias','text','filter')}):
        rows=[r for r in dev if r['method']==method and r['beta']==beta]
        edits=[e for r in rows for e in r.get('hook',{}).get('residual',[])]
        b=[baseline[(r['case_id'],json.dumps(r['control'],sort_keys=True))] for r in rows]
        item={'method':method,'beta':beta,'total':len(rows),'correct':sum(r['correct'] for r in rows),
          'prediction_changed':sum(r['prediction']!=z['prediction'] for r,z in zip(rows,b)),
          'corrected':sum(r['correct'] and not z['correct'] for r,z in zip(rows,b)),
          'damaged':sum(z['correct'] and not r['correct'] for r,z in zip(rows,b))}
        if edits:
            norms=[e['requested_norm']/beta for e in edits] if beta else []
            item['edits']=len(edits);item['requested_norm_median']=float(np.median([e['requested_norm'] for e in edits]))
            item['actual_relative_norm_median']=float(np.median([e['relative_norm'] for e in edits]))
            if method=='route' and beta:
                bnorm=geometry['route_control_component_norm_beta1']
                item['common_component_norm_lower_bound_median']=float(np.median([max(0,x-bnorm) for x in norms]))
        if task=='N2':
            by={cid:[r for r in rows if r['case_id']==cid] for cid in {r['case_id'] for r in rows}}
            item['same_answer_for_both_controls']=sum(x[0]['prediction']==x[1]['prediction'] for x in by.values())
            item['both_correct']=sum(all(r['correct'] for r in x) for x in by.values())
            if method=='route' and beta:
                a=[];bsize=geometry['route_control_component_norm_beta1']
                for rs in by.values():
                    ns=[r['hook']['residual'][0]['requested_norm']/beta for r in rs]
                    a.append(max(0,(ns[0]**2+ns[1]**2)/2-bsize**2)**.5)
                item['exact_common_component_norm_median']=float(np.median(a))
        if task=='N5':
            good=lambda r: all(s['state']==next(z['state'] for z in r['reference_states'] if z['rule_id']==s['rule_id']) for s in r['signals'])
            item['A1_strata']={str(flag):{'correct':sum(r['correct'] for r in rows if good(r)==flag),'total':sum(good(r)==flag for r in rows)} for flag in [True,False]}
            if beta and method=='route':
                margins=[r['candidate_margin'] for r in rows if r.get('candidate_margin') is not None]
                item['candidate_margin_min_median_max']=[min(margins),float(np.median(margins)),max(margins)]
        grid.append(item)
    x={'geometry':geometry,'dev_grid':grid}
    if task=='N3':
        x['test_by_DQ']={m:{state:{'correct':sum(r['correct'] for r in test if r['method']==m and r['DQ']==state),'total':sum(r['method']==m and r['DQ']==state for r in test)} for state in ('D','Q')} for m in ('baseline','mean','route','text')}
    if task=='N5':
        labels={r['case_id']:r for r in read_jsonl(ROOT/'data_v2_ready/labels.jsonl')}
        pairs=read_json(RUN/task/'n5_pair_audit.json')['pairs'];pairs_reference=Counter()
        for pair in pairs:
            states=[]
            for side in ['positive','negative']:
                o=pair[side];states.append(next(s['state'] for s in labels[o['case_id']]['reference_states'] if s['rule_id']==o['rule_id']))
            pairs_reference[str(tuple(states))]+=1
        x['paired_reference_posthoc_only']=dict(pairs_reference)
        for split,rows in [('dev',dev),('test',test)]:
            base=[r for r in rows if r['method']=='baseline'];good=lambda r:all(s['state']==next(z['state'] for z in r['reference_states'] if z['rule_id']==s['rule_id']) for s in r['signals'])
            x[split+'_A1_strata']={str(flag):{'correct':sum(r['correct'] for r in base if good(r)==flag),'total':sum(good(r)==flag for r in base)} for flag in [True,False]}
    result['tasks'][task]=x
assert before=={str(p):sha256_file(p) for p in RUN.rglob('*') if p.is_file()}
result['source_files_unchanged']=True;result['source_file_count']=len(before)
write_json(OUT/'analysis.json',result);write_json(OUT/'input_sha256.json',before)
print(json.dumps(result,ensure_ascii=False,indent=2))
