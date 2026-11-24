"""Two-stage audit of explicitly numeric conditions using an analyst's private blind span file."""
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT=Path(os.environ.get('ASTER_ARITHMETIC_ROOT',Path.home()/'aster-v3-review-followup-private/arithmetic'))
QUEUE=Path(os.environ.get('ASTER_BLIND_QUEUE',ROOT.parent/'queue'))
OUT=ROOT/'shareable_alt'
labels={'yes','no','indeterminate'}
def read(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f)) if p.suffix=='.csv' else [json.loads(x) for x in f]
def write(p,rs):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
def decision(op, nums, thresholds):
    val=nums[0]
    if op=='<':return val<thresholds[0]
    if op=='>':return val>thresholds[0]
    if op=='>=':return val>=thresholds[0]
    if op=='open_interval':return thresholds[0]<val<thresholds[1]
    if op=='closed_interval':return thresholds[0]<=val<=thresholds[1]
    if op=='ratio_ge':return nums[0]/nums[1]>=thresholds[0]
    raise ValueError('unknown operator')
def spans(s):
    return [tuple(map(int,x.split('-'))) for x in s.split(',') if x]
def prepare():
    # Reconstruct the blind working specifications without inspecting labels or outcomes.
    keys={(x['case_id'],x['gid']):x['blind_id'] for x in read(QUEUE/'coordinator/key.csv')}
    specs=read(Path(__file__).resolve().parents[3]/'reports/aster_v3_selective_20261124/followup/arithmetic/span_protocol.csv')
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[dict(key=keys[row['case_id'],row['gid']][:6], **{k:v for k,v in row.items() if k not in ('case_id','gid')}) for row in specs]
    assert len({v[:6] for v in keys.values()})==len(keys)
    write(ROOT/'blind_span_spec.csv',rows)
def lock():
    start=time.process_time()
    cards=read(QUEUE/'reviewer/cards.jsonl')
    qs=sorted({r['condition'] for r in cards})
    specs={r['key']:r for r in read(ROOT/'blind_span_spec.csv')}
    result=[]
    for r in cards:
        spec=specs.get(r['blind_id'][:6]);q=r['condition'];t=r['patient']
        if not spec:
            result.append(dict(blind_id=r['blind_id'],question=q,status='outside',decision='',math_branch='',metric='',op='',reason='outside_explicit_single_variable_threshold',value_spans='',condition_spans='',context_spans=''));continue
        assert qs[int(spec['qindex'])]==q
        qnums=[m.span() for m in re.finditer(r'\d+(?:\.\d+)?',q)]
        limitspans=[qnums[int(i)] for i in spec['threshold_indices'].split('|') if i]
        vspans=spans(spec['values'])
        nums=[Decimal(t[a:b]) for a,b in vspans]
        thresh=[Decimal(q[a:b]) for a,b in limitspans]
        computed=('yes' if decision(spec['op'],nums,thresh) else 'no') if spec['op'] else ''
        status=spec['status']
        if status=='resolved': assert computed in {'yes','no'}
        ctx=spans(spec['context'])
        assert ctx and all(0<=a<b<=len(t) for a,b in ctx)
        assert all(any(a<=i<j<=b for a,b in ctx) for i,j in vspans)
        result.append(dict(blind_id=r['blind_id'],question=q,status=status,decision=computed if status=='resolved' else 'indeterminate',math_branch=computed,
                     metric=spec['metric'],op=spec['op'],reason=spec['reason'],value_spans=spec['values'],
                     condition_spans=','.join(f'{a}-{b}' for a,b in limitspans),context_spans=spec['context']))
    assert sum(r['status']!='outside' for r in result)==len(specs)
    with (ROOT/'locked.jsonl').open('x') as f:
        for r in result:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (ROOT/'lock_cost.json').write_text(json.dumps(dict(cpu_seconds=time.process_time()-start)))
def link():
    start=time.process_time(); rs=read(ROOT/'locked.jsonl')
    keys={r['blind_id']:r for r in read(QUEUE/'coordinator/key.csv')}
    assert set(keys)=={r['blind_id'] for r in rs}
    original=Path.home()/'TRIAGE/data/MEGA-CDP/train_val/case_paths.jsonl'
    srcids={int(r['case_id'].split(':')[-2]) for r in keys.values()}
    sourcepaths={};linenos={}
    with original.open() as f:
        for n,line in enumerate(f,1):
            a=json.loads(line);g=int(a['guideline_id'])
            if g in srcids:sourcepaths[g]=a['paths'];linenos[g]=n
    v3=Path.home()/'aster-v3-selective-20261124/reports/aster_v3_selective_20261124/s1/first_errors.csv'
    eligible={(r['case_id'],r['gid']) for r in read(v3) if r['evidence_status']=='pending_review'}
    sources=defaultdict(Counter); public=[]
    for r in rs:
        k=keys[r['blind_id']]; src=k['source_id']; c=sources[src];c['screened']+=1
        assert (k['case_id'],k['gid']) in eligible
        if r['status']=='outside':c['outside']+=1;continue
        c['arithmetic_candidates']+=1;c[r['status']]+=1
        g=int(k['case_id'].split(':')[-2]); j=int(k['case_id'].split(':')[-1])-1
        z=[(step,b['decision_outcome']) for step,b in enumerate(sourcepaths[g][j]) if b.get('condition')==r['question']]
        assert len(z)==1 and z[0][1] in {'yes','no'}
        agree=(r['decision']==z[0][1]) if r['status']=='resolved' else None
        if agree is not None:c['agree' if agree else 'disagree']+=1
        out={**{i:k[i] for i in ['source_id','case_id','gid']},**{i:v for i,v in r.items() if i not in ['blind_id','question']},'archived_label':z[0][1],
             'agreement': 'not_scored' if agree is None else ('agrees' if agree else 'disagrees'),
             'source_paths_line':linenos[g],'case_path_index':j,'step_index':z[0][0]}
        public.append(out)
    write(OUT/'node_findings.csv', sorted(public,key=lambda x:(x['source_id'],x['case_id'],x['gid'])))
    metrics=('screened','arithmetic_candidates','resolved','pending','agree','disagree','outside')
    write(OUT/'by_source.csv',[dict(source_id=s,**{m:c[m] for m in metrics}) for s,c in sorted(sources.items())])
    c=sum(sources.values(),Counter()); totals={m:c[m] for m in metrics}
    (OUT/'status.json').write_text(json.dumps(dict(**totals,source_count=len(sources),
      candidate_sources=sum(c['arithmetic_candidates']>0 for c in sources.values()),resolved_sources=sum(c['resolved']>0 for c in sources.values()),
      discrepancy_sources=sum(c['disagree']>0 for c in sources.values()),distinct_conditions=len({r['question'] for r in rs}),
      candidate_conditions=len({r['question'] for r in rs if r['status']!='outside'}),
      prior_qualified_verdicts_reused=0,clinical_adjudications=0,labels_changed=0,new_model_calls=0,gpu_hours=0,
      cpu_seconds=json.loads((ROOT/'lock_cost.json').read_text())['cpu_seconds']+time.process_time()-start),indent=2)+'\n')
if __name__=='__main__':
    os.umask(0o077)
    {'prepare':prepare,'lock':lock,'link':link}[sys.argv[1]]()
