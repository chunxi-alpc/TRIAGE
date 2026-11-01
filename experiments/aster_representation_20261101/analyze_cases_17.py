import json
from pathlib import Path
from collections import Counter
from experiments.aster_representation_20261101.data import OUT, OLD, read, write


def rows(path):
    return {(r['case_id'], r['view']): r for r in read(path) if r['status']=='complete'}


def correct(row):
    s=row['score'];return bool(s['exact_path'] and s['leaf_correct'])


def path(row):
    return row['score']['path']


def first_wrong(predicted,gold):
    return next((i for i,r in enumerate(gold) if i>=len(predicted) or predicted[i]!=r),None)


cases={c['case_id']:c for c in read(OUT/'development.jsonl')}
selected=json.loads((OUT/'full_layer18/selected.json').read_text())
edited=rows(Path(selected['checkpoint']).parent/f'development_epoch{selected["epoch"]}.jsonl')
zero=rows(OUT/'development_zero.jsonl');blind=rows(OUT/'development_blind_lookup.jsonl');fusion=rows(OUT/'development_fusion.jsonl')
direct={}
for f in OLD.glob('routing_[0-3].jsonl'):
    for r in read(f):
        if r.get('status')=='complete' and r['case_id'] in cases:direct[r['case_id']]=r['result']['decisions']['direct']
selected_cases=[]
for cid,case in cases.items():
    key=cid,'original'
    if correct(edited[key])==correct(blind[key]):continue
    groupmap={g['group_id']:g for g in case['guideline']['groups']}
    details={d['group']:d for d in edited[key]['result']['diagnostics']}
    swapped_details={d['group']:d for d in edited[cid,'swap']['result']['diagnostics']}
    decisions=[]
    for depth,gold in enumerate(case['gold_rules']):
        gid='G'+gold[1:-1];group=groupmap[gid];probs=direct[cid][gid]['probs']
        yes,no='R'+gid[1:]+'Y','R'+gid[1:]+'N'
        yesprob,noprob,unknown=probs[yes],probs[no],probs['U']
        blind_choice=yes if yesprob>=noprob else no
        item={'depth':depth,'gid':gid,'gold':gold,'condition':group['rules'][0]['condition'],
              'ancestors':group.get('ancestor_conditions',[]),'rules':group['rules'],
              'blind_choice':blind_choice,'blind_correct':blind_choice==gold,
              'blind_probabilities':[yesprob,noprob,unknown],'blind_state':direct[cid][gid]['state'],
              'zero_on_same_prefix':path(zero[key])[:depth]==case['gold_rules'][:depth],
              'zero_choice_at_depth':path(zero[key])[depth] if depth<len(path(zero[key])) else None}
        for label,diags in [('original',details),('swap',swapped_details)]:
            if gid not in diags:continue
            d=diags[gid];clean=d['clean_logits'];after=d['edited_logits'];sign=1 if gold.endswith('Y') else -1
            cleanchoice=yes if clean[0]>clean[1] else no
            edchoice=yes if after[0]>after[1] else no
            item[label]={'clean_choice':cleanchoice,'edited_choice':edchoice,'clean_correct':cleanchoice==gold,
                         'edited_correct':edchoice==gold,'clean_margin':sign*(clean[0]-clean[1]),
                         'edited_margin':sign*(after[0]-after[1]),'margin_change':sign*(after[0]-after[1]-clean[0]+clean[1]),
                         'gate':d['gate'],'magnitude':d['magnitude'],'blind_state':d['blind_state']}
        decisions.append(item)
    row={'case_id':cid,'source_id':case['source_id'],'kind':'corrected' if correct(edited[key]) else 'regressed',
         'patient':case['patient'],'gold_rules':case['gold_rules'],'gold_leaf':case['gold_leaf'],
         'gold_conclusion':case.get('gold_conclusion'),'question':case['question'],
         'paths':{},'scores':{},'first_errors':{},'decisions':decisions,
         'full_original_diagnostics':edited[key]['result']['diagnostics']}
    for name,results in [('blind',blind),('zero',zero),('edited',edited),('fusion',fusion)]:
        for view in ['original','swap']:
            key=cid,view;label=name+'/'+view
            row['paths'][label]=path(results[key]);row['scores'][label]=correct(results[key])
            row['first_errors'][label]=first_wrong(path(results[key]),case['gold_rules'])
    selected_cases.append(row)
analysis=OUT/'case_analysis_17';analysis.mkdir(exist_ok=True)
write(analysis/'cases.json',selected_cases)
for i,row in enumerate(selected_cases,1):
    important=[d for d in row['decisions'] if not d['blind_correct'] or ('original' in d and (not d['original']['clean_correct'] or not d['original']['edited_correct']))]
    small={k:row[k] for k in ['case_id','source_id','kind','patient','gold_rules','gold_conclusion','paths','scores','first_errors']}
    small['important_decisions']=important
    write(analysis/f'{i:02}.json',small)
summary={'n':len(selected_cases),'kinds':dict(Counter(r['kind'] for r in selected_cases)),
         'sources':dict(Counter(r['source_id'] for r in selected_cases))}
write(analysis/'summary.json',summary)
print(json.dumps(summary))
for i,r in enumerate(selected_cases,1):
    nodes=[d for d in r['decisions'] if not d['blind_correct'] or ('original' in d and (not d['original']['clean_correct'] or not d['original']['edited_correct']))]
    print(i,r['case_id'],r['kind'],'zero',r['scores']['zero/original'],'swap',r['scores']['edited/swap'],
          [(d['gid'],d['gold'],d['blind_choice'],d['original']['clean_choice'] if 'original' in d else None,d['original']['edited_choice'] if 'original' in d else None) for d in nodes])
