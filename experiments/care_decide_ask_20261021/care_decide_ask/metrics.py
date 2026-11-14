"""Straightforward, honest, denominator-preserving decide/ask evaluation."""
from __future__ import annotations
from collections import defaultdict
from random import Random


def evaluate(rows,gold,*,bootstrap_repetitions=2000,seed=261021):
    """Oracle gold is in a separate file. Do not filter difficult or REVIEW rows."""
    if not isinstance(gold,dict) or not gold:raise ValueError('independent gold required')
    found={};summary=[]
    for r in rows:
        cid=r.get('case_id')
        if cid in found or cid not in gold:raise ValueError('duplicate or unknown case id')
        found[cid]=True;label=gold[cid]
        required=label.get('requires_ask')
        action=label.get('gold_action')
        if not isinstance(required,bool) or not isinstance(label.get('family_id'),str):
            raise ValueError('gold must contain independent requires_ask and family_id')
        if required and action is not None:raise ValueError('Q label cannot include a unique gold_action')
        if not required and (not isinstance(action,str) or not action):raise ValueError('D requires gold_action')
        status=r.get('status')
        if status not in ('DECIDE','ASK','REVIEW'):raise ValueError('invalid execution status')
        wrong=int(status=='DECIDE' and (required or r.get('action')!=action))
        overask=int(status=='ASK' and not required)
        hit=int(status=='ASK' and required and r.get('ask_attribute') in label.get('acceptable_questions',[]))
        summary.append({'case_id':cid,'family_id':label['family_id'],'requires_ask':required,
                        'status':status,'wrong_decision':wrong,'unnecessary_ask':overask,
                        'question_hit':hit,'decided':int(status=='DECIDE'),
                        'asked':int(status=='ASK'),'reviewed':int(status=='REVIEW'),
                        'correct_task_behavior':int((not required and status=='DECIDE' and r.get('action')==action)
                                      or (required and status=='ASK' and bool(hit)))})
    if set(found)!=set(gold):raise ValueError('missing evaluated records; refuse denominator shrink')
    n=len(summary)
    p=lambda rows,key:sum(r[key] for r in rows)/len(rows) if rows else None
    result={'n_cases':n,'n_families':len({r['family_id'] for r in summary}),
            'wrong_decision_count':sum(r['wrong_decision'] for r in summary),
            'wrong_decision_rate_all_cases':p(summary,'wrong_decision'),
            'wrong_decision_rate_conditional_on_deciding':p([r for r in summary if r['decided']],'wrong_decision'),
            'unnecessary_ask_rate_among_decidable':p([r for r in summary if not r['requires_ask']],'unnecessary_ask'),
            'question_hit_rate_among_need_to_ask':p([r for r in summary if r['requires_ask']],'question_hit'),
            'ask_rate':p(summary,'asked'),'review_rate':p(summary,'reviewed'),
            'task_behavior_accuracy':p(summary,'correct_task_behavior'),
            'scored_records':summary}
    family=defaultdict(list)
    for r in summary:family[r['family_id']].append(r)
    fid=sorted(family)
    if bootstrap_repetitions>0:
        rng=Random(seed);samples=[]
        for _ in range(bootstrap_repetitions):
            groups=[family[rng.choice(fid)] for _ in fid]
            rows_boot=[x for group in groups for x in group]
            samples.append(p(rows_boot,'wrong_decision'))
        samples.sort()
        result['family_bootstrap95_wrong_decision_rate']=[samples[int(.025*(len(samples)-1))],samples[int(.975*(len(samples)-1))]]
        result['family_bootstrap_repetitions']=bootstrap_repetitions
        result['family_bootstrap_seed']=seed
    return result
