#!/usr/bin/env python3
"""Full-denominator accuracy, counterfactual adherence and paired corrections."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

CONDITIONS=('baseline','direct_rule','applicable_only','unsigned','signed')
VARIANTS=('A_original','B_applicable_rule_changed','C_sibling_rule_changed')

def summarize(rows,expected_ids=None):
    keyed={}
    for r in rows:
        key=(r['id'],r['variant'],r['condition'])
        if key in keyed:raise ValueError('duplicate prediction key')
        keyed[key]=r
    ids=sorted(set(expected_ids) if expected_ids is not None else {r['id'] for r in rows})
    def correct(r):return bool(r and r.get('prediction') is not None and r['prediction']==r['gold_answer'])
    arms={};paired={}
    for variant in VARIANTS:
        for condition in (('closed_book',)+CONDITIONS if variant==VARIANTS[0] else CONDITIONS):
            rs=[keyed.get((i,variant,condition)) for i in ids]
            present=[r for r in rs if r];n=len(ids)
            valid=sum(r.get('prediction') is not None for r in present)
            count=sum(correct(r) for r in rs)
            stubborn=[r for r in present if r['gold_answer']!=r['original_answer']]
            memory=[r for r in stubborn if r.get('closed_book_prediction')==r['original_answer']]
            misuse=sum(r.get('prediction') is not None and r['prediction']==r.get('sibling_action_label')
                       and r['sibling_action_label']!=r['gold_answer'] for r in present)
            arms[f'{variant}/{condition}']={'n_expected':n,'n_present':len(present),'missing':n-len(present),
                'correct':count,'accuracy':count/n if n else None,'invalid_or_missing':n-valid,
                'stubborn_errors':sum(r.get('prediction')==r['original_answer'] for r in stubborn),
                'stubborn_eligible':len(stubborn),'closed_book_original_eligible':len(memory),
                'stubborn_given_closed_book_original':sum(r.get('prediction')==r['original_answer'] for r in memory)/len(memory) if memory else None,
                'sibling_branch_misuse':misuse,'sibling_branch_misuse_rate':misuse/n if n else None}
            if condition not in ('baseline','closed_book'):
                pairs=[(keyed.get((i,variant,'baseline')),keyed.get((i,variant,condition))) for i in ids]
                paired[f'{variant}/{condition}']={'n_expected':n,'n_paired':sum(bool(b and r) for b,r in pairs),
                  'corrected':sum(not correct(b) and correct(r) for b,r in pairs if b and r),
                  'broken':sum(correct(b) and not correct(r) for b,r in pairs if b and r)}
    consistency={}
    for cond in CONDITIONS:
        triples=[tuple(keyed.get((i,v,cond)) for v in VARIANTS) for i in ids]
        full=[t for t in triples if all(t)]
        consistency[cond]={'n_expected':len(ids),'n_complete':len(full),
            'all_three_correct':sum(all(correct(r) for r in t) for t in full),
            'A_C_same_prediction':sum(t[0].get('prediction') is not None and t[0]['prediction']==t[2]['prediction'] for t in full),
            'B_follows_edited_rule':sum(correct(t[1]) for t in full)}
    return {'n_examples':len(ids),'n_predictions':len(rows),'expected_predictions':len(ids)*16,
            'complete':len(rows)==len(ids)*16 and all(v['missing']==0 for v in arms.values()),
            'arms':arms,'paired_changes':paired,'counterfactual_consistency':consistency}

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dataset',required=True,help='prepared dataset supplies expected IDs, including missing runs')
    a=p.parse_args()
    rows=[json.loads(s) for s in Path(a.input).read_text().splitlines() if s.strip()]
    expected={json.loads(s)['id'] for s in Path(a.dataset).read_text().splitlines() if s.strip()}
    result=summarize(rows,expected)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print('complete:',result['complete'],'predictions:',len(rows))

if __name__=='__main__':main()
