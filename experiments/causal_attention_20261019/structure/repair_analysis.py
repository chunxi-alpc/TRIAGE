#!/usr/bin/env python3
"""Decompose repair effects and attribute remaining errors after prediction."""
import collections,copy,json
from pathlib import Path
from prepare_structures import *

def main():
    out=HERE/'results';cases=[json.loads(l) for l in open(REPO/'experiments/care_reliability_20261019/data/mechanism_cases.jsonl')];cases=[c for c in cases if c['split']=='confirmation']
    requests,cache,old,_=load_old('/home/gcx/care_reliability_runs/joint_formal_work_02/structures')
    recovered={k:recover_output(r,cache[k]) for k,r in requests.items()}
    final={r['case_id']:r['structure'] for r in map(json.loads,open(out/'fixed.jsonl'))}
    stages={stage:[] for stage in ['original_v7_ungated','mapping_only','reparse_only','mapping_and_reparse','fixed']}
    changes=[]
    for case in cases:
        inputs=prediction_input(case);raw={r['rule_id']:r for r in old[case['case_id']]['raw']['rules']}
        for stage in stages:
            if stage=='original_v7_ungated':st=copy.deepcopy(old[case['case_id']]['normalized'])
            elif stage=='fixed':st=copy.deepcopy(final[case['case_id']])
            else:
                rules=[]
                for rule in inputs['guideline']:
                    atoms={a['condition_id']:a for a in raw[rule['rule_id']]['conditions']+raw[rule['rule_id']]['exceptions']};states={}
                    for role,kind in [('conditions','inclusion'),('exceptions','exclusion')]:
                        for atom in rule[role]:
                            a=atoms[atom['condition_id']];num=a.get('numeric');tm=a['trialmatch'];key=request_key(inputs['patient'],atom,kind)
                            req={'request_key':key,'criterion_kind':kind,'condition':{'text':atom['text']},'uses':[{'condition_id':atom['condition_id']}]}
                            if stage=='mapping_only':
                                state=LABELS[kind].get(tm.get('raw_label'),tm.get('state','U')) if tm.get('parse_status')=='accepted' else tm.get('state','U')
                            else:
                                r=recover_output(req,tm)
                                if stage=='mapping_and_reparse' and r['needs_generation'] and not recovered[key]['needs_generation']:r=recovered[key]
                                state=r['state']
                                if stage=='reparse_only' and kind=='exclusion' and r['raw_label']=='Not Violated':state='U'
                            if num and num.get('status') in ('T','F'):state=num['status']
                            states[atom['condition_id']]=state
                    state=tri_and([eval_logic(rule['logic'],states),tri_not(eval_logic(rule.get('exception_logic',{'op':'OR','args':[]}),states))])
                    rules.append({'rule_id':rule['rule_id'],'state':state})
                st={'rules':rules,'status':'ok'}
            st['source']=stage;stages[stage].append({'case_id':case['case_id'],'structure':st})
    metrics={};predictions={}
    for stage,rows in stages.items():metrics[stage],predictions[stage]=summarize(cases,rows)
    for left,right in [('original_v7_ungated','mapping_only'),('original_v7_ungated','reparse_only'),('mapping_only','mapping_and_reparse'),('mapping_and_reparse','fixed')]:
        before={r['case_id']:r['structure'] for r in stages[left]};after={r['case_id']:r['structure'] for r in stages[right]};change=corrected=damaged=0;bychange=collections.Counter()
        for case in cases:
            b={r['rule_id']:r['state'] for r in before[case['case_id']]['rules']};a={r['rule_id']:r['state'] for r in after[case['case_id']]['rules']};g={r['rule_id']:r['state'] for r in case['reference_structure']['rules']}
            for rid in b:
                if a[rid]!=b[rid]:
                    change+=1;bychange[b[rid]+'->'+a[rid]]+=1;corrected+=a[rid]==g[rid];damaged+=b[rid]==g[rid]
        changes.append({'from':left,'to':right,'changed_rules':change,'corrected_rules':corrected,'damaged_rules':damaged,'state_changes':dict(bychange)})
    errors=[]
    for case in cases:
        g={r['rule_id']:r['state'] for r in case['reference_structure']['rules']};ga=case['reference_structure']['atoms']
        for rule in final[case['case_id']]['rules']:
            if rule['state']==g[rule['rule_id']]:continue
            wrong=[{'condition_id':a['condition_id'],'predicted':a['state'],'reference':ga[a['condition_id']]['state'],'method':a['applicability_source'],'label':a['raw_label'],'kind':a['criterion_kind']} for a in rule['atoms'] if a['state']!=ga[a['condition_id']]['state']]
            errors.append({'case_id':case['case_id'],'variant':case['variant'],'rule_id':rule['rule_id'],'predicted':rule['state'],'reference':g[rule['rule_id']],'wrong_atoms':wrong,'patient':case['patient']})
    checks={'not_violated_exclusion_is_false':LABELS['exclusion']['Not Violated']=='F',
        'not_met_inclusion_is_false':LABELS['inclusion']['Not Met']=='F',
        'negation_applied_once':eval_logic({'op':'NOT','args':['E2']},{'E2':'F'})=='T',
        'unknown_kept_unknown':LABELS['exclusion']['Unclear']=='U',
        'all_sources_have_all_192':all(len(rs)==192 for rs in stages.values())}
    save(out/'repair_decomposition.json',{'stages':metrics,'changes':changes,'checks':checks})
    write_jsonl(out/'remaining_errors.jsonl',errors)
    sample=[]
    for key,r in recovered.items():
        if r['method']=='recovered_completed_single_atom_json' and len(sample)<3:
            sample.append({'request_key':key,'request':requests[key], 'recovery':r,'raw_model_output':cache[key]['raw_output']})
    save(out/'recovery_examples.json',sample)
    print(json.dumps({'stage_metrics':{k:{m:v[m] for m in ['rule_correct','execution_correct','primary_pair_correct']} for k,v in metrics.items()},'changes':changes,'remaining_errors':len(errors),'error_atom_methods':dict(collections.Counter(a['method'] for r in errors for a in r['wrong_atoms']))}))
if __name__=='__main__':main()
