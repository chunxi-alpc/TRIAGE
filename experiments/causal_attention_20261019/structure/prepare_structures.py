#!/usr/bin/env python3
"""Experimental atomic-label repair, isolated from frozen V7. CPU cache replay.
Prediction code receives only patient/rules/question and never reference labels.
"""
import argparse, collections, copy, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO/'experiments/care_reliability_20261019')]
from joint.context import prediction_input
from joint.run_structures import collect_requests, request_key
from src.module1.care_v6 import eval_logic, tri_and, tri_not
from src.module1.trialmatch_parser import strip_thinking

LABELS = {'inclusion': {'Met':'T','Not Met':'F','Unclear':'U','Irrelevant':'U'},
          'exclusion': {'Violated':'T','Not Violated':'F','Unclear':'U','Irrelevant':'U'}}
ARRAYS = {'inclusion':'Inclusion_Criteria_Evaluation','exclusion':'Exclusion_Criteria_Evaluation'}

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def write_jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def recover_output(request, output):
    """Accept a known single-atom request's completed final JSON, even if a
    subsequent repeated response was truncated. Never parse a thinking label.
    The request manifest supplies condition_id; text copying isn't the key.
    """
    kind=request['criterion_kind'];allowed=LABELS[kind]
    result={'request_key':request['request_key'],'condition_ids':sorted({u['condition_id'] for u in request['uses']}),
            'criterion_kind':kind,'old_parse_status':output.get('parse_status'),
            'old_state':output.get('state','U'),'old_label':output.get('raw_label'),
            'old_truncated':output.get('truncated',False)}
    if output.get('parse_status')=='accepted' and output.get('raw_label') in allowed:
        label=output['raw_label']; result.update(state=allowed[label],raw_label=label,
            method='accepted_cached_label_atomic_mapping',needs_generation=False)
        return result
    clean=strip_thinking(output.get('raw_output',''))
    decoder=json.JSONDecoder(); matches=[]
    for hit in re.finditer(r'\{',clean):
        try: obj,_=decoder.raw_decode(clean,hit.start())
        except (ValueError,json.JSONDecodeError): continue
        if not isinstance(obj,dict):continue
        target=obj.get(ARRAYS[kind])
        if not isinstance(target,list):continue
        entries=[e for e in target if isinstance(e,dict) and e.get('Classification') in allowed]
        exact=[e for e in entries if re.sub(r'\s+',' ',str(e.get('Criterion',''))).strip()==re.sub(r'\s+',' ',request['condition']['text']).strip()]
        chosen=exact if len(exact)==1 else entries if len(entries)==1 else []
        if len(chosen)==1:matches.append(chosen[0])
    labels={e['Classification'] for e in matches}
    if len(labels)==1:
        label=next(iter(labels));result.update(state=allowed[label],raw_label=label,
            method='recovered_completed_single_atom_json',needs_generation=False,
            completed_objects=len(matches),matched_criterion=matches[-1].get('Criterion'),
            justification=matches[-1].get('Justification'))
    else:result.update(state='U',raw_label=None,method='missing_or_ambiguous_final_json',
        needs_generation=True,completed_objects=len(matches))
    return result

def load_old(root):
    requests={};cache={};structures={};sources=[]
    for directory in sorted(Path(root).glob('shard*')):
        sources.append(str(directory))
        for r in json.load(open(directory/'request_manifest.json')):
            if r['request_key'] in requests:
                requests[r['request_key']]['uses'].extend(r['uses'])
            else:requests[r['request_key']]=copy.deepcopy(r)
        for line in open(directory/'trialmatch_generations.jsonl'):
            r=json.loads(line);cache.setdefault(r['request_key'],r['output'])
        for f in (directory/'structures/v7').glob('*.json'):
            v=json.load(open(f));structures[v['case_id']]=v
    return requests,cache,structures,sources

def fixed_structure(inputs, old, recovered):
    # Only prediction_input's allowlisted keys enter this function.
    rawrules={r['rule_id']:r for r in old['raw']['rules']}
    outputs=[];atom_records=[]
    for rule in inputs['guideline']:
        oldrule=rawrules[rule['rule_id']]
        oldatoms={r['condition_id']:r for r in oldrule['conditions']+oldrule['exceptions']}
        states={};rows=[]
        for role,kind in [('conditions','inclusion'),('exceptions','exclusion')]:
            for atom in rule[role]:
                key=request_key(inputs['patient'],atom,atom.get('criterion_kind',kind));r=copy.deepcopy(recovered[key]);num=oldatoms[atom['condition_id']].get('numeric')
                local_request={'request_key':key,'condition':{'text':atom['text']},'criterion_kind':atom.get('criterion_kind',kind),'uses':[{'condition_id':atom['condition_id']}]}
                local_recovered=recover_output(local_request,oldatoms[atom['condition_id']]['trialmatch'])
                if not local_recovered['needs_generation']:r=local_recovered
                elif not r['needs_generation']:r['reused_exact_input_cache']=True
                state=r['state'];method=r['method']
                if num and num.get('status') in ('T','F'):
                    state=num['status'];method='retained_v7_deterministic_numeric'
                states[atom['condition_id']]=state
                item=dict(r,rule_id=rule['rule_id'],condition_id=atom['condition_id'],role=role,
                    atom_text=atom['text'],state=state,applicability_source=method)
                rows.append(item);atom_records.append(item)
        state=tri_and([eval_logic(rule['logic'],states),tri_not(eval_logic(rule.get('exception_logic',{'op':'OR','args':[]}),states))])
        outputs.append({'rule_id':rule['rule_id'],'state':state,'action':rule['action'],
            'support_evidence':oldrule.get('support_evidence',[]),'refute_evidence':oldrule.get('refute_evidence',[]),
            'allow_attention_intervention':state in ('T','F'),'atoms':rows})
    return {'source':'fixed','status':'ok','rules':outputs,
        'parse_failure':any(r['needs_generation'] for r in atom_records),
        'gate_policy':'hard_T_plus_F_minus_U_unchanged_no_evidence_gate'},atom_records

def deterministic_answer(case,structure):
    actions={r['rule_id']:r['action'] for r in case['rules']}
    applicable=[actions[r['rule_id']] for r in structure['rules'] if r['state']=='T']
    action=applicable[0] if len(applicable)==1 else 'Request review'
    return next((key for key,value in case['option_map'].items() if value==action),None)

def summarize(cases,structures):
    byid={r['case_id']:r['structure'] for r in structures};rules_correct=0;rules_total=0;preds=[];states=collections.Counter();confusion=collections.Counter()
    for case in cases:
        st=byid[case['case_id']];gold={r['rule_id']:r['state'] for r in case['reference_structure']['rules']}
        for r in st['rules']:
            rules_correct+=r['state']==gold[r['rule_id']];rules_total+=1;states[r['state']]+=1;confusion[gold[r['rule_id']]+'->'+r['state']]+=1
        pred=deterministic_answer(case,st)
        preds.append({'case_id':case['case_id'],'source':st['source'],'prediction':pred,'gold_answer':case['gold_answer'],
            'correct':pred==case['gold_answer'],'pair_id':case['pair_id'],'pair_eligible':case['pair_eligible'],'variant':case['variant'],
            'is_model_prediction':False,'method':'deterministic_rule_execution_on_given_structure'})
    pairs=collections.defaultdict(list)
    for r in preds:
        if r['pair_eligible']:pairs[r['pair_id']].append(r)
    canonical=[rs for rs in pairs.values() if rs[0]['variant']=='canonical']
    metrics={'cases':len(cases),'rule_correct':rules_correct,'rules':rules_total,'rule_accuracy':rules_correct/rules_total,
        'states':dict(states),'confusion_gold_to_prediction':dict(confusion),'execution_correct':sum(p['correct'] for p in preds),
        'execution_accuracy':sum(p['correct'] for p in preds)/len(preds),'pairs':len(pairs),
        'pair_correct':sum(all(r['correct'] for r in rs) for rs in pairs.values()),
        'primary_pairs':len(canonical),'primary_pair_correct':sum(all(r['correct'] for r in rs) for rs in canonical),
        'parse_failure_cases':sum(r['structure'].get('parse_failure',False) for r in structures)}
    return metrics,preds

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old-root',default='/home/gcx/care_reliability_runs/joint_formal_work_02/structures');ap.add_argument('--output',type=Path,default=HERE/'results');ap.add_argument('--compact-cache',type=Path,default=HERE/'compact_generations.jsonl');args=ap.parse_args()
    cases=[json.loads(l) for l in open(REPO/'experiments/care_reliability_20261019/data/mechanism_cases.jsonl')];cases=[c for c in cases if c['split']=='confirmation']
    requests,cache,old,sources=load_old(args.old_root)
    recovered={k:recover_output(r,cache[k]) for k,r in requests.items()}
    if args.compact_cache.exists():
        for line in open(args.compact_cache):
            r=json.loads(line)
            if r.get('state') in ('T','F','U') and r.get('parse_status')=='accepted':
                recovered[r['request_key']].update(state=r['state'],method='compact_atomic_model_output',needs_generation=False,compact_generation=r)
    missing=[r for k,r in requests.items() if recovered[k]['needs_generation']]
    if not (HERE/'missing_requests.jsonl').exists():write_jsonl(HERE/'missing_requests.jsonl',missing)
    write_jsonl(args.output/'unresolved_requests.jsonl',missing)
    write_jsonl(args.output/'request_recovery.jsonl',list(recovered.values()))
    outputs={k:[] for k in ['gold','original_v7_ungated','fixed']};atomic=[]
    for case in cases:
        gold=copy.deepcopy(case['reference_structure']);gold['source']='gold'
        original=copy.deepcopy(old[case['case_id']]['normalized']);original['source']='original_v7_ungated'
        for r in original['rules']:r['allow_attention_intervention']=r['state'] in ('T','F');r.pop('confidence',None);r.pop('gate_weight',None)
        original['gate_policy']='hard_T_plus_F_minus_U_unchanged_no_evidence_gate'
        fixed,atoms=fixed_structure(prediction_input(case),old[case['case_id']],recovered)
        for atom in atoms:atomic.append(dict(atom,case_id=case['case_id']))
        for name,st in [('gold',gold),('original_v7_ungated',original),('fixed',fixed)]:outputs[name].append({'case_id':case['case_id'],'structure':st})
    summary={};executions=[]
    for source,rows in outputs.items():
        write_jsonl(args.output/(source+'.jsonl'),rows);summary[source],pred=summarize(cases,rows);executions+=pred
    write_jsonl(args.output/'atomic_mapping.jsonl',atomic);write_jsonl(args.output/'deterministic_predictions.jsonl',executions)
    summary['cache']={'unique_requests':len(requests),'old_statuses':dict(collections.Counter(o.get('parse_status') for o in cache.values())),
        'recovery_methods':dict(collections.Counter(r['method'] for r in recovered.values())),
        'unresolved_requests':len(missing),'old_cache_paths':sources,
        'not_violated_mapped_to_atomic_false':sum(r['old_label']=='Not Violated' and r['criterion_kind']=='exclusion' for r in recovered.values())}
    save(args.output/'metrics.json',summary)
    save(args.output/'mapping_contract.json',{'experiment':'fixed_atomic_interface_v1','old_V7_unchanged':True,'reference_used_only_for_evaluation':True,
        'atomic_label_maps':LABELS,'meaning':'Each condition text is a proposition. Exclusion Violated means proposition true; Not Violated means proposition false. Rule-level NOT and UNLESS are applied exactly once by the logic tree.',
        'prediction_inputs':['patient','guideline','question'],'cache_recovery':'Use completed final JSON for the one requested atom. Complete repeated outputs must agree. condition_id comes from the exact cached request manifest; no exact copied-text requirement.',
        'compact_output':'{"condition_id":"C1","state":"T|F|U"}',
        'unknown_policy':'Unclear/Irrelevant/missing/conflicting model final labels remain U unless supplemented by actual compact generation. Existing V7 numeric precedence retained.',
        'dev_scope':'96 dev remains the attention development set; this automatic structure comparison uses all 192 confirmation cases with existing V7 outputs.',
        'interpretation':'Exploratory reused-data experiment, not independent clinical validation.'})
    print(json.dumps(summary))
if __name__=='__main__':main()
