"""Gold-aware offline evaluation after parser freeze; no model or GPU calls."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.module1.care_v5 import eval_logic
EXP = ROOT/'experiments/care_module1_v5'
OPS = {'eq':'==','lt':'<','le':'<=','gt':'>','ge':'>=','ne':'!='}


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def macro_f1(gold, pred):
    values=[]
    for label in ('T','F','U'):
        tp=sum(g==label and p==label for g,p in zip(gold,pred))
        fp=sum(g!=label and p==label for g,p in zip(gold,pred))
        fn=sum(g==label and p!=label for g,p in zip(gold,pred))
        values.append(2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.0)
    return sum(values)/3


def atom_truth(atom, observed):
    """Faithful pure-Python port of V4's frozen build_dataset.atom_truth."""
    if observed is None:
        return 'U'
    op=atom['op']; threshold=atom['value']
    def compare(value):
        return {'lt':lambda:value<threshold, 'le':lambda:value<=threshold,
            'gt':lambda:value>threshold, 'ge':lambda:value>=threshold,
            'eq':lambda:value==threshold, 'ne':lambda:value!=threshold}[op]()
    if not isinstance(observed,dict):
        return 'T' if compare(observed) else 'F'
    cuts=sorted({-100.0,float(threshold),*(float(value) for value in observed.values())})
    cuts.append(max(cuts)+100.0)
    points=set(cuts)|{(left+right)/2 for left,right in zip(cuts,cuts[1:])}
    def bound(value, bound_op, limit):
        return {'lt':value<limit,'le':value<=limit,'gt':value>limit,'ge':value>=limit}[bound_op]
    compatible=[value for value in points if all(bound(value,bop,float(limit)) for bop,limit in observed.items())]
    if not compatible:
        return 'U'
    outcomes={compare(value) for value in compatible}
    return 'T' if outcomes=={True} else 'F' if outcomes=={False} else 'U'


def metric(rows, pred_key):
    gold=[row['gold'] for row in rows]; pred=[row[pred_key] for row in rows]
    known=[i for i,value in enumerate(gold) if value in ('T','F')]
    counts=collections.Counter()
    for g,p in zip(gold,pred):
        if g==p: counts['correct']+=1
        elif p=='U' and g!='U': counts['known_to_unknown']+=1
        elif g=='U' and p!='U': counts['unknown_to_known']+=1
        elif g=='T' and p=='F': counts['false_rejection_T_to_F']+=1
        elif g=='F' and p=='T': counts['false_acceptance_F_to_T']+=1
        else: counts['known_polarity_error']+=1
    return {'n':len(rows),'correct':counts['correct'],'accuracy':counts['correct']/len(rows),
        'macro_f1':macro_f1(gold,pred),'gold_known_n':len(known),
        'gold_known_correct':sum(gold[i]==pred[i] for i in known),
        'gold_known_accuracy':sum(gold[i]==pred[i] for i in known)/len(known),
        'gold_known_to_unknown':sum(pred[i]=='U' for i in known),
        'gold_unknown_n':len(rows)-len(known),'gold_unknown_correct':sum(gold[i]=='U' and pred[i]=='U' for i in range(len(rows))),
        'confusion_order':['T','F','U'],'confusion':[[sum(g==a and p==b for g,p in zip(gold,pred)) for b in ('T','F','U')] for a in ('T','F','U')],
        'error_categories':dict(counts)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--conditions',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--rules',type=Path,default=EXP/'data/rules_v5.jsonl')
    p.add_argument('--labels',type=Path,default=EXP/'data/labels_public.jsonl')
    p.add_argument('--reference-facts',type=Path,required=True)
    p.add_argument('--reparse-freeze',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    freeze=json.loads(a.reparse_freeze.read_text())
    if freeze.get('gold_labels_read') is not False or freeze.get('model_calls')!=0:
        raise ValueError('Reparse must be frozen before gold-aware evaluation')
    source_paths=[Path(path) for path in freeze['source_sha256']]
    for path in source_paths:
        if not path.exists() or sha(path)!=freeze['source_sha256'][str(path)]:
            raise ValueError('A source frozen before parsing changed: '+str(path))
    for relative,digest in freeze['code_sha256'].items():
        if sha(ROOT/relative)!=digest:
            raise ValueError('Parser/evaluator code changed after reparse freeze: '+relative)
    rules={r['rule_id']:r for r in read(a.rules)}
    labels={r['sample_id']:r for r in read(a.labels)}
    facts={r['sample_id']:r for r in read(a.reference_facts)}
    conditions=read(a.conditions); predictions=read(a.predictions)
    if len(conditions)!=492 or len(predictions)!=192 or set(labels)!=set(facts):
        raise ValueError('Expected exact 492 conditions and matched 192 frozen reference observations')
    rows=[]; seen=set(); by_sample=collections.defaultdict(dict)
    atoms={}
    for rid,rule in rules.items():
        for atom in rule['conditions']+rule.get('exceptions',[]):
            atoms[(rid,atom['condition_id'])]=atom
    for item in conditions:
        key=(item['sample_id'],item['rule_id'],item['condition_id'])
        if key in seen: raise ValueError('Duplicate condition output')
        seen.add(key)
        atom=atoms[(item['rule_id'],item['condition_id'])]
        ref=facts[item['sample_id']]
        if ref['rule_id']!=item['rule_id']:
            raise ValueError('Reference rule does not match sample')
        value=ref.get('gold_observations',{}).get(atom['field'])
        if atom.get('numeric'):
            numeric=atom['numeric']; operator={v:k for k,v in OPS.items()}[numeric['operator']]
            gold=atom_truth({'op':operator,'value':numeric['value']},value)
        else:
            target='positive' if item['criterion_text'].strip().casefold().endswith(' is positive') else 'negative'
            gold='U' if value is None else 'T' if value==target else 'F'
        row={k:item[k] for k in ('sample_id','rule_id','condition_id','criterion_kind','parse_status','parse_error','unknown_reason','raw_label','old_state','state','truncated','raw_output_sha256')}
        row.update(gold=gold,old_correct=item['old_state']==gold,new_correct=item['state']==gold,
            changed=item['old_state']!=item['state'],corrected=item['old_state']!=gold and item['state']==gold,
            regressed=item['old_state']==gold and item['state']!=gold,
            parsed_confident_correct=item['state']==gold if item['parse_status']=='accepted' and item['state'] in ('T','F') else None)
        rows.append(row); by_sample[item['sample_id']][item['condition_id']]=gold
    # Independently verify the atom-level annotation composes to every frozen
    # sample label before reporting model performance.
    for sid,label in labels.items():
        rule=rules[label['rule_id']]
        if eval_logic(rule['logic'],by_sample[sid])!=label['label']:
            raise ValueError('Gold atom states do not compose to frozen sample label: '+sid)
    old_metric=metric(rows,'old_state'); new_metric=metric(rows,'state')
    accepted=[r for r in rows if r['parse_status']=='accepted']
    accepted_known=[r for r in accepted if r['gold'] in ('T','F')]
    confident=[r for r in accepted if r['state'] in ('T','F')]
    statuses=collections.defaultdict(list)
    variants=collections.defaultdict(list)
    for row in rows:
        statuses[row['parse_status']].append(row)
        variants[labels[row['sample_id']].get('variant','unknown')].append(row)
    status_metrics={name:metric(values,'state') for name,values in sorted(statuses.items())}
    variant_metrics={name:metric(values,'state') for name,values in sorted(variants.items())}
    gold_hash=sha(a.reference_facts)
    result={'protocol':'post-freeze offline reparse; no model calls; gold observations used only in this evaluator',
        'parser_version':freeze['parser_version'],'created_utc':datetime.now(timezone.utc).isoformat(),
        'model_calls':0,'gpu_used':False,'gold_reference_sha256':gold_hash,
        'condition_count':len(rows),'labels_composed_from_gold_atoms_verified':len(labels),
        'parser_status_counts':dict(collections.Counter(r['parse_status'] for r in rows)),
        'unknown_reason_counts':dict(collections.Counter(r['unknown_reason'] for r in rows if r['unknown_reason'])),
        'target_label_counts':dict(collections.Counter(r['raw_label'] for r in accepted)),
        'old_parser':old_metric,'new_parser':new_metric,
        'delta':{'additional_correct_conditions':sum(r['corrected'] for r in rows)-sum(r['regressed'] for r in rows),
            'newly_correct':sum(r['corrected'] for r in rows),'newly_incorrect':sum(r['regressed'] for r in rows),
            'changed_condition_states':sum(r['changed'] for r in rows)},
        'accepted_condition_metrics':metric(accepted,'state'),
        'accepted_gold_known_metrics':metric(accepted_known,'state') if accepted_known else None,
        'accepted_confident_label_metrics':metric(confident,'state') if confident else None,
        'by_parse_status':status_metrics,'by_scenario':variant_metrics,
        'condition_errors':{k:sum(not r['new_correct'] and r['gold']==k for r in rows) for k in ('T','F','U')},
        'input_hashes':{'conditions_public':sha(a.conditions),'predictions_public':sha(a.predictions),
            'rules':sha(a.rules),'labels_public':sha(a.labels),'reference_facts':gold_hash},
        'limitations':['192-case V4 data was used for method design; not independent validation.',
            'Gold observations are source-read synthetic profiles and controlled edits; no clinician review.']}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'condition_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (a.output/'condition_predictions_public.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (a.output/'by_parse_status.json').write_text(json.dumps(status_metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'old':old_metric,'new':new_metric,'delta':result['delta'],
        'parser_status_counts':result['parser_status_counts'],'unknown_reason_counts':result['unknown_reason_counts'],
        'accepted_confident_label_metrics':result['accepted_confident_label_metrics']},ensure_ascii=False))

if __name__=='__main__':main()
