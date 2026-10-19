#!/usr/bin/env python3
"""Evaluate completed predictions only; never imported by prediction code."""
import argparse, collections, csv, datetime, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATES = ('T', 'F', 'U')

def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

def rules(row):
    return row.get('rules', row.get('structure', {}).get('rules', []))

def evaluate(cases, predictions):
    byid = {r['case_id']: r for r in predictions}
    confusion = {g: {p: 0 for p in (*STATES, 'ERROR')} for g in STATES}
    records, pairs = [], collections.defaultdict(list)
    valid_cases = calls = nonempty = located = evidence_all = full_correct = 0
    elapsed = []
    for c in cases:
        row = byid.get(c['case_id'], {})
        pr = {r['rule_id']: r for r in rules(row)}
        gr = {r['rule_id']: r['state'] for r in c['reference_structure']['rules']}
        case_ok = valid = True
        for rid, gold in gr.items():
            r = pr.get(rid, {})
            p = r.get('state') if r.get('state') in STATES else 'ERROR'
            confusion[gold][p] += 1
            case_ok &= p == gold
            valid &= p != 'ERROR'
            quote = r.get('evidence_quote') or ''
            loc = bool(quote and quote in c['patient'])
            if 'evidence_quote' in r and r.get('evidence_available',True):
                evidence_all += 1
                nonempty += bool(quote)
                located += loc
            records.append(dict(case_id=c['case_id'], family_id=c['family_id'], rule_id=rid,
                gold=gold, predicted=p, correct=p==gold, evidence_quote=quote,
                evidence_located=loc, reason_short=r.get('reason_short'),
                status=row.get('status', row.get('structure',{}).get('status', 'missing'))))
        valid_cases += valid
        full_correct += case_ok
        calls += row.get('model_call_count', 0)
        if row.get('inference_seconds') is not None:
            elapsed.append(row['inference_seconds'])
        if c.get('pair_eligible') and c.get('pair_id'):
            pairs[c['pair_id']].append((c,pr))
    pair_n = pair_correct = flip_rules_n = flip_rules_correct = pair_all_n = pair_all_correct = 0
    canonical_n = canonical_correct = 0
    for pid, xs in pairs.items():
        if len(xs) != 2: continue
        (a, ap), (b, bp) = xs
        ag = {r['rule_id']:r['state'] for r in a['reference_structure']['rules']}
        bg = {r['rule_id']:r['state'] for r in b['reference_structure']['rules']}
        rid = a.get('target_rule_id','R_TARGET')
        if {ag.get(rid), bg.get(rid)} == {'T','F'}:
            ok = ap.get(rid,{}).get('state') == ag[rid] and bp.get(rid,{}).get('state') == bg[rid]
            pair_n += 1; pair_correct += ok
            if a.get('variant') == b.get('variant') == 'canonical':
                canonical_n += 1; canonical_correct += ok
        pair_all_n += 1
        pair_all_correct += all(ap.get(r,{}).get('state')==ag[r] and bp.get(r,{}).get('state')==bg[r] for r in ag)
        for r in ag:
            if {ag[r],bg[r]} == {'T','F'}:
                flip_rules_n += 1
                flip_rules_correct += ap.get(r,{}).get('state')==ag[r] and bp.get(r,{}).get('state')==bg[r]
    total = len(records); correct = sum(confusion[s][s] for s in STATES)
    per_state = {}
    for s in STATES:
        tp = confusion[s][s]; support = sum(confusion[s].values())
        predicted = sum(confusion[g][s] for g in STATES)
        precision = tp/predicted if predicted else 0
        recall = tp/support if support else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        per_state[s] = dict(precision=precision,recall=recall,f1=f1,support=support,predicted=predicted)
    m = dict(cases=len(cases),rules=total,correct=correct,accuracy=correct/total if total else 0,
        macro_f1=sum(v['f1'] for v in per_state.values())/3,confusion=confusion,per_state=per_state,
        F_to_T=confusion['F']['T'],F_to_T_denominator=sum(confusion['F'].values()),
        valid_rules=total-sum(confusion[g]['ERROR'] for g in STATES),valid_cases=valid_cases,
        all_rules_correct_cases=full_correct,model_calls=calls,
        target_pair_correct=pair_correct,target_pair_total=pair_n,
        canonical_target_pair_correct=canonical_correct,canonical_target_pair_total=canonical_n,
        flipped_rule_pair_correct=flip_rules_correct,flipped_rule_pair_total=flip_rules_n,
        all_rules_pair_correct=pair_all_correct,all_rules_pair_total=pair_all_n,
        evidence_rules_with_field=evidence_all,evidence_nonempty=nonempty,evidence_exact_located=located,
        evidence_exact_rate_of_nonempty=located/nonempty if nonempty else None,
        summed_case_inference_seconds=sum(elapsed),mean_case_inference_seconds=sum(elapsed)/len(elapsed) if elapsed else None)
    return m, records

def main():
    p=argparse.ArgumentParser();p.add_argument('--select',action='store_true');p.add_argument('--new-cases',type=Path);p.add_argument('--new-gold',type=Path)
    args=p.parse_args()
    history=read(REPO/'experiments/care_reliability_20261019/data/mechanism_cases.jsonl')
    datasets={s:[c for c in history if c['split']==s] for s in ['dev','confirmation']}
    if args.new_cases:
        new_cases=read(args.new_cases)
        gold_path=args.new_gold or args.new_cases.with_name('new72_gold.jsonl')
        if gold_path.exists():
            byid={r['case_id']:r for r in read(gold_path)}
            new_cases=[dict(c,**{k:v for k,v in byid[c['case_id']].items() if k!='case_id'}) for c in new_cases]
        datasets['new72']=new_cases
    summaries={};details={};family=[]
    for split,cases in datasets.items():
        sources={arm:HERE/'results'/f'{split}_{arm}'/'predictions.jsonl' for arm in ['A0','A1']}
        if split=='confirmation':sources['A2']=HERE/'data/a2_confirmation_fixed.jsonl'
        if split=='new72':sources['A2']=HERE/'results/new72_A2/predictions.jsonl'
        for arm,path in sources.items():
            if not path.exists():continue
            rows=read(path)
            # Running results are not counted as completed; failures within a completed file remain.
            if len({r['case_id'] for r in rows}) != len(cases):continue
            key=f'{split}_{arm}';m,rs=evaluate(cases,rows)
            m.update(split=split,arm=arm,predictions_file=str(path.relative_to(HERE)))
            manifest=path.parent/'manifest.json'
            if manifest.exists():
                m['run_manifest']=json.loads(manifest.read_text())
                if arm=='A2' and 'actual_atomic_calls' in m['run_manifest']:
                    m['sum_case_referenced_atomic_calls']=m['model_calls']
                    m['model_calls']=m['run_manifest']['actual_atomic_calls']
            summaries[key]=m;details[key]=rs
            for fid in sorted(set(c['family_id'] for c in cases)):
                fm,_=evaluate([c for c in cases if c['family_id']==fid],rows)
                family.append(dict(split=split,arm=arm,family_id=fid,**fm))
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    write(out/'metrics.json',summaries);write(out/'family_metrics.json',family)
    with (out/'rule_predictions.jsonl').open('w') as f:
        for key,rs in details.items():
            for r in rs:f.write(json.dumps(dict(experiment=key,**r),ensure_ascii=False)+'\n')
    comparisons={}
    for split in datasets:
        for arm in ['A0','A1']:
            left=f'{split}_{arm}';right=f'{split}_A2'
            if left not in details or right not in details:continue
            l={(r['case_id'],r['rule_id']):r for r in details[left]}
            r={(r['case_id'],r['rule_id']):r for r in details[right]}
            changes=[];counts=collections.Counter()
            for k,a in l.items():
                b=r[k];counts[('both_correct' if a['correct'] and b['correct'] else 'semantic_only_correct' if a['correct'] else 'A2_only_correct' if b['correct'] else 'both_wrong')]+=1
                if a['predicted']!=b['predicted']:changes.append(dict(**a,A2_predicted=b['predicted'],A2_correct=b['correct']))
            comparisons[left+'_vs_A2']=dict(counts=counts,changes=changes)
    write(out/'paired_comparison.json',comparisons)
    fields=['split','arm','cases','rules','correct','accuracy','macro_f1','F_to_T','F_to_T_denominator','valid_rules','valid_cases','target_pair_correct','target_pair_total','evidence_exact_located','evidence_nonempty','model_calls']
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(summaries.values())
    if args.select:
        assert all('dev_'+a in summaries for a in ['A0','A1']), 'Both full dev outputs required'
        ranked=sorted(['A0','A1'],key=lambda a:(-summaries['dev_'+a]['accuracy'],summaries['dev_'+a]['F_to_T'],-summaries['dev_'+a]['macro_f1'],a))
        selection=dict(selected=ranked[0],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            based_only_on='dev96',criteria=json.loads((HERE/'protocol.json').read_text())['selection_order'],
            metrics={a:summaries['dev_'+a] for a in ['A0','A1']})
        target=HERE/'selection.json'
        if target.exists():assert json.loads(target.read_text())['selected']==ranked[0], 'Selection already frozen'
        else:write(target,selection)
    print(json.dumps({k:{x:m[x] for x in ['correct','rules','macro_f1','F_to_T','valid_cases','target_pair_correct','target_pair_total']} for k,m in summaries.items()},ensure_ascii=False))

if __name__=='__main__':main()
