"""Read-only routing/outcome audit; no model loading or official-result writes."""
import argparse,collections,json,random,re
from pathlib import Path
from src.aster.manuscript_core import plan_gains
from experiments.aster_manuscript_clinical_20261030.engine import decisions_from,direct_text,parse_prediction

SEED=20261030

def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def oracle_decisions(case):
    truth={'G'+rid[1:-1]:rid for rid in case['gold_rules']}
    # NONE off-path is a routing placeholder only; never a semantic gold label.
    return {g['group_id']:{'prediction':truth.get(g['group_id'],'NONE'),'abstain':False,
            'diagnostic_only':True,'label_scope':'reference_path' if g['group_id'] in truth else 'off_path_placeholder'}
            for g in case['guideline']['groups']}

def strict_parse(text):
    leaves=re.findall(r'FINAL\s*[:：]\s*(L\d+|NONE)\b',text,re.I)
    sections=re.findall(r'PATH\s*[:：]\s*([^\n]*)',text,re.I)
    path=re.findall(r'R\d+[YN]',sections[-1]) if sections else []
    leaf=leaves[-1].upper() if leaves else None
    nodes=[rid[:-1] for rid in path]
    invalid=leaf is None or not sections or len(set(nodes))!=len(nodes)
    return leaf,path,invalid

def strict_score(case,text):
    leaf,path,bad=strict_parse(text)
    return {'leaf_correct':leaf==case['gold_leaf'],'root_correct':bool(path) and path[0]==case['gold_rules'][0],
            'exact_path':not bad and path==case['gold_rules'],'invalid':bad,'leaf':leaf,'path':path}

def paired(rows,method,other):
    index={(r['case_id'],r['method']):r for r in rows};pairs=[]
    for r in rows:
        if r['method']!=method or (r['case_id'],other) not in index:continue
        b=index[r['case_id'],other]
        pairs.append((r['source_id'],int(r['ok'])-int(b['ok']),r['ok'],b['ok'],r['status']=='complete' and b['status']=='complete'))
    grouped=collections.defaultdict(list)
    for source,d,*_ in pairs:grouped[source].append(d)
    keys=sorted(grouped);n=len(pairs);ci=None;macro_ci=None
    if len(keys)>=2:
        rng=random.Random(SEED);values=[];macros=[]
        for _ in range(2000):
            sampled=[rng.choice(keys) for _ in keys];ds=[d for k in sampled for d in grouped[k]]
            values.append(sum(ds)/len(ds));macros.append(sum(sum(grouped[k])/len(grouped[k]) for k in sampled)/len(sampled))
        values.sort();macros.sort();ci=[values[49],values[1949]];macro_ci=[macros[49],macros[1949]]
    success=[d for _,d,_,_,s in pairs if s]
    return {'comparison':method+' minus '+other,'paired_cases':n,'source_clusters':len(keys),
            'case_weighted_delta':sum(x[1] for x in pairs)/n if n else None,
            'source_weighted_delta':sum(sum(v)/len(v) for v in grouped.values())/len(keys) if keys else None,
            'case_weighted_cluster_bootstrap_ci':ci,'source_weighted_cluster_bootstrap_ci':macro_ci,
            'interval_scope':'exploratory_few_clusters' if len(keys)<10 else 'source_cluster_bootstrap',
            'corrected':sum(a and not b for _,_,a,b,_ in pairs),'regressed':sum(b and not a for _,_,a,b,_ in pairs),
            'paired_success_cases':len(success),'paired_success_coverage':len(success)/n if n else None,
            'paired_success_delta':sum(success)/len(success) if success else None}

def load_units(run,phase):
    units=[];classes={}
    for shard in [0,1]:
        for d in sorted((run/(phase+str(shard))/'units').glob('*')):
            if not d.is_dir():continue
            cp=d/'classification.json'
            cls=json.loads(cp.read_text()) if cp.exists() else None
            for f in d.glob('*.json'):
                if f.name=='classification.json':continue
                row=json.loads(f.read_text())
                if 'method' in row:
                    units.append(row)
                    if cls is not None:classes[row['case_id']]=cls
    return units,classes

def routing_row(case,raw,freeze):
    decisions=decisions_from(raw,freeze);plan=plan_gains(case['guideline'],decisions,g_plus=freeze['g_plus'],g_minus=freeze['g_minus'])
    truth={'G'+r[1:-1]:r for r in case['gold_rules']};first_error=None;path_raw_correct=0;path_enabled_correct=0
    by_depth=[]
    for depth,rid in enumerate(case['gold_rules']):
        gid='G'+rid[1:-1];decision=decisions[gid];s=plan['groups'][gid]
        raw_ok=decision is not None and decision['prediction']==rid and not decision['abstain']
        gated_ok=s['enabled'] and s['prediction']==rid
        path_raw_correct+=raw_ok;path_enabled_correct+=gated_ok
        if not gated_ok and first_error is None:first_error=depth
        by_depth.append({'depth':depth,'raw_correct':raw_ok,'gated_correct':gated_ok,'reason':s['reason'],'reachable':s['reachable']})
    score=strict_score(case,direct_text(case,plan));root=plan['groups']['G0'];counts=collections.Counter(s['reason'] for s in plan['groups'].values())
    blocked_with_prediction=sum(not s['reachable'] and s['prediction'] not in [None,'NONE'] for s in plan['groups'].values())
    illegal_edit=sum(any(plan['gains'][r['rule_id']]!=0 for r in g['rules']) for g in case['guideline']['groups'] if not plan['groups'][g['group_id']]['reachable'])
    return {'case_id':case['case_id'],'source_id':case['source_id'],'gold_depth':len(truth),
            'root_accepted':root['enabled'],'root_correct':root['enabled'] and root['prediction']==case['gold_rules'][0],
            'gold_path_raw_correct_groups':path_raw_correct,'gold_path_gated_correct_groups':path_enabled_correct,
            'gold_path_fully_covered':path_enabled_correct==len(truth),'first_gated_error_depth':first_error,
            'route_leaf_correct':score['leaf_correct'],'reason_counts':dict(counts),'blocked_nonNONE_predictions':blocked_with_prediction,
            'unreachable_group_edits':illegal_edit,'by_depth':by_depth}

def summarize_routing(rows):
    reasons=collections.Counter();depth=collections.defaultdict(list)
    for r in rows:
        reasons.update(r['reason_counts'])
        for d in r['by_depth']:depth[d['depth']].append(d)
    n=len(rows);accepted=sum(r['root_accepted'] for r in rows)
    return {'cases':n,'source_clusters':len({r['source_id'] for r in rows}),
        'root_coverage':accepted/n if n else None,'root_correct':sum(r['root_correct'] for r in rows),
        'root_accuracy':sum(r['root_correct'] for r in rows)/n if n else None,
        'root_selective_risk':sum(r['root_accepted'] and not r['root_correct'] for r in rows)/accepted if accepted else None,
        'full_gold_path_coverage':sum(r['gold_path_fully_covered'] for r in rows)/n if n else None,
        'direct_map_leaf_accuracy':sum(r['route_leaf_correct'] for r in rows)/n if n else None,
        'gold_path_groups':sum(r['gold_depth'] for r in rows),'gold_path_raw_correct_groups':sum(r['gold_path_raw_correct_groups'] for r in rows),
        'gold_path_gated_correct_groups':sum(r['gold_path_gated_correct_groups'] for r in rows),
        'reason_counts':dict(reasons),'unreachable_group_edits':sum(r['unreachable_group_edits'] for r in rows),
        'depth_counts_descriptive_only':{str(k):{'groups':len(v),'raw_correct':sum(x['raw_correct'] for x in v),'gated_correct':sum(x['gated_correct'] for x in v)} for k,v in depth.items()}}

def audit(run,phase,out):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    freeze=json.loads((run/'frozen.json').read_text());cases=read(run/'cases'/(phase+'.jsonl'));bycase={c['case_id']:c for c in cases}
    units,classes=load_units(run,phase);route_rows=[];oracle_pass=0;route_mismatch=0;outcomes=[];score_differences=[];missing_classes=[]
    for c in cases:
        oracle=oracle_decisions(c);plan=plan_gains(c['guideline'],oracle,g_plus=freeze['g_plus'],g_minus=freeze['g_minus'])
        score=strict_score(c,direct_text(c,plan));oracle_pass+=score['leaf_correct'] and score['exact_path']
        cls=classes.get(c['case_id'])
        if not cls or 'raw_decisions' not in cls:missing_classes.append(c['case_id']);continue
        rr=routing_row(c,cls['raw_decisions'],freeze);route_rows.append(rr)
    routes={r['case_id']:r for r in route_rows}
    for r in units:
        if r['case_id'] not in bycase or r['method']=='native_official':continue
        c=bycase[r['case_id']];score=strict_score(c,r.get('result',{}).get('text',''));legacy=r.get('score',{});complete=r['status']=='complete'
        ok=complete and bool(legacy.get('final_leaf_correct',False));rr=routes.get(r['case_id'],{})
        outcomes.append({'case_id':r['case_id'],'source_id':r['source_id'],'method':r['method'],'status':r['status'],'ok':ok,
           'strict_leaf_correct':complete and score['leaf_correct'],'strict_exact_path':complete and score['exact_path'],
           'strict_root_correct':complete and score['root_correct'],'strict_invalid':complete and score['invalid'],
           'routing_root_correct':rr.get('root_correct'),'routing_leaf_correct':rr.get('route_leaf_correct'),
           'classification_interface_failures':len(r.get('classification_failures',{}))})
        if complete and (legacy.get('exact_path')!=score['exact_path'] or legacy.get('invalid_generation')!=score['invalid']):
            score_differences.append({'case_id':r['case_id'],'method':r['method'],'legacy_exact_path':legacy.get('exact_path'),
                                     'strict_exact_path':score['exact_path'],'legacy_invalid':legacy.get('invalid_generation'),'strict_invalid':score['invalid']})
        stored=r.get('result',{}).get('plan')
        if complete and stored and r['case_id'] in classes:
            raw=classes[r['case_id']]['raw_decisions'];expected=plan_gains(c['guideline'],decisions_from(raw,freeze),g_plus=r['gain'],g_minus=r['gain'])
            route_mismatch+=stored!=expected
    comparisons=[paired(outcomes,m,'baseline') for m in sorted({r['method'] for r in outcomes}) if m!='baseline']
    methods=[]
    for m in sorted({r['method'] for r in outcomes}):
        rs=[r for r in outcomes if r['method']==m];n=len(rs)
        strata={}
        for field in ['routing_root_correct','routing_leaf_correct']:
            strata[field]={str(b):{'cases':sum(r[field] is b for r in rs),'correct':sum(r['ok'] and r[field] is b for r in rs)} for b in [True,False]}
        methods.append({'method':m,'denominator':len(cases),'observed_units':n,'pending_units':len(cases)-n,
          'leaf_correct':sum(r['ok'] for r in rs),'leaf_accuracy_failures_as_zero':sum(r['ok'] for r in rs)/len(cases),
          'technical_failures':sum(r['status']!='complete' for r in rs),'units_with_classification_interface_failure':sum(r['classification_interface_failures']>0 for r in rs),
          'strict_root_correct':sum(r['strict_root_correct'] for r in rs),'strict_exact_path':sum(r['strict_exact_path'] for r in rs),
          'strict_invalid':sum(r['strict_invalid'] for r in rs),'routing_strata':strata})
    summary={'phase':phase,'scope':'existing_reference_path_only; no physician three-state clinical gold',
             'official_results_used_for_tuning':False,'frozen_probability_threshold':freeze['thresholds']['probability'],
             'frozen_margin_threshold':freeze['thresholds']['margin'],'registered_cases':len(cases),'source_clusters':len({c['source_id'] for c in cases}),
             'oracle_direct_map_correct':oracle_pass,'oracle_direct_map_gate_pass':oracle_pass==len(cases),
             'missing_classification_cases':len(missing_classes),'stored_vs_recomputed_plan_mismatches':route_mismatch,
             'strict_vs_legacy_score_difference_units':len(score_differences),'routing':summarize_routing(route_rows),'methods':methods,'comparisons':comparisons,
             'statistical_limit':'Only two source clusters in current confirm; bootstrap intervals are exploratory and cannot establish general cross-source benefit.'}
    write(out/'summary.json',summary);write(out/'routing_cases.json',route_rows);write(out/'paired_outcomes.json',outcomes);write(out/'scoring_differences.json',score_differences)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase',default='confirm',choices=['confirm']);p.add_argument('--output',required=True);a=p.parse_args()
    s=audit(a.run,a.phase,a.output);print(json.dumps({k:s[k] for k in ['registered_cases','source_clusters','oracle_direct_map_correct','stored_vs_recomputed_plan_mismatches','strict_vs_legacy_score_difference_units','routing']}))
if __name__=='__main__':main()
