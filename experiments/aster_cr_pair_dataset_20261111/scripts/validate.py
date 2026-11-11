"""Independent structural validation; never turns uncertain semantics into PASS."""
from runtime_paths import ROOT, DATA, RAW, REPO, SCRIPTS, SCHEMA, source_document_path
import argparse, collections, csv, datetime, difflib, json, pathlib, re
from jsonschema import Draft202012Validator
from restore_historical import DATA, ROOT, RAW, rows, write_json, write_rows
KINDS=['A','P','C','F']
def eval_rule(expr,facts):
    op=expr['op']
    if op in ['and','or']:
        vals=[eval_rule(x,facts) for x in expr['args']]
        if op=='and': return False if any(x is False for x in vals) else None if any(x is None for x in vals) else True
        return True if any(x is True for x in vals) else None if any(x is None for x in vals) else False
    if op=='not':
        v=eval_rule(expr['arg'],facts);return None if v is None else not v
    value=facts.get(expr['field']);target=expr['value']
    if value is None:return None
    if op=='<':return value<target
    if op=='<=':return value<=target
    if op=='>':return value>target
    if op=='>=':return value>=target
    if op=='==':return value==target
    if op=='!=':return value!=target
    if op=='in':return value in target
    if op=='not_in':return value not in target
    raise ValueError('unknown operator')
def flat_leaves(e):
    if e['op'] in ['and','or']:
        return sum([flat_leaves(x) for x in e['args']],[])
    if e['op']=='not':return flat_leaves(e['arg'])
    return [e]
def predicate_differences(a,b):
    if type(a)!=type(b):return 1
    if isinstance(a,dict):
        ignore={'source','number_source','comparator_source','name','unit'}
        return sum(predicate_differences(a.get(k),b.get(k)) for k in (a.keys()|b.keys())-ignore)
    if isinstance(a,list):
        if len(a)!=len(b):return 1
        return sum(predicate_differences(x,y) for x,y in zip(a,b))
    return int(a!=b)
def normalize(s):return re.sub(r'\s+','',s).casefold()
def semantic_signature(r):
    a,b=r['original'],r['modified']
    return (r['kind'],normalize(a['patient']),normalize(a['condition']),normalize(a['guide']),normalize(b['patient']),normalize(b['condition']),normalize(b['guide']),a['label'],b['label'])
def near_text(r):
    a,b=r['original'],r['modified']
    # Compare the clinical contrast, not the full common guideline shared by unrelated cases.
    return normalize(a['patient'])+'\n'+normalize(a['condition'])+'\n'+normalize(b['patient'])+'\n'+normalize(b['condition'])+'\n'+'\n'.join(normalize(e['old'])+'>'+normalize(e['new']) for e in r['edits'])
def find_duplicates(records):
    seen={};exact={};near={};buckets=collections.defaultdict(list)
    for r in records:
        sig=semantic_signature(r)
        if sig in seen:exact[r['pair_id']]=seen[sig]
        else:seen[sig]=r['pair_id']
        buckets[(r['source_id'],r['kind'],r['node_id'])].append(r)
    # Near-duplicates identify the same clinical original and edit under minor formatting/word changes.
    # Distinct original cases or different edited values are reported as clusters, not silently merged.
    for bucket in buckets.values():
        for i,r in enumerate(bucket):
            if r['pair_id'] in exact:continue
            for s in bucket[:i]:
                if s['pair_id'] in exact:continue
                ra,sa=r['original'],s['original'];rb,sb=r['modified'],s['modified']
                if ra['label']!=sa['label'] or rb['label']!=sb['label']:continue
                a,b=near_text(r),near_text(s)
                if abs(len(a)-len(b))>max(len(a),len(b))*.02:continue
                ratio=difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()
                if ratio>=.995:
                    near[r['pair_id']]=dict(other=s['pair_id'],similarity=round(ratio,6),same_case=r['case_id']==s['case_id'])
                    break
    return exact,near
def run(stage,finalize=False):
    records=rows(ROOT/'historical_pairs.jsonl')
    generated=ROOT/'generated_candidates.jsonl'
    if stage=='all':
        if not generated.exists():raise RuntimeError('generated_candidates.jsonl is not present; all-stage cannot be declared complete')
        records+=rows(generated)
    splits=json.loads((ROOT/'source_splits.json').read_text())
    source_sets={k:set(v) for k,v in splits.items()}
    schema=Draft202012Validator(json.loads((SCHEMA).read_text()))
    cases={r['case_id']:r for r in rows(DATA/'path_cases.jsonl') if r['split']=='fit'}
    all_source_rows=rows(DATA/'sources.jsonl')
    source_by_gid={g:r['source_id'] for r in all_source_rows for g in r['guideline_ids']}
    exact,near=find_duplicates(records)
    checks=[]
    for r in records:
        failures=[];unknowns=[];pid=r['pair_id'];a,b=r['original'],r['modified'];kind=r['kind']
        for err in schema.iter_errors(r):failures.append('schema:'+str(list(err.path))+':'+err.message)
        if r.get('origin')=='generated' and (r.get('provenance',{}).get('requested_model')!='gpt-6.1-sol' or r.get('provenance',{}).get('requested_reasoning_effort')!='high'):failures.append('generation_model_or_reasoning_configuration_mismatch')
        if r['source_id'] not in source_sets['fit'] or r['source_id'] in source_sets['dev']|source_sets['test']:failures.append('source_split_overlap_or_nonfit')
        if source_by_gid.get(r['guideline_id'])!=r['source_id']:failures.append('source_guideline_id_mismatch')
        c=cases.get(r['case_id'])
        if c is None:failures.append('unknown_fit_case')
        else:
            gold={x['gid']:x['label'] for x in c['gold_path']}
            if a['patient']!=c['patient']:failures.append('original_patient_not_released_case')
            if gold.get(r['node_id'])!=a['label']:failures.append('original_label_not_gold_path')
            if r['source_id']!=c['source_id']:failures.append('case_source_mismatch')
        if not source_document_path(r['source_document']).is_file():failures.append('missing_guideline_document')
        if a['node_id']!=r['node_id'] or b['node_id']!=r['node_id']:failures.append('target_node_changed')
        if a['definitions']!=b['definitions']:failures.append('definitions_changed')
        diff={k:a[k]!=b[k] for k in ['patient','condition','guide']}
        if not any(diff.values()):failures.append('no_edit')
        if kind in ['A','P'] and a['label']!=b['label']:failures.append('invariance_label_mismatch')
        if kind in ['C','F'] and a['label']==b['label']:failures.append('counterfactual_label_did_not_flip')
        if kind in ['C','F'] and not r['counterfactual']:failures.append('counterfactual_not_marked')
        if kind=='A':
            if diff['patient'] or diff['condition']:failures.append('A_patient_or_condition_changed')
            # Only terminal action lines may change, preserving all decision-node conditions and edges.
            left=[line for line in a['guide'].splitlines() if not re.match(r'^L\d+:',line)]
            right=[line for line in b['guide'].splitlines() if not re.match(r'^L\d+:',line)]
            if left!=right:failures.append('A_nonaction_tree_text_changed')
        if kind=='P' and (diff['condition'] or diff['guide'] or not diff['patient']):failures.append('P_structural_scope_violation')
        if kind=='F' and (diff['condition'] or diff['guide'] or not diff['patient']):failures.append('F_structural_scope_violation')
        if kind=='C':
            if diff['patient'] or not diff['condition']:failures.append('C_structural_scope_violation')
            # Guide conditions must reflect the exact same one target edit.
            if a['guide'].replace(r['node_id']+': '+a['condition'],r['node_id']+': '+b['condition'],1)!=b['guide']:failures.append('C_other_guide_content_changed')
        rule=r.get('program_rule')
        left_calc=right_calc=None
        if kind in ['C','F'] and rule:
            try:
                pred=rule['predicate'];facts=rule['facts']
                left_calc=eval_rule(pred,facts)
                newpred=r.get('program_modified_predicate') or pred
                newfacts=r.get('program_modified_facts') or facts
                right_calc=eval_rule(newpred,newfacts)
                if left_calc is None or right_calc is None:unknowns.append('rule_requires_unknown_facts')
                elif int(left_calc)!=a['label'] or int(right_calc)!=b['label']:failures.append('program_recomputed_label_mismatch')
                if kind=='C' and (facts!=newfacts or predicate_differences(pred,newpred)!=1):failures.append('C_multiple_rule_factor_change')
                if kind=='F':
                    changed=[k for k in facts.keys()|newfacts.keys() if facts.get(k)!=newfacts.get(k)]
                    if len(changed)!=1 or pred!=newpred:failures.append('F_multiple_fact_factor_change')
            except (KeyError,TypeError,ValueError) as ex:unknowns.append('rule_not_computable:'+str(ex))
        elif kind in ['C','F']:unknowns.append('no_executable_rule_semantics_left_to_agent_B')
        # An uncomputable rule is not passed by this program; independent B decides semantics.
        if pid in exact:failures.append('exact_duplicate_of:'+exact[pid])
        if pid in near:unknowns.append('near_duplicate_requires_resolution:'+near[pid]['other'])
        checks.append(dict(pair_id=pid,kind=kind,status='FAIL' if failures else 'UNCERTAIN' if pid in near else 'PASS',status_scope='deterministic_structure_and_duplicate_checks',rule_semantics_status=('FAIL' if any('program_recomputed_label_mismatch' in f or 'multiple_rule_factor_change' in f or 'multiple_fact_factor_change' in f for f in failures) else 'PASS' if kind in ['C','F'] and rule and left_calc is not None and right_calc is not None else 'UNCERTAIN'),
            failures=failures,uncertainties=unknowns,rule_recomputed=rule is not None and kind in ['C','F'],program_left_label=None if left_calc is None else int(left_calc),program_right_label=None if right_calc is None else int(right_calc),
            semantic_review='required_independent_agent_B',near_duplicate=near.get(pid)))
    write_rows(ROOT/('validation_historical.jsonl' if stage=='historical' else 'validation_results.jsonl'),checks)
    write_json(ROOT/'reports/duplicate_report.json',dict(exact=exact,near=near,method='exact normalized complete pairs; within-source/kind/node clinical contrast SequenceMatcher>=0.995',near_policy='UNCERTAIN; no automatic acceptance'))
    summary=dict(total=len(records),status=dict(collections.Counter(x['status'] for x in checks)),failures=dict(collections.Counter(e for x in checks for e in x['failures'])),unknowns=dict(collections.Counter(e.split(':')[0] for x in checks for e in x['uncertainties'])))
    write_json(ROOT/'reports/validation_summary.json',summary)
    if finalize:finalize_results(records,checks,stage)
    print(json.dumps(summary,ensure_ascii=False))
def export_csv(p,fields,rs):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
def write_human_audit(accepted,stage):
    sample=[]
    for k in KINDS:
        rs=sorted([r for r in accepted if r['kind']==k],key=lambda r:(r['source_id'],r['case_id'],r['pair_id']))
        bysource=collections.defaultdict(list)
        for r in rs:bysource[r['source_id']].append(r)
        limit=min(len(rs),50 if k in ['C','F'] else 25)
        chosen=[]
        while len(chosen)<limit and any(bysource.values()):
            for sid in sorted(bysource):
                if bysource[sid] and len(chosen)<limit:chosen.append(bysource[sid].pop(0))
        sample.extend(dict(pair_id=r['pair_id'],kind=k,source_id=r['source_id'],case_id=r['case_id'],selection_reason='source-stratified; prioritize C/F',reviewer='',clinical_qualification='',review_date='',human_status='PENDING',disagreement='',revision_or_exclusion_reason='') for r in chosen)
    fields=['pair_id','kind','source_id','case_id','selection_reason','reviewer','clinical_qualification','review_date','human_status','disagreement','revision_or_exclusion_reason']
    export_csv(ROOT/('human_audit_historical.csv' if stage=='historical' else 'human_audit.csv'),fields,sample)
    write_json(ROOT/'reports/human_audit_status.json',dict(status='awaiting_human_review',selected=len(sample),actually_reviewed=0,actual_review_fraction=0,human_pass_rate=None,clinical_expert_involved=False))
def select_balanced(accepted):
    """Stratify by source/language/complexity, preferring unused cases in every class."""
    counts=collections.Counter(r['kind'] for r in accepted)
    n=min(counts[k] for k in KINDS)
    balanced=[]
    for kind in KINDS:
        buckets=collections.defaultdict(list)
        for r in accepted:
            if r['kind']==kind:
                size='small' if r['tree_complexity']<=15 else 'medium' if r['tree_complexity']<=30 else 'large'
                buckets[(r['source_id'],r['language'],size)].append(r)
        for bucket in buckets.values():
            bucket.sort(key=lambda r:(r['case_id'],r['pair_id']))
        used_cases=set()
        selected=[]
        while len(selected)<n:
            progress=False
            for key in sorted(buckets):
                bucket=buckets[key]
                index=next((i for i,r in enumerate(bucket) if r['case_id'] not in used_cases),None)
                if index is None:
                    continue
                item=bucket.pop(index)
                selected.append(item)
                used_cases.add(item['case_id'])
                progress=True
                if len(selected)==n:
                    break
            if not progress:
                # All available cases in this class have already been represented.
                for key in sorted(buckets):
                    if buckets[key] and len(selected)<n:
                        selected.append(buckets[key].pop(0))
                        progress=True
                if not progress:
                    raise ValueError('Insufficient accepted pairs for balanced selection')
        balanced.extend(selected)
    return balanced

def finalize_results(records,checks,stage):
    rp=ROOT/('review_results_historical.jsonl' if stage=='historical' else 'review_results.jsonl')
    if not rp.exists():raise RuntimeError('independent semantic review is not finished')
    reviewed=rows(rp);rev={r['pair_id']:r for r in reviewed};val={r['pair_id']:r for r in checks}
    expected={r['pair_id'] for r in records}
    wrong_protocol=[r['pair_id'] for r in reviewed if r.get('review_model')!='gpt-6.1-sol' or r.get('reasoning_effort')!='xhigh' or r.get('round')!=1]
    if wrong_protocol:raise RuntimeError('formal review must use Sol/xhigh round 1; mismatches: '+str(wrong_protocol[:5]))
    if len(rev)!=len(reviewed) or set(rev)!=expected:raise RuntimeError('review must contain exactly one result for every candidate')
    accepted=[];rejected=[]
    for r in records:
        rv=rev[r['pair_id']];vc=val[r['pair_id']]
        left=rv.get('independent_left_label');right=rv.get('independent_right_label')
        # Missing independent labels or unvalidated semantic checklist must not leak into training.
        criteria=rv.get('checks',{})
        missing=[k for k in ['factor_matches_class','no_other_decisive_change','labels_correct','source_support','medical_plausibility','protected_content_preserved','no_hidden_second_factor'] if criteria.get(k) not in [True,'PASS']]
        if rv.get('status')=='PASS' and vc['status']=='PASS' and left==r['original']['label'] and right==r['modified']['label'] and not missing:
            accepted.append(r)
        else:
            rejected.append(dict(pair=r,review_status=rv.get('status'),program_status=vc['status'],reason=rv.get('reason'),program_failures=vc['failures'],program_uncertainties=vc['uncertainties'],independent_labels=[left,right],missing_pass_checks=missing))
    write_human_audit(accepted,stage)
    suffix='_historical' if stage=='historical' else ''
    write_rows(ROOT/('accepted'+suffix+'_pairs.jsonl' if stage=='historical' else 'accepted_pairs.jsonl'),accepted)
    write_rows(ROOT/('rejected'+suffix+'_pairs.jsonl' if stage=='historical' else 'rejected_pairs.jsonl'),rejected)
    stats=[];coverage=[];counts=collections.Counter(r['kind'] for r in accepted)
    for k in KINDS:
        src=[r for r in records if r['kind']==k];acc=[r for r in accepted if r['kind']==k]
        stats.append(dict(kind=k,candidates=len(src),ai_pass=sum(rev[r['pair_id']].get('status')=='PASS' for r in src),accepted=len(acc),acceptance_rate=len(acc)/len(src) if src else 0,sources=len({r['source_id'] for r in acc}),cases=len({r['case_id'] for r in acc}),nodes=len({r['group_id'] for r in acc}),needed_to_500=max(0,500-len(acc))))
        for sid in sorted({r['source_id'] for r in records}):
            ss=[r for r in src if r['source_id']==sid];aa=[r for r in acc if r['source_id']==sid]
            coverage.append(dict(kind=k,source_id=sid,candidates=len(ss),accepted=len(aa),cases=len({r['case_id'] for r in aa})))
    export_csv(ROOT/('pair_statistics_historical.csv' if stage=='historical' else 'pair_statistics.csv'),list(stats[0]),stats)
    export_csv(ROOT/('source_coverage_historical.csv' if stage=='historical' else 'source_coverage.csv'),list(coverage[0]),coverage)
    write_json(ROOT/('reports/generation_gaps.json' if stage=='historical' else 'reports/final_target_gaps.json'),dict(stage='after_one_round_historical_review' if stage=='historical' else 'after_single_new_candidate_review',target_per_class=500,counts=dict(counts),classes=stats,generation_started=stage=='all',allowed_sources=json.loads((ROOT/'source_splits.json').read_text())['fit'],priority=['P','C','F','A'],review_loop=False))
    plot(stats,coverage,suffix)
    if stage=='all':
        write_rows(ROOT/'full_pool_train.jsonl',accepted)
        balanced=select_balanced(accepted)
        write_rows(ROOT/'balanced_train.jsonl',balanced)
    write_json(ROOT/('reports/completion_'+stage+'.json'),dict(stage=stage,review_rounds=1,accepted=len(accepted),rejected=len(rejected),counts=dict(counts),clinical_expert_review=False,remaining='Historical scope complete; generation and new-candidate review not started in this window' if stage=='historical' else None))
    import subprocess,sys
    result=subprocess.run([sys.executable,str(SCRIPTS/'report_manifest.py')],capture_output=True,text=True)
    (ROOT/'logs/report_manifest.log').write_text(result.stdout+result.stderr)

def plot(stats,coverage,suffix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    x=np.arange(4);axes[0].bar(x-.18,[r['candidates'] for r in stats],.36,label='Candidates');axes[0].bar(x+.18,[r['accepted'] for r in stats],.36,label='Accepted')
    axes[0].set_xticks(x,KINDS);axes[0].set_ylabel('Pairs');axes[0].legend()
    axes[1].bar(KINDS,[100*r['acceptance_rate'] for r in stats]);axes[1].set_ylabel('Accepted (%)');axes[1].set_ylim(0,100)
    fig.tight_layout();fig.savefig(ROOT/'reports'/('pair_counts_and_acceptance'+suffix+'.png'),dpi=180);plt.close(fig)
    sources=sorted({r['source_id'] for r in coverage});z=np.zeros((4,len(sources)))
    for r in coverage:z[KINDS.index(r['kind']),sources.index(r['source_id'])]=r['accepted']
    fig,ax=plt.subplots(figsize=(16,4));im=ax.imshow(z,aspect='auto',cmap='Blues');ax.set_yticks(range(4),KINDS);ax.set_xticks(range(len(sources)),[s.split(':')[-1] for s in sources],rotation=90,fontsize=6);ax.set_xlabel('Guideline source cluster (train_val ID)');fig.colorbar(im,ax=ax,label='Accepted pairs');fig.tight_layout();fig.savefig(ROOT/'reports'/('source_distribution'+suffix+'.png'),dpi=180);plt.close(fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['historical','all'],default='historical');p.add_argument('--finalize',action='store_true');args=p.parse_args();run(args.stage,args.finalize)
