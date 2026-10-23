"""CPU evidence audit. Public output contains IDs, labels, indices and hashes only."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from step0_extended_audit import branch_action_labels, indexed, normalized_action, private_case_audit
from step0_reanalysis import ROOT, read_jsonl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def counts(rows, key):
    return dict(sorted(collections.Counter(r[key] for r in rows).items()))


def retained(case, variant, branches):
    indices = [i for b in branches for i in case['branches'][b]['rule_indices']]
    actions = [case['variant_rules'][variant][i]['action'] for i in indices]
    return {'branches': branches, 'rule_indices': indices,
            'allowed_option_labels': sorted(branch_action_labels(case, variant, branches)),
            'exact_option_labels': sorted(k for k, v in case['options'].items() if v in actions),
            'distinct_actions': len(set(map(normalized_action, actions))),
            'action_hashes': [digest(normalized_action(x)) for x in actions]}


def transition(a, d, gold, sibling):
    if a == d:
        return 'unchanged'
    if a == gold:
        return 'gold_to_sibling' if d == sibling else 'gold_to_other'
    if d == gold:
        return 'sibling_to_gold' if a == sibling else 'other_to_gold'
    return 'wrong_to_wrong_changed'


def structural(cases, scores, ids):
    base = private_case_audit(cases, scores, ids)
    ix = indexed(scores)
    cohorts = {'A24': [], 'D6': [], 'B_oracle_errors': [], 'D_oracle_errors': []}
    structure, directions, top = [], [], []
    for c in sorted(cases, key=lambda c:c['id']):
        i = c['id']; st = ix[i,'A','branch_hidden']['stratum']
        for v in 'ABCD':
            rules = c['variant_rules'][v]
            all_indices = [j for b in c['branches'] for j in b['rule_indices']]
            if sorted(all_indices) != list(range(len(rules))):
                raise ValueError('branch rules do not partition rules: '+i)
            for b in c['branches']:
                if any(rules[j]['condition'] != b['condition'] for j in b['rule_indices']):
                    raise ValueError('branch condition linkage mismatch: '+i)
            for method in ['visible_prune','hidden_prune','hidden_top2','gold_branch_prune']:
                r = ix[i,v,method]
                if r['gold'] != c['gold_answer'][v] or bool(r['correct']) != (r['prediction']==r['gold']):
                    raise ValueError('action gold/score mismatch: '+i)
        ra, rd = c['variant_rules']['A'], c['variant_rules']['D']
        sib_a = retained(c,'A',[c['sib_branch']]); sib_d = retained(c,'D',[c['sib_branch']])
        d_actions = [normalized_action(r['action']) for r in rd]
        structural_row = dict(id=i, stratum=st, sibling_rules=len(sib_d['rule_indices']),
            sibling_actions_before=sib_a['distinct_actions'], sibling_actions_after=sib_d['distinct_actions'],
            sibling_allowed_before=sib_a['allowed_option_labels'], sibling_allowed_after=sib_d['allowed_option_labels'],
            sibling_duplicate_actions=len(sib_d['rule_indices'])>sib_d['distinct_actions'],
            any_repeated_action_across_rules=len(set(d_actions))<len(d_actions),
            conditions_identical=[r['condition'] for r in ra]==[r['condition'] for r in rd],
            shared_patient_options_branch_order=True,
            gold_branch_single_rule=len(c['branches'][c['app_branch']]['rule_indices'])==1)
        structure.append(structural_row)
        a,d = ix[i,'A','branch_visible'],ix[i,'D','branch_visible']
        directions.append(dict(id=i,stratum=st,A=a['prediction'],D=d['prediction'],
            gold=chr(65+c['app_branch']),sibling=chr(65+c['sib_branch']),
            transition=transition(a['prediction'],d['prediction'],chr(65+c['app_branch']),chr(65+c['sib_branch']))))
        for v in 'ABCD':
            r=ix[i,v,'hidden_prune']; b=ix[i,v,'branch_hidden']; o=ix[i,v,'gold_branch_prune']
            for name,row,bs in [(v+'24' if v=='A' else v+'6', r,[r['predicted_branch']])] if v in 'AD' and not b['correct'] and r['correct'] else []:
                rr=retained(c,v,bs)
                item=dict(id=i,stratum=st,variant=v,prediction=row['prediction'],gold=row['gold'],original=c['original_answer'],branch_correct=False,retained=rr,
                    literal_class=1 if row['gold'] in rr['allowed_option_labels'] else 2,
                    semantic_review='pending',exact_match=row['prediction'] in rr['exact_option_labels'])
                if v=='A':
                    dr=ix[i,'D','hidden_prune']; drules=retained(c,'D',[dr['predicted_branch']])
                    item.update(D_prediction=dr['prediction'],D_retained=drules,
                        original_in_A=c['original_answer'] in rr['allowed_option_labels'],
                        original_in_D=c['original_answer'] in drules['allowed_option_labels'])
                cohorts[name].append(item)
            if v in 'BD' and not o['correct']:
                rr=retained(c,v,[c['gold_branch'][v]])
                cohorts[v+'_oracle_errors'].append(dict(id=i,stratum=st,prediction=o['prediction'],gold=o['gold'],retained=rr,
                    permitted=o['prediction'] in rr['allowed_option_labels'],original=o['prediction']==c['original_answer'],semantic_review='pending'))
            t=ix[i,v,'hidden_top2']; first=t['predicted_branch']; second=t['runner_up']
            f=branch_action_labels(c,v,[first]); s=branch_action_labels(c,v,[second]) if second is not None else set()
            origin=('both' if t['prediction'] in s else 'first') if t['prediction'] in f else ('second' if t['prediction'] in s else 'neither')
            outcome='rescued' if t['correct'] and not r['correct'] else 'damaged' if r['correct'] and not t['correct'] else 'unchanged_correct' if t['correct'] else 'unchanged_wrong'
            top.append(dict(id=i,variant=v,stratum=st,baseline_prediction=r['prediction'],prediction=t['prediction'],prediction_changed=r['prediction']!=t['prediction'],
                outcome=outcome,origin=origin,first=retained(c,v,[first]),second=retained(c,v,[second]) if second is not None else None,
                runner_is_gold=second==c['gold_branch'][v],original=t['prediction']==c['original_answer'],
                designated_original_sibling=t['prediction']==c['source_variants']['A']['original_sibling_action_label'],
                selected_in_current_sibling=t['prediction'] in branch_action_labels(c,v,[c['sib_branch']]),
                semantic_review='pending'))
    grouped={g:counts([r for r in directions if g=='all' or r['stratum']==g],'transition') for g in ['all','K','U']}
    top_summary={}
    for v in 'ABCD':
        rows=[r for r in top if r['variant']==v]
        top_summary[v]={}
        for outcome in ['rescued','damaged']:
            rr=[r for r in rows if r['outcome']==outcome]
            top_summary[v][outcome]=dict(n=len(rr),origin=counts(rr,'origin'),original=sum(r['original'] for r in rr),designated_original_sibling=sum(r['designated_original_sibling'] for r in rr),selected_in_current_sibling=sum(r['selected_in_current_sibling'] for r in rr))
        rr=[r for r in rows if r['runner_is_gold']]
        top_summary[v]['runner_gold']=dict(n=len(rr),outcome=counts(rr,'outcome'))
        top_summary[v]['all_prediction_changes']=sum(r['prediction_changed'] for r in rows)
    return dict(extended_private_checks=base,cohorts=cohorts,structure_rows=structure,direction_rows=directions,
        direction_summary=grouped,top2_summary=top_summary,top2_changed_rows=[r for r in top if r['prediction_changed']],
        classification_policy='1/2 are normalized literal inclusion/absence. Categories 3 semantic equivalence and 4 unresolved mapping require review; no clinician labels inferred.',
        semantic_categories_3_4={'status':'not_adjudicated','counts':None})


def source_audit(cases, source_root, expand_results):
    paths=json.loads((source_root/'all_paths.json').read_text()); meta=json.loads((source_root/'sources.json').read_text())
    manifest=json.loads((expand_results/'manifest.json').read_text())
    for name in ['all_paths.json','train.parquet']:
        if sha(source_root/name)!=meta[name]['sha256'] or sha(source_root/name)!=manifest['source_hashes'][name]:
            raise ValueError('source hash mismatch '+name)
    import pandas as pd
    table=pd.read_parquet(source_root/'train.parquet')
    mappings=[]
    for c in sorted(cases,key=lambda c:c['id']):
        r=c['source_variants']['A']; tree=c['disease']; source=table.iloc[r['source_row']]
        if list(source['path'])!=r['path'] or source['disease']!=tree:
            raise ValueError('source row mismatch '+c['id'])
        for field,idx in [('path','source_path_index'),('sibling_path','sibling_path_index')]:
            if paths[tree][r[idx]]!=r[field]:raise ValueError('path index mismatch '+c['id'])
        mappings.append(dict(id=c['id'],disease=tree,dataset_tree_id=tree,source_row=r['source_row'],source_path_index=r['source_path_index'],sibling_path_index=r['sibling_path_index'],
            source_path_sha256=digest(r['path']),sibling_path_sha256=digest(r['sibling_path']),tree_path_set_sha256=digest(paths[tree]),
            clinical_guideline_source=None,clinical_guideline_version=None,shared_upper_source=None))
    tree_counts=counts(mappings,'dataset_tree_id')
    expanded=read_jsonl(expand_results/'selection.jsonl')
    return dict(source_files=meta,tree_counts=tree_counts,n_dataset_trees=len(tree_counts),max_tree_fraction=max(tree_counts.values())/len(cases),
        case_mapping=mappings,overlap_with_800=len({c['id'] for c in cases}&{r['id'] for r in expanded}),
        independent_clinical_sources=None,source_level_intervals='not_computed_no_verified_upper_source_mapping',
        tree_level_statistics='dataset_tree_id equals disease key exactly; reuse extended_public.json disease-cluster intervals',
        source_schema_has_guideline_version=False)


def first_step_mask_bound(option_logps, gold, beta):
    """True proves masking; False is inconclusive without the vocabulary maximum."""
    return option_logps[gold] < max(option_logps.values()) + math.log(beta)


def select_cad(rows, ids):
    subset=[r for r in rows if r['id'] in ids and r['variant'].startswith('B')
            and r['method']=='cad' and not r.get('assigned_wrong',False)]
    if len(subset)!=len(ids) or len({r['id'] for r in subset})!=len(ids):
        raise ValueError('CAD diagnostic cohort mismatch')
    return {r['id']:r for r in subset}


def cad_audit(results, runroot):
    frozen=json.loads((results/'step0_review/summary.json').read_text())['cad_change']['B']
    ids=frozen['old_only_ids']+frozen['filtered_only_ids']; tables=[]; inventories=[]
    for p in [runroot/'qwen_full_cad128.jsonl',runroot/'cad200/original_filtered.jsonl']:
        table=select_cad(read_jsonl(p),ids); subset=list(table.values())
        tables.append(table)
        inventories.append(dict(path=str(p),sha256=sha(p),keys=sorted(set().union(*(r.keys() for r in subset)))))
    records=[]
    for i in ids:
        a,b=tables[0][i],tables[1][i]
        loss=i in frozen['old_only_ids']
        if bool(a['correct'])!=loss or bool(b['correct'])==loss:raise ValueError('CAD frozen transition mismatch')
        gold=b['gold_answer']; logps=b['option_logprobs']
        # max over options <= max over full vocabulary: sufficient, not necessary, masking evidence.
        certainly_masked=first_step_mask_bound(logps,gold,b['plausibility_beta'])
        records.append(dict(id=i,outcome='lost' if loss else 'gained',old_prediction=a['prediction'],filtered_prediction=b['prediction'],gold=gold,
            original=b['original_answer'],beta=b['plausibility_beta'],old_first_token_sha256=digest(a['unconstrained_first_token']),
            filtered_first_token_sha256=digest(b['unconstrained_first_token']),first_token_changed=a['unconstrained_first_token']!=b['unconstrained_first_token'],
            first_step_option_logps_identical=a['option_logprobs']==b['option_logprobs'],
            old_first_token_is_gold=a['unconstrained_first_token'].strip()==gold,
            filtered_first_token_is_gold=b['unconstrained_first_token'].strip()==gold,
            old_generated_tokens=a['generated_tokens'],filtered_generated_tokens=b['generated_tokens'],old_stop_reason=a['stop_reason'],filtered_stop_reason=b['stop_reason'],
            old_completion_sha256=digest(a['completion']),filtered_completion_sha256=digest(b['completion']),old_parse_failure=a['prediction'] is None,filtered_parse_failure=b['prediction'] is None,
            first_step_gold_certainly_masked_from_option_lower_bound=certainly_masked,
            first_step_gold_logp=logps[gold],max_option_logp=max(logps.values()),
            exact_global_threshold_available=False,per_step_masks_available=False))
    probes=[]
    for p in [results.parents[1]/'results/run500/cad_diagnosis/first_token_probe.json',runroot/'cad_cause_probe.json',runroot/'cad_failure_probe.json']:
        rows=json.loads(p.read_text()); probes.append(dict(path=str(p),sha256=sha(p),rows=len(rows),ids=sorted({r['id'] for r in rows}),diagnostic_overlap=sorted({r['id'] for r in rows}&set(ids))))
    return dict(frozen_case_ids=ids,raw_inventory=inventories,existing_probes=probes,cases=records,new_gpu_calls=0,
        limitation='Stored option_logprobs are first-step conditional log probabilities, not full-vocabulary or later-step traces. A sufficient mask bound does not establish whole-sequence causality.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=Path('/home/gcx/aster_medguide_20261022/private'))
    p.add_argument('--expand-results',type=Path,default=Path('/home/gcx/TRIAGE-worktrees/medguide-expand-cs/experiments/medguide_expand_cs_20261023/results'))
    a=p.parse_args();cases=read_jsonl(a.cases);ids=json.loads((ROOT/'priority200/selection.json').read_text())['ids']
    manifest=json.loads(a.cases.with_name('qwen_complete_parser_v2.manifest.json').read_text())
    if sha(a.cases)!=manifest['input_sha256'] or not manifest['complete']:raise ValueError('private inference input fingerprint mismatch')
    scores_path=ROOT/'priority200/qwen/full_parser_v2/scores.jsonl'
    scores=read_jsonl(scores_path)
    if len(ids)!=200 or len(set(ids))!=200 or {r['id'] for r in scores}!=set(ids):
        raise ValueError('fixed200 ID contract failed')
    raw_path=a.cases.with_name('qwen_complete_parser_v2.jsonl')
    raw=indexed(read_jsonl(raw_path)); public=indexed(scores)
    if set(raw)!=set(public):raise ValueError('private/public inference keys differ')
    for key,row in public.items():
        for field in ['prediction','gold','correct','stratum','disease','predicted_branch','runner_up']:
            if row.get(field)!=raw[key].get(field):raise ValueError('private/public score field differs: '+field)
    result=structural(cases,scores,ids)
    result['sources']=source_audit(cases,a.source_root,a.expand_results)
    result['cad']=cad_audit(ROOT,a.cases.parents[2])
    result['provenance']=dict(n=len(ids),private_input_sha256=sha(a.cases),scores_sha256=sha(scores_path),
        script_sha256=sha(__file__),extended_script_sha256=sha(Path(__file__).with_name('step0_extended_audit.py')),
        source_head='525514a33083a50e06042382d143067263288b9e',new_model_calls=0,
        private_predictions_sha256=sha(raw_path),private_public_prediction_rows_verified=len(scores),
        prepare_sha256=sha(Path(__file__).with_name('prepare.py')),run_sha256=sha(Path(__file__).with_name('run.py')),
        frozen_public_sha256=sha(ROOT/'step0_review/extended_public.json'),frozen_summary_sha256=sha(ROOT/'step0_review/summary.json'))
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Private structure audit complete:',len(ids),'cases; public-safe output written')

if __name__=='__main__':main()
