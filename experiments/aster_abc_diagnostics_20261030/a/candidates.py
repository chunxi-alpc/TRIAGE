"""Source-isolated, explicitly UNADJUDICATED non-age clinical candidate pairs."""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from .common import BASE,read_jsonl,write_json

ROOT=Path('/home/gcx/TRIAGE/data/MEGA-CDP')
# Suggestions follow visible root numeric thresholds; they are never gold labels.
SPECS={
    275:(r'(?:BMI|体质指数)[^\d\n]{0,14}(\d+(?:\.\d+)?)','25.0','23.0','BMI'),
    576:(r'心率[^\d\n]{0,12}(\d+(?:\.\d+)?)','121','80','heart rate'),
    1127:(r'静脉血浆葡萄糖[^\d\n]{0,12}(\d+(?:\.\d+)?)','2.7','3.8','venous plasma glucose'),
    1479:(r'fasting plasma glucose[^\d\n]{0,12}(\d+(?:\.\d+)?)','110','90','fasting plasma glucose'),
    1670:(r'收缩压[^\d\n]{0,12}(\d+(?:\.\d+)?)','141','130','office systolic BP'),
    2659:(r'body mass index[^\d\n]{0,12}(\d+(?:\.\d+)?)','26','24','BMI'),
}

def make_candidates(base,root):
    clusters=json.loads((base/'source_clusters.json').read_text());excluded=set(clusters['official_test_clusters']+clusters['pr40_clusters']+clusters['previous_mechanism_clusters'])
    for r in read_jsonl(base/'data02/candidate_pairs.jsonl'):excluded.add(r['source_id'])
    by=defaultdict(list)
    for r in read_jsonl(root/'train_val/vignette_cases.jsonl'):by[int(r['guideline_id'])].append(r)
    candidates=[];seen=set();skipped=[]
    for source in read_jsonl(root/'train_val/decision_trees.jsonl'):
        gid=int(source['guideline_id']);key=f'train_val:{gid}'
        if gid not in SPECS:continue
        cluster=clusters['records'][key]['cluster']
        if cluster in excluded or cluster in seen:
            skipped.append({'guideline_id':gid,'reason':'excluded source cluster'});continue
        tree=json.loads(source['text'])['decision_tree'];pattern,value_a,value_b,measurement=SPECS[gid]
        anchor=next((r for r in by[gid] if re.search(pattern,r['text'],re.I)),None)
        if anchor is None:
            skipped.append({'guideline_id':gid,'reason':'no explicit matching measurement anchor'});continue
        patient=anchor['text'];match=re.search(pattern,patient,re.I);start,end=match.span(1)
        states={name:patient[:start]+value+patient[end:] for name,value in [('a',value_a),('b',value_b),('missing','NOT_RECORDED')]}
        candidates.append({'candidate_id':f'nonage:{gid}:{anchor["path_id"]}','source_id':cluster,'source_guideline_id':gid,
            'source_filename':source['filename'],'partition':['fit','validation','confirm'][len(candidates)%3],
            'source_condition':tree['description'],'source_tree':tree,'original_patient':patient,'patient_proposals':states,
            'proposed_measurement':measurement,'edit_span':[start,end],'original_measurement':match.group(1),
            'labels':{'a':None,'b':None,'missing':None},'ancestors_reachable':None,'status':'unadjudicated',
            'required_manual_work':['resolve all remaining qualitative/derived descriptions of changed measurement','check synchronized repeated measurements and comorbid causal constraints','check exact source predicate and ancestor reachability','two independent blinded labels plus third adjudicator for disagreements'],
            'warning':'numeric substitution suggestions may leave contradictions; do not use as gold or clinically coherent pairs before adjudication'})
        seen.add(cluster)
    return candidates,{'source_method':clusters['method'],'official_test_clusters_excluded':True,'pr40_clusters_excluded':True,
                       'previous_mechanism_and_clinical_clusters_excluded':True,'selected_sources':sorted(seen),'skipped':skipped,
                       'candidate_count':len(candidates),'physician_adjudicated_pairs':0,'labels_generated_by_model':False}

def validate_annotation(candidate,annotations):
    """Minimal input contract; does not turn machine output into physician gold."""
    reviews=[r for r in annotations if r.get('candidate_id')==candidate['candidate_id']]
    independent={r.get('reviewer_id') for r in reviews if r.get('role')=='independent_reviewer'}
    if None in independent or len(independent)<2:raise ValueError('two named independent reviewers are required')
    for r in reviews:
        if r.get('role') not in ('independent_reviewer','adjudicator'):raise ValueError('invalid reviewer role')
        if r.get('reviewer_type')!='human_clinician':raise ValueError('machine review is auxiliary and cannot satisfy physician adjudication')
        if set(r.get('labels',{}))!={'a','b','missing'} or any(v not in ('true','false','unknown') for v in r['labels'].values()):raise ValueError('three explicit labels are required')
        if not isinstance(r.get('ancestors_reachable'),bool) or not r.get('evidence'):raise ValueError('reachability and source evidence are required')
        if not r.get('corrected_patient_texts') or set(r['corrected_patient_texts'])!={'a','b','missing'}:raise ValueError('review all three final patient texts')
    independent_reviews=[r for r in reviews if r['role']=='independent_reviewer']
    disagree=any(r['labels']!=independent_reviews[0]['labels'] or r['corrected_patient_texts']!=independent_reviews[0]['corrected_patient_texts'] or r['ancestors_reachable']!=independent_reviews[0]['ancestors_reachable'] for r in independent_reviews[1:])
    adjudicators=[r for r in reviews if r['role']=='adjudicator' and r['reviewer_id'] not in independent]
    if disagree and not adjudicators:raise ValueError('disagreement requires an independent adjudicator')
    final=adjudicators[-1] if disagree else independent_reviews[0]
    if {final['labels']['a'],final['labels']['b']}!={'true','false'} or final['labels']['missing']!='unknown':raise ValueError('eligible triplet requires a/b true-false and missing unknown')
    return {'candidate_id':candidate['candidate_id'],'source_id':candidate['source_id'],'partition':candidate['partition'],'status':'physician_adjudicated',
            'labels':final['labels'],'patient_texts':final['corrected_patient_texts'],'ancestors_reachable':final['ancestors_reachable'],'evidence':final['evidence'],'reviewer_ids':sorted({r['reviewer_id'] for r in reviews}),
            'eligible_for_direction':final['ancestors_reachable'],'exclusion_reason':None if final['ancestors_reachable'] else 'ancestor_not_reachable'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default=str(BASE));p.add_argument('--root',default=str(ROOT));p.add_argument('--output',required=True);p.add_argument('--annotations');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);rows,audit=make_candidates(Path(args.base),Path(args.root))
    with (out/'candidate_pairs.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (out/'annotation_template.jsonl').open('x') as f:
        for row in rows:
            for reviewer in ['reviewer_1','reviewer_2']:
                f.write(json.dumps({'candidate_id':row['candidate_id'],'reviewer_id':reviewer,'reviewer_type':None,'role':'independent_reviewer','labels':{'a':None,'b':None,'missing':None},'ancestors_reachable':None,'evidence':None,'corrected_patient_texts':row['patient_proposals'],'blinded_to_model_results':True},ensure_ascii=False)+'\n')
    write_json(out/'source_manifest.json',audit)
    adjudicated_count=eligible_count=0
    if args.annotations:
        accepted=[];errors=[];annotations=read_jsonl(args.annotations)
        for row in rows:
            try:accepted.append(validate_annotation(row,annotations))
            except ValueError as exc:errors.append({'candidate_id':row['candidate_id'],'reason':str(exc)})
        with (out/'adjudicated_pairs.jsonl').open('x') as f:
            for row in accepted:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        adjudicated_count=len(accepted);eligible_count=sum(r['eligible_for_direction'] for r in accepted)
        write_json(out/'annotation_status.json',{'physician_adjudicated':len(accepted),'eligible_for_direction':sum(r['eligible_for_direction'] for r in accepted),'adjudicated_exclusions':[{'candidate_id':r['candidate_id'],'reason':r['exclusion_reason']} for r in accepted if not r['eligible_for_direction']],'pending':errors})
    print(json.dumps({'status':'complete','unadjudicated_candidates':len(rows),'source_clusters':len(audit['selected_sources']),'physician_adjudicated':adjudicated_count,'eligible_for_direction':eligible_count}))
if __name__=='__main__':main()
