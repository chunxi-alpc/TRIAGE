"""Reconcile the two reported source discrepancies. No scores, fitting, or label writes."""
import csv
import json
import os
import re
import time
import uuid
from decimal import Decimal
from pathlib import Path

ROOT = Path.home()/"aster-v3-review-followup-private/disputes"
HOME = Path.home()
MEGA = HOME/'TRIAGE/data/MEGA-CDP/train_val'
PATHS = HOME/'aster_cr_repair_v2_run_20261108/data/path_cases.jsonl'
V3 = HOME/'aster-v3-selective-20261124/reports/aster_v3_selective_20261124/s1'
AUDIT = HOME/'aster_cr_pair_dataset_20261111'
CASES = {'mega:train_val:1857:4': 'G3', 'mega:train_val:2744:10':'G1'}


def decoded(path):
    with path.open(encoding='utf-8-sig') as stream:
        for n, line in enumerate(stream, 1):
            yield n,json.loads(line)


def get_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def to_csv(path, fields, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        w=csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(records)


def original_question(group):
    return re.sub(r'（判断为[是否]）$','',group['rules'][0]['condition'])


def main():
    for relative in ('reviewer/guides','coordinator','shareable'):
        (ROOT/relative).mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    cpu=time.process_time()
    case_rows={r['case_id']:(n,r) for n,r in decoded(PATHS) if r['case_id'] in CASES}
    target_sources={r['guideline_id'] for _,r in case_rows.values()}
    raw_vignettes={(r['guideline_id'],int(r['path_id'])):(n,r) for n,r in decoded(MEGA/'vignette_cases.jsonl') if r['guideline_id'] in target_sources}
    paths={r['guideline_id']:(n,r) for n,r in decoded(MEGA/'case_paths.jsonl') if r['guideline_id'] in target_sources}
    old=list(get_table(V3/'first_errors.csv'))
    ids=ROOT/'coordinator/key.csv'
    prior={r['case_id']:r['blind_id'] for r in get_table(ids)} if ids.exists() else {}
    results=[]
    keys=[]
    cards=[]
    for case_id,gid in CASES.items():
        row,r=case_rows[case_id]
        case_ix=int(case_id.split(':')[-1])
        number,raw=raw_vignettes[r['guideline_id'],case_ix]
        path_no,p=paths[r['guideline_id']]
        patient=raw['text']
        assert patient==r['patient'] and raw['filename']==p['filename']==r['source_filename']
        questions={g['group_id']:original_question(g) for g in r['guideline']['groups']}
        q=questions[gid]
        raw_nodes=[(i,e) for i,e in enumerate(p['paths'][case_ix-1]) if e.get('condition')==q]
        assert len(raw_nodes)==1
        step,ref=raw_nodes[0]
        status=next(v['label'] for v in r['gold_path'] if v['gid']==gid)
        assert status==int(ref['decision_outcome']=='yes')
        old_rows=[o for o in old if o['case_id']==case_id and o['gid']==gid]
        assert len(old_rows)==4 and all(int(o['gold'])==status for o in old_rows)
        if gid=='G3':
            temporal=re.search(r"at the time.{0,30}considered",patient,re.I)
            begin,end=(int(old_rows[0][k]) for k in ('text_start','text_end'))
            finding=re.search(r"ratio.{0,15}?([01]\.\d+)",patient[begin:end],re.I)
            neg=re.search(r"not\s+(?:below|under)\s+([01]\.\d+)",patient)
            assert temporal and finding and neg and temporal.end()<=begin and (begin-temporal.end())<15
            value_span=(begin+finding.start(1),begin+finding.end(1))
            context_span=(temporal.start(),value_span[1])
            value=finding[1]
            unit='dimensionless'
            obj='TOFR'
            claim=True
            reason='当时点、测量对象和数值均明确；同句比较结论与数值判断一致。'
            question='请核对指南所指时间点与病例测量时间，并独立给出节点答案。病例和原始参考路径的冲突需由临床审核者裁决。'
        else:
            begin,end=(int(old_rows[0][k]) for k in ('text_start','text_end'))
            finding=re.search(r"(\d+(?:\.\d+)?)\s*mmol/L", patient[begin:end])
            neg=re.search(r"(?:未|不)(?:满足|符合|达到).{0,4}≥\s*(\d+)\s*mmol/L",patient)
            assert finding and neg and '汗' in patient[max(0,begin-28):begin]
            value_span=(begin+finding.start(1),begin+finding.end(1))
            context_span=(begin,end)
            value=finding[1]
            obj='sweat_chloride'
            unit='mmol/L'
            claim=False
            reason='同一检验的报告数值和紧接其后的文字判定给出了相反分支。'
            question='请核对检验值与病例文字判定，记录原文能否唯一支持该节点以及应由哪一侧更正。'
        threshold=re.search(r'≥\s*(\d+(?:\.\d+)?)',q)[1]
        assert Decimal(neg[1])==Decimal(threshold)
        val=Decimal(value); cut=Decimal(threshold)
        numeric=val>=cut
        results.append(dict(case_id=case_id,gid=gid,measurement=obj,unit=unit,comparison='>=',value=str(val),threshold=str(cut),
            difference=str(val-cut),numeric_branch=int(numeric),text_claim_branch=int(claim),
            upstream_reference_branch=status,working_reference_branch=status,source_case_path_index=case_ix-1,source_path_step_index=step,
            upstream_raw_node_index=ref['index'],working_copy_matches_source=True,patient_copy_matches_source=True,
            source_vignette_line_1based=number,source_case_paths_line_1based=path_no,working_path_cases_line_1based=row,
            value_span=list(value_span),time_or_measurement_context_span=list(context_span),comparison_claim_span=list(neg.span()),
            same_case_versions=len(old_rows),clinical_adjudication='pending',label_change=0,
            evidence=reason,adjudication_question=question))
        bid=prior.get(case_id) or uuid.uuid4().hex
        source_doc=Path(r['source_document'])
        guide_content=source_doc.read_text()
        if patient in guide_content or any(token in guide_content for token in ('gold_path','decision_outcome',case_id)):
            raise ValueError('Case-specific answer found in source document')
        guide_path=Path('guides')/(bid+'.md')
        blind_doc=ROOT/'reviewer'/guide_path
        if not blind_doc.exists():
            blind_doc.write_text(guide_content)
        blind_doc.chmod(0o400)
        cards.append(dict(guide_path=str(guide_path),blind_id=bid,condition=q,patient=patient,guideline_conditions=sorted(set(questions.values())),
                          guideline_definitions=r['definitions']))
        keys.append(dict(blind_id=bid,source_id=r['source_id'],case_id=case_id,gid=gid))
    assert len(cards)==2 and all(not x['guideline_definitions'] for x in cards)
    cards.sort(key=lambda c:c['blind_id'])
    keyset={c['blind_id'] for c in cards}
    assert len(keyset)==2
    with (ROOT/'reviewer/cards.jsonl').open('w') as f:
        for c in cards: f.write(json.dumps(c,ensure_ascii=False)+'\n')
    to_csv(ids,['blind_id','source_id','case_id','gid'],keys)
    forms=ROOT/'reviewer/responses.csv'
    if not forms.exists():
        to_csv(forms,['blind_id','decision','evidence_start','evidence_end','explanation','reviewer_id','cause'],
               [dict(blind_id=c['blind_id']) for c in cards])
    elif {r['blind_id'] for r in get_table(forms)}!=keyset:
        raise ValueError('Existing response form has different item IDs')
    human=[]
    for f in ('human_audit.csv','human_audit_historical.csv'):
        matches=[r for r in get_table(AUDIT/f) if r['case_id'] in CASES]
        human.append(dict(file=f,matching_cases=len(matches),signed_and_qualified_rows=sum(
                    bool(x['reviewer'] and x['clinical_qualification'] and x['review_date']) for x in matches)))
    out=dict(results=results,offsets='zero-based half-open Unicode code points in upstream patient text; JSONL source positions are one-based',
             inputs=dict(source_dataset='MEGA-CDP/train_val',source_vignettes='vignette_cases.jsonl',source_paths='case_paths.jsonl',
                         processed_cases=str(PATHS),v3_first_errors=str(V3/'first_errors.csv')),
             review_record_check=human,match_policy='match raw guideline, path identity, identical patient, exact question and structured source branch',
             clinical_adjudications=0,cases=2,versions=8,label_changes=0,gpu_hours=0,new_model_calls=0,
             cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.perf_counter()-started)
    (ROOT/'coordinator/disputes_detailed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    # Per-patient measurements stay next to the private coordinator materials.
    public=dict(out)
    public['results']=[{k:v for k,v in row.items() if k not in {'value','threshold','difference','unit'}}
                       for row in results]
    (ROOT/'shareable/disputes.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    os.umask(0o077)
    main()
