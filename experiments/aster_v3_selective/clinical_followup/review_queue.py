"""Build blinded clinician materials and align the released MEGA rationale to each vignette."""
import argparse
import csv
import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from decimal import Decimal
from difflib import SequenceMatcher
from pathlib import Path

HOME = Path.home()
ARCHIVE = HOME / 'aster_cr_repair_v2_run_20261108/data'
MEGA = HOME / 'TRIAGE/data/MEGA-CDP/train_val'
STUDY = HOME / 'aster-v3-selective-20261124/reports/aster_v3_selective_20261124/s1'
FIELDNAMES = ['blind_id','decision','evidence_start','evidence_end','explanation','reviewer_id','cause']
UNIT_TEXT = r'mmol\s*/\s*[lL]|[µμu]?mol\s*/\s*[lL]|(?:[µμu]|m|n)?g\s*/\s*(?:mL|dL|L)|mm\s*Hg|mm\s*H2O|cm\s*H2O|kg\s*/\s*m[²2]|years?(?:\s+old)?|months?|weeks?|days?|hours?|minutes?|seconds?|bpm|beats\s*/\s*min|kg|mm|cm|mL|mg|°C|%|岁|年|个月|周|天|小时|分钟|秒'
MEASURE = re.compile(r'(?<![\d.])(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>'+UNIT_TEXT+r')(?![a-z])', re.I)
NUM = re.compile(r'(?<![\d.])\d+(?:\.\d+)?(?![\d.])')
UNK = '仅有短字串或无逐字重合'
DOC_PART = re.compile(r'摘要|引言|病例介绍部分|临床检查部分|讨论部分|是否.*(?:报告中|文中)|\b(?:abstract|introduction|case report|manuscript)\b', re.I)


def load_lines(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, data, fields=None, mode='w'):
    with path.open(mode, newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(data[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def write_lines(path, records):
    with path.open('w') as f:
        for row in records:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')


def question(group):
    return re.sub(r'（判断为[是否]）$','',group['rules'][0]['condition'])


def condensed(text):
    # Removing spacing and punctuation permits deterministic character anchoring
    # across English/Chinese typography. Offsets always refer to the original.
    inds=[i for i,c in enumerate(text) if c.isalnum()]
    return ''.join(text[i].lower() for i in inds),inds


def anchor(patient, explanation):
    fragment=explanation.strip()
    exact=patient.find(fragment)
    if exact>=0:
        offset=explanation.index(fragment)
        return '整段发布依据逐字出现在病例',exact,exact+len(fragment),offset,offset+len(fragment)
    p,pi=condensed(patient); e,ei=condensed(explanation)
    if not p or not e: return UNK,'','','',''
    block=SequenceMatcher(None,p,e,autojunk=False).find_longest_match()
    if not block.size: return UNK,'','','',''
    a,b=pi[block.a],pi[block.a+block.size-1]+1
    c,d=ei[block.b],ei[block.b+block.size-1]+1
    n_cjk=len(re.findall(r'[\u3400-\u9fff]',patient[a:b]))
    n_words=len(re.findall(r'[A-Za-z]+',patient[a:b]))
    if block.size==len(e):
        label='整段发布依据规范化后匹配'
    elif n_cjk>=8 or n_words>=4:
        label='存在较长连续字串'
    else:
        label=UNK
    return label,a,b,c,d


def measurements(text):
    found=defaultdict(list)
    for m in MEASURE.finditer(text):
        u=re.sub(r'\s+','',m['unit'].lower()).replace('μ','u').replace('µ','u')
        if u in ('year','years','yearold','yearsold','岁'):u='age_or_duration_year'
        value=str(Decimal(m['number']).normalize())
        found[value,u].append(m.span())
    return found


def approved_fragment(reason):
    if 'anchor outside current condition' in reason:
        return '先前提取的规则文字超出当前条件'
    if 'JSON' in reason or "'source'" in reason:
        return '先前结构化输出解析问题'
    return '先前记录为条件提取问题'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=HOME/'aster-v3-review-followup-private/queue')
    parser.add_argument('--public',type=Path)
    args=parser.parse_args()
    os.umask(0o077)
    begin,cpu=time.perf_counter(),time.process_time()
    work=args.work
    reviewer,coord=work/'reviewer',work/'coordinator'
    public=args.public or work/'deliver'
    for path in (reviewer/'guidelines',coord,public):
        path.mkdir(parents=True,exist_ok=True)
    pending={}
    for r in read_csv(STUDY/'first_errors.csv'):
        key=(r['case_id'],r['gid'])
        if r['evidence_status']=='pending_review': pending[key]=r
    case_ids={k[0] for k in pending}
    cases={c['case_id']:c for c in load_lines(ARCHIVE/'path_cases.jsonl') if c['case_id'] in case_ids}
    gids={r['guideline_id'] for r in cases.values()}
    upstream={(r['guideline_id'],r['path_id']):r for r in load_lines(MEGA/'vignette_cases.jsonl') if r['guideline_id'] in gids}
    explanations={(r['guideline_id'],r['path_id']):r for r in load_lines(MEGA/'gt_paths.jsonl') if r['guideline_id'] in gids}
    trees={r['guideline_id']:r for r in load_lines(MEGA/'case_paths.jsonl') if r['guideline_id'] in gids}
    node_ids={n['group_id']:(n['case_id'],n['gid']) for n in load_lines(ARCHIVE/'nodes.jsonl') if (n['case_id'],n['gid']) in pending}
    rules=defaultdict(list)
    for r in load_lines(ARCHIVE/'rules.jsonl'):
        if r['group_id'] in node_ids:
            rules[node_ids[r['group_id']]].append(r)
    attempts=defaultdict(list)
    for r in load_lines(ARCHIVE/'extraction_review.jsonl'):
        if r['group_id'] in node_ids: attempts[node_ids[r['group_id']]].append(r)
    keyfile=coord/'key.csv'
    if keyfile.exists():
        ids={(r['case_id'],r['gid']):r['blind_id'] for r in read_csv(keyfile)}
        if set(ids)!=set(pending) or len(ids)!=len(set(ids.values())):
            raise ValueError('Existing blind batch differs from the selected cohort.')
    else:
        ids={k:uuid.uuid4().hex for k in pending}
        write_csv(keyfile,[dict(blind_id=i,source_id=pending[k]['source_id'],case_id=k[0],gid=k[1]) for k,i in sorted(ids.items())],mode='x')
    original_docs=sorted({Path(c['source_document']) for c in cases.values()})
    doc_refs={}
    doc_patients=defaultdict(set)
    for c in cases.values(): doc_patients[Path(c['source_document'])].add(c['patient'])
    for i,path in enumerate(original_docs,1):
        if path.parent.parent.resolve()!=(MEGA/'guideline_md').resolve() or path.parent.name!=path.stem:
            raise ValueError('Unrecognized original guideline location')
        data=path.read_text()
        if any(v.strip() and v in data for v in doc_patients[path]):
            raise ValueError('Original guideline contains a complete target vignette; omit document until reviewed.')
        fname=f'guidelines/doc_{i:03}.txt'
        dest=reviewer/fname
        if dest.is_file():
            if dest.read_text()!=data:
                raise ValueError('Previously issued guideline reference changed')
        else:
            dest.write_text(data)
        doc_refs[path]=fname
    cards,details,public_rows=[],[],[]
    for (cid,gid), baseline in sorted(pending.items()):
        key=cid,gid
        c=cases[cid]; raw=c['patient']; blind=ids[key]
        if c['definitions']: raise ValueError('Definitions require schema inspection before entering blinded materials.')
        guide={g['group_id']:question(g) for g in c['guideline']['groups']}
        q=guide[gid]
        guide_id=c['guideline_id']; path_id=int(cid.rsplit(':',1)[1]); data_key=guide_id,path_id
        original=upstream[data_key]; rationale=explanations[data_key]
        table=trees[guide_id]
        if not (original['filename']==c['source_filename']==rationale['filename']==table['filename'] and
                original['text']==raw and Path(c['source_document']).stem==c['source_filename']):
            raise ValueError('Source, patient, or file identity conflict at '+cid)
        steps=table['paths'][path_id-1]
        reference=json.loads(rationale['output_text'])
        if len(steps)!=len(reference) or len(steps)!=len(c['gold_path'])+1:
            raise ValueError('Path/rationale step mismatch at '+cid)
        for j,step in enumerate(c['gold_path']):
            original_step=steps[j]
            if original_step['condition']!=guide[step['gid']] or original_step['decision_outcome'].lower()!=('yes' if step['label'] else 'no'):
                raise ValueError('Released step and archived step conflict at '+cid)
        j=next(i for i,s in enumerate(c['gold_path']) if s['gid']==gid)
        exp=reference[j]
        if not isinstance(exp,str) or not exp.strip(): raise ValueError('Missing released rationale')
        kind,a,b,x,y=anchor(raw,exp)
        pm,rm,qm=map(measurements,(raw,exp,q))
        shared=sorted(pm.keys() & rm.keys())
        qu={k[1] for k in qm}
        q_units={k[1] for k in shared if k[1] in qu}
        # A value absent as a numeral may be paraphrased or spelled out.
        numerals=lambda t:{Decimal(m.group()) for m in NUM.finditer(t)}
        nonmatched=numerals(exp)-numerals(raw)-numerals(q)
        evidence=[]
        for rule in rules[key]:
            if rule['original_condition']!=q or rule['case_id']!=cid:continue
            for name,obs in rule['observations'].items():
                for event in obs.get('evidence',[]):
                    lo,hi=event.get('start'),event.get('end')
                    if type(lo) is int and type(hi) is int and 0<=lo<hi<=len(raw) and event.get('text')==raw[lo:hi]:
                        evidence.append((lo,hi,rule['method']))
        attempts_status=sorted({('先前提取结果分歧' if z['state']=='extraction_disagreement' else approved_fragment(str(z.get('reason','')))) for z in attempts[key]})
        row=dict(source_id=c['source_id'],case_id=cid,gid=gid,
                 released_path_step=j+1,original_patient_exact=True,original_condition_exact=True,original_branch_exact=True,
                 release_rationale_available=True,original_guideline_available=True,definition_count=len(c['definitions']),
                 original_text_match=kind,patient_start=a,patient_end=b,rationale_start=x,rationale_end=y,
                 shared_value_unit_pairs=len(shared),shared_units_also_in_question=len(q_units),
                 rationale_other_numeric_values=len(nonmatched),archived_fact_spans=len(set((l,h) for l,h,_ in evidence)),
                 prior_extraction_state='|'.join(attempts_status),references_to_report_sections=bool(DOC_PART.search(q)),
                 cause='待临床盲审')
        public_rows.append(row)
        cards.append(dict(blind_id=blind,condition=q,patient=raw,guideline_conditions=sorted(set(guide.values())),
                          guideline_definitions=[],guideline_document=doc_refs[Path(c['source_document'])]))
        details.append(dict(blind_id=blind,**row,released_rationale=exp,archived_grounded_fact_offsets=evidence,
                            shared_value_unit_occurrences=[dict(value=k[0],unit=k[1],patient_offsets=pm[k],rationale_offsets=rm[k]) for k in shared]))
    # Keep each source and all of its cases in one batch. Four groups are
    # an operational convenience, independent of clinical labels or scores.
    frequencies=Counter(r['source_id'] for r in public_rows)
    batch_sizes=[0]*4
    assigned={}
    for source,size in sorted(frequencies.items(), key=lambda entry:(-entry[1],entry[0])):
        idx=min(range(4),key=lambda i:(batch_sizes[i],i))
        assigned[source]=idx+1
        batch_sizes[idx]+=size
    assignments=sorted((dict(blind_id=blind,batch=assigned[pending[k]['source_id']]) for k,blind in ids.items()),
                       key=lambda row:(row['batch'],row['blind_id']))
    batchfile=reviewer/'batches.csv'
    if batchfile.exists():
        if read_csv(batchfile)!=[dict(blind_id=r['blind_id'],batch=str(r['batch'])) for r in assignments]:
            raise ValueError('Issued source batch assignments changed')
    else:write_csv(batchfile,assignments,mode='x')
    cards.sort(key=lambda r:r['blind_id'])
    cards_path=reviewer/'cards.jsonl'
    if cards_path.exists():
        if list(load_lines(cards_path))!=cards: raise ValueError('Previously issued blind cards differ')
    else:write_lines(cards_path,cards)
    reply=reviewer/'responses.csv'
    if not reply.exists():write_csv(reply,[dict(blind_id=c['blind_id']) for c in cards],FIELDNAMES,mode='x')
    elif [r['blind_id'] for r in read_csv(reply)]!=[c['blind_id'] for c in cards]:
        raise ValueError('Existing response template has a different batch')
    write_lines(coord/'released_evidence.jsonl',details)
    write_csv(public/'evidence_index.csv',public_rows)
    group_rows=[]
    for src in ['ALL']+sorted({r['source_id'] for r in public_rows}):
        rr=[r for r in public_rows if src=='ALL' or r['source_id']==src]
        counts=Counter(r['original_text_match'] for r in rr)
        measures=dict(original_rationale=len(rr),shared_value_and_unit=sum(r['shared_value_unit_pairs']>0 for r in rr),
                      shared_unit_also_in_predicate=sum(r['shared_units_also_in_question']>0 for r in rr),
                      unmatched_other_numeral=sum(r['rationale_other_numeric_values']>0 for r in rr),
                      indexed_archived_fact_spans=sum(r['archived_fact_spans']>0 for r in rr),prior_extraction_record=sum(bool(r['prior_extraction_state']) for r in rr),
                      report_section_reference=sum(r['references_to_report_sections'] for r in rr))
        for typ,n in measures.items():group_rows.append(dict(source=src,measure=typ,n=n,denominator=len(rr)))
        for typ in sorted({r['original_text_match'] for r in public_rows}):
            group_rows.append(dict(source=src,measure=typ,n=counts[typ],denominator=len(rr)))
    write_csv(public/'by_source.csv',group_rows)
    out=dict(records=len(public_rows),sources=len({r['source_id'] for r in public_rows}),cases=len(cases),guide_documents=len(original_docs),
             unique_questions=len({r['condition'] for r in cards}),blinded_cards=len(cards),new_human_reviews=0,
             batches=[dict(batch=i+1,n=n,sources=sum(v==i+1 for v in assigned.values())) for i,n in enumerate(batch_sizes)],
             metrics={r['measure']:dict(count=r['n'],positive_sources=sum(t['source']!='ALL' and t['measure']==r['measure'] and t['n']>0 for t in group_rows)) for r in group_rows if r['source']=='ALL'},
             data_sources=['MEGA-CDP/train_val/vignette_cases.jsonl','MEGA-CDP/train_val/case_paths.jsonl','MEGA-CDP/train_val/gt_paths.jsonl','MEGA-CDP/train_val/guideline_md','aster_cr_repair_v2_run_20261108/data','aster_v3_selective_20261124/s1/first_errors.csv'],
             cpu_seconds=time.process_time()-cpu,elapsed_seconds=time.perf_counter()-begin,gpu_hours=0,model_forwards=0,full_path_replays=0)
    (public/'queue_status.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
