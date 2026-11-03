#!/usr/bin/env python3
"""Post-hoc final-answer format compatibility; raw and original strict outputs are immutable."""
import ast,collections,json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(name):return [json.loads(s) for s in (ROOT/name).read_text().splitlines() if s.strip()]
def save(name,rows):(ROOT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
tree=ast.parse((ROOT/'src/run_free.py').read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='parse_answer')
ns={'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original_strict_parser>','exec'),ns);strict_parse=ns['parse_answer']

def clean_line(line):
    if re.fullmatch(r'\s*```[\w-]*\s*',line):return ''
    line=line.replace('**','').replace('`','').strip()
    line=re.sub(r'^#{1,6}\s*','',line)
    def decorative(char):return char.isspace() or char in '#>*•' or unicodedata.category(char).startswith('S') or unicodedata.category(char) in ('Cf','Mn')
    while line and (decorative(line[0]) or line[0]=='-'):line=line[1:]
    while line and decorative(line[-1]):line=line[:-1]
    return line.strip()

def format_parse(raw,options):
    lines=[clean_line(s) for s in raw.splitlines()]
    lines=[s for s in lines if s and not re.fullmatch(r'[-=_]+',s)]
    if not lines:return 'INVALID','empty',[]
    marker=re.compile(r'\b(?:(?:the\s+)?(?:(?:final|correct)\s+)?answer|correct\s+action)\s*(?:is\b\s*|[:=]\s*|$)',re.I)
    last=None
    for index,line in enumerate(lines):
        for match in marker.finditer(line):last=(index,match.end())
    if last is None:
        pred,reason=strict_parse(lines[-1],options)
        return pred,'final_line_'+reason,[lines[-1]]
    index,offset=last;block=[lines[index][offset:].strip()]+lines[index+1:]
    block=[x for x in block if x]
    candidates=[]
    for line in block:
        pred,reason=strict_parse(line,options)
        if reason=='label_action_mismatch':return 'INVALID','final_block_label_action_mismatch',block
        if pred!='INVALID':candidates.append(pred)
    unique=set(candidates)
    if len(unique)==1:return next(iter(unique)),'explicit_final_answer_block',block
    return 'INVALID','conflicting_final_answers' if len(unique)>1 else 'no_explicit_parseable_final_answer',block

cases={r['item_id']:r for r in read('data/cases.jsonl')}
original=read('results/free_predictions.jsonl');updated=[];changes=[]
for row in original:
    pred,reason,block=format_parse(row['raw_output'],cases[row['item_id']]['options'])
    valid=pred in cases[row['item_id']]['options']
    if row['valid_answer']:assert pred==row['prediction'],'Compatibility changed an originally valid answer'
    action=cases[row['item_id']]['options'].get(pred)
    new=dict(row,prediction=pred,parse_reason=reason,valid_answer=valid,correct=pred==cases[row['item_id']]['gold'],
             new_action_selected=(cases[row['item_id']]['new_action'] in action) if valid else None,
             old_action_selected=(cases[row['item_id']]['options'][cases[row['item_id']]['old_action_label']] in action) if valid else None,
             scoring='free_generation_final_answer_block_posthoc_format_compatible',parser_version='posthoc_final_answer_block_v1',
             strict_prediction=row['prediction'],strict_parse_reason=row['parse_reason'],strict_valid_answer=row['valid_answer'],
             normalized_answer_block=block,posthoc_format_sensitivity=True)
    updated.append(new)
    if not row['valid_answer'] and valid:
        changes.append({'item_id':row['item_id'],'method':row['method'],'retained_answer_block':block,
                        'parsed_label':pred,'parsed_action':cases[row['item_id']]['options'][pred],'parse_reason':reason,
                        'raw_output_unchanged':True,'label_action_consistency_enforced':True})
save('results/free_format_normalized_predictions.jsonl',updated)
save('audit/free_format_normalization_changes.jsonl',changes)
full=read('results/c_full_predictions.jsonl');assert len(full)==20 and all(r['alpha']==0 and r['signed_gate']==0 for r in full)
summary={}
for name,source in [('strict',original),('normalized',updated)]:
    bykey={(r['method'],r['item_id']):r for r in source};derived=[]
    for gate in full:
        baseline=bykey[('A',gate['item_id'])]
        copy=dict(baseline,method='C_V6',alpha=0.0,signed_gate=0.0,executed=False,reused_from='A',reused_completion=True,
                  actual_additional_generation_seconds=0.0,gate_source='results/c_full_predictions.jsonl',
                  gate_seconds=gate['gate_seconds'],gate_applicability_state=gate.get('module1_state'),
                  module1_state=gate.get('module1_state'),module1_evidence_status=gate.get('module1_evidence_status'),
                  c_v6_generation_provenance='Exact reuse of A: independently executed module one returned zero gate for this item')
        derived.append(copy)
    filename='results/c_full_free_predictions.jsonl' if name=='strict' else 'results/c_full_free_format_normalized_predictions.jsonl'
    save(filename,derived)
    subset_ids={r['item_id'] for r in full};summary[name]={}
    for method in ['A','B','C_reduced','C_V6']:
        rr=derived if method=='C_V6' else [r for r in source if r['method']==method and r['item_id'] in subset_ids]
        summary[name][method]={'n':len(rr),'correct':sum(r['correct'] for r in rr),'accuracy':sum(r['correct'] for r in rr)/len(rr),
                               'valid':sum(r['valid_answer'] for r in rr),'coverage':sum(r['valid_answer'] for r in rr)/len(rr),'families':len({r['rule_family_id'] for r in rr})}
(ROOT/'results/c_full_free_subset_summary.json').write_text(json.dumps({'descriptive_only':True,'subset_size':20,'families':2,'no_additional_generation':True,'results':summary},ensure_ascii=False,indent=2)+'\n')
counts={}
for split in ['test','transfer']:
 counts[split]={}
 for method in ['A','B','C_reduced','D_adapt']:
  rr=[r for r in updated if r['method']==method and r['split']==split]
  counts[split][method]={'n':len(rr),'correct':sum(r['correct'] for r in rr),'valid':sum(r['valid_answer'] for r in rr),'invalid':sum(not r['valid_answer'] for r in rr)}
(ROOT/'audit/free_format_normalization_summary.json').write_text(json.dumps({'posthoc':True,'parser_version':'posthoc_final_answer_block_v1','raw_generations_unchanged':True,'originally_valid_answers_unchanged':True,'invalid_to_valid_rows':len(changes),'normalization':'Markdown, code fences, decorative symbols; last explicit Answer/Final answer/Correct answer or correct action block, otherwise an exact final line; no gold access; conflicting letters/actions remain INVALID','summary':counts},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'normalized':counts,'c_v6_subset':summary,'format_recoveries':len(changes)},ensure_ascii=False))
