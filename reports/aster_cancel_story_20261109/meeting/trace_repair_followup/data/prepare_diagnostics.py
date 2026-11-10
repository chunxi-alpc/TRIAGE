"""Grounded numeric C/F and literal background P diagnostics; never score models."""
import sys,re,json,copy,random
from pathlib import Path
from collections import defaultdict,Counter
HERE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(HERE))
from data import BIG,OUT,SEED,rows,dump,emit,task_id,original_task,canonical_guide,prompt,condition
from repair_background import background_from_measurements
NUM=r'-?\d+(?:\.\d+)?'
OP=r'>=|<=|≥|≤|>|<|不低于|不超过|至少|超过|低于|高于|小于|大于|不少于|至多|greater than|less than|at least|at most|more than'
MAP={'≥':'>=','≤':'<=','不低于':'>=','不少于':'>=','至少':'>=','不超过':'<=','至多':'<=','超过':'>','高于':'>','大于':'>','低于':'<','小于':'<','greater than':'>','less than':'<','at least':'>=','at most':'<=','more than':'>'}
FIELDS=[('年龄',['年龄','现年','age','aged'],'岁'),('收缩压',['收缩压','systolic blood pressure','SBP'],'mmHg'),('舒张压',['舒张压','diastolic blood pressure','DBP'],'mmHg'),('心率',['心率','heart rate'],'次/分'),('呼吸频率',['呼吸频率','respiratory rate'],'次/分'),('体温',['体温','body temperature','temperature'],'℃'),('BMI',['BMI','body mass index'],'kg/m²'),('血糖',['血糖','blood glucose'],'mmol/L')]

def truth(op,x,t):return {'>=':x>=t,'<=':x<=t,'>':x>t,'<':x<t}[op]
def numeric_observation(q,patient):
 ms=list(re.finditer(r'(?P<op>'+OP+r')\s*(?P<number>'+NUM+r')',q,re.I))
 if len(ms)!=1:raise ValueError('condition_not_single_explicit_comparison')
 if re.search(r'且|或|并|\band\b|\bor\b',q,re.I):raise ValueError('compound_condition_requires_more_facts')
 m=ms[0];prefix=q[:m.start()];found=[]
 for field,aliases,unit in FIELDS:
  aliases=[a for a in aliases if a.lower() in prefix.lower()]
  if aliases:found.append((field,aliases,unit))
 if len(found)!=1:raise ValueError('no_unambiguous_supported_measurement_field')
 field,aliases,unit=found[0]
 if field=='年龄' and re.search(r'月|month',q,re.I):raise ValueError('age_unit_not_years')
 # Do not infer a second clinical predicate from a qualifying noun phrase.
 rest=prefix
 for alias in aliases:rest=re.sub(re.escape(alias),'',rest,flags=re.I)
 rest=re.sub(r'患者|患儿|儿童|病人|的|是否|当前|数值|水平|为|是|\b(?:Is|the|patient|patients|s|is|value|of|a|an)\b|[\W_]+','',rest,flags=re.I)
 if rest:raise ValueError('unparsed_condition_qualifier')
 full_aliases=next(a for f,a,u in FIELDS if f==field)
 pats=[]
 for alias in full_aliases:
  pat=re.escape(alias)+r'(?:为|是|测得为|测得|水平为|约为|约|达到|达|:|：|\s|\bis\b|\bof\b)*(?P<value>'+NUM+r')'
  pats.extend(re.finditer(pat,patient,re.I))
 unique={(x.start('value'),x.end('value')):x for x in pats}
 if len(unique)!=1:raise ValueError('measurement_absent_or_repeated')
 observed=next(iter(unique.values()));value=float(observed['value']);threshold=float(m['number']);op=MAP.get(m['op'].lower(),m['op'])
 return dict(field=field,aliases=full_aliases,unit=unit,match=m,observation=observed,value=value,threshold=threshold,op=op,label=int(truth(op,value,threshold)))

def changed_guide(guide,gid,old,new):
 g=copy.deepcopy(guide)
 for node in g['groups']:
  if node['group_id']==gid:
   for rule in node['rules']:rule['condition']=rule['condition'].replace(old,new,1)
 return g

def build(split):
 cases=[c for c in rows(BIG/f'cases_{split}.jsonl') if c['variant']=='original'];grouped=defaultdict(list)
 for c in cases:
  for gold in c['gold_path']:grouped[c['source_id']].append((c,gold))
 rng=random.Random(SEED+17);counts=Counter();reasons=Counter();source_counts=Counter();attempts=0
 with (BIG/f'tasks_diagnostic_{split}.jsonl').open('w') as tf,(BIG/f'pairs_diagnostic_{split}.jsonl').open('w') as pf,(OUT/f'diagnostic_edits_{split}.jsonl').open('w') as ef:
  for source in sorted(grouped):
   candidates=grouped[source];rng.shuffle(candidates);seen=set();selected=[]
   for c,gold in candidates:
    key=(c['guideline_id'],gold['gid'])
    if key in seen:continue
    seen.add(key);selected.append((c,gold))
    if len(selected)>=10:break
   for c,gold in selected:
    attempts+=1;gid=gold['gid'];label=gold['label'];q=next(n['condition'] for n in canonical_guide(c['guideline'])['nodes'] if n['id']==gid)
    allnodes=canonical_guide(c['guideline'])['nodes'];left=task_id(c,gid,'original');left_emitted=False
    def add(kind,patient=None,guide=None,newlabel=None,**metadata):
     nonlocal left_emitted
     if not left_emitted:
      emit(tf,original_task(c,gid,'original',c['guideline'],label));left_emitted=True
     variant='diagnostic_'+kind;right=task_id(c,gid,variant);nl=label if newlabel is None else int(newlabel)
     task=dict(id=right,case_id=c['case_id'],source_id=source,split=split,variant=variant,gid=gid,label=nl,kind=kind,prompt=prompt(guide if guide is not None else c['guideline'],c['definitions'],c['patient'] if patient is None else patient,gid))
     emit(tf,task);pair=dict(pair_id=c['case_id']+':'+gid+':'+kind,left=left,right=right,kind=kind,source_id=source,case_id=c['case_id'],gid=gid,split=split,left_label=label,right_label=nl,**metadata)
     emit(pf,pair);emit(ef,pair);counts[kind]+=1;source_counts[kind+':'+source]+=1
    # P protects all tree conditions, stronger than only checking the gold ancestors.
    g=dict(c,gid=gid,ancestors=[dict(gid=n['id'],condition=n['condition']) for n in allnodes if n['id']!=gid])
    bg=background_from_measurements(g)
    if bg:
     patient=c['patient'].replace(bg['old'],bg['new'])
     add('P',patient=patient,edit_locations=[dict(old=bg['old'],new=bg['new'])],changed_field=bg['field'],relevance_reason='Original directly located measurement rule, variable families absent from all tree conditions; same side of stated comparisons.',scope='node_pair_only_no_modified_path_claim')
    else:reasons['P:no_literal_unrelated_measurement']+=1
    try:obs=numeric_observation(q,c['patient'])
    except ValueError as e:
     reasons['CF:'+str(e)]+=1;continue
    if obs['label']!=label:
     reasons['CF:computed_original_disagrees_with_released_label']+=1;continue
    m=obs['match'];value=obs['value'];threshold=obs['threshold'];op=obs['op'];step=.1 if '.' in m['number'] else 1.
    new_threshold=next((v for v in [value-step,value,value+step] if v>=0 and truth(op,value,v)!=bool(label)),None)
    if new_threshold is not None:
     nq=q[:m.start('number')]+format(new_threshold,'.12g')+q[m.end('number'):]
     guide=changed_guide(c['guideline'],gid,q,nq)
     add('C',guide=guide,newlabel=int(truth(op,value,new_threshold)),field=obs['field'],facts={obs['field']:value},rule_before=dict(op=op,threshold=threshold),rule_after=dict(op=op,threshold=new_threshold),edit_locations=[dict(old=q,new=nq)],truth_method='single_explicit_comparison_and_located_patient_measurement',patient_unchanged=True)
    else:reasons['C:no_valid_flipping_threshold']+=1
    ancestors_gids={x['gid'] for x in c['gold_path'][:next(i for i,x in enumerate(c['gold_path']) if x['gid']==gid)]}
    ancestors_text='\n'.join(n['condition'] for n in allnodes if n['id'] in ancestors_gids)
    if any(a.lower() in ancestors_text.lower() for a in obs['aliases']):
     reasons['F:changed_measurement_may_change_ancestor']+=1;continue
    o=obs['observation'];sentence_start=max(c['patient'].rfind('。',0,o.start()),c['patient'].rfind('.',0,o.start()),c['patient'].rfind('\n',0,o.start()))+1
    sentence_end=min([i for i in [c['patient'].find('。',o.end()),c['patient'].find('\n',o.end())] if i>=0] or [len(c['patient'])])
    sentence=c['patient'][sentence_start:sentence_end]
    if re.search(r'高龄|低龄|偏高|偏低|升高|降低|过快|过慢|未达|不符合|不属于|不超过|未超过|超过|高于|低于|[<>≥≤]',sentence):
     reasons['F:qualitative_or_comparative_restatement']+=1;continue
    if obs['field']=='年龄':
     remainder=c['patient'][:o.start()]+c['patient'][o.end():]
     if re.search(r'高龄|年迈|幼儿|婴儿|儿童|成人|老年|月龄|\d+岁',remainder):
      reasons['F:age_category_or_repeated_age_requires_annotation']+=1;continue
    new_value=next((v for v in [threshold-step,threshold,threshold+step] if v>=0 and truth(op,v,threshold)!=bool(label)),None)
    if new_value is None:reasons['F:no_valid_flipping_value']+=1;continue
    patient=c['patient'][:o.start('value')]+format(new_value,'.12g')+c['patient'][o.end('value'):]
    add('F',patient=patient,newlabel=int(truth(op,new_value,threshold)),field=obs['field'],facts_before={obs['field']:value},facts_after={obs['field']:new_value},rule=dict(op=op,threshold=threshold),edit_locations=[dict(old=o['value'],new=format(new_value,'.12g'),start=o.start('value'),end=o.end('value'))],truth_method='single_explicit_comparison_and_located_patient_measurement',ancestor_check='measurement aliases absent from all original gold ancestors',scope='node_pair_only_no_modified_path_claim')
 dump(OUT/f'diagnostic_coverage_{split}.json',dict(split=split,attempted_gold_nodes=attempts,maximum_attempts_per_source=10,pairs=counts,pair_sources={kind:len({k.split(':',1)[1] for k in source_counts if k.startswith(kind+':')}) for kind in counts},unavailable_reasons=reasons,truth_policy='No model annotations or guessed labels. C/F only single explicit numerical comparison, uniquely located measurement and agreement with released original label. Unavailable cases remain unavailable.',P_policy='Literal existing unrelated measurement; aliases absent from entire decision tree; no arbitrary appended task marker.',scope='P/C/F are node-pair diagnostics, not new natural path benchmarks.',tasks=str(BIG/f'tasks_diagnostic_{split}.jsonl'),pairs_file=str(BIG/f'pairs_diagnostic_{split}.jsonl'),final_scoring_allowed=False if split=='final' else True))
 return counts

def main():
 for split in ['dev','final']:build(split)
if __name__=='__main__':main()
