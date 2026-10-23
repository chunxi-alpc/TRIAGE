"""Independent branch/action mechanism dataset; no existing data modified."""
import json,re,hashlib,collections
from pathlib import Path
BASE=Path('/home/gcx/aster_validation_20261022/private');OUT=BASE/'run500/mechanism';OUT.mkdir(exist_ok=True)
# Human-readable review policy, reviewed by two AI agents; not expert clinical labels.
def condition_type(c):
 low=c.lower()
 if re.search(r'transplant candidate|resectab|rt (?:not )?possible|cytoreduction predicted|good ps|poor ps|declining ps',low):return 'judgment','feasibility/prediction or qualitative status'
 if re.search(r'rt (?:not )?required',low):return 'treatment_judgment','treatment-necessity judgment; separate from main judgment category'
 if re.search(r'planned|observe|consider resection|therapy|thyroidectomy|monitoring',low) and 'from completion of primary therapy' not in low:return 'ambiguous','plan/action or mixed action/condition'
 if re.search(r'yes to all|no to at least one|oligometastatic|lung only|^negative$',low):return 'ambiguous','missing referent or potentially overlapping classes'
 if re.search(r'[<>≤≥]|ps \d|molecular diagnostic|prior wbrt|margins|metastases \(m[01]\)|diagnosis:|csf|cerebrospinal|morphology',low):return 'objective_candidate','explicit measurement/history/report required; not clinical expert verification'
 return 'ambiguous','insufficient evidence for an unambiguous type'
def parse_rules(r):
 text=r['prompt_text'];end=text.index('Patient:\n');starts=[m.start() for m in re.finditer('IF the clinical sequence is: ',text[:end])];rules=[]
 for i,start in enumerate(starts):
  chunk=text[start:starts[i+1] if i+1<len(starts) else end].strip();c,a=chunk.split('\nTHEN the next step is: ',1);rules.append({'condition':c.removeprefix('IF the clinical sequence is: '),'action':a[:-1] if a.endswith('.') else a,'original_start':start})
 return rules
cs_counts=collections.Counter();label_table={};coverage={}
for split,file in [('test',BASE/'run500/all.jsonl'),('dev',BASE/'dev.jsonl')]:
 source=list(map(json.loads,file.read_text().splitlines()));groups=collections.defaultdict(dict)
 for r in source:groups[r['id']][r['variant'][0]]=r
 result=[]
 for id,versions in groups.items():
  a=versions['A'];rules=parse_rules(a);app=next(i for i,x in enumerate(rules) if x['original_start']==a['applicable_char_span'][0]);sib=next(i for i,x in enumerate(rules) if x['original_start']==a['sibling_char_span'][0]);conds=list(dict.fromkeys(r['condition'] for r in rules));branches=[{'condition':c,'rule_indices':[i for i,r in enumerate(rules) if r['condition']==c]} for c in conds];appb=conds.index(rules[app]['condition']);sibb=conds.index(rules[sib]['condition']);assert appb!=sibb
  typ,reason=condition_type(a['condition_text']);label_table[a['condition_text']]={'type':typ,'reason':reason,'review':'AI-reviewed policy and pair review; not clinical-expert gold'}
  variant_rules={v:parse_rules(versions[v]) for v in 'ABC'};d=[dict(r) for r in rules];d[app]['action'],d[sib]['action']=d[sib]['action'],d[app]['action'];variant_rules['D']=d
  gold={v:versions[v]['gold_answer'] for v in 'ABC'};gold['D']=a['original_sibling_action_label'];gold_branch={v:appb for v in 'ABCD'}
  cs={'eligible':False,'reason':'no verified complementary scalar pair'}
  # Strict CS: fixed 6->12 month threshold. Require unique old/new action branches.
  c1=a['path'][-2];c2=a['sibling_path'][-2]
  if {c1,c2}=={'<6 mo from completion of primary therapy','≥6 mo from completion of primary therapy'}:
   if any(len(branches[b]['rule_indices'])!=1 for b in (appb,sibb)):cs['reason']='same threshold condition has multiple action leaves; unique new action unverifiable'
   else:
    patient=a['closed_book_prompt'].split('\n\nWhat is')[0]
    matches=re.findall(r'(\d+(?:\.\d+)?) months? after completing (?:his |her |the )?primary therapy',patient,re.I)
    if len(matches)!=1:cs['reason']='no unique explicit recurrence interval with verified event anchor'
    else:
     x=float(matches[0]);oldlow=x<6;newlow=x<12
     if oldlow==newlow:cs['reason']='fixed 6-to-12 threshold does not change branch'
     elif oldlow!=(c1.startswith('<')):cs['reason']='patient value disagrees with source branch'
     else:
      cr=[dict(r) for r in rules]
      for r in cr:r['condition']=r['condition'].replace('<6 mo from completion','<12 mo from completion').replace('≥6 mo from completion','≥12 mo from completion')
      variant_rules['CS']=cr;gold['CS']=a['original_sibling_action_label'];gold_branch['CS']=sibb;cs={'eligible':True,'value_months':x,'old_threshold':6,'new_threshold':12,'old_branch':appb,'new_branch':sibb,'mutually_exclusive':True,'action_source':'unchanged source sibling action','synthetic_threshold_change':True}
  # Same prespecified doubling 6->12, millimetres, only solitary pure ground-glass pair.
  if {c1,c2}=={'<6 mm','≥6 mm'}:
   patient=a['closed_book_prompt'].split('\n\nWhat is')[0]
   hits=re.findall(r'(?:single |lesion measuring )(\d+(?:\.\d+)?)[ -]mm',patient,re.I)
   if len(hits)!=1 or re.search(r'several|multiple nodules',patient,re.I) or 'Solitary pure ground-glass nodules' not in a['path']:cs={'eligible':False,'reason':'nodule measurement/type/singularity not unambiguous'}
   elif any(len(branches[b]['rule_indices'])!=1 for b in (appb,sibb)):cs={'eligible':False,'reason':'multiple action leaves for nodule threshold branch'}
   else:
    x=float(hits[0]);oldlow=x<6;newlow=x<12
    if oldlow==newlow:cs={'eligible':False,'reason':'fixed threshold doubling does not change branch'}
    elif oldlow!=(c1.startswith('<')):cs={'eligible':False,'reason':'nodule size disagrees with source branch'}
    else:
     cr=[dict(r) for r in rules]
     for i in [app,sib]:cr[i]['condition']=cr[i]['condition'].replace(' -> <6 mm',' -> <12 mm').replace(' -> ≥6 mm',' -> ≥12 mm')
     assert rules[app]['condition'].rsplit(' -> ',1)[0]==rules[sib]['condition'].rsplit(' -> ',1)[0]
     variant_rules['CS']=cr;gold['CS']=a['original_sibling_action_label'];gold_branch['CS']=sibb;cs={'eligible':True,'value_mm':x,'old_threshold':6,'new_threshold':12,'old_branch':appb,'new_branch':sibb,'mutually_exclusive':True,'action_source':'unchanged published sibling action','synthetic_threshold_change':True,'evidence':hits[0]+' mm solitary pure ground-glass nodule','review':'AI source-profile and rule review; not a real clinical threshold update'}
  if split=='test':cs_counts[cs.get('reason','eligible')]+=1
  oldaction=a['options'][a['original_answer']];dold=[j for j,b in enumerate(branches) if any(d[i]['action']==oldaction for i in b['rule_indices'])]
  record={'id':id,'split':split,'disease':a['disease'],'options':a['options'],'closed_book_prompt':a['closed_book_prompt'],'patient':a['closed_book_prompt'].split('\n\nWhat is the appropriate next step?')[0],'original_answer':a['original_answer'],'app_rule':app,'sib_rule':sib,'app_branch':appb,'sib_branch':sibb,'branches':branches,'variant_rules':variant_rules,'gold_answer':gold,'gold_branch':gold_branch,'type':typ,'condition_text':a['condition_text'],'cs_validation':cs,'D_original_action_branches':dold,'single_action_gold_branch':len(branches[appb]['rule_indices'])==1,'source_variants':versions}
  result.append(record)
 target=OUT/f'{split}.jsonl';assert not target.exists();target.write_text(''.join(json.dumps(r)+'\n' for r in result));coverage[split]={'cases':len(result),'D_cases':len(result),'CS_cases':sum(r['cs_validation']['eligible'] for r in result),'single_action_gold_branch':sum(r['single_action_gold_branch'] for r in result),'types':dict(collections.Counter(r['type'] for r in result)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 assert split!='test' or len(result)==500
(OUT/'condition_labels.json').write_text(json.dumps(label_table,indent=2));(OUT/'manifest.json').write_text(json.dumps({'coverage':coverage,'CS_exclusions':dict(cs_counts),'branch_gold':'equivalence class of identical full condition paths','CS_policy':'Fixed threshold doubling 6-to-12 months or mm; explicit anchored patient scalar, complementary pair, unique action per branch; no per-case threshold tuning','label_review':'AI review, not clinical-expert review'},indent=2));print(json.dumps(coverage));print(dict(cs_counts))
