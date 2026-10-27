"""Freeze repaired development inputs; never load predictions or test cases."""
from pathlib import Path
import collections,copy,datetime,hashlib,json,re
D=Path(__file__).resolve().parent;OLD=D.parent/'aster_prelim_20261027'
SOURCE=Path('/home/gcx/aster_dynamic_review_20261025/dev.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def split_action(a):
 # Exact transformations for the finite guideline action vocabulary, not LLM guesses.
 q=[];s=a
 if ' if ' in s:
  s,c=s.split(' if ',1);q.append(c)
 elif ' (if ' in s:
  s,c=s.split(' (if ',1);q.append(c.rstrip(')'))
 if ' in selected patients who are low risk' in s:
  s=s.replace(' in selected patients who are low risk','');q.append('Selected patients are low risk.')
 if ' based on other risk factors' in s:
  s=s.replace(' based on other risk factors','');q.append('Other risk factors inform the treatment decision.')
 if ' (not previously used)' in s:
  s=s.replace(' (not previously used)','');q.append('The regimen has not previously been used.')
 if ' previously administered' in s:
  s=s.replace(' previously administered','');q.append('The regimen was previously administered.')
 marker=', and consider comprehensive regional nodal irradiation (RNI) in patients with '
 if marker in s:
  base,guard=s.split(marker,1);s='Component 1: '+base+'; component 2: consider comprehensive regional nodal irradiation (RNI)';q.append('Component 2 applies in patients with '+guard)
 marker=' (use of comprehensive RNI with or without intentional inclusion of axilla at the discretion of the radiation oncologist)'
 if marker in s:
  s=s.replace(marker,'; optional comprehensive RNI with or without intentional inclusion of axilla');q.append('The optional component is at the discretion of the radiation oncologist.')
 return s.strip(),q

# Only target-treatment mentions in the interrogative clause are hidden. The entire
# source patient text is retained, including Given/Considering factual sentences.
TERMS=[r'boost(?: to (?:the )?tumor bed)?',r'whole breast radiotherapy',r'whole breast radiation therapy',r'whole breast radiation',r'breast irradiation',r'comprehensive regional nodal irradiation',r'regional nodal irradiation',r'radiation therapy',r'radiation treatment',r'radiotherapy',r'radiation',r'systemic chemotherapy',r'systemic treatment',r'systemic therapy',r'adjuvant endocrine therapy',r'endocrine therapy',r'hematopoietic (?:cell|stem cell) transplant(?:ation)?',r'bone marrow transplant(?:ation)?',r're-excision',r'chemotherapy',r'\b(?:HCT|APBI|PBI|EBRT|RNI|RT)\b']
pattern=re.compile(r'(?<!\w)(?:'+'|'.join(TERMS)+r')(?!\w)',re.I)
def hide_question_treatments(text):
 shadow=list(text)
 for m in re.finditer(r'\b(?:Mrs|Mr|Ms|Dr)\.|(?<=\d)\.(?=\d)',text):
  for k in range(m.start(),m.end()):
   if shadow[k]=='.':shadow[k]=' '
 shadow=''.join(shadow);ed=shadow.rfind('?')
 if ed<0:return text,[]
 st=max(shadow.rfind('.',0,ed),shadow.rfind('!',0,ed),shadow.rfind('?',0,ed))+1
 clause=text[st:ed+1]
 ask=re.search(r'\b(?:what|which|how|should|do you|would you|is adjuvant)\b',clause,re.I)
 if ask is None:return text,[]
 st+=ask.start()
 # Preserve post-question contextual clauses (past treatments are patient facts).
 tail=re.search(r'\b(?:given|considering|following|after|before|because)\b',text[st:ed],re.I)
 target_end=st+tail.start() if tail else ed
 spans=[{'start':st+m.start(),'end':st+m.end(),'text':m[0]} for m in pattern.finditer(text[st:target_end])]
 result=text
 for x in reversed(spans):result=result[:x['start']]+'[treatment]'+result[x['end']:]
 return result,spans

def main():
 target=D/'private/cases.jsonl';assert not target.exists(),'Frozen data already exist; create a new revision instead of overwriting'
 source=[json.loads(s) for s in SOURCE.read_text().splitlines()];src={(r['id'],r['variant']):r for r in source}
 original=[json.loads(s) for s in (OLD/'private/cases.jsonl').read_text().splitlines()]
 split=json.loads((OLD/'results/split.json').read_text());dev=set(split['dev_trees']);test=set(split['test_trees']);assert not dev&test
 assert all(r['split']=='dev' and r['tree'] in dev and r['tree'] not in test for r in source+original)
 groups=collections.defaultdict(dict)
 for r in original:groups[r['id']][r['variant']]=r
 excluded=[];rows=[];audit=[];action_map={}
 for cid,vs in groups.items():
  n=vs['NC']
  if {q['condition'] for q in n['rules']}=={'Negative margins','Negative margins for invasive disease'}:
   excluded.append({'id':cid,'tree':n['tree'],'reason':'overlapping_entry_conditions_without_disambiguating_context'});continue
  full=src[cid,'NC']['patient'];blind,redactions=hide_question_treatments(full)
  canonical={q['bid']:split_action(q['action'])[1] for q in n['rules']}
  # Conditions/qualifications stay attached to the branch; the intervention swaps
  # only treatment content. We test entry-node membership, not downstream eligibility.
  for variant,old in vs.items():
   r=copy.deepcopy(old);r.update(patient=full,blind_patient=blind,source_question='',source_patient_sha256=hashlib.sha256(full.encode()).hexdigest(),repair_version='v2_facts_and_roles',branch_label_scope='source entry-node membership; execution qualifications shown in both visibility conditions')
   for q in r['rules']:
    raw=q['action'];clean,qual=split_action(raw);action_map[raw]={'action':clean,'execution_qualifications':qual};q['source_action']=raw;q['action']=clean;q['execution_qualifications']=canonical[q['bid']]
   options=[]
   for raw in old['options']:
    clean,qual=split_action(raw);action_map[raw]={'action':clean,'execution_qualifications':qual}
    if clean not in options:options.append(clean)
   r['options']=options
   assert all(q['action'] in options for q in r['rules'])
   assert r['rules'][int(r['gold'])-1]['bid']==r['gold']
   rows.append(r)
  audit.append({'id':cid,'tree':n['tree'],'question_redactions':redactions,'full_patient_preserved':True,'qualification_by_branch':canonical,'gold_source_path':n['source_gold_rule'],'branch_conditions':[q['condition'] for q in n['rules']]})
 target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 (D/'private/repair_trace.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
 dump(D/'results/split.json',split);dump(D/'results/exclusions.json',excluded);dump(D/'results/action_role_map.json',action_map)
 (D/'results/selection.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['id','tree','split','variant','gold','source_gold_rule','source_decoy_rule','source']})+'\n' for r in rows))
 public=[{'id':r['id'],'tree':r['tree'],'question_redaction_count':len(r['question_redactions']),'full_patient_preserved':True,'qualifications_visible_in_both':True} for r in audit];dump(D/'results/repair_trace.json',public)
 manifest={'status':'prepared','version':'repaired_v2','frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':dict(collections.Counter(r['variant'] for r in rows)),'trees':sorted({r['tree'] for r in rows}),'base_count':len(audit),'excluded_base_count':len(excluded),'tree_counts':dict(collections.Counter(r['tree'] for r in rows if r['variant']=='NC')),'source':str(SOURCE),'source_sha256':sha(SOURCE),'prior_cases_sha256':sha(OLD/'private/cases.jsonl'),'cases_sha256':sha(target),'script_sha256':sha(__file__),'role_map_sha256':sha(D/'results/action_role_map.json'),'test_case_files_opened':[],'test_tree_intersection':[],'question_redacted_cases':sum(bool(r['question_redactions']) for r in audit),'qualification_cases':sum(any(r['qualification_by_branch'].values()) for r in audit),'CS_requested':200,'CS_actual':sum(r['variant']=='CS' for r in rows),'selection_based_on_model_outputs':False,'clinical_human_review':False,'scope':'entry-node branch judgment; downstream execution qualifications retained as separate visible context, not converted into new branch labels'}
 dump(D/'results/data_audit.json',manifest);print(json.dumps(manifest))
if __name__=='__main__':main()
