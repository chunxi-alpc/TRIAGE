"""Clinical vignette minimal pairs, deterministic age rules, and source audit."""
import argparse,copy,json,random,re,unicodedata
from pathlib import Path
from collections import Counter,defaultdict
from difflib import SequenceMatcher
import torch
from transformers import AutoTokenizer
from src.aster.manuscript_runtime import PromptBuilder,aligned_triplet
ROOT=Path('/home/gcx/TRIAGE/data/MEGA-CDP')
BASE=Path('/home/gcx/aster_manuscript_20261030/clinical01')
MODEL=BASE/'models/Qwen3-8B-2602814'
REV='26028140be3ee69b82b1d1450179ab71bb1121b9'
# Explicit author-checked root predicates, bounded to current age in years.
SPECS={13:('le',18),134:('range',2,5),178:('range',25,84),212:('range',45,74),259:('le',16),301:('lt',6),342:('ge',45),400:('lt',10),403:('ge',65),438:('ge',80),453:('ge',18),505:('range',2,17),564:('ge',6),604:('ge',16),665:('ge',12),670:('range',50,74),711:('range',1,18),786:('ge',50),807:('ge',19),851:('lt',18),869:('lt',6),916:('ge',18),995:('gt',75),997:('le',2),1001:('lt',18),1015:('range',2,17),1109:('gt',60),1153:('lt',18),1205:('gt',65),1224:('gt',70),1308:('gt',18),1310:('ge',60),1313:('ge',18),1320:('ge',12),1350:('range',50,80),1409:('gt',80),1424:('ge',18),1507:('ge',18),1546:('ge',13),1581:('gt',65),1589:('ge',65),1626:('ge',18),1664:('ge',18),1678:('lt',18),1859:('ge',60),2040:('ge',50),2054:('ge',18),2064:('ge',80),2068:('ge',18),2192:('ge',6),2209:('ge',6),2409:('lt',18),2531:('range',45,50),2564:('le',1),2719:('ge',75)}
def read(p):return [json.loads(s) for s in Path(p).open() if s.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def jsonl(p,rows):
 with Path(p).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def norm(t):return re.sub(r'\s+','',unicodedata.normalize('NFKC',str(t))).casefold()
def tree_guideline(tree):
 nodes=[]
 def visit(n,parent=None,branch=None):
  i=len(nodes);nodes.append((n,parent,branch))
  if any(isinstance(n.get(b),dict) for b in ['yes','no']):
   if not all(isinstance(n.get(b),dict) for b in ['yes','no']):raise ValueError('nonbinary tree')
   visit(n['yes'],i,'Y');visit(n['no'],i,'N')
 visit(tree);loc={id(n):i for i,(n,_,_) in enumerate(nodes)}
 groups=[];mapping={}
 for i,(n,parent,branch) in enumerate(nodes):
  original=str(n['index']);mapping[original]={'node':i,'leaf':not any(isinstance(n.get(b),dict) for b in ['yes','no']),'description':n.get('description','')}
  if not any(isinstance(n.get(b),dict) for b in ['yes','no']):continue
  ancestors=[];cur=i
  while nodes[cur][1] is not None:
   pi=nodes[cur][1];br=nodes[cur][2];pn=nodes[pi][0]
   ancestors.append(pn['description']+('（判断为是）' if br=='Y' else '（判断为否）'));cur=pi
  group={'group_id':f'G{i}','mutually_exclusive':True,'ancestor_conditions':list(reversed(ancestors)),'rules':[]}
  if parent is not None:group.update(parent_group_id=f'G{parent}',parent_rule_id=f'R{parent}{branch}')
  for b,B in [('yes','Y'),('no','N')]:
   child=n[b];j=loc[id(child)];leaf=not any(isinstance(child.get(k),dict) for k in ['yes','no'])
   action=f'L{j}：'+child.get('conclusion',child.get('description','')) if leaf else f'继续评估规则组 G{j}。'
   group['rules'].append({'rule_id':f'R{i}{B}','condition':n['description']+('（判断为是）' if B=='Y' else '（判断为否）'),'action':action})
  groups.append(group)
 if not groups:raise ValueError('no decision groups')
 return {'groups':groups},mapping

def source_audit():
 records=[]
 for split in ['train_val','test']:
  for r in read(ROOT/split/'decision_trees.jsonl'):
   p=ROOT/split/'guideline_md'/r['filename']/(r['filename']+'.md')
   text=p.read_text(errors='replace') if p.exists() else ''
   lines=[re.sub(r'^#+\s*','',s).strip() for s in text.splitlines() if len(s.strip())>20]
   title=next((s for s in lines[:12] if len(s)<300),' '.join(lines[:3])[:300])
   title=norm(re.sub(r'(19|20)\d\d','',title))
   doi=re.findall(r'10\.\d{4,9}/[A-Za-z0-9._;()/:+-]+',text[:2500])
   records.append({'key':split+':'+str(r['guideline_id']),'split':split,'guideline_id':r['guideline_id'],'filename':r['filename'],'title':title,'doi':doi,'normalized_content':norm(text),'text':text,'row':r})
 parent={r['key']:r['key'] for r in records};reasons=[]
 def find(k):
  while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
  return k
 def union(a,b,why):
  if find(a)!=find(b):parent[find(b)]=find(a);reasons.append({'a':a,'b':b,'basis':why})
 seen={}
 for r in records:
  for key in [('filename',r['filename']),('content',r['normalized_content'])]+[('doi',norm(x)) for x in r['doi']]:
   if not key[1]:continue
   if key in seen:union(r['key'],seen[key],key[0])
   else:seen[key]=r['key']
 valid=[r for r in records if len(r['title'])>25]
 for i,r in enumerate(valid):
  for s in valid[:i]:
   if abs(len(r['title'])-len(s['title']))>30:continue
   if SequenceMatcher(None,r['title'],s['title'],autojunk=False).ratio()>=.92:union(r['key'],s['key'],'normalized title near-duplicate')
 pr40=json.loads((BASE/'pr40_exposure.json').read_text());exposed=set(pr40['all_tree_ids'])
 old=json.loads(Path('/home/gcx/aster_manuscript_20261030/run03/data/data_audit.json').read_text());old_ids={int(x['guideline_id']) for x in old['source_map']}
 for r in records:r['cluster']=find(r['key'])
 test_clusters={r['cluster'] for r in records if r['split']=='test'}
 pr40_clusters={r['cluster'] for r in records if r['filename'] in exposed}
 old_clusters={r['cluster'] for r in records if r['split']=='train_val' and r['guideline_id'] in old_ids}
 out={r['key']:{k:r[k] for k in ['split','guideline_id','filename','title','doi','cluster']} for r in records}
 write(BASE/'source_clusters.json',{'method':'normalized full guideline content, original filename, front-matter DOI and near-duplicate title; no file hashes','records':out,'merges':reasons,'pr40_tree_ids':pr40,'pr40_clusters':sorted(pr40_clusters),'previous_mechanism_clusters':sorted(old_clusters),'official_test_clusters':sorted(test_clusters),'cross_corpus_policy':'MedGUIDE 17 NCCN cancer families are grouped across all numbered pages; NCCN/oncology source documents excluded from clinical direction development; source-heldout transfer, not a claim of independent external provenance'})
 return records,test_clusters,pr40_clusters,old_clusters

AGE=re.compile(r'(?P<num>\d{1,3}(?:\.\d+)?)\s*(?:(?:[-– ]?(?:years?|yrs?)[-– ]old)|(?:周岁|岁))',re.I)
def truth(v,spec):
 if v is None:return None
 kind,a,*rest=spec
 return {'ge':lambda:v>=a,'gt':lambda:v>a,'le':lambda:v<=a,'lt':lambda:v<a,'range':lambda:a<=v<=rest[0]}[kind]()
def clean_anchor(patient):
 matches=list(AGE.finditer(patient))
 if not matches or matches[0].start()>140:raise ValueError('no explicit current-age anchor near start')
 vals={float(m.group('num')) for m in matches}
 if len(vals)!=1:raise ValueError('multiple age values or family/history age references')
 age=vals.pop()
 if age<0:raise ValueError('negative age')
 body=AGE.sub('',patient)
 body=re.sub(r'\b(elderly|adult|adolescent|pediatric|paediatric|young|middle-aged|older|aged)\b','',body,flags=re.I)
 body=re.sub(r'\b(child|boy|girl|infant|baby|teenager)\b','patient',body,flags=re.I)
 body=re.sub(r'老年|中年|青年|成年|儿童|患儿|婴幼儿|婴儿','患者',body)
 body=re.sub(r'\bAn (?=man|woman|male|female|patient)', 'A ',body,flags=re.I)
 body=re.sub(r'[,，]{2,}','，',body);body=re.sub(r'[ \t]+',' ',body).strip()
 return age,body

def clinical_validator(row):
 spec=row['clinical_verification']['age_predicate'];values={}
 for state,p in row['states'].items():
  match=re.search(r'\nAge at current encounter: (\d+(?:\.\d+)?) years\.\s*$',p)
  values[state]=float(match.group(1)) if match else None
  body=p.split('\nAge at current encounter:')[0]
  if AGE.search(body):raise ValueError('unsynchronized age evidence remains')
 expected=row['clinical_verification']['expected_truth']
 actual={s:truth(v,spec) for s,v in values.items()}
 if actual!=expected:raise ValueError('independent clinical age predicate failed')
 return actual

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
 records,testclusters,pr40clusters,oldclusters=source_audit()
 cases=read(ROOT/'train_val/vignette_cases.jsonl');byid=defaultdict(list)
 for case in cases:byid[int(case['guideline_id'])].append(case)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);builder=PromptBuilder(tok,chat_template_kwargs={'enable_thinking':False})
 candidates=[];excluded=[];eligible=[]
 for r in records:
  if r['split']!='train_val' or int(r['guideline_id']) not in SPECS:continue
  if r['cluster'] in testclusters or r['cluster'] in pr40clusters:excluded.append({'guideline_id':r['guideline_id'],'reason':'official-test or PR40 exposed source cluster'});continue
  if re.search(r'NCCN|cancer|carcinoma|oncolog|neoplas|肿瘤|癌|白血病',r['text'][:3500],re.I):excluded.append({'guideline_id':r['guideline_id'],'reason':'oncology/NCCN source reserved for cross-corpus audit'});continue
  eligible.append(r)
 rng=random.Random(20261030);rng.shuffle(eligible)
 used=set();accepted_sources=[]
 for source in eligible:
  if source['cluster'] in used:continue
  gid=int(source['guideline_id']);spec=SPECS[gid]
  try:tree=json.loads(source['row']['text'])['decision_tree'];guide,mapping=tree_guideline(tree)
  except Exception as e:excluded.append({'guideline_id':gid,'reason':str(e)});continue
  success=None
  for anchor in byid[gid]:
   try:
    age,body=clean_anchor(anchor['text'])
    # Closest opposite status at integer boundaries, preserving all other facts.
    kind,cut,*rest=spec
    pool=sorted(set([age,cut-1,cut,cut+1]+([rest[0]-1,rest[0],rest[0]+1] if rest else [])))
    positive=min((v for v in pool if v>=0 and truth(v,spec)),key=lambda v:abs(v-age))
    negative=min((v for v in pool if v>=0 and not truth(v,spec)),key=lambda v:abs(v-age))
    agefmt=lambda v:format(v,'g')
    patients={'positive':body+'\nAge at current encounter: '+agefmt(positive)+' years.',
              'negative':body+'\nAge at current encounter: '+agefmt(negative)+' years.'}
    for unknown in ['not recorded this visit.','not available this visit.','not documented this visit.','not recorded at admission.','not available in this record.','not recorded for this encounter.','not documented in this record.','not recorded.','not known.','not available.']:
     row={'kind':'clinical_counterfactual_vignette','pair_id':f'clinical:{gid}:{anchor["path_id"]}:Y','source_id':source['cluster'],'split':'pending','guideline':guide,
      'target_rule_id':'R0Y','question':'请依据完整指南给出患者的临床决策路径及最终处置。最后输出 PATH: 经过的规则编号；FINAL: 终点处置的L编号。',
      'states':{**patients,'unknown':body+'\nAge at current encounter: '+unknown},
      'verification':{'verifier':'independent_age_rule_parser_v1','reference':f'MEGA-CDP/{gid}/root; original vignette with age-only counterfactual edits',
       'positive_rule_id':'R0Y','negative_rule_id':'R0N','unknown_label':'NONE','ancestors_reachable':True,'independent_of_classifier':True},
      'clinical_verification':{'type':'rule-verified clinical counterfactual, not clinician-adjudicated','age_predicate':list(spec),'source_condition':tree['description'],'original_age':age,
       'original_case_id':f'train_val:{gid}:{anchor["path_id"]}','expected_truth':{'positive':True,'negative':False,'unknown':None},
       'normalization':'remove synchronized explicit age and derived age-category terms; preserve other clinical facts','validator_rule_version':'age-predicate-v1'},
      'source_guideline_id':gid,'source_filename':source['filename'],'original_patient':anchor['text'],'tree_mapping':mapping,'source_tree':tree}
     try:
      clinical_validator(row);aligned_triplet(builder,row)
      if max(builder.prepare(guide,v,row['question'],online=False).input_ids.shape[1] for v in row['states'].values())>40960:raise ValueError('context exceeds backbone maximum')
     except ValueError:continue
     success=row;break
    if success:break
   except (ValueError,StopIteration):continue
  if success is None:excluded.append({'guideline_id':gid,'reason':'no natural-age triplet passing independent facts and token alignment'});continue
  candidates.append(success);used.add(source['cluster']);accepted_sources.append(source)
  if len(candidates)==30:break
 write(out/'construction_audit.json',{'accepted_sources':len(candidates),'eligible_sources':len(eligible),'exclusions':excluded})
 if len(candidates)<5:raise RuntimeError('insufficient source clusters for five disjoint data partitions')
 # Source assignment fixed before model verification. Known exposure never enters new confirmation.
 clean=[r for r in candidates if r['source_id'] not in oldclusters];known=[r for r in candidates if r['source_id'] in oldclusters]
 n=len(candidates);confirm_n=max(1,n//7);confirm=clean[-confirm_n:];rest=[r for r in candidates if r not in confirm]
 sizes={'calibration':max(1,n//7),'selection':max(1,n//5),'development':max(1,n//7)}
 assignments={r['source_id']:'confirm' for r in confirm};cursor=0
 for split,count in sizes.items():
  for r in rest[cursor:cursor+count]:assignments[r['source_id']]=split
  cursor+=count
 for r in rest[cursor:]:assignments[r['source_id']]='fit'
 final=[]
 for row in candidates:
  row['split']=assignments[row['source_id']];clinical_validator(row);final.append(row)
  reverse=copy.deepcopy(row);reverse['pair_id']=row['pair_id'][:-1]+'N';reverse['target_rule_id']='R0N'
  reverse['states']['positive'],reverse['states']['negative']=row['states']['negative'],row['states']['positive']
  reverse['verification'].update(positive_rule_id='R0N',negative_rule_id='R0Y')
  reverse['clinical_verification']['expected_truth']={'positive':False,'negative':True,'unknown':None};clinical_validator(reverse);final.append(reverse)
 jsonl(out/'candidate_pairs.jsonl',final)
 write(out/'data_audit.json',{'candidate_age_source_specs':len(SPECS),'eligible_sources_before_alignment':len(eligible),'accepted_source_clusters':len(candidates),
  'submitted_triplets':len(final),'source_splits':dict(Counter(assignments.values())),'triplet_splits':dict(Counter(r['split'] for r in final)),'exclusions':excluded,
  'clinical_origin':'original MEGA-CDP clinical vignettes with single age counterfactual and missing age; no Boolean patient codes','independent_validator':'clinical_validator, age-predicate-v1',
  'clinician_review':False,'LLM_consistency_review':'pending independent Llama; authoritative numerical labels are deterministic','model_revision':REV,
  'prior_confirm_sources_excluded':True,'previous_synthetic_exposure_allowed_in_confirmation':False,'model_predictions_used_to_construct_labels':False})
 write(out/'protocol_preverification.json',{'seed':20261030,'rank_candidates':[1,2,4,8],'top_k':4,'gain_candidates':[.25,1.,4.],
  'rank_selection':'selection all-difference energy minus rank/128','gain_selection':'development final leaf/path accuracy, tie smaller gain','temperature':'independent calibration NLL',
  'probability_grid':[0.,.7,.9],'margin_grid':[0.,.1],'evaluation':'all original clinical cases, no gold correctness routing filter',
  'primary_metrics':['final_leaf_id_accuracy','path_rule_id_f1','exact_path'],'official_PCC':'separate Qwen-judged adapted official scorer; not GPT clinical main-table score',
  'max_new_tokens':256,'sampling':'greedy','clinical_population':'MEGA-CDP original vignettes; data are benchmark-generated clinical scenarios, not prospective patient records'})
 print(json.dumps({'accepted_sources':len(candidates),'triplets':len(final),'splits':dict(Counter(assignments.values()))}))
if __name__=='__main__':main()

