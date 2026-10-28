"""Construct exact-name counterfactual drafts from the development-only lexical scan."""
import sys,json,re,hashlib,collections
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
sys.path.insert(0,str(D))
import build_candidates as b
from scan_exact_treatments_v2 import PATTERNS,terms
ACTION=re.compile(r'recommend|treat|administ|\buse[ds]?\b|consider|undergo|prefer|indicat|therapy|therapies|offer|prescri|receiv|perform|suggest|建议|推荐|使用|给予|选用|采用|治疗|予以|可行|进行|应行',re.I)
COND=re.compile(r'\b(if|unless|when|provided that)\b|如果|若|当|除非',re.I)
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def tokens(s):return set(re.findall(r'[a-z]{3,}|[\u4e00-\u9fff]',s.lower()))
def locate(guide,term,conclusion,mappings):
 options=[];pattern=PATTERNS[term];tt=tokens(conclusion)
 for m in pattern.finditer(guide):
  lo=max(guide.rfind('\n',0,m.start()),guide.rfind('.',0,m.start()),guide.rfind('。',0,m.start()),guide.rfind(';',0,m.start()),guide.rfind('；',0,m.start()))+1
  ends=[guide.find(x,m.end()) for x in ['\n','.','。',';','；']]
  hi=min([x for x in ends if x>=0] or [len(guide)])
  sentence=guide[lo:hi]
  if not ACTION.search(sentence):continue
  before=guide[lo:m.start()]
  cond=list(COND.finditer(before))
  # Do not edit a name inside an unfinished conditional clause.
  if cond and not re.search(r'[,，:：]',before[cond[-1].end():]):continue
  sim=len(tt&tokens(sentence))/max(1,len(tt|tokens(sentence)))
  support=max([len(tokens(sentence)&tokens(q))/max(1,len(tokens(q))) for q in mappings] or [0.])
  options.append({'start':m.start(),'end':m.end(),'text':m.group(),'context_start':lo,'context_end':hi,'context':sentence,'alignment_score':sim+support,'source_quote_support':support})
 return sorted(options,key=lambda x:(-x['alignment_score'],x['start']))
def main():
 scan=[json.loads(l) for l in (B/'private/p6_lexical_scan_v2/candidates.jsonl').read_text().splitlines()]
 wanted={tuple(x['source_case_key']):x for x in scan};files={k[2] for k in wanted}
 vignettes={b.key(r):r for r in b.read(b.ROOT/'train_val/vignette_cases.jsonl') if b.key(r) in wanted}
 trees={r['filename']:json.loads(r['text']) for r in b.read(b.ROOT/'train_val/decision_trees.jsonl') if r['filename'] in files}
 paths={r['filename']:r['paths'] for r in b.read(b.ROOT/'train_val/case_paths.jsonl') if r['filename'] in files}
 refs={b.key(r):json.loads(r['output_text']) for r in b.read(b.ROOT/'train_val/gt_paths.jsonl') if b.key(r) in wanted}
 nodes={f:{int(n['index']):(n,p,e) for n,p,e in b.walk(t['decision_tree'])} for f,t in trees.items()}
 out=B/'private/p6_auto_exact_v2';out.mkdir(exist_ok=False);(out/'guidelines').mkdir();(out/'decoy_guidelines').mkdir()
 records=[];failures=[]
 for key,sc in sorted(wanted.items(),key=lambda kv:b.value_digest([20261028,list(kv[0])])):
  r=vignettes[key];f=r['filename'];guide=b.safe_guideline_path(b.ROOT/'train_val/guideline_md',f).read_text()
  path=paths[f][int(key[1])-1];n,parent,edge=nodes[f][int(path[-1]['index'])]
  other=next(v for v in parent.values() if isinstance(v,dict) and 'index' in v and v['index']!=n['index'])
  mapping=trees[f].get('node_guideline_mapping',{})
  def qs(node):
   v=mapping.get(str(node['index']),[]);return [v] if isinstance(v,str) else [s for s in v if isinstance(s,str)]
  protected=set().union(*(terms(x.get('condition','')) for x in path))
  pairs=[]
  for a in sc['applicable_terms']:
   if a in protected:continue
   for bb in sc['sibling_terms']:
    if bb in protected:continue
    aa=locate(guide,a,n['conclusion'],qs(n));bs=locate(guide,bb,other['conclusion'],qs(other))
    for x in aa[:3]:
     for y in bs[:3]:
      if max(x['start'],y['start'])<min(x['end'],y['end']):continue
      pairs.append((x['alignment_score']+y['alignment_score'],a,bb,x,y))
  if not pairs:failures.append({'source_case_key':list(key),'reason':'no_action_context_match_or_treatment_in_condition'});continue
  _,a,bb,x,y=max(pairs,key=lambda z:(z[0],z[1],z[2],-z[3]['start'],-z[4]['start']))
  original_ref=refs[key];edited_ref=[];refchanges=[]
  for i,s in enumerate(original_ref):
   new,count=PATTERNS[a].subn(lambda _:y['text'],s);edited_ref.append(new)
   if count:refchanges.append({'step':i,'before':s,'after':new,'count':count})
  if not refchanges:failures.append({'source_case_key':list(key),'reason':'no_reference_action_match'});continue
  edits=[{'start':x['start'],'end':x['end'],'before':x['text'],'after':y['text'],'branch':edge},{'start':y['start'],'end':y['end'],'before':y['text'],'after':x['text'],'branch':'no' if edge=='yes' else 'yes'}]
  edited=guide
  for e in sorted(edits,key=lambda z:z['start'],reverse=True):
   assert guide[e['start']:e['end']]==e['before'];edited=edited[:e['start']]+e['after']+edited[e['end']:]
  pid=b.value_digest(['P6-auto-exact',list(key),a,bb])[:20]
  (out/'guidelines'/(f+'.md')).write_text(guide);(out/'decoy_guidelines'/(pid+'.md')).write_text(edited)
  record={'pair_id':pid,'source_case_key':list(key),'guideline_id':sc['guideline_id'],'source_split':'train_val/development','language':r.get('language','en'),'patient_facts':r['text'],'question':'Follow the supplied full guideline and generate the complete decision path.','native_guideline_file':'guidelines/'+f+'.md','decoy_guideline_file':'decoy_guidelines/'+pid+'.md','native_guideline_sha256':sha(guide),'decoy_guideline_sha256':sha(edited),'native_reference_path':original_ref,'decoy_reference_path':edited_ref,'source_decision_path':path,'target_condition':parent['description'],'target_branch':edge,'native_action':n['conclusion'],'competing_action':other['conclusion'],'native_action_term':x['text'],'decoy_action_term':y['text'],'guide_edits':edits,'reference_edits':refchanges,'source_contexts':[x,y],'decision_nodes':sc['decision_nodes'],'validation':{'patient_facts_unchanged':True,'question_unchanged':True,'outside_selected_terms_byte_identical':True,'selected_terms_not_in_tree_conditions':True,'treatment_lexicon_only':True,'source_action_context_match':True,'clinical_semantic_validity':'NOT_HUMAN_VERIFIED','automatic_only':True}}
  records.append(record)
 (out/'pairs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
 result={'status':'AUTOMATIC_CONSTRUCTION_ONLY_NOT_FROZEN','candidates':len(scan),'constructed_pairs':len(records),'guidelines':len({x['guideline_id'] for x in records}),'complex_pairs':sum(x['decision_nodes']>2 for x in records),'failures':failures,'test_files_read':False,'method_generation_inputs':0,'human_review_required_for_pilot':False,'clinical_semantic_validity':'UNVERIFIED','private_package':str(out)}
 (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='failures'}))
if __name__=='__main__':main()
