"""Read source locally on the server, retain complete cases and add explicit corrections.
Gold observations are manually source-read or controlled interventions, never Qwen labels.
"""
import argparse,collections,hashlib,itertools,json,pathlib,re
import pandas as pd
from specs import RULES,CASES,FIELDS,CATEGORY,NAMES,UNITS
def sha(x):return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def split(s):return re.split(r'(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<=[.!?])\s+(?=[A-Z*])',s)
def atom_truth(c,v):
 if v is None:return 'U'
 t=c['value'];op=c['op']
 def cmp(a):return {'lt':lambda:a<t,'le':lambda:a<=t,'gt':lambda:a>t,'ge':lambda:a>=t,'eq':lambda:a==t}[op]()
 if not isinstance(v,dict):return 'T' if cmp(v) else 'F'
 cuts=sorted({-100.,float(t),*[float(b) for b in v.values()]});cuts.append(max(cuts)+100.)
 points=set(cuts)|{(a+b)/2 for a,b in zip(cuts,cuts[1:])}
 good=[p for p in points if all({'lt':p<b,'le':p<=b,'gt':p>b,'ge':p>=b}[o] for o,b in v.items())]
 assert good
 ys={cmp(p) for p in good};return 'T' if ys=={True} else 'F' if ys=={False} else 'U'
def gold(rule,vals):
 ys=[atom_truth(c,vals.get(c['field'])) for c in rule['conditions']]
 return ('F' if 'F' in ys else 'U' if 'U' in ys else 'T') if rule['mode']=='all' else ('T' if 'T' in ys else 'U' if 'U' in ys else 'F')
def fields(rule):return list(dict.fromkeys(c['field'] for c in rule['conditions']))
def point(rule,f,wanted):
 cs=[c for c in rule['conditions'] if c['field']==f]
 pool=['positive','negative'] if f in CATEGORY else sorted({0,1,*[c['value'] for c in cs],*[max(0,c['value']-1) for c in cs],*[c['value']+1 for c in cs]})
 for v in pool:
  if all(atom_truth(c,v)=='T' for c in cs)==wanted:return v
 raise ValueError((f,wanted))
def state(rule,wanted):
 fs=fields(rule);vals={f:point(rule,f,rule['mode']=='all') for f in fs}
 if (rule['mode']=='all' and wanted=='F') or (rule['mode']=='any' and wanted=='T'):vals[fs[0]]=point(rule,fs[0],wanted=='T')
 assert gold(rule,vals)==wanted
 return vals
def proofs(rule,vals,ev):
 label=gold(rule,vals)
 if label=='U':return [[]]
 fs=fields(rule)
 if (rule['mode']=='all' and label=='T') or (rule['mode']=='any' and label=='F'):
  return [sorted({i for group in groups for i in group}) for groups in itertools.product(*(ev[f] for f in fs))]
 return [group for c in rule['conditions'] if atom_truth(c,vals.get(c['field']))==label for group in ev[c['field']]]
def statement(f,v):
 return f'{NAMES[f]}: {v} {UNITS[f]}.'.replace(' .','.')
def build(source,out,prior):
 out.mkdir(parents=True,exist_ok=False);(out/'private').mkdir()
 df=pd.read_parquet(source/'data/train-00000-of-00001.parquet');old=json.loads((prior/'prior_rows.json').read_text());oldr=[json.loads(s) for s in (prior/'prior_rules.jsonl').read_text().splitlines()]
 oldnodes={r['source_node_sha256'] for r in oldr};norm=lambda r:json.dumps({'mode':r['mode'],'conditions':sorted(r['conditions'],key=lambda x:json.dumps(x,sort_keys=True)),'exceptions':r.get('exceptions',[])},sort_keys=True)
 oldsig={norm(r['schema_gold']) for r in oldr};inputs=[];labels=[];pairs=[];rm=[];review=[];new=[]
 for r in RULES:
  rule=r['schema_gold'];fs=fields(rule);node=df.iloc[r['source_row']].path[r['source_node_index']]
  assert sha(node) not in oldnodes and norm(rule) not in oldsig
  rm.append(r|{'source_node_sha256':sha(node)})
  for pos,(row,original,oev) in enumerate(CASES[r['rule_id']]):
   assert row not in old and row not in new
   profile=df.iloc[row].profile;sents=split(profile);assert ' '.join(sents)==re.sub(r'\s+',' ',profile).strip() or ''.join(sents).replace(' ','')==profile.replace(' ','')
   for f,v in original.items():assert (v is None and not oev[f]) or (v is not None and oev[f] and all(1<=i<=len(sents) for g in oev[f] for i in g))
   new.append(row);aid=f'C{len(pairs)+1:02}';native=gold(rule,original);variants=[('original',original,sents,oev,{})]
   full=state(rule,'T' if pos==0 else 'F');flipped=state(rule,'F' if native=='T' else 'T')
   missing=fs[pos%len(fs)];part={f:point(rule,f,rule['mode']=='all') for f in fs};part[missing]=None
   if pos==1 and len(fs)>1:part[next(f for f in fs if f!=missing)]=point(rule,next(f for f in fs if f!=missing),rule['mode']=='any')
   for kind,vals in [('counterfactual',flipped),('partial_missing',part),('multiple_values',full),('negation',full),('contradiction',{f:point(rule,f,rule['mode']=='all') for f in fs})]:
    vals=dict(vals);text=list(sents);ev={};extra={}
    text.append('Record correction: all earlier statements about the following items were copied in error and are withdrawn: '+ '; '.join(NAMES[f] for f in fs)+'. The corrected entries below replace those items for the exact assessment or historical event specified; all other history remains unchanged.')
    for f,v in list(vals.items()):
     start=len(text)+1;name=NAMES[f];u=UNITS[f];opp=point(rule,f,not all(atom_truth(c,v)=='T' for c in rule['conditions'] if c['field']==f)) if v is not None else None
     if kind=='contradiction' and f==missing:
      va,vb=point(rule,f,True),point(rule,f,False)
      text += [f'Corrected report A for the specified assessment/event: {name} is {va} {u}.',f'Corrected report B for the SAME assessment/event and SAME specimen: {name} is {vb} {u}. Both reports remain signed; neither is withdrawn or superseded.']
      vals[f]=None;ev[f]=[];extra={'conflict_field':f,'conflict_evidence_ids':[start,start+1],'conflicting_values':[va,vb]};continue
     if v is None:
      text.append(f'The corrected record has no verified result for {name}; the entry is unavailable, and the withdrawn value cannot be used.');ev[f]=[];continue
     if kind=='multiple_values':
      text.append(f'The worksheet includes {name} recorded as {opp} {u} for a DIFFERENT specimen/person; that entry does not belong to this patient. For this patient at the specified assessment/event, the verified {name} is {v} {u}; an unrelated laboratory value is 11.2 and the sample accession number is 150.')
     elif kind=='negation':
      if f in CATEGORY:
       text.append(f'It is not correct to describe {name} as {opp}. On reassessment of this patient, with an unequivocal result, the correct description is {v}.')
      else:text.append(f'The proposed value of {opp} {u} for {name} was rejected. The same measurement was verified, and its actual value is {v} {u}.')
     else:text.append('Verified corrected entry: '+statement(f,v))
     ev[f]=[[start]]
    if kind=='partial_missing':extra|={'partial_at_rule_level':len(fs)>1,'decisive_other_field':pos==1 and len(fs)>1,'missing_field':missing}
    variants.append((kind,vals,text,ev,extra))
   for kind,vals,text,ev,extra in variants:
    sid=aid+'_'+kind;label=gold(rule,vals);relevant=sorted({i for groups in ev.values() for group in groups for i in group}|set(extra.get('conflict_evidence_ids',[])))
    inputs.append({'sample_id':sid,'rule_id':r['rule_id'],'sentences':text})
    labels.append({'sample_id':sid,'rule_id':r['rule_id'],'source_row':row,'anchor_id':aid,'variant':kind,'label':label,'gold_observations':vals,'field_evidence_groups':ev,'acceptable_evidence_proofs':proofs(rule,vals,ev),'all_relevant_evidence_ids':relevant,'source_profile_sha256':sha(profile),'source_sentence_count':len(sents),'retained_source_sentence_ids':list(range(1,len(sents)+1)),'word_count':len(' '.join(text).split()),'input_sha256':sha(json.dumps(inputs[-1],sort_keys=True))}|extra)
   pairs.append({'pair_id':aid,'source_row':row,'rule_id':r['rule_id'],'original_id':aid+'_original','counterfactual_id':aid+'_counterfactual'})
   review.append({'rule_id':r['rule_id'],'source_row':row,'facts':original,'evidence':{f:[[sents[i-1] for i in g] for g in gs] for f,gs in oev.items()}})
 for name,items in [('private/inputs.jsonl',inputs),('private/rules.jsonl',rm),('labels.jsonl',labels),('pairs.jsonl',pairs),('rules_manifest.jsonl',rm)]:
  (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
 (out/'private/annotation_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
 def grams(s):
  ws=re.findall('[a-z]+',s.lower());return set(zip(ws,ws[1:],ws[2:]))
 og=[grams(df.iloc[i].profile) for i in old];ng=[grams(df.iloc[i].profile) for i in new]
 audit={'source_revision':'1460ec478280a7295040b92f7eb985327f77dcbe','source_parquet_sha256':sha((source/'data/train-00000-of-00001.parquet').read_bytes()),'previous_source_rows':old,'new_source_rows':new,'previous_rule_count':len(oldr),'new_rules':len(rm),'new_cases':len(new),'samples':len(labels),'row_overlap':len(set(old)&set(new)),'node_hash_overlap':0,'schema_signature_overlap':0,'profile_hash_overlap':len({sha(df.iloc[i].profile) for i in old}&{sha(df.iloc[i].profile) for i in new}),'max_trigram_jaccard_to_previous':max(len(a&b)/len(a|b) for a in ng for b in og),'labels':dict(collections.Counter(x['label'] for x in labels)),'native_full_cases':32,'controlled_addendum_cases':160,'all_original_sentences_retained':True,'partial_multifield':sum(x.get('partial_at_rule_level',False) for x in labels),'partial_decisive':sum(x.get('decisive_other_field',False) for x in labels),'annotation':'Codex source-read, no clinician review; source profiles synthetic; recommendations never used as gold','word_counts':{v:{'min':min(x['word_count'] for x in labels if x['variant']==v),'mean':sum(x['word_count'] for x in labels if x['variant']==v)/32,'max':max(x['word_count'] for x in labels if x['variant']==v)} for v in sorted({x['variant'] for x in labels})}}
 (out/'dataset_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ['new_rules','new_cases','samples','labels','max_trigram_jaccard_to_previous']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('data'));p.add_argument('--prior',type=pathlib.Path,default=pathlib.Path('prerequisites'));a=p.parse_args();build(a.source,a.out,a.prior)
