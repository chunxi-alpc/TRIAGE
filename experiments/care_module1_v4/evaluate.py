"""Offline metrics from fixed reference labels; no model calls or source prose needed."""
import argparse,collections,csv,json,pathlib
import numpy as np
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def macro(g,p):
 return float(np.mean([2*np.sum((g==c)&(p==c))/(np.sum(g==c)+np.sum(p==c)) if np.sum(g==c)+np.sum(p==c) else 0 for c in ['T','F','U']]))
def main(root):
 labels=read(root/'labels.jsonl');pr=read(root/'model/predictions.jsonl');pairs=read(root/'pairs.jsonl');pb={x['sample_id']:x for x in pr};lb={x['sample_id']:x for x in labels};assert set(pb)==set(lb) and len(pb)==len(pr)
 methods=['direct','structured_no_guard','structured'];g=np.array([x['label'] for x in labels]);arrays={};eh={};out={'status':'executed_frozen_holdout','n':len(labels),'methods':{}}
 def evidence_hit(x,m):
  ids=set(pb[x['sample_id']][m]['evidence_sentence_ids'])
  if x.get('conflict_evidence_ids'):return ids==set(x['conflict_evidence_ids'])
  if x['label']=='U':return not ids
  return any(set(p)<=ids for p in x['acceptable_evidence_proofs']) and ids<=set(x['all_relevant_evidence_ids'])
 stage={s:sum(x['seconds'] for x in json.loads((root/f'model/{s}_timing.json').read_text())) for s in ['direct','facts','rule_extraction','verification']}
 for m in methods:
  p=np.array([pb[x['sample_id']][m]['label'] for x in labels]);arrays[m]=p;hits=np.array([evidence_hit(x,m) for x in labels]);eh[m]=hits
  known=np.array([x['label']!='U' for x in labels]);total_ids=sum(len(pb[x['sample_id']][m]['evidence_sentence_ids']) for x in labels);supported_ids=sum(len(set(pb[x['sample_id']][m]['evidence_sentence_ids'])&set(x['all_relevant_evidence_ids'])) for x in labels)
  v={'accuracy':float(np.mean(g==p)),'correct':int(np.sum(g==p)),'macro_f1':macro(g,p),'invalid':int(np.sum(p=='INVALID')),'known_n':int(known.sum()),'known_to_unknown':int(np.sum(known&(p=='U'))),'unknown_n':int((~known).sum()),'unknown_correct':int(np.sum((~known)&(p=='U'))),'evidence_known_correct':int(hits[known].sum()),'evidence_known_accuracy':float(hits[known].mean()),'evidence_all_correct':int(hits.sum()),'evidence_all_accuracy':float(hits.mean()),'joint_label_and_evidence_correct':int(np.sum(hits&(p==g))),'joint_accuracy':float(np.mean(hits&(p==g))),'supported_citation_count':supported_ids,'citation_count':total_ids,'reference_supported_citation_precision':supported_ids/total_ids if total_ids else None,'pair_n':len(pairs),'pair_correct':sum(all(pb[z[k]][m]['label']==lb[z[k]]['label'] for k in ['original_id','counterfactual_id']) for z in pairs),'conflict_evidence_correct':sum(evidence_hit(x,m) for x in labels if x['variant']=='contradiction'),'generation_seconds':stage['direct'] if m=='direct' else stage['rule_extraction']+stage['facts']+(stage['verification'] if m=='structured' else 0),'by_variant':{},'confusion_order':['T','F','U','INVALID']}
  v['confusion']=[[int(np.sum((g==a)&(p==b))) for b in v['confusion_order']] for a in v['confusion_order']]
  for typ in sorted({x['variant'] for x in labels}):
   ix=np.array([x['variant']==typ for x in labels]);v['by_variant'][typ]={'n':int(ix.sum()),'correct':int(np.sum((g==p)&ix)),'evidence_correct':int(hits[ix].sum()),'joint_correct':int(np.sum((g==p)&hits&ix))}
  v['pair_accuracy']=v['pair_correct']/len(pairs);out['methods'][m]=v
 out['stage_generation_seconds']=stage;out['paired']={}
 for a,b in [('structured','direct'),('structured','structured_no_guard')]:
  out['paired'][a+'_vs_'+b]={'rescued':int(np.sum((arrays[a]==g)&(arrays[b]!=g))),'introduced':int(np.sum((arrays[a]!=g)&(arrays[b]==g)))}
 rng=np.random.default_rng(20261018);out['bootstrap']={}
 for unit in ['rule_id','source_row']:
  groups=collections.defaultdict(list)
  for i,x in enumerate(labels):groups[x[unit]].append(i)
  cs=list(groups.values());ds=[];js=[]
  for _ in range(2000):
   ix=[i for j in rng.integers(0,len(cs),len(cs)) for i in cs[j]];ds.append(macro(g[ix],arrays['structured'][ix])-macro(g[ix],arrays['direct'][ix]));js.append(float(np.mean((g[ix]==arrays['structured'][ix])&eh['structured'][ix])-np.mean((g[ix]==arrays['direct'][ix])&eh['direct'][ix])))
  out['bootstrap'][unit]={'clusters':len(cs),'replications':2000,'f1_difference_ci95':np.quantile(ds,[.025,.975]).tolist(),'joint_difference_ci95':np.quantile(js,[.025,.975]).tolist()}
 def norm(r):
  if not isinstance(r,dict):return r
  return {'mode':'all' if len(r.get('conditions',[]))==1 else r.get('mode'),'conditions':sorted(r.get('conditions',[]),key=lambda c:json.dumps(c,sort_keys=True)),'exceptions':r.get('exceptions',[])}
 rr={r['rule_id']:r['schema_gold'] for r in read(root/'rules_manifest.jsonl')};rx=read(root/'model/rule_extraction_raw.jsonl')
 out['rule_extraction']={'n':len(rx),'exact_normalized':sum(norm(x['parsed'])==norm(rr[x['id']]) for x in rx),'mismatches':[x['id'] for x in rx if norm(x['parsed'])!=norm(rr[x['id']])]}
 records=read(root/'model/normalized_records.jsonl');out['verification_audit']={}
 for arm in ['unverified','verified']:
  audits=[a for r in records for aus in r[arm]['alignment'].values() for a in aus];out['verification_audit'][arm]={'relocated_citations':sum(a['relocated'] for a in audits),'rejected_citations':sum(a['rejected'] for a in audits),'citations':len(audits)}
 errors=[{k:x[k] for k in ['sample_id','rule_id','source_row','variant','label','gold_observations']}|{m:pb[x['sample_id']][m] for m in methods}|{'evidence_correct':{m:evidence_hit(x,m) for m in methods}} for x in labels if any(pb[x['sample_id']][m]['label']!=x['label'] or not evidence_hit(x,m) for m in methods)]
 (root/'errors.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in errors));(root/'metrics.json').write_text(json.dumps(out,indent=2)+'\n')
 with (root/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['method','correct','n','accuracy','macro_f1','evidence_all_correct','joint_correct','pair_correct','generation_seconds'])
  for m,v in out['methods'].items():w.writerow([m,v['correct'],len(labels),v['accuracy'],v['macro_f1'],v['evidence_all_correct'],v['joint_label_and_evidence_correct'],v['pair_correct'],v['generation_seconds']])
 print(json.dumps({m:{k:v[k] for k in ['correct','macro_f1','evidence_all_correct','joint_label_and_evidence_correct','pair_correct','generation_seconds']} for m,v in out['methods'].items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,default=pathlib.Path('data'));main(p.parse_args().data)
