from pathlib import Path
import json, collections, statistics
ROOT=Path('/home/gcx/TRIAGE'); D=ROOT/'experiments/care_module1_v6/data'; R=ROOT/'experiments/care_module1_v6/report'; R.mkdir(parents=True,exist_ok=True)
T,F,U='T','F','U'; LABELS=(T,F,U)
def load(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def tri_and(v): return F if F in v else U if U in v else T
def tri_or(v): return T if T in v else U if U in v else F
def tri_not(v): return {T:F,F:T}.get(v,U)
def logic(e,s):
 if isinstance(e,str): return s.get(e,U)
 a=[logic(x,s) for x in e.get('args',[])]; return tri_and(a) if e.get('op','AND')=='AND' else tri_or(a) if e.get('op')=='OR' else tri_not(a[0]) if e.get('op')=='NOT' and len(a)==1 else U
def f1(g,p):
 return sum((2*sum(a==b==x for a,b in zip(g,p)))/(2*sum(a==x and b==x for a,b in zip(g,p))+sum(a!=x and b==x for a,b in zip(g,p))+sum(a==x and b!=x for a,b in zip(g,p))) if (2*sum(a==x and b==x for a,b in zip(g,p))+sum(a!=x and b==x for a,b in zip(g,p))+sum(a==x and b!=x for a,b in zip(g,p))) else 0 for x in LABELS)/3
def evidence_ids(row):
 ids=set()
 for c in row.get('conditions',[]):
  for k in ('support_spans','refute_spans','numeric_evidence_spans'): ids.update(x.get('sentence_id') for x in c.get(k,[]) if x.get('sentence_id') is not None)
 return ids
def evidence_ok(ids,label):
 if label['label'] in (T,F): return any(set(p).issubset(ids) for p in label.get('acceptable_evidence_proofs',[])) and ids.issubset(set(label.get('all_relevant_evidence_ids',[])))
 conflicts=set(label.get('conflict_evidence_ids',[])); return conflicts==ids if conflicts else not ids
def summarize(labels,preds,method,pair_ids):
 by={x['sample_id']:x for x in preds}; gold=[x['label'] for x in labels]; pred=[by[x['sample_id']][method] for x in labels]; ev=[evidence_ok(evidence_ids(by[x['sample_id']]),x) for x in labels]
 known=[i for i,x in enumerate(labels) if x['label'] in (T,F)]; pairs=[(a,b) for a,b in pair_ids if a in by and b in by]
 pair_acc=sum(by[a][method]==next(x['label'] for x in labels if x['sample_id']==a) and by[b][method]==next(x['label'] for x in labels if x['sample_id']==b) for a,b in pairs)/len(pairs) if pairs else None
 return {'n':len(labels),'accuracy':sum(a==b for a,b in zip(gold,pred))/len(labels),'macro_f1':f1(gold,pred),'known_evidence_accuracy':sum(ev[i] for i in known)/len(known) if known else None,'joint_accuracy':sum(a==b and e for a,b,e in zip(gold,pred,ev))/len(labels),'counterfactual_pair_accuracy':pair_acc,'mean_elapsed_seconds':statistics.mean([by[x['sample_id']].get('elapsed_seconds',0) for x in labels]),'unknown_predictions':sum(x==U for x in pred),'correct_known_to_unknown':sum(x['label']!=U and by[x['sample_id']][method]==U for x in labels),'evidence_correct_n':sum(ev),'allow_intervention_n':sum(bool(by[x['sample_id']].get('allow_attention_intervention')) for x in labels)}
def old_rows():
 labels=load(D/'labels_public.jsonl'); v5=load(D/'v5_predictions_public.jsonl'); rules={x['rule_id']:x for x in load(D/'rules_v5.jsonl')}; out=[]
 for row in v5:
  rule=rules[row['rule_id']]; statesA={}; statesB={}; evstat=[]
  for c in row['conditions']:
   a=c.get('trialmatch_state','U'); n=c.get('numeric_status','U'); b=n if n in (T,F) else a; statesA[c['condition_id']]=a; statesB[c['condition_id']]=b
   hasS=bool(c.get('support_spans') or c.get('numeric_evidence_spans')); hasR=bool(c.get('refute_spans')); evstat.append('disputed' if hasS and hasR else 'supported' if hasS or hasR else 'missing')
  A=row.get('trialmatch_model_only_applicability',logic(rule['logic'],statesA)); B=logic(rule['logic'],statesB)
  status='disputed' if 'disputed' in evstat else 'supported' if 'supported' in evstat else 'missing'
  allow=B==T and status=='supported' and 'disputed' not in evstat
  out.append({'sample_id':row['sample_id'],'rule_id':row['rule_id'],'A':A,'B':B,'C':B,'elapsed_seconds':row.get('elapsed_seconds',0),'conditions':row.get('conditions',[]),'evidence_status':status,'allow_attention_intervention':allow})
 return labels,out
new_inputs=load(D/'new_inputs_100.jsonl'); new_labels=load(D/'new_labels_100.jsonl'); newpred=[]
for p in sorted((ROOT/'experiments/care_module1_v6/runs/new100').glob('pred_*.jsonl')): newpred+=load(p)
new=[]
for r in newpred:
 new.append({'sample_id':r['sample_id'],'rule_id':r['rule_id'],'A':r['trialmatch_model_only_applicability'],'B':r['applicability'],'C':r['applicability'],'elapsed_seconds':r['elapsed_seconds'],'conditions':r['conditions'],'evidence_status':r['evidence_status'],'allow_attention_intervention':r['allow_attention_intervention']})
old_labels,old=old_rows();
def pairs_for(labels):
 by=collections.defaultdict(dict)
 for x in labels: by[x.get('anchor_id',x['sample_id'].rsplit('_',1)[0])][x.get('variant','')]=x['sample_id']
 return [(v['original'],v['counterfactual']) for v in by.values() if 'original' in v and 'counterfactual' in v]
metrics={'protocol':'CARE module1 V6; A=TrialMatchAI, B=valid numeric comparison, C=B+BioLinkBERT evidence-only','gpu_plan':'4x NVIDIA A100-SXM4-80GB for new-set run','new_set_manifest':json.loads((D/'new_set_manifest.json').read_text()),'methods':{}}
for name,labels,preds in [('old_192',old_labels,old),('new_100',new_labels,new)]:
 pairs=pairs_for(labels)
 for m in ('A','B','C'): metrics['methods'][f'{name}_{m}']=summarize(labels,preds,m,pairs)
 # intervention and degradation counts
 c_to_u=sum(x['A'] in (T,F) and x['C']==U for x in preds); b_to_u=sum(x['B'] in (T,F) and x['C']==U for x in preds)
 metrics.setdefault('diagnostics',{})[name]={'correct_TF_closed_intervention':sum(x['C']==next(l['label'] for l in labels if l['sample_id']==x['sample_id']) in (T,F) and not x['allow_attention_intervention'] for x in preds),'known_to_unknown_C_vs_A':c_to_u,'incorrect_activation':sum(x['C']==T and next(l['label'] for l in labels if l['sample_id']==x['sample_id'])==F for x in preds),'reliable_intervention_rate':sum(x['allow_attention_intervention'] and x['C']==next(l['label'] for l in labels if l['sample_id']==x['sample_id']) for x in preds)/max(1,sum(next(l['label'] for l in labels if l['sample_id']==x['sample_id'])==T for x in preds))}
(D/'v6_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
for name,labels,preds in [('old_192',old_labels,old),('new_100',new_labels,new)]:
 with (D/f'{name}_predictions_v6.jsonl').open('w') as f:
  for x in preds:f.write(json.dumps(x,ensure_ascii=False)+'\n')
# failure samples
fails=[]
for lab in old_labels+new_labels:
 pred=next((x for x in old+new if x['sample_id']==lab['sample_id']),None)
 if pred and (pred['C']!=lab['label'] or not evidence_ok(evidence_ids(pred),lab)): fails.append({'sample_id':lab['sample_id'],'rule_id':lab['rule_id'],'gold':lab['label'],'A':pred['A'],'B':pred['B'],'C':pred['C'],'evidence_status':pred['evidence_status'],'evidence_ids':sorted(evidence_ids(pred))})
(D/'v6_failure_samples.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in fails[:80])+'\n')
print(json.dumps({k:v for k,v in metrics['methods'].items() if k.endswith('_C')},ensure_ascii=False))
