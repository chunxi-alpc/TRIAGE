"""Post-hoc scoring; evaluation labels are loaded only here, not by the checker."""
import collections,datetime,hashlib,json
from pathlib import Path
E=Path(__file__).parent;ROOT=E.parents[1];O=E/'results';P=Path('/home/gcx/cs44_explicit_check_20261024/private');OLD=ROOT/'experiments/cs44_threshold_order_20261024/results'
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def jsonl(p,x):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
def main():
 evaluation={r['request_key']:r for r in read(P/'evaluation.jsonl')};base={r['request_key']:r for r in read(OLD/'scores.jsonl')};rows=[];workers=[]
 for k in range(2):
  m=json.loads((O/f'worker{k}.json').read_text());assert m['complete'] and m['output_sha256']==sha(P/f'outputs{k}.jsonl');workers.append(m);rows+=read(P/f'outputs{k}.jsonl')
 assert len(rows)==176 and len({r['cell_key'] for r in rows})==176 and set(evaluation)=={r['cell_key'] for r in rows}
 for r in rows:
  e=evaluation[r['cell_key']];b=base[r['cell_key']];r.update({**{k:e[k] for k in ['id','stratum','threshold_state','rule_order','gold_answer','gold_slot','focus18']},'correct':r['prediction']==e['gold_answer'],'check_correct':r['selected_slot']==e['gold_slot'],'baseline_prediction':b['prediction'],'baseline_correct':b['correct'],'closed_book':b['closed_book_prediction'],'old_gold':b['old_gold'],'new_gold':b['new_gold']})
 summary={'complete':True,'cases':44,'cells':176,'check_calls':sum(w['check_calls'] for w in workers),'new_final_calls':sum(w['new_final_calls'] for w in workers),'final_cached_cells':sum(r['final_reused'] for r in rows),'fallback_cells':sum(r['fallback_two_rules'] for r in rows),'groups':{}}
 for st in ['all','K','U','focus18']:
  rr=[r for r in rows if st=='all' or (r['focus18'] if st=='focus18' else r['stratum']==st)];cells={}
  for t in ['original','changed']:
   for o in ['current','reversed']:
    a=[r for r in rr if r['threshold_state']==t and r['rule_order']==o];cells[t+'/'+o]={'n':len(a),'baseline_correct':sum(r['baseline_correct'] for r in a),'correct':sum(r['correct'] for r in a),'check_correct':sum(r['check_correct'] for r in a),'gained':sum(not r['baseline_correct'] and r['correct'] for r in a),'lost':sum(r['baseline_correct'] and not r['correct'] for r in a),'wrong_check':sum(r['selected_slot'] is not None and not r['check_correct'] for r in a),'fallback':sum(r['fallback_two_rules'] for r in a),'right_check_wrong_answer':sum(r['check_correct'] and not r['correct'] for r in a),'old_answer':sum(r['prediction']==r['old_gold'] for r in a)}
  ix={(r['id'],r['threshold_state'],r['rule_order']):r for r in rr};percase=[]
  for i in sorted({r['id'] for r in rr}):
   a=[ix[i,t,o] for t in ['original','changed'] for o in ['current','reversed']];orig=[r for r in a if r['threshold_state']=='original'];new=[r for r in a if r['threshold_state']=='changed'];percase.append({'id':i,'all4_correct':all(r['correct'] for r in a),'original_both_correct':all(r['correct'] for r in orig),'changed_both_correct':all(r['correct'] for r in new),'changed_any_correct':any(r['correct'] for r in new),'fixed_old_all4':all(r['prediction']==r['old_gold'] for r in a),'check_all4_correct':all(r['check_correct'] for r in a)})
  tp=[(ix[i,'original',o],ix[i,'changed',o]) for i in {r['id'] for r in rr} for o in ['current','reversed']];op=[(ix[i,t,'current'],ix[i,t,'reversed']) for i in {r['id'] for r in rr} for t in ['original','changed']]
  summary['groups'][st]={'cases':len(percase),'cells':cells,'all4_correct':sum(r['all4_correct'] for r in percase),'original_both_correct':sum(r['original_both_correct'] for r in percase),'changed_both_correct':sum(r['changed_both_correct'] for r in percase),'changed_any_correct':sum(r['changed_any_correct'] for r in percase),'fixed_old_all4':sum(r['fixed_old_all4'] for r in percase),'correct_threshold_pairs':sum(a['correct'] and b['correct'] for a,b in tp),'threshold_pairs':len(tp),'order_changed_pairs':sum(a['prediction']!=b['prediction'] for a,b in op),'order_pairs':len(op),'per_case':percase}
 assert summary['check_calls']==176
 jsonl(O/'scores.jsonl',[{k:v for k,v in r.items() if k not in ['check_completion','completion']} for r in rows]);dump(O/'summary.json',summary)
 # Keep full factual checks on the server; compact examples include no patient prompt.
 error_examples=[{k:r[k] for k in ['id','stratum','threshold_state','rule_order','selected_slot','gold_slot','prediction','gold_answer','check_completion']} for r in rows if not r['correct']];jsonl(P/'error_examples.jsonl',error_examples)
 dump(O/'STATUS.json',{'status':'complete','cases':44,'cells':176,'checker_calls':summary['check_calls'],'new_final_calls':summary['new_final_calls'],'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({**summary,'groups':{k:{a:b for a,b in v.items() if a!='per_case'} for k,v in summary['groups'].items()}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
