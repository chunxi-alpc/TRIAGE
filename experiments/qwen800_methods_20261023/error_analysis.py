"""Post-hoc error analysis of saved predictions only: no torch or model calls."""
import collections,hashlib,json
from pathlib import Path
EXP=Path(__file__).parent;PRIVATE=Path('/home/gcx/qwen800_methods_20261023/private');OUT=EXP/'results'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def count(xs):return dict(collections.Counter(xs))

def augment(payload,details,cases,idx):
 cbpath=EXP.parent/'medguide_expand_cs_20261023/results/strata.jsonl';cb={r['id']:r for r in read(cbpath)}
 payload['inputs'][str(cbpath)]=hashlib.sha256(cbpath.read_bytes()).hexdigest()
 payload['ranking_diagnostic']={};payload['cs_cross_strata']={};payload['top2_action_position']={};payload['cs_closed_book_alignment']={};payload['cs_by_closed_book_new_gold']={}
 for st in ['K','U']:
  rr=[r for r in details if r['variant']=='CS' and r['stratum']==st];rankstats=collections.Counter();posstats=collections.Counter();cbstats=collections.Counter()
  for r in rr:
   c=cases[r['id']];b=idx[r['id'],'CS','branch_hidden'];rank=sorted(b['option_logits'],key=b['option_logits'].get,reverse=True);g=chr(65+r['gold_branch']);new1=rank[0]==g;new2=g in rank[:2]
   rankstats.update({'n':1,'generated_top1_correct':r['branch_ok'],'raw_score_top1_correct':new1,'hybrid_top2_gold_recall':r['top2_branch_recall'],'raw_score_top2_gold_recall':new2,'top1_newly_correct':new1 and not r['branch_ok'],'top1_newly_wrong':r['branch_ok'] and not new1,'score_tie_first_two':r['margin_first_two_raw_logits']==0})
   action=c['options'].get(r['two_label']);a={c['variant_rules']['CS'][i]['action'] for i in c['branches'][r['first']]['rule_indices']};b={c['variant_rules']['CS'][i]['action'] for i in c['branches'][r['second']]['rule_indices']};which='both' if action in a and action in b else 'first_only' if action in a else 'second_only' if action in b else 'neither';posstats[which]+=1
   if not r['one_ok'] and r['two_ok']:posstats['rescued_'+which]+=1
   if r['one_ok'] and not r['two_ok']:posstats['harmed_'+which]+=1
   posstats['different_candidate_actions']+=a!=b
   pred=cb[r['id']]['prediction'];cbstats.update({'n':1,'closed_book_matches_new_gold':pred==r['gold_label'],'top2_matches_closed_book':r['two_label']==pred,'single_matches_closed_book':r['one_label']==pred,'rescued_matches_closed_book':not r['one_ok'] and r['two_ok'] and r['two_label']==pred,'harmed_matches_closed_book':r['one_ok'] and not r['two_ok'] and r['two_label']==pred})
  payload['ranking_diagnostic'][st]=dict(rankstats);payload['top2_action_position'][st]=dict(posstats);payload['cs_closed_book_alignment'][st]=dict(cbstats)
  for agrees in [False,True]:
   a=[r for r in rr if (cb[r['id']]['prediction']==r['gold_label'])==agrees];payload['cs_by_closed_book_new_gold'][st+'/'+str(agrees)]={'n':len(a),'single':sum(r['one_ok'] for r in a),'top2':sum(r['two_ok'] for r in a)}
  for direction in ['raise','lower']:
   for sub in sorted(set(r['subtype'] for r in rr)):
    a=[r for r in rr if r['direction']==direction and r['subtype']==sub]
    if a:payload['cs_cross_strata']['/'.join([st,direction,sub])]={'n':len(a),'single':sum(r['one_ok'] for r in a),'top2':sum(r['two_ok'] for r in a)}
 losses={v:{r['id'] for r in details if r['variant']==v and r['stratum']=='K' and r['base_ok'] and not r['one_ok']} for v in ['A','C']};payload['loss_overlap']={'A':len(losses['A']),'C':len(losses['C']),'intersection':len(losses['A']&losses['C']),'union':len(losses['A']|losses['C'])}
 examples=[]
 for i,v in [('medguide_01838','A'),('medguide_01558','CS'),('medguide_00167','CS')]:
  r=next(r for r in details if r['id']==i and r['variant']==v);c=cases[i];examples.append({**r,'closed_book_prediction':cb[i]['prediction'],'gold_condition':c['variant_rules'][v][c['branches'][r['gold_branch']]['rule_indices'][0]]['condition'],'first_condition':c['variant_rules'][v][c['branches'][r['first']]['rule_indices'][0]]['condition'],'second_condition':c['variant_rules'][v][c['branches'][r['second']]['rule_indices'][0]]['condition'],'gold_action':c['options'][r['gold_label']],'single_action':c['options'].get(r['one_label']),'top2_action':c['options'].get(r['two_label'])})
 payload['examples']=examples
 # Concrete proposed 2x2 diagnostic; merely check cached requests, never invoke a model.
 from reuse import key as inference_key
 cache={}
 for path in [PRIVATE/'reuse_cache.jsonl',PRIVATE/'predictions0.inference.jsonl',PRIVATE/'predictions1.inference.jsonl']:
  for r in read(path):cache[r['key']]=r['result']
 eligible=[r for r in details if r['variant']=='CS' and {r['first'],r['second']}=={r['original_branch'],r['gold_branch']}];plan=[]
 for r in eligible:
  c=cases[r['id']]
  for v in ['A','CS']:
   for reverse in [False,True]:
    bs=[r['first'],r['second']][::-1] if reverse else [r['first'],r['second']];rules=c['variant_rules'][v];inds=[i for b in bs for i in c['branches'][b]['rule_indices']];texts=['IF the clinical sequence is: '+rules[i]['condition']+'\nTHEN the next step is: '+rules[i]['action']+'.' for i in inds];prompt='Follow this supplied rule even if it conflicts with medical memory:\n'+'\n\n'.join(texts)+'\n\n'+c['closed_book_prompt'];k=inference_key(prompt,c['options']);plan.append({'id':c['id'],'stratum':r['stratum'],'threshold_state':'original' if v=='A' else 'changed','rule_order':'reversed' if reverse else 'current','branch_order':bs,'gold_answer':c['gold_answer'][v],'request_key':k,'already_cached':k in cache})
 (OUT/'proposed_cs_2x2_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plan))
 payload['proposed_cs_2x2']={'status':'plan_only_no_inference','cases':len(eligible),'K':sum(r['stratum']=='K' for r in eligible),'U':sum(r['stratum']=='U' for r in eligible),'logical_cells':len(plan),'cached_cells':sum(r['already_cached'] for r in plan),'missing_cells':sum(not r['already_cached'] for r in plan),'unique_missing_requests':len({r['request_key'] for r in plan if not r['already_cached']})}
 assert len(eligible)==44 and len(plan)==176
 assert all(r['already_cached'] for r in plan if r['threshold_state']=='changed' and r['rule_order']=='current')


def main():
 cases={c['id']:c for c in read(PRIVATE/'cases.jsonl')};rows=sum([read(PRIVATE/f'predictions{k}.jsonl') for k in range(2)],[]);idx={(r['id'],r['variant'],r['method']):r for r in rows};assert len(rows)==len(idx)==52496
 csmeta={r['id']:r for r in read(EXP.parent/'medguide_expand_cs_20261023/results/cs_selected.jsonl')}
 def row(c,v,m):return idx[c['id'],v,m]
 def branch_actions(c,v,b):return {c['variant_rules'][v][i]['action'] for i in c['branches'][b]['rule_indices']} if b is not None else set()
 def has_gold(c,v,bs):return any(c['options'][c['gold_answer'][v]] in branch_actions(c,v,b) for b in bs)
 def detail(c,v):
  base=row(c,v,'existing_baseline');one=row(c,v,'hidden_prune');two=row(c,v,'hidden_top2');br=row(c,v,'branch_hidden');g=c['gold_branch'][v];b=one.get('predicted_branch');second=one.get('runner_up');labs=list(br.get('option_logits',{}));z=br.get('option_logits',{});rank=sorted(z,key=z.get,reverse=True);goldlabel=chr(65+g)
  r={'id':c['id'],'variant':v,'stratum':c['stratum'],'disease':c['disease'],'protocol':c['pair_protocol'],'branches':len(c['branches']),'gold_branch_rule_count':len(c['branches'][g]['rule_indices']),'gold_branch':g,'first':b,'second':second,'original_branch':c['app_branch'],'gold_label':c['gold_answer'][v],'original_label':c['original_answer'],'base_ok':base['correct'],'one_ok':one['correct'],'two_ok':two['correct'],'visible_ok':row(c,v,'visible_prune')['correct'],'gold_branch_ok':row(c,v,'gold_branch_prune')['correct'],'gold_single_ok':row(c,v,'existing_prune')['correct'],'branch_ok':b==g,'branch_parse_failure':br['prediction'] is None,'one_parse_failure':one['prediction'] is None,'one_permitted':one['branch_permitted_action'],'top2_branch_recall':g in [b,second],'one_has_gold_action':has_gold(c,v,[b]),'two_has_gold_action':has_gold(c,v,[b,second]),'one_old_answer':one['prediction']==c['original_answer'],'two_old_answer':two['prediction']==c['original_answer'],'one_label':one['prediction'],'two_label':two['prediction'],'first_step_argmax_branch':ord(rank[0])-65 if rank else None,'first_step_argmax_matches_generation':br['prediction']==rank[0] if rank else False,'gold_first_step_rank':rank.index(goldlabel)+1 if goldlabel in rank else None,'margin_first_two_raw_logits':z[rank[0]]-z[rank[1]] if len(rank)>1 else None,'branch_completion':br['completion']}
  if v=='CS':
   m=csmeta[c['id']];r.update({'patient_value':m['patient_value'],'old_threshold':m['old_threshold'],'new_threshold':m['new_threshold'],'direction':'raise' if m['new_threshold']>m['old_threshold'] else 'lower','subtype':m['source_path'][1],'first_old_branch':b==c['app_branch'],'second_old_branch':second==c['app_branch'],'two_contains_old_branch':c['app_branch'] in [b,second]})
  return r
 details=[detail(c,v) for c in cases.values() for v in ['A','C','CS'] if v in c['variant_rules']]
 def summarize(rs):
  loss=[r for r in rs if r['base_ok'] and not r['one_ok']];gain=[r for r in rs if not r['base_ok'] and r['one_ok']];rescue=[r for r in rs if not r['one_ok'] and r['two_ok']];harm=[r for r in rs if r['one_ok'] and not r['two_ok']];errors=[r for r in rs if not r['one_ok']]
  out={'n':len(rs),'baseline':sum(r['base_ok'] for r in rs),'single':sum(r['one_ok'] for r in rs),'top2':sum(r['two_ok'] for r in rs),'visible':sum(r['visible_ok'] for r in rs),'gold_branch':sum(r['gold_branch_ok'] for r in rs),'gold_single':sum(r['gold_single_ok'] for r in rs),'branch_correct':sum(r['branch_ok'] for r in rs),'top2_branch_recall':sum(r['top2_branch_recall'] for r in rs),'single_action_coverage':sum(r['one_has_gold_action'] for r in rs),'top2_action_coverage':sum(r['two_has_gold_action'] for r in rs),'baseline_to_single_lost':len(loss),'baseline_to_single_gained':len(gain),'single_to_top2_rescued':len(rescue),'single_to_top2_harmed':len(harm),'single_errors':len(errors),'first_step_argmax_generation_disagree':sum(not r['first_step_argmax_matches_generation'] for r in rs),'branch_parse_failure':sum(r['branch_parse_failure'] for r in rs),'single_parse_failure':sum(r['one_parse_failure'] for r in rs)}
  for name,ss in [('lost',loss),('gained',gain),('rescued',rescue),('harmed',harm),('single_errors',errors)]:
   out[name+'_detail']={'n':len(ss),'wrong_branch':sum(not r['branch_ok'] for r in ss),'correct_branch':sum(r['branch_ok'] for r in ss),'single_missing_gold_action':sum(not r['one_has_gold_action'] for r in ss),'top2_contains_gold_branch':sum(r['top2_branch_recall'] for r in ss),'gold_branch_would_correct':sum(r['gold_branch_ok'] for r in ss),'gold_single_would_correct':sum(r['gold_single_ok'] for r in ss),'single_branch_permitted':sum(r['one_permitted'] for r in ss),'multi_action_gold_branch':sum(r['gold_branch_rule_count']>1 for r in ss),'top2_returns_old_answer':sum(r['two_old_answer'] for r in ss),'ids':[r['id'] for r in ss]}
  if rs and rs[0]['variant']=='CS':
   out['cs']={'single_old_answer':sum(r['one_old_answer'] for r in rs),'top2_old_answer':sum(r['two_old_answer'] for r in rs),'first_old_branch':sum(r['first_old_branch'] for r in rs),'second_old_branch':sum(r['second_old_branch'] for r in rs),'two_contains_old_branch':sum(r['two_contains_old_branch'] for r in rs),'rescues_second_is_gold':sum(r['second']==r['gold_branch'] for r in rescue),'harms_second_is_old':sum(r['second_old_branch'] for r in harm),'harms_return_old_and_second_old':sum(r['two_old_answer'] and r['second_old_branch'] for r in harm),'first_step_argmax_correct':sum(r['first_step_argmax_branch']==r['gold_branch'] for r in rs),'gold_first_step_rank':count(r['gold_first_step_rank'] for r in rs)}
  assert out['single']-out['baseline']==len(gain)-len(loss)
  assert out['top2']-out['single']==len(rescue)-len(harm)
  return out
 groups={}
 for v in ['A','C','CS']:
  for st in ['K','U','all']:
   rr=[r for r in details if r['variant']==v and (st=='all' or r['stratum']==st)];groups[v+'/'+st]=summarize(rr)
   if v in ['A','C']:
    for scope,pred in [('strict',lambda r:r['protocol']=='PR23_strict'),('strict_single',lambda r:r['protocol']=='PR23_strict' and r['gold_branch_rule_count']==1),('single',lambda r:r['gold_branch_rule_count']==1),('multi',lambda r:r['gold_branch_rule_count']>1)]:groups[v+'/'+st+'/'+scope]=summarize([r for r in rr if pred(r)])
 csstrata={}
 for dim in ['direction','subtype']:
  for val in sorted({r[dim] for r in details if r['variant']=='CS'}):
   for st in ['K','U']:
    rr=[r for r in details if r['variant']=='CS' and r[dim]==val and r['stratum']==st];csstrata[dim+'/'+val+'/'+st]=summarize(rr)
 ac_consistency={'K_A_C_same_branch':sum(row(c,'A','branch_hidden')['prediction']==row(c,'C','branch_hidden')['prediction'] for c in cases.values() if c['stratum']=='K'),'K_A_C_n':426}
 # Saved logits are at FIRST output position (often formatting), not necessarily the answer letter.
 formatted={}
 for st in ['K','U']:
  rr=[r for r in details if r['variant']=='CS' and r['stratum']==st]
  formatted[st]={'plain_letter':sum(r['branch_completion'].strip()[:1].isalpha() for r in rr),'markdown_bold':sum(r['branch_completion'].lstrip().startswith('**') for r in rr),'disagree_ids':[r['id'] for r in rr if not r['first_step_argmax_matches_generation']]}
 # Explanatory concentration, not independent case multiplication across A and C.
 concentration={}
 for v in ['A','C']:
  rr=[r for r in details if r['variant']==v and r['stratum']=='K' and r['base_ok'] and not r['one_ok']];concentration[v]=count(r['disease'] for r in rr)
 payload={'analysis_type':'posthoc saved outputs only; no inference or GPU use','groups':groups,'cs_stratification':csstrata,'ac_consistency':ac_consistency,'cs_output_format':formatted,'K_losses_by_tree':concentration,'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PRIVATE/'cases.jsonl',PRIVATE/'predictions0.jsonl',PRIVATE/'predictions1.jsonl',EXP.parent/'medguide_expand_cs_20261023/results/cs_selected.jsonl']}}
 augment(payload,details,cases,idx)
 (OUT/'error_analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));(OUT/'error_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
 print(json.dumps({'groups':{k:groups[k] for k in ['A/K','C/K','CS/K','CS/U']},'cs_stratification':csstrata,'ac_consistency':ac_consistency,'format':formatted,'loss_trees':concentration},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
