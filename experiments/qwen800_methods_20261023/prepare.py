"""Frozen PR24 cohort, PR23-compatible pairs plus explicitly marked extensions."""
import ast,collections,hashlib,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'experiments/aster_medguide_20261022'
sys.path.insert(0,str(OLD))
from prepare_dataset import build_example,norm
EXP=Path(__file__).parent;OUT=EXP/'results';PRIVATE=Path('/home/gcx/qwen800_methods_20261023/private')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def main():
 import pyarrow.parquet as pq
 rows=pq.read_table('/home/gcx/aster_medguide_20261022/private/train.parquet').to_pylist();paths=json.load(open('/home/gcx/aster_medguide_20261022/private/all_paths.json'));cohort=list(map(json.loads,Path('/home/gcx/medguide_expand_cs_20261023/private/cohort.jsonl').read_text().splitlines()));prior=ROOT/'experiments/medguide_expand_cs_20261023/results';strata={r['id']:r['stratum'] for r in map(json.loads,(prior/'strata.jsonl').read_text().splitlines())};cs={r['id']:r for r in map(json.loads,(prior/'cs_selected.jsonl').read_text().splitlines())};oldids=set(json.load(open(ROOT/'experiments/aster_validation_20261022/mechanism_20261023/results/priority200/selection.json'))['ids'])
 cases=[];inventory=[]
 for c in cohort:
  r=rows[c['source_row']];ps=list({tuple(p):p for p in paths[c['tree']]}.values());app=ps.index(r['path']);strict=False
  try:versions=build_example(r,paths,c['source_row'],20261022);strict=True;sib=versions[0]['sibling_path_index']
  except ValueError:
   def shared(p):return next((j for j,(a,b) in enumerate(zip(p,r['path'])) if a!=b),min(len(p),len(r['path'])))
   others=[j for j,p in enumerate(ps) if p[:-1]!=r['path'][:-1]]
   aligned=[j for j in others if norm(ps[j][-1])!=norm(r['answer_text']) and any(norm(ps[j][-1])==norm(v) for v in r['options'])]
   candidates=aligned or others;sib=min(candidates,key=lambda j:(-shared(ps[j]),j)) if candidates else None
  order=list(range(len(ps)));random.Random(20261022+c['source_row']).shuffle(order);appidx=order.index(app);sibidx=order.index(sib) if sib is not None else None
  rules=[{'condition':' -> '.join(ps[j][:-1]),'action':ps[j][-1]} for j in order];conds=list(dict.fromkeys(x['condition'] for x in rules));branches=[{'condition':s,'rule_indices':[j for j,x in enumerate(rules) if x['condition']==s]} for s in conds];appb=conds.index(rules[appidx]['condition']);sibb=conds.index(rules[sibidx]['condition']) if sibidx is not None else None
  opts=c['options'];gold=r['answer'];siblabel=next((k for k,v in opts.items() if sibidx is not None and norm(v)==norm(rules[sibidx]['action'])),None);replacement=next((k for k in opts if k not in (gold,siblabel)),next(k for k in opts if k!=gold))
  vr={'A':[dict(x) for x in rules],'B':[dict(x) for x in rules]};vr['B'][appidx]['action']=opts[replacement];golds={'A':gold,'B':replacement};gb={'A':appb,'B':appb};gr={'A':appidx,'B':appidx};missing={}
  if sibidx is not None:
   vr['C']=[dict(x) for x in rules];clabel=replacement if opts[replacement]!=rules[sibidx]['action'] else gold;vr['C'][sibidx]['action']=opts[clabel];golds['C']=gold;gb['C']=appb;gr['C']=appidx
  else:missing['C']='no different condition branch in published tree'
  if siblabel is not None and siblabel!=gold:
   vr['D']=[dict(x) for x in rules];vr['D'][appidx]['action'],vr['D'][sibidx]['action']=vr['D'][sibidx]['action'],vr['D'][appidx]['action'];golds['D']=siblabel;gb['D']=appb;gr['D']=appidx
  else:missing['D']='no different off-path action among original options'
  if c['id'] in cs:
   x=cs[c['id']];j=x['depth'];prefix=x['source_path'][:j];edited=[]
   for pi in order:
    p=ps[pi].copy()
    if p[:j]==prefix and p[j] in x['rewritten_nodes']:p[j]=x['rewritten_nodes'][p[j]]
    edited.append({'condition':' -> '.join(p[:-1]),'action':p[-1]})
   newpath=x['new_execution']['paths'][0];newidx=order.index(ps.index(newpath));newb=next(k for k,b in enumerate(branches) if newidx in b['rule_indices']);vr['CS']=edited;golds['CS']=x['new_gold'];gb['CS']=newb;gr['CS']=newidx
  # Source-variant prompt is byte-compatible with PR23 for strict cases.
  prefix='Use the supplied decision rules for this offline hypothetical task. If a supplied recommendation conflicts with medical memory, follow the supplied rule.\n\nDecision rules:\n'
  source={v:{'prompt_text':prefix+'\n\n'.join('IF the clinical sequence is: '+x['condition']+'\nTHEN the next step is: '+x['action']+'.' for x in rs)+'\n\n'+c['closed_book_prompt']} for v,rs in vr.items()}
  if strict:
   for old in versions:assert source[old['variant'][0]]['prompt_text']==old['prompt_text']
  case={'id':c['id'],'source_row':c['source_row'],'disease':c['tree'],'stratum':strata[c['id']],'options':opts,'closed_book_prompt':c['closed_book_prompt'],'patient':'Patient:\n'+c['profile'].strip(),'original_answer':gold,'app_rule':appidx,'sib_rule':sibidx,'app_branch':appb,'sib_branch':sibb,'branches':branches,'variant_rules':vr,'source_variants':source,'gold_answer':golds,'gold_branch':gb,'gold_rule':gr,'type':'unreviewed','single_action_gold_branch':len(branches[appb]['rule_indices'])==1,'pair_protocol':'PR23_strict' if strict else 'extended_offpath','missing_variants':missing,'original200_overlap':c['id'] in oldids}
  cases.append(case);inventory.append({k:case[k] for k in ('id','source_row','disease','stratum','pair_protocol','single_action_gold_branch','missing_variants','original200_overlap')}|{'variants':list(vr),'gold_answer':golds})
 PRIVATE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True);write(PRIVATE/'cases.jsonl',cases);write(OUT/'case_manifest.jsonl',inventory)
 for k in range(2):write(PRIVATE/f'shard{k}.jsonl',cases[k::2])
 m={'pr24_head':'c615f8330652d5399a4f696f8b5df3b0cfb12bd8','pr23_head':'3e4fac744bf63564cd32045f15c102390ac57c28','cases':len(cases),'strata':dict(collections.Counter(c['stratum'] for c in cases)),'protocols':dict(collections.Counter(c['pair_protocol'] for c in cases)),'variants':dict(collections.Counter(v for c in cases for v in c['variant_rules'])),'single_action_branches':sum(c['single_action_gold_branch'] for c in cases),'overlap200':sum(c['original200_overlap'] for c in cases),'input_sha256':sha(PRIVATE/'cases.jsonl'),'pr24_cohort_sha256':sha('/home/gcx/medguide_expand_cs_20261023/private/cohort.jsonl'),'gain':0.5,'attention':'eager; K/U frozen from PR24 SDPA closed-book','main_tokens':8,'cad_tokens':128,'filtered_cad_beta':0.1,'methods':['existing_baseline','existing_prune','existing_app','existing_signed','gold_branch_prune','visible_prune','hidden_prune','hidden_top2','hidden_signed','hidden_unsigned','CAD8','CAD128','CAD_filtered128'],'branch_methods':['branch_hidden','branch_visible','branch_hide_inapp'],'expected_rows':sum(len(c['variant_rules'])*16 for c in cases),'extended_policy':'nearest shared-prefix off-path condition with option-aligned different action; third option preferred; binary C uses original action; no fabricated D for missing option-aligned alternative'}
 (OUT/'dataset_manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m))
if __name__=='__main__':main()
