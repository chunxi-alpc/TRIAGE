"""Six explicitly experimental forms; private text never exported to git."""
import collections,hashlib,json,random,sys
from pathlib import Path
import pyarrow.parquet as pq
from threshold_source import build,atom,evaluate
ROOT=Path('/home/gcx/aster_e0_e8_20261024/private'); OUT=Path(__file__).parent/'results'
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
def lines(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rank(x):return hashlib.sha256(str(x).encode()).hexdigest()
def main():
 src=Path('/home/gcx/aster_medguide_20261022/private'); rows=pq.read_table(src/'train.parquet').to_pylist(); paths=json.loads((src/'all_paths.json').read_text())
 trees=sorted(paths,key=lambda t:rank('split0:'+t));dev=set(trees[:10]); inventory,pairs,cs,audit=build(rows,paths)
 csmap={r['source_row']:r for r in cs if r['patient_point_value_verified'] and r['branch_flipped'] and r['label_flipped'] and r['preserving_label_ok'] and r['preserving_threshold']!=r['old_threshold']}
 candidates=[];excluded=[]
 for i,row in enumerate(rows):
  ps=paths[row['disease']]; path=row['path']; reason=None
  matches=[j for j,p in enumerate(ps) if p==path]
  if len(matches)!=1:reason='gold_path_not_unique'
  elif row['options'][ord(row['answer'])-65].strip()!=path[-1].strip():reason='source_option_mismatch'
  if reason:excluded.append({'id':i,'reason':reason});continue
  gi=matches[0]; siblings=[j for j,p in enumerate(ps) if p[:-2]==path[:-2] and p[-2]!=path[-2] and p[-1]!=path[-1]]
  if not siblings:excluded.append({'id':i,'reason':'no_same_parent_sibling'});continue
  sibling_actions={ps[j][-1] for j in siblings}; replacements=sorted({p[-1] for p in ps}-{path[-1]}-sibling_actions)
  if not replacements:excluded.append({'id':i,'reason':'no_AC_action_outside_all_siblings'});continue
  si=sorted(siblings,key=lambda j:rank(f'{i}:{j}'))[0];rep=replacements[0]
  options=list(dict.fromkeys([path[-1],ps[si][-1],rep]+row['options']))
  if len(options)>26:excluded.append({'id':i,'reason':'too_many_options'});continue
  candidates.append({'id':f'medguide_{i:05d}','source_row':i,'tree':row['disease'],'split':'dev' if row['disease'] in dev else 'test','patient':row['profile'],'paths':ps,'gold_rule':gi,'decoy_rule':si,'replacement':rep,'options_base':options,'original_action':path[-1],'depth':len(path)-1,'siblings':len(siblings),'cs':csmap.get(i)})
 selected=[]
 for split,n in [('dev',200),('test',1000)]:
  buckets={t:sorted([r for r in candidates if r['tree']==t and r['split']==split],key=lambda r:(r['cs'] is None,rank(r['id']))) for t in trees}
  while sum(r['split']==split for r in selected)<n and any(buckets.values()):
   for t in trees:
    if buckets[t] and sum(r['split']==split for r in selected)<n:selected.append(buckets[t].pop(0))
 dataset=[]
 for c in selected:
  for seed in [0,1,2]:
   options=c['options_base'][:];random.Random(f"{seed}:{c['id']}").shuffle(options)
   for variant in ['NC','AC','IC','SW','CS','CS-null']:
    if variant.startswith('CS') and c['cs'] is None:continue
    ps=[p[:] for p in c['paths']];g=c['gold_rule'];s=c['decoy_rule'];target=g;proof=None
    if variant=='AC':ps[g][-1]=c['replacement']
    if variant=='IC':ps[s][-1]=c['replacement']
    if variant=='SW':ps[g][-1],ps[s][-1]=ps[s][-1],ps[g][-1]
    if variant.startswith('CS'):
     z=c['cs'];threshold=z['new_threshold'] if variant=='CS' else z['preserving_threshold'];depth=z['depth']
     for p in ps:
      if p[:depth]==z['source_path'][:depth] and p[depth] in z['nodes']:
       a=atom(p[depth]);p[depth]=p[depth][:a['span'][0]]+a['op']+f"{threshold:g}"+(' '+a['unit'] if a['unit'] else '')+p[depth][a['span'][1]:]
     destination=z['new_execution']['paths'][0] if variant=='CS' else z['source_path'];target=c['paths'].index(destination)
     proof={'value':z['patient_value'],'old_threshold':z['old_threshold'],'threshold':threshold,'unit':z['unit'],'evidence':z['evidence'],'branch_flipped':variant=='CS','execution_scope':z['execution_scope']}
    rules=[{'rid':j,'condition':' > '.join(p[:-1]),'action':p[-1],'source_path_index':j} for j,p in enumerate(ps)]
    gold=options.index(ps[target][-1]);assert ps[target][-1] in options
    dataset.append({'id':c['id'],'tree':c['tree'],'split':c['split'],'seed':seed,'variant':variant,'patient':c['patient'],'rules':rules,'gold_rule':target,'decoy_rule':s,'options':options,'gold_option':gold,'original_action':c['original_action'],'label_kind':'source_path_aligned' if variant=='NC' else 'experimental_counterfactual_not_NCCN','threshold_proof':proof,'depth':c['depth'],'siblings':c['siblings']})
 ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
 lines(ROOT/'cases.jsonl',dataset);lines(ROOT/'exclusions.jsonl',excluded);lines(ROOT/'threshold_audit.jsonl',audit)
 lines(OUT/'selection.jsonl',[{k:c[k] for k in ['id','tree','split','source_row']} for c in selected]);dump(OUT/'split.json',{'seed':0,'dev_trees':sorted(dev),'test_trees':sorted(set(trees)-dev),'tree_definition':'MedGUIDE disease entry; not independently verified full NCCN tree'})
 review=sorted([r for r in dataset if r['seed']==0],key=lambda r:rank(r['id']+r['variant']))[:max(150,int(len(selected)*.1))]
 for name in ['A','B']:
  lines(ROOT/f'annotation_{name}.jsonl',[dict(r,review={'source_alignment':None,'condition_applicable':None,'counterfactual_valid':None,'gold_correct':None,'reason':''}) for r in review])
 summary={'rows':len(rows),'trees':len(trees),'eligible':len(candidates),'selected':dict(collections.Counter(c['split'] for c in selected)),'forms':dict(collections.Counter(r['variant'] for r in dataset)),'CS_verified_source_cases':len(csmap),'exclusions':dict(collections.Counter(r['reason'] for r in excluded)),'human_review':'pending','closed_book':'pending','atoms':'numeric threshold proof only; full gold atoms pending','status':'partial_E0_not_passed','source_sha256':{n:sha(src/n) for n in ['train.parquet','all_paths.json']},'data_sha256':sha(ROOT/'cases.jsonl'),'review_rows_each':len(review),'limitations':['all published paths retained; source path gold is not independently verified patient applicability','CS source-conditioned traversal excludes unresolved downstream branches','AC excludes every same-parent sibling action','option set expanded identically across variants; letters randomized, not guaranteed exact balance','decoy chosen deterministically before closed-book inference']}
 dump(OUT/'E0.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
