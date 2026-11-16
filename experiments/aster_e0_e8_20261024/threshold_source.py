"""Deterministic tree-stratified cohort and patient-conditioned threshold execution."""
import argparse,collections,hashlib,json,random,re
from pathlib import Path
SEED=20261023
ATOM=re.compile(r'([<>≤≥]=?)\s*(\d+(?:\.\d+)?)\s*(mm|cm|mo|months?|years?|y|%)?')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def atom(s):
 m=list(ATOM.finditer(s))
 if len(m)!=1:return None
 m=m[0];return dict(op=m[1],threshold=float(m[2]),unit=m[3] or '',span=list(m.span()),template=s[:m.start()]+'{}'+s[m.end():])
def evaluate(op,x,t):return {'<':x<t,'>':x>t,'≤':x<=t,'≥':x>=t,'<=':x<=t,'>=':x>=t}[op]
def complementary(a,b):return {a,b} in ({'<','≥'},{'≤','>'},{'<','>='},{'<=','>'})
def values(profile,a):
 unit=a['unit'];text=profile.lower();template=a['template'].lower()
 if unit in ('mo','month','months'):
  if 'psadt' in template:pat=r'(?:psa doubling time|psadt)(?:\s+of|\s+is|\s+was|\s+approximately|\s*:)*\s*(\d+(?:\.\d+)?)\s*months?'
  else:
   # Explicit response/recurrence interval; never substitute age or surveillance duration.
   pat=r'(\d+(?:\.\d+)?)\s*months?\s+(?:after|following)\s+(?:(?:completing|completion of|finishing|his|her|the|primary|initial|therapy|treatment|chemotherapy)\b[ ,]*)+'
 elif unit in ('mm','cm'):
  pat=r'(\d+(?:\.\d+)?)\s*[- ]?'+unit+r'\b'
 else:return []
 hits=[]
 for m in re.finditer(pat,text):
  if unit in ('mm','cm'):
   context=text[max(0,m.start()-85):min(len(text),m.end()+50)]
   if not re.search(r'nodule|lesion|mass|tumou?r|metasta|margin',context):continue
  hits.append({'value':float(m[1]),'evidence':profile[m.start():m.end()],'span':[m.start(),m.end()]})
 return list({h['value']:h for h in hits}.values())
def execute(paths,source_path,depth,pair,value,threshold):
 """Traverse changed scalar branch, conditioning other decisions on source path facts.
 No destination facts are invented: a different downstream condition stays unresolved.
 """
 facts=set(source_path[:-1]);prefix=source_path[:depth];active=[]
 for p in paths:
  if p[:depth]!=prefix or len(p)<=depth or p[depth] not in pair:continue
  spec=atom(p[depth])
  if evaluate(spec['op'],value,threshold) and all(n in facts for n in p[depth+1:-1]):active.append(p)
 actions=sorted({p[-1] for p in active})
 return {'paths':active,'actions':actions,'action':actions[0] if len(actions)==1 else None}
def build(rows,paths):
 inventory=[];pairs=[]
 for tree,ps in paths.items():
  nodes=sorted({(tuple(p[:j]),p[j]) for p in ps for j in range(len(p)-1)})
  for prefix,node in nodes:
   a=atom(node)
   if ATOM.search(node) or re.search(r'\d+\s*(?:[–-]\s*\d+\s*)?(?:mo\b|months?\b|years?\b|mm\b|cm\b|wks?\b)',node):inventory.append({'tree':tree,'prefix':list(prefix),'node':node,'atoms':[m.group(0) for m in ATOM.finditer(node)],'executable_scalar':a})
  for i,(prefix,node) in enumerate(nodes):
   a=atom(node)
   if not a:continue
   for otherprefix,other in nodes[i+1:]:
    b=atom(other)
    if b and prefix==otherprefix and a['template']==b['template'] and a['unit']==b['unit'] and a['threshold']==b['threshold'] and complementary(a['op'],b['op']):
     pairs.append({'tree':tree,'prefix':list(prefix),'nodes':[node,other],'threshold':a['threshold'],'unit':a['unit']})
 cs=[];audit=[]
 bytree=collections.defaultdict(list)
 for p in pairs:bytree[p['tree']].append(p)
 for i,r in enumerate(rows):
  for pair in bytree[r['disease']]:
   j=len(pair['prefix']);path=r['path']
   if path[:j]!=pair['prefix'] or path[j] not in pair['nodes']:continue
   a=atom(path[j]);hits=values(r['profile'],a);reason=None
   if len(hits)!=1:reason='no unique explicit patient scalar'
   else:
    hit=hits[0];x=hit['value'];t=a['threshold']
    if not evaluate(a['op'],x,t):reason='patient scalar disagrees with published path'
    else:
     delta=max(abs(x)*0.1,0.1);new=x-delta if x<t else x+delta
     if new<=0:new=x/2 if x>0 else delta
     old=execute(paths[r['disease']],path,j,pair['nodes'],x,t)
     changed=execute(paths[r['disease']],path,j,pair['nodes'],x,new)
     control=execute(paths[r['disease']],path,j,pair['nodes'],x,(t+x)/2)
     if old['action']!=r['answer_text']:reason='original action not uniquely recovered'
     elif changed['action'] is None:reason='destination downstream facts or unique action unavailable'
     else:
      labels=[chr(65+k) for k,v in enumerate(r['options']) if v==changed['action']]
      if len(labels)!=1:reason='destination action absent from original options'
      else:
       rewritten={n:n[:atom(n)['span'][0]]+atom(n)['op']+f'{new:g}'+(' '+a['unit'] if a['unit'] else '')+n[atom(n)['span'][1]:] for n in pair['nodes']}
       cs.append({'id':f'medguide_{i:05d}','source_row':i,'tree':r['disease'],'source_path':path,'depth':j,'nodes':pair['nodes'],'rewritten_nodes':rewritten,'patient_value':x,'unit':a['unit'],'evidence':hit,'old_threshold':t,'new_threshold':new,'preserving_threshold':(t+x)/2,'old_gold':r['answer'],'new_gold':labels[0],'label_flipped':labels[0]!=r['answer'],'branch_flipped':not evaluate(a['op'],x,new),'preserving_action':control['action'],'preserving_label_ok':control['action']==r['answer_text'],'old_execution':old,'new_execution':changed,'synthetic':True,'execution_scope':'source path facts condition all unchanged nodes; unresolved alternative subtree is excluded'})
   audit.append({'id':f'medguide_{i:05d}','tree':r['disease'],'nodes':pair['nodes'],'reason':reason or 'eligible'})
 for r in cs:
  profile=rows[r['source_row']]['profile'];start=r['evidence']['span'][0]
  r['patient_point_value_verified']=not bool(re.search(r'(?:less than|greater than|more than|at least|at most|up to|under|over|greater than or equal to|less than or equal to|[<>≤≥])\s*$',profile[max(0,start-40):start],re.I))
  if not r['patient_point_value_verified']:
   for item in audit:
    if item['id']==r['id'] and item['nodes']==r['nodes']:item['reason']='patient evidence gives a bound rather than a point value'
 return inventory,pairs,cs,audit

def main():
 import pyarrow.parquet as pq
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--private',required=True);p.add_argument('--dev',required=True);p.add_argument('--size',type=int,default=800);a=p.parse_args()
 src=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);private=Path(a.private);private.mkdir(parents=True,exist_ok=True)
 rows=pq.read_table(src/'train.parquet').to_pylist();paths=json.loads((src/'all_paths.json').read_text());inventory,pairs,cs,audit=build(rows,paths)
 dev={json.loads(s)['id'] for s in Path(a.dev).read_text().splitlines()};write(out/'dev_exclusions.jsonl',[{'id':i} for i in sorted(dev)]);target={r['source_row'] for r in cs};rng=random.Random(SEED);groups={}
 for tree in paths:
  ids=[i for i,r in enumerate(rows) if r['disease']==tree and f'medguide_{i:05d}' not in dev];rng.shuffle(ids);ids.sort(key=lambda i:i not in target);groups[tree]=ids
 selected=sorted(i for i in target if f'medguide_{i:05d}' not in dev)
 for tree in groups:groups[tree]=[i for i in groups[tree] if i not in selected]
 while len(selected)<a.size:
  progressed=False
  for tree,ids in groups.items():
   if ids and len(selected)<a.size:selected.append(ids.pop(0));progressed=True
  if not progressed:break
 cohort=[{'id':f'medguide_{i:05d}','source_row':i,'tree':rows[i]['disease'],'original_answer':rows[i]['answer'],'options':{chr(65+k):v for k,v in enumerate(rows[i]['options'])},'profile':rows[i]['profile'],'path':rows[i]['path'],'closed_book_prompt':'Patient:\n'+rows[i]['profile'].strip()+'\n\nWhat is the appropriate next step?\nOptions:\n'+'\n'.join(f'{chr(65+k)}: {v}' for k,v in enumerate(rows[i]['options']))+'\nReturn only one option letter.'} for i in selected]
 write(private/'cohort.jsonl',cohort)
 write(out/'selection.jsonl',[{k:r[k] for k in ('id','source_row','tree','original_answer')} for r in cohort]);write(out/'threshold_nodes.jsonl',inventory);write(out/'threshold_pairs.jsonl',pairs);write(out/'cs_candidates.jsonl',cs);write(out/'cs_all.jsonl',[r for r in cs if r['patient_point_value_verified']]);write(out/'cs_selected.jsonl',[r for r in cs if r['source_row'] in selected and r['patient_point_value_verified']]);write(out/'cs_audit.jsonl',audit)
 m={'base_head':'3e4fac744bf63564cd32045f15c102390ac57c28','seed':SEED,'source_rows':len(rows),'trees':len(paths),'selected':len(selected),'trees_without_selected_rows':[t for t in paths if not any(r['tree']==t for r in cohort)],'source_tree_counts':dict(collections.Counter(r['disease'] for r in rows)),'selected_tree_counts':dict(collections.Counter(r['tree'] for r in cohort)),'dev_excluded':len(dev),'threshold_nodes':len(inventory),'complementary_pairs':len(pairs),'cs_candidates':len(cs),'cs_all':sum(r['patient_point_value_verified'] for r in cs),'cs_selected':sum(r['source_row'] in selected and r['patient_point_value_verified'] for r in cs),'cs_reasons':dict(collections.Counter(r['reason'] for r in audit)),'source_hashes':{f:sha(src/f) for f in ('train.parquet','all_paths.json')},'cohort_sha256':sha(private/'cohort.jsonl'),'sampling':'include threshold CS candidates before point-value validation, then round-robin by tree with seeded order; no model outputs used','K_definition':'Qwen3-8B parsed greedy closed-book answer equals original answer; pending inference'}
 (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));print(json.dumps(m))
if __name__=='__main__':main()
